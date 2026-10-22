"""AFD-TP3 partial sum: the two GX ranks' routed-expert slices.

The routed experts of every main layer are column-split over three ranks
(``2304 = 576 + 864 + 864``).  P7 runs its own 576 columns through vLLM's
kernel; the other two slices run on the GX pair.  Every expert kernel is linear
in the intermediate dimension, so each rank returns a partial hidden state and
the layer output is their sum.

Two calls per main layer, deliberately split (see :mod:`afd_session`):

``post``    after routing, before the local kernel.  The activation and the
            routing leave for both ranks while the local 576-wide GEMMs run.
``collect`` once the local routed output exists, inside
            ``MoERunner._maybe_apply_routed_scale_to_output``.  That hook runs
            after the output is finalized (shared-expert overlap, padding
            trimming, latent transform inputs) and before ``routed_scaling_factor``
            is applied, so the remote partials are added *unscaled* and the
            single 1.5 multiply covers all three slices.

The wire engine transfers a fixed ``rows`` per layer, so a step with fewer
tokens zero-pads; padded rows carry expert id 0 with weight 0 and contribute
nothing.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.afd")

_DUMP_LAYERS = (0, 1, 2, 3, 14, 30)

_T0 = time.perf_counter

# AFD_NVTX=1 brackets post/collect phases in NVTX ranges for nsys. Unset, every branch is dead.
_NVTX = os.environ.get("AFD_NVTX") == "1"
if _NVTX:
    import torch.cuda.nvtx as _nvtx

# AFD_PROF_CTL=<file>: a daemon thread polls the file every 20 ms; "start"/"stop" drive
# cudaProfilerStart/Stop. Pairs with `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat`.
# Unset (or AFD_NVTX unset), no thread exists.
_PROF_CTL = os.environ.get("AFD_PROF_CTL") if _NVTX else None


def _prof_thread() -> None:
    on = False
    cr = torch.cuda.cudart()
    while True:
        try:
            with open(_PROF_CTL) as f:
                want = f.read().strip() == "start"
        except OSError:
            want = False
        if want != on:
            cr.cudaProfilerStart() if want else cr.cudaProfilerStop()
            on = want
            logger.warning("AFD profiler %s", "start" if want else "stop")
        time.sleep(0.02)


_prof_started = False


def _start_prof_thread() -> None:
    # Started from the first post, i.e. only in the process that runs the model
    # (EngineCore). The API server imports this module too but never posts.
    global _prof_started
    if _PROF_CTL is None or _prof_started:
        return
    _prof_started = True
    import threading

    threading.Thread(target=_prof_thread, daemon=True, name="afd-prof").start()


class Bridge:
    """Two transport peers, one sequence number per step."""

    def __init__(self) -> None:
        with open(os.environ.get("AFD_CONFIG", "/opt/afd/afd.json")) as f:
            cfg = json.load(f)
        self.layers = int(cfg.get("layers", 40))
        self.rows = int(os.environ.get("AFD_ROWS", cfg.get("rows", 1)))
        self.peer_cfg = cfg["peers"]
        self.sessions: list = []
        self._pool: ThreadPoolExecutor | None = None
        self._stream = None
        self._x = None
        self._ids = None
        self._tw = None
        self._pending: set[int] = set()
        self._warned = False
        # Diagnostics: AFD_TIMING prints the step's wire cost split per phase,
        # AFD_DEBUG dumps one layer's partial norms and routing.  Both are off in
        # production; a step line costs two perf_counter reads per layer.
        self._timing = os.environ.get("AFD_TIMING") == "1"
        self._debug = int(os.environ.get("AFD_DEBUG", "0"))
        # AFD_DUMP=<host dir> writes the exact inputs and received partials of
        # the first step's traced layers, so an offline run over the
        # un-sliced checkpoint can be compared against the wire's arithmetic.
        self._dump = os.environ.get("AFD_DUMP") or None
        self._t_post = 0.0
        self._t_wait = [0.0] * 8
        self._t_fetch = [0.0] * 8
        # Per-step shape and local-compute accounting: the step line is what
        # says whether a layer is wire-bound or bound by the attention/MoE work
        # this side does while the ranks compute.
        self._t_local = 0.0
        self._t_last = 0.0
        # The other half of a layer's period: from this layer's collect to the
        # next post. That interval is the model's own attention and router on
        # this side -- the part AFD cannot shorten, and the yardstick the wire
        # cost has to be judged against.
        self._t_gap = 0.0
        self._t_after = 0.0
        self._step_rows: set[int] = set()
        self._step_layers = 0
        self._step = 0

    # -- setup -------------------------------------------------------------
    def _connect(self, x: torch.Tensor) -> None:
        from afd_session import K, TOPK, PeerSession

        device = x.device.index or 0
        for p in self.peer_cfg:
            self.sessions.append(
                PeerSession(
                    p["dev"],
                    int(p["gid"]),
                    p["ip"],
                    int(p["port"]),
                    layers=self.layers,
                    rows=self.rows,
                    iters=0,
                    device=device,
                )
            )
        # Each peer owns its own QP, registered slab and CUDA streams, and the
        # C++ post/wait/fetch release the GIL, so the peers' wire time overlaps
        # instead of adding up: at 4096 rows the two posts are ~15 ms serialized
        # against the worker's 26 ms.
        #
        # Everything below is staging, written from the wire threads and from
        # the engine's forward, and vLLM runs that forward under
        # ``torch.inference_mode``. Allocating here would inherit the mode and
        # make these inference tensors, which only the calling thread may then
        # write, so the allocation is taken out of the mode explicitly.
        with torch.inference_mode(False):
            self._pool = ThreadPoolExecutor(
                max_workers=max(1, len(self.sessions)), thread_name_prefix="afd-wire"
            )
            self._x = torch.zeros(self.rows, K, dtype=torch.bfloat16, device=x.device)
            self._ids = torch.zeros(self.rows, TOPK, dtype=torch.int32, device=x.device)
            self._tw = torch.zeros(self.rows, TOPK, dtype=torch.float32, device=x.device)
        logger.warning(
            "AFD: transport up to %d ranks, rows=%d layers=%d",
            len(self.sessions),
            self.rows,
            self.layers,
        )

    # -- per layer ---------------------------------------------------------
    def post(self, layer: int, x: torch.Tensor, tw: torch.Tensor, ids: torch.Tensor) -> None:
        if self._timing and self._t_after:
            self._t_gap += _T0() - self._t_after
        if not self.sessions:
            self._connect(x)
        if layer == 0:
            _start_prof_thread()
            # begin_round re-binds each peer's staging stream, and the fetch
            # H2D runs in a pool thread, where torch's current stream is the
            # legacy default one. Pin the calling stream so the received rows
            # land ordered with the kernels that consume them.
            self._stream = torch.cuda.current_stream(x.device)
            for s in self.sessions:
                s.begin_round()
        t = x.shape[0]
        if t > self.rows:
            # `rows` is the wire's capacity (AFD_ROWS, and the GX worker's
            # --rows). The row count itself is per post, so a prefill chunk up
            # to the capacity rides the same slot as a one-row decode step.
            raise RuntimeError(f"AFD: step of {t} tokens exceeds capacity rows={self.rows}")
        if x.shape[-1] != self._x.shape[-1]:
            # A backend may pad the hidden dim for its kernel; the padding is
            # tail zeros, so the wire carries the first HIDDEN columns.
            if x.shape[-1] < self._x.shape[-1]:
                raise RuntimeError(
                    f"AFD: hidden {x.shape[-1]} < {self._x.shape[-1]}"
                )
            x = x[..., : self._x.shape[-1]]
        # The peer copies the rows straight from here into the registered slab,
        # so staging through ``_x`` first would move the same 40 MiB to the host
        # twice (measured 7.4 ms/layer at 4096 rows, half of it redundant). The
        # staging buffer is only needed when the tensor cannot be handed over
        # as-is: a non-contiguous view, or a debug/dump run that reads it back.
        src = x if x.is_contiguous() and not (self._dump or self._debug) else None
        if src is None:
            self._x[:t].copy_(x)
            src = self._x[:t]
        self._ids[:t].copy_(ids)
        self._tw[:t].copy_(tw)
        t0 = _T0()
        if _NVTX:
            _nvtx.range_push(f"post_L{layer}")
        list(self._pool.map(
            lambda s: s.post(layer, src, self._ids, self._tw, n=t), self.sessions
        ))
        if _NVTX:
            _nvtx.range_pop()
        if self._timing:
            self._t_post += _T0() - t0
            self._step_rows.add(t)
            self._step_layers += 1
            self._t_last = _T0()
        self._pending.add(layer)

    def collect(self, layer: int, out: torch.Tensor) -> None:
        """Add both remote partials to ``out[tokens]`` in fp32."""
        if layer not in self._pending:
            if not self._warned:
                self._warned = True
                logger.warning("AFD: layer %d has no posted step; add skipped", layer)
            return
        self._pending.discard(layer)
        if self._timing:
            # Everything this side did between handing the layer over and
            # asking for the result: attention, the local expert slice, the
            # launch of the next layer. Wire time is invisible here.
            self._t_local += _T0() - self._t_last
        t = out.shape[0]
        # The accumulation buffer is `out` itself when it is already fp32 (the
        # routed seam hands over fp32); otherwise `.float()` made a copy, so the
        # partials have to be copied back or the remote ranks contribute nothing.
        acc = out.float()
        write_back = acc.data_ptr() != out.data_ptr()
        if acc.shape[-1] != self._x.shape[-1]:
            if acc.shape[-1] < self._x.shape[-1]:
                raise RuntimeError(f"AFD: output width {acc.shape[-1]} too narrow")
            out = out[..., : self._x.shape[-1]]
            acc = acc[..., : self._x.shape[-1]]
        trace = self._debug > 0 and layer in _DUMP_LAYERS and self._step < 2
        dump = self._dump is not None and layer in _DUMP_LAYERS and self._step < 8
        if trace or dump:
            xf = self._x[:t].float()
            loc = acc.clone()

        stream = self._stream

        def fetch(i: int, s) -> torch.Tensor:
            t0 = _T0()
            if _NVTX:
                _nvtx.range_push(f"wait_gx{i}")
            s.wait(layer)
            if _NVTX:
                _nvtx.range_pop()
            t1 = _T0()
            if _NVTX:
                _nvtx.range_push(f"fetch_gx{i}")
            with torch.cuda.stream(stream):
                part = s.fetch(layer, t).float()
            if _NVTX:
                _nvtx.range_pop()
            if self._timing:
                self._t_wait[i] += t1 - t0
                self._t_fetch[i] += _T0() - t1
            return part

        if _NVTX:
            _nvtx.range_push(f"collect_L{layer}")
        parts = list(self._pool.map(lambda a: fetch(*a), enumerate(self.sessions)))
        if _NVTX:
            _nvtx.range_pop()
            _nvtx.range_push("combine")
        for part in parts:
            acc += part
        if write_back:
            out.copy_(acc)
        if _NVTX:
            _nvtx.range_pop()
        if self._timing:
            self._t_after = _T0()
        if dump:
            import numpy as np

            np.savez(
                f"{self._dump}/step{self._step}_L{layer}.npz",
                x=self._x[:t].float().cpu().numpy(),
                ids=self._ids[:t].cpu().numpy(),
                tw=self._tw[:t].cpu().numpy(),
                local=loc.cpu().numpy(),
                **{f"p{i}": p.cpu().numpy() for i, p in enumerate(parts)},
            )
            logger.warning("AFD dump: %s/step%d_L%d.npz", self._dump, self._step, layer)
        if trace:
            ref = out.float()
            p1 = parts[0] if parts else None
            cos = []
            cos.append(float((p1 * ref).sum() / (p1.norm() * ref.norm()).clamp_min(1e-9)))
            if len(parts) > 1:
                p2 = parts[1]
                cos.append(float((p2 * ref).sum() / (p2.norm() * ref.norm()).clamp_min(1e-9)))
                cos.append(float((p2 * p1).sum() / (p1.norm() * p2.norm()).clamp_min(1e-9)))
            logger.warning(
                "AFD dbg step=%d L=%d |x|=%.4f xsum=%.4f |local|=%.4f parts=%s "
                "cos(l,p1|p2,p1p2)=%s ids=%s tw=%s",
                self._step, layer, float(xf.norm()), float(xf.sum()), float(ref.norm()),
                " ".join(f"{float(v.norm()):.4f}" for v in parts),
                " ".join(f"{v:.3f}" for v in cos),
                self._ids[:t].tolist(),
                [round(v, 4) for v in self._tw[:t].reshape(-1).tolist()],
            )
        if self._timing and layer == self.layers - 1:
            # One layer's period = local_before (attention + router, this side
            # only) + post + wait/fetch. `local_between` is what the seam costs
            # this side after the hand-off, so the wire's share of the period is
            # post + wait + fetch - local_between.
            logger.warning(
                "AFD step %d: rows=%s layers=%d post %.2f ms local_between %.2f ms "
                "local_before %.2f ms | %s ms",
                self._step, sorted(self._step_rows), self._step_layers,
                self._t_post * 1e3, self._t_local * 1e3, self._t_gap * 1e3,
                " ".join(
                    f"r{i} wait {self._t_wait[i] * 1e3:.2f} fetch {self._t_fetch[i] * 1e3:.2f}"
                    for i in range(len(self.sessions))
                ),
            )
            self._t_post = 0.0
            self._t_local = 0.0
            self._t_gap = 0.0
            self._t_wait = [0.0] * 8
            self._t_fetch = [0.0] * 8
            self._step_rows = set()
            self._step_layers = 0
        if layer == self.layers - 1:
            self._step += 1

    def close(self) -> None:
        for s in self.sessions:
            s.close()
        self.sessions.clear()


_bridge: Bridge | None = None


def bridge() -> Bridge:
    global _bridge
    if _bridge is None:
        _bridge = Bridge()
    return _bridge
