#!/usr/bin/env python3
"""W4: GX expert worker -- NVFP4 grouped-MoE arena and CUDA-graph step.

Runs one process per GX host. Holds 40 layers of this host's column slice of the
routed experts (C = 864 of 2304 intermediate channels) and executes, per layer,
the grouped NVFP4 MoE for a row count of R <= GRAPH_MAX rows (exactly 6, the
power-of-two buckets otherwise), returning the partial
hidden state that P7 sums across the three AFD-TP ranks.

Layout facts this file is built on (validated 2026-10-20 on GX1, GB10 sm_121):

  1. cutlass_fp4_moe_mm (grouped) requires the *contraction* dimension of each
     GEMM to be a multiple of 128. If it is not, every expert after the first
     returns exactly zero with no error and no diagnostic.
        stage 1 contraction = K = 5120          (already a multiple of 128)
        stage 2 contraction = C = 864           (NOT -> must become 896)
     The fused helper derives both GEMM shapes from a single `n`, so n = CP = 896
     and the stage-1 output width follows as 2*CP = 1792.
  2. Gate and up halves are separated at CP: rows [0, CP) are gate, rows
     [CP, 2*CP) are up. Padding 1728 -> 1792 therefore inserts 32 zero rows
     *between* the halves, not at the end. The same rule applies to the scales.
  3. alpha = weight_global_scale / activation_global_scale. With the pack's
     convention (dequant = code * block_scale * weight_global_scale,
     activation dequant = code * block_scale / act_global_scale,
     act_global_scale = 6 / max|x|) that is the only consistent value;
     1/(a_gs*g) is wrong by g^2.
  4. Scales are linear on disk ([E, 2*C, K/16]); the kernel wants them swizzled.

Verified against a per-expert reference on real pack bytes: cos = 0.999997.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import struct
import time

import torch
from safetensors import safe_open

from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import cutlass_moe as cm
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)

K = 5120          # hidden
E = 384           # routed experts
TOPK = 6
C = 864           # this host's column slice of moe_intermediate_size = 2304
CP = 896          # C rounded up to a multiple of 128
NLAYERS = 40
CHUNK = 64        # experts per host->device staging copy


def log(msg: str) -> None:
    print(f"[gx-worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# GX_NVTX=1 brackets the serve loop's phases (flag_wait, rout_stage, load_x_h2d, replay, sync_d2h, out_flag)
# in NVTX ranges. GX_PROF_CTL=<file> ("start"/"stop") drives cudaProfilerStart/Stop from a daemon thread
# polling the file every 20 ms. Unset, every branch is dead.
_NVTX = os.environ.get("GX_NVTX") == "1"
if _NVTX:
    import torch.cuda.nvtx as _nvtx
_PROF_CTL = os.environ.get("GX_PROF_CTL") if _NVTX else None


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
            log(f"profiler {'start' if want else 'stop'}")
        time.sleep(0.02)


if _PROF_CTL is not None:
    import threading

    threading.Thread(target=_prof_thread, daemon=True, name="gx-prof").start()


def swiglu(h: torch.Tensor, cp: int) -> torch.Tensor:
    """Reference SwiGLU, gate first, then up.

    Unclamped on purpose: the serving path is ``run_cutlass_moe_fp4``, whose
    fused SiLU+NVFP4-quant op has no clamp argument, so this reference and the
    ``act_amax`` calibration sweep must quantize exactly what the kernel
    computes. Applying the checkpoint's ``swiglu_limit`` here would desync the
    scales from the served activations; it needs a clamp-capable fused kernel
    (see PLAN W6, correctness).
    """
    gate = h[:, :cp].float()
    up = h[:, cp:].float()
    return torch.nn.functional.silu(gate) * up


def mem_report() -> str:
    free, total = torch.cuda.mem_get_info()
    return f"free={free / 2**30:.1f}GiB total={total / 2**30:.1f}GiB"


def pack_identity(pack: str) -> str:
    """Name the column slice a pack holds, by content.

    Both GX packs are 864 columns wide and the same byte size, so neither the
    file size nor the manifest declares which rank a pack belongs to -- only
    its bytes do.  Two workers once ran the *same* slice for hours because a
    copy went to the wrong host; the payload hash below makes that mismatch
    visible on one line of the startup log.
    """
    split = "?"
    try:
        with open(os.path.join(pack, "MANIFEST.json")) as f:
            split = json.load(f).get("split", "?")
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(pack, "experts-L00.safetensors"), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            name = next(k for k in hdr if k.endswith(".w2.weight"))
            f.seek(8 + n + hdr[name]["data_offsets"][0])
            fp = hashlib.sha256(f.read(1 << 16)).hexdigest()[:16]
    except (OSError, ValueError, KeyError, StopIteration) as e:
        return f"{pack}: split={split} unreadable ({e})"
    return f"{pack}: split={split} {name} sha256[:16]={fp}"


class Arena:
    """Resident NVFP4 expert arena: padded codes, swizzled block scales."""

    def __init__(self, device: torch.device, layers: int = NLAYERS, cp: int = CP):
        self.dev = device
        self.cp = cp
        self.layers = layers
        self.w13: list[torch.Tensor | None] = [None] * layers
        self.s13: list[torch.Tensor | None] = [None] * layers
        self.w2: list[torch.Tensor | None] = [None] * layers
        self.s2: list[torch.Tensor | None] = [None] * layers
        self.g13: list[torch.Tensor | None] = [None] * layers
        self.g2: list[torch.Tensor | None] = [None] * layers
        self.a2: list[torch.Tensor | None] = [None] * layers

    # -- size accounting ---------------------------------------------------
    def bytes_per_layer(self) -> int:
        cp = self.cp
        return (
            E * 2 * cp * (K // 2)          # w13 codes
            + E * 2 * cp * (K // 16)       # w13 scales (padded to 2*cp rows)
            + E * K * (cp // 2)            # w2 codes, K padded along the slice
            + E * K * (cp // 16)           # w2 scales
            + 4 * E * 2                    # two fp32 global scales
        )

    def alloc_layer(self, li: int) -> None:
        d, cp = self.dev, self.cp
        self.w13[li] = torch.zeros(E, 2 * cp, K // 2, dtype=torch.uint8, device=d)
        self.w2[li] = torch.zeros(E, K, cp // 2, dtype=torch.uint8, device=d)
        # scales are produced by swizzle_blockscale below; allocate the arena
        # slots lazily there so no unswizzled copy is ever resident.
        self.g13[li] = torch.zeros(E, dtype=torch.float32, device=d)
        self.g2[li] = torch.zeros(E, dtype=torch.float32, device=d)

    def load_layer(self, li: int, pack: str) -> None:
        d, cp = self.dev, self.cp
        path = os.path.join(pack, f"experts-L{li:02d}.safetensors")
        self.alloc_layer(li)
        f = safe_open(path, framework="pt", device="cpu")

        c13 = f.get_slice(f"L{li}.w13.weight")          # [E, 2C, K/2] uint8
        c2 = f.get_slice(f"L{li}.w2.weight")            # [E, K, C/2] uint8
        s13l = f.get_slice(f"L{li}.w13.weight_scale")   # [E, 2C, K/16] fp8
        s2l = f.get_slice(f"L{li}.w2.weight_scale")     # [E, K, C/16] fp8
        lin13 = torch.zeros(E, 2 * cp, K // 16, dtype=torch.float8_e4m3fn, device=d)
        lin2 = torch.zeros(E, K, cp // 16, dtype=torch.float8_e4m3fn, device=d)

        for e0 in range(0, E, CHUNK):
            e1 = min(e0 + CHUNK, E)
            src = c13[e0:e1].to(d)
            dst = self.w13[li][e0:e1]
            dst[:, :C].copy_(src[:, :C])          # gate half
            dst[:, cp : cp + C].copy_(src[:, C:])  # up half, after the padding gap
            del src, dst

            src = c2[e0:e1].to(d)
            self.w2[li][e0:e1, :, : C // 2].copy_(src)
            del src

            src = s13l[e0:e1].to(d)
            lin13[e0:e1, :C].copy_(src[:, :C])
            lin13[e0:e1, cp : cp + C].copy_(src[:, C:])
            del src

            src = s2l[e0:e1].to(d)
            lin2[e0:e1, :, : C // 16].copy_(src)
            del src

        self.s13[li] = swizzle_blockscale(lin13)
        del lin13
        self.s2[li] = swizzle_blockscale(lin2)
        del lin2

        self.g13[li].copy_(f.get_tensor(f"L{li}.w13.weight_scale_2").to(d))
        self.g2[li].copy_(f.get_tensor(f"L{li}.w2.weight_scale_2").to(d))
        del f
        # The pack is mmap'd by safetensors and read once. Those pages stay in
        # page cache, and on GB10 page cache and the arena draw on the same
        # 121.6 GiB, so without an eviction the arena cannot be filled.
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def load(self, pack: str, layers: int) -> None:
        need = layers * self.bytes_per_layer()
        free, total = torch.cuda.mem_get_info()
        log(
            f"arena needs {need / 2**30:.1f}GiB, gpu free {free / 2**30:.1f}GiB "
            f"of {total / 2**30:.1f}GiB"
        )
        if need > free:
            raise SystemExit(
                f"refusing to start: arena {need / 2**30:.1f}GiB > free "
                f"{free / 2**30:.1f}GiB ({mem_report()})"
            )
        t0 = time.time()
        for li in range(layers):
            self.load_layer(li, pack)
            done = (li + 1) * self.bytes_per_layer() / 2**30
            log(f"layer {li:02d} resident, {done:.1f}GiB, {mem_report()}")
        log(f"pack {pack_identity(pack)}")
        log(f"arena loaded in {time.time() - t0:.1f}s")

    # -- calibration -------------------------------------------------------
    def calibrate_a2(self, sigma: float = 1.0, safety: float = 1.5) -> None:
        """Per-layer activation global scale for the SwiGLU intermediate.

        The fused kernel needs a2_gscale up front, so it cannot be derived from
        the intermediate it is about to produce. Estimate each layer's
        max|silu(gate)*up| on a synthetic residual-stream draw and divide by the
        safety factor: a larger assumed max (smaller gscale) only costs code
        resolution, a smaller one clips.
        """
        d = self.dev
        g = torch.Generator(device="cpu").manual_seed(1234)
        x = (torch.randn(8, K, generator=g) * sigma).to(d).to(torch.bfloat16)
        ids = torch.arange(TOPK, device=d, dtype=torch.int32).repeat(8, 1)
        for li in range(self.layers):
            if self.w13[li] is None:   # partially loaded arena (wire check)
                continue
            ags = (6.0 / x.abs().max().clamp_min(1e-6)).float()
            xq, xsf = scaled_fp4_quant(x, ags.view(1).contiguous(), True)
            a2 = torch.zeros(E, dtype=torch.float32, device=d)
            amax = 0.0
            for e in range(TOPK):
                h = cutlass_scaled_fp4_mm(
                    xq,
                    self.w13[li][e],
                    xsf,
                    self.s13[li][e],
                    self.g13[li][e] / ags,
                    torch.bfloat16,
                ).float()
                act = swiglu(h, self.cp)
                amax = max(amax, act.abs().max().item())
            a2.fill_(6.0 / (amax * safety))
            self.a2[li] = a2
        log(f"a2 calibrated (sigma={sigma}, safety={safety})")

    def set_a2(self, li: int, value: float) -> None:
        """Rewrite one layer's scale in place: the captured graphs read this
        buffer on every replay, so a calibration result needs no recapture."""
        if self.a2[li] is None:
            self.a2[li] = torch.full((E,), value, dtype=torch.float32, device=self.dev)
        else:
            self.a2[li].fill_(value)

    def save_a2(self, path: str) -> None:
        vals = {str(li): float(a[0]) for li, a in enumerate(self.a2) if a is not None}
        with open(path, "w") as f:
            json.dump(vals, f)
        log(f"a2 sidecar written: {path} ({len(vals)} layers)")

    def load_a2(self, path: str) -> bool:
        """Take the SwiGLU scales measured on real activations, if on disk."""
        if not os.path.exists(path):
            return False
        with open(path) as f:
            vals = json.load(f)
        n = 0
        if self.w13 is None:
            return False
        for li, v in vals.items():
            L = int(li)
            if L < self.layers and self.w13[L] is not None:
                self.set_a2(L, float(v))
                n += 1
        log(f"a2 sidecar loaded: {path} ({n} layers)")
        return n > 0


class Worker:
    """Grouped-MoE step over the arena, one CUDA graph per (layer, row count)."""

    def __init__(self, arena: Arena, max_rows: int = 8):
        self.a = arena
        self.dev = arena.dev
        self.max_rows = max_rows
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        # One pool for every layer graph: a private pool per (layer, rows) costs
        # tens of MiB each, which at 40 layers is the difference between a
        # resident fleet and an OOM kill. Replays are strictly sequential.
        self.pool = torch.cuda.graph_pool_handle()
        # Activation sets are created per row count on first use, not for every
        # count up to the wire's capacity: one buffer set at 1024 rows is
        # ~140 MiB (ws13 alone is 123 KiB/row), so `for r in range(1, max_rows)`
        # would be quadratic in capacity and OOM the arena on start.
        self.x: dict[int, torch.Tensor] = {}
        self.ids: dict[int, torch.Tensor] = {}
        self.tw: dict[int, torch.Tensor] = {}
        self.a1g: dict[int, torch.Tensor] = {}
        self.out: dict[int, torch.Tensor] = {}
        self.ws13: dict[int, torch.Tensor] = {}
        self.ws2: dict[int, torch.Tensor] = {}

    def bufs(self, r: int) -> None:
        """Allocate the activation set for `r` rows (idempotent)."""
        if r in self.x:
            return
        if r < 1 or r > self.max_rows:
            raise ValueError(f"row count {r} outside 1..{self.max_rows}")
        self.x[r] = torch.zeros(r, K, dtype=torch.bfloat16, device=self.dev)
        self.ids[r] = torch.zeros(r, TOPK, dtype=torch.int32, device=self.dev)
        self.tw[r] = torch.zeros(r, TOPK, dtype=torch.float32, device=self.dev)
        # Per-token activation scale, set from the arriving payload before each
        # replay. It has to be a device buffer that the captured graph reads, not
        # a value computed during capture: capture runs on the still-empty x
        # buffer, so `6/max|x|` there is 6e6, and every replay then quantizes the
        # activation to a saturated sign pattern (measured: output exactly 0).
        self.a1g[r] = torch.full((E,), 6.0, dtype=torch.float32, device=self.dev)
        self.out[r] = torch.zeros(r, K, dtype=torch.bfloat16, device=self.dev)
        self.ws13[r] = torch.zeros(r * TOPK * 2 * K, dtype=torch.bfloat16, device=self.dev)
        self.ws2[r] = torch.zeros(r * TOPK * self.a.cp, dtype=torch.bfloat16, device=self.dev)

    def _launch(self, li: int, r: int) -> None:
        x = self.x[r]
        a1g = self.a1g[r]
        a2g = self.a.a2[li]
        # The per-layer activation scale belongs inside the graph: computed on
        # the host it cost 2.4 ms/layer (a 40 MiB scan of the staged payload)
        # and forced the payload through a pinned staging buffer first. Here it
        # is one reduction over the rows already resident on the device, and it
        # reads whatever `x` holds on *this* replay, so capture-time emptiness
        # no longer matters.
        a1g.fill_(6.0 / x.abs().amax().clamp_min(1e-6))
        cm.run_cutlass_moe_fp4(
            self.out[r],
            x,
            a1g,
            self.a.w13[li],
            self.a.s13[li],
            self.a.g13[li] / a1g,
            a2g,
            self.a.w2[li],
            self.a.s2[li],
            self.a.g2[li] / a2g,
            self.tw[r],
            self.ids[r],
            MoEActivation.SILU,
            self.ws13[r],
            self.ws2[r],
            r,
            self.a.cp,
            K,
            E,
            self.dev,
        )

    def capture(self, li: int, r: int, dst: torch.Tensor | None = None) -> None:
        self.bufs(r)
        # warm the allocator outside capture so the shared pool stays small
        self._launch(li, r)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self._launch(li, r)
            if dst is not None:
                # host mapping registered before capture: the D2H lands in the
                # wire out-slot, so no staging copy and no stream drain per layer
                dst.copy_(self.out[r], non_blocking=True)
        self.graphs[(li, r)] = g

    def step(self, li: int, x: torch.Tensor, ids: torch.Tensor, tw: torch.Tensor) -> torch.Tensor:
        r = x.shape[0]
        self.bufs(r)
        if (li, r) not in self.graphs:
            self.capture(li, r)
        self.x[r].copy_(x)
        self.ids[r].copy_(ids)
        self.tw[r].copy_(tw)
        self.graphs[(li, r)].replay()
        return self.out[r]

    def act_amax(self, li: int, r: int) -> float:
        """max|silu(gate)*up| over this layer's routed rows, on the real payload.

        The fused SiLU+quant path never materializes the intermediate, so
        calibration re-runs GEMM1 for the selected experts only: one row and one
        cutlass call per picked expert, paid for a handful of rounds. The padded
        columns of a pack slice are zero codes, so they contribute exactly 0 to
        the max and need no special case.
        """
        x = self.x[r]
        a1g = self.a1g[r]
        ids = self.ids[r]
        amax = 0.0
        for row in range(r):
            for j in range(TOPK):
                e = int(ids[row, j])
                xq, xsf = scaled_fp4_quant(
                    x[row : row + 1], a1g[e].view(1).contiguous(), True
                )
                h = cutlass_scaled_fp4_mm(
                    xq,
                    self.a.w13[li][e],
                    xsf,
                    self.a.s13[li][e],
                    self.a.g13[li][e] / a1g[e],
                    torch.bfloat16,
                ).float()
                act = swiglu(h, self.a.cp)
                amax = max(amax, act.abs().max().item())
        return amax


def verify(pack: str, li: int = 0) -> None:
    """Arena path vs per-expert reference on real pack bytes."""
    dev = torch.device("cuda:0")
    a = Arena(dev, layers=li + 1)
    a.load_layer(li, pack)
    # calibrate this layer's intermediate scale
    g = torch.Generator(device="cpu").manual_seed(7)
    x = (torch.randn(1, K, generator=g) * 1.0).to(dev).to(torch.bfloat16)
    ids = torch.arange(TOPK, device=dev, dtype=torch.int32).unsqueeze(0)
    tw = torch.full((1, TOPK), 1.0 / TOPK, device=dev, dtype=torch.float32)
    # reference a2 for this x
    ags = (6.0 / x.abs().max().clamp_min(1e-6)).float()
    xq, xsf = scaled_fp4_quant(x, ags.view(1).contiguous(), True)
    per = []
    a2s = torch.zeros(E, device=dev)
    for e in range(TOPK):
        h = cutlass_scaled_fp4_mm(
            xq, a.w13[li][e], xsf, a.s13[li][e], a.g13[li][e] / ags, torch.bfloat16
        ).float()
        act = swiglu(h, CP)
        a2s[e] = 6.0 / act.abs().max().item()
        q, sf = scaled_fp4_quant(act.to(torch.bfloat16), a2s[e].view(1).contiguous(), True)
        per.append(
            cutlass_scaled_fp4_mm(
                q, a.w2[li][e], sf, a.s2[li][e], a.g2[li][e] / a2s[e], torch.bfloat16
            ).float()
        )
    a2f = torch.full((E,), float(a2s[:TOPK].max()), device=dev)
    a2f[:TOPK] = a2s[:TOPK]
    a.a2[li] = a2f
    w = Worker(a)
    out = w.step(li, x, ids, tw).float().clone()
    ref = sum(per) / TOPK
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), 0).item()
    rel = ((out - ref).norm() / ref.norm()).item()
    log(f"verify layer {li}: cos={cos:.6f} rel={rel:.2e} in_norm={x.float().norm():.4f} out_norm={out.norm():.4f}")


def bench(pack: str, layers: int, buckets: list[int], iters: int = 20, load: bool = True) -> None:
    dev = torch.device("cuda:0")
    a = Arena(dev, layers=layers)
    a.load(pack, layers) if load else None
    if load:
        a.calibrate_a2()
    w = Worker(a, max_rows=max(buckets))
    g = torch.Generator(device="cpu").manual_seed(3)
    for r in buckets:
        x = (torch.randn(r, K, generator=g)).to(dev).to(torch.bfloat16)
        ids = torch.randint(0, E, (r, TOPK), generator=g).to(dev).to(torch.int32)
        tw = torch.rand(r, TOPK, generator=g).to(dev)
        tw = (tw / tw.sum(-1, keepdim=True)).to(torch.float32)
        t0 = time.time()
        for li in range(layers):
            w.step(li, x, ids, tw)
        torch.cuda.synchronize()
        cap = time.time() - t0
        for _ in range(3):
            for li in range(layers):
                w.step(li, x, ids, tw)
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            for li in range(layers):
                w.step(li, x, ids, tw)
            torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        per_layer = dt / layers * 1e6
        log(
            f"R={r}: capture {cap:.1f}s  token {dt * 1e3:.2f} ms  "
            f"layer {per_layer:.1f} us  -> {1 / dt:.1f} tok/s expert-side ceiling"
        )



# ---- transport integration ------------------------------------------------
# Layout mirrored from src/qmesh_afd_edge.cpp: the C++ process talks verbs and
# owns nothing else; this process reads the in-flags, runs the layer, raises the
# out-flags in the same /dev/shm pages.
# `--rows` is the wire's row capacity, fixed at open and mirrored as R_MAX in
# src/qmesh_afd_peer.cpp / qmesh_afd_edge.cpp. Each post states its own valid
# row count in a header at the head of its layer's routing slot, so a decode
# step carries one row and a prefill chunk carries the whole batch over the
# same connection.
WIRE_ROWS = 4096
ROUT_HDR = 16     # int32 seq, int32 n, 8 B pad
SLOT = K * 2 * WIRE_ROWS
TRANSPORT_LAYERS = 40
# One in-slot and one out-slot serve every layer, exactly as in the two C++
# sides: the client posts layer L+1 only after it saw L's out-flag, and this
# loop raises that flag after it copied L's payload out of the in-slot, so a
# single pair is enough and keeps the shm region at 2 slots (84 MiB at 4096
# rows) instead of 2*TRANSPORT_LAYERS slots, which no longer fits beside the
# resident expert arena.  The flag plane stays per layer.
SLAB = 2 * SLOT
IN_BASE = 0
OUT_BASE = SLOT
FSTRIDE = 64
FOUT_OFF = TRANSPORT_LAYERS * FSTRIDE
FLAGS = 2 * TRANSPORT_LAYERS * FSTRIDE
SEQ_SER = 100
SEQ_PIPE = 1000
SESSION_OFF = 4   # unused word in layer 0's flag stride: wire-engine epoch
# The worker's half of the connect handshake. A client that connects reads the
# epoch, clears the flag plane, bumps the epoch and then waits for this word to
# carry the new epoch before it posts anything. Words 0 and 1 of layer 0 hold
# its in-flag and the epoch, so word 2 is the first spare one.
ACK_OFF = 8
ROUT_ROW = TOPK * 4 * 2   # ids[n][TOPK] int32 then tw[n][TOPK] f32
ROUT_SLOT = ROUT_HDR + WIRE_ROWS * ROUT_ROW
ROUT = ROUT_SLOT


def routing_fail(seq: int, msg: str) -> None:
    """Fail closed: log why a post cannot be served and raise out of serve().

    A round is answered by releasing a flag, and the ingest's release sits
    inside the graph, so a post must be refused *before* any compute: a partial
    output cannot be taken back once published. Raising ends the process, so
    the client sees a dead peer and fails the request instead of summing
    fabricated numbers.
    """
    import gx_ingest as gxi
    log(f"round {seq}: FATAL {msg}")
    raise gxi.RoutingError(f"round {seq}: {msg}")


def routing_gate(seq: int, layer: int, n: int, ids_np) -> int:
    """Padding lanes in this post; refuse the post if an id names no expert.

    Id -1 is vLLM's unrouted/padded sentinel: a producer warm-up posts nothing
    else, so it is normal traffic, served as expert 0 with weight 0 (a zero
    contribution) and reported as `pad`. Any other id outside [0, E) is
    corruption: `get_cutlass_moe_mm_data` drops such a slot, so its routing-map
    entry is never written and `shuffle_rows` gathers x at whatever integer the
    allocator left there (measured: 6 of 96 slots for one -1 row; a gather at
    one of them is an illegal address). Such a post is refused, not repaired.
    """
    import gx_ingest as gxi
    pad, bad = gxi.classify_ids(ids_np)
    if bad:
        routing_fail(seq, f"layer {layer}: {bad} of {n * TOPK} routing slots name "
                          f"no expert (ids must be -1 or in [0, {gxi.E})); "
                          "refusing to serve a partial result")
    return pad


def serve(shm: str, pack: str, layers: int, rows: int, iters: int, mode: str,
          calib: int = 0, a2_path: str | None = None, ingest: str = "legacy",
          swiglu_limit: float = 10.0, round_graph: bool = False,
          round_rows: int = 0, decode_kernel: str = "clamped") -> None:
    """Serve the RDMA flag protocol with real expert compute.

    One round = `layers` sequential layer activations. The client runs `iters`
    serialized rounds (seq = SEQ_SER + r) and then one pipelined round
    (seq = SEQ_PIPE), so the worker serves `iters + 1` rounds of `layers` layers.
    """
    if rows != WIRE_ROWS:
        # The shm regions are sized from WIRE_ROWS on both sides of the wire.
        raise SystemExit(f"--rows {rows} != wire capacity {WIRE_ROWS}")
    if decode_kernel not in ("clamped", "cuda-swz"):
        raise SystemExit(f"unknown --decode-kernel {decode_kernel!r}")
    if decode_kernel != "clamped":
        # Fail before the arena load rather than after it: this flag picks the
        # graph body, and the two flags it depends on are fixed at startup.
        if ingest != "graph":
            raise SystemExit(
                f"--decode-kernel {decode_kernel} needs --ingest graph: the "
                f"legacy path has no small-bucket graph body to replace")
        if swiglu_limit <= 0:
            raise SystemExit(
                f"--decode-kernel {decode_kernel} needs --swiglu-limit > 0: "
                f"the kernel always applies the checkpoint clamp")
    import ctypes
    import mmap

    import numpy as np

    def region(suffix: str, size: int) -> mmap.mmap:
        # The worker may start first; the wire engine ftruncates to the same
        # size and zeroes the region when it starts, so create-if-absent is safe.
        fd = os.open(shm + suffix, os.O_RDWR | os.O_CREAT, 0o666)
        # The wire engine runs as the host user, the worker as root in the
        # container; umask 022 would otherwise make the region owner-only.
        os.fchmod(fd, 0o666)
        if os.fstat(fd).st_size != size:
            os.ftruncate(fd, size)
        return mmap.mmap(fd, size)

    def pin_region(m: mmap.mmap, size: int, name: str) -> int:
        # The captured graph writes the out-slot straight into this mapping, and
        # a pageable destination is not capturable; registering it also removes
        # the staging copy plus stream drain that a pageable D2H forces.
        # Returns the base address (device address == host address on GB10).
        addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
        rc = torch.cuda.cudart().cudaHostRegister(addr, size, 0)
        log(f"registered {name} {size}B for device access ({rc})")
        return addr

    slab = region(".slab", SLAB)
    flags = region(".flags", FLAGS)
    f32 = np.ndarray((FLAGS // 4,), dtype=np.int32, buffer=flags)

    def pub_ack(e: int) -> None:
        """Publish that this worker reads nothing from round state on epoch `e`.

        The connect handshake waits for this word before its first post, so it
        must only be written once no device graph can still be reading the
        mapped slab or routing. Every caller either sits between rounds or
        synchronizes the device first. Monotonic epochs make a stale value
        distinguishable: the client waits for its own new epoch, not just for
        the word to become nonzero.
        """
        f32[ACK_OFF // 4] = e

    rout = region(".rout", ROUT)

    def rout_count() -> int:
        """Valid rows in the post being served; the client writes it with the routing."""
        hdr = np.ndarray((2,), dtype=np.int32, buffer=rout, offset=0)
        return int(hdr[1])

    def rout_scale() -> float:
        """1/scale of the fp8 payload, or 1.0 for a bf16 post.

        The client halves the payload by sending e4m3 rows under one per-post
        scale; header word 3 tags the format and defaults to 0 (bf16) so an
        older engine's post still reads as bf16 here.
        """
        hdr = np.ndarray((4,), dtype=np.int32, buffer=rout, offset=0)
        inv = np.ndarray((1,), dtype=np.float32, buffer=rout, offset=8)
        return float(inv[0]) if int(hdr[3]) == 1 else 1.0


    def rout_slot(n: int) -> tuple[np.ndarray, np.ndarray]:
        """Router output for the post being served, as the wire engine wrote it."""
        base = ROUT_HDR
        ids = np.ndarray((n, TOPK), dtype=np.int32, buffer=rout, offset=base)
        tw = np.ndarray((n, TOPK), dtype=np.float32, buffer=rout,
                        offset=base + n * TOPK * 4)
        return ids, tw


    def slot_vec(base: int, n: int) -> torch.Tensor:
        """Out-slot rows, bf16: the partial the attention host sums in fp32."""
        a = np.ndarray((n, K), dtype=np.uint16, buffer=slab, offset=base)
        return torch.from_numpy(a).view(torch.bfloat16)


    def slot_x(base: int, n: int) -> torch.Tensor:
        """In-slot rows as the client sent them: e4m3, one byte per element."""
        a = np.ndarray((n, K), dtype=np.uint8, buffer=slab, offset=base)
        return torch.from_numpy(a).view(torch.float8_e4m3fn)


    def load_x(base: int, n: int, inv: float) -> torch.Tensor:
        """Dequantized activation rows for one post, ready for the graph.

        The graph's first op is the per-tensor fp4 quantizer, so the fp8 rows
        have to be bf16 before replay; the multiply rides the H2D.
        """
        worker.bufs(n)
        x = worker.x[n]
        x.copy_(slot_x(base, n), non_blocking=True)
        if inv != 1.0:
            x.mul_(inv)
        return x

    slab_addr = pin_region(slab, SLAB, "slab")
    rout_addr = pin_region(rout, ROUT, "rout")
    flags_addr = pin_region(flags, FLAGS, "flags")
    out_addr = ctypes.addressof(ctypes.c_char.from_buffer(slab)) + OUT_BASE
    # Per-layer seq cells the graph reads at run time: one int32 per layer in a
    # registered anonymous page. The host writes the round's seq here before replay.
    cells_arr = torch.zeros(layers, dtype=torch.int32)
    cells_host = cells_arr.numpy()
    cells_addr = cells_arr.data_ptr()
    cr = torch.cuda.cudart()
    cr.cudaHostRegister(cells_addr, layers * 4, 0)
    # Out-flag index per layer: the int32 word of layer L in the flag region.
    fout_idx = [FOUT_OFF // 4 + L * FSTRIDE // 4 for L in range(layers)]

    # A previous run leaves nonzero flags in the region, and this worker may
    # open it before the wire engine has reset it. If it is dirty now, serve
    # only after the engine's epoch bump; if it is already clean, the engine has
    # either not run yet or already reset, and every later flag is the client's.
    dirty = any(int(f32[L * FSTRIDE // 4]) != 0 for L in range(layers))
    epoch = int(f32[SESSION_OFF // 4])
    worker = None
    if mode == "moe":
        arena = Arena(torch.device("cuda:0"), layers=layers)
        arena.load(pack, layers)
        if a2_path is None or not arena.load_a2(a2_path):
            arena.calibrate_a2()
        worker = Worker(arena, max_rows=rows)
        # ids/tw arrive per layer through the routing plane; the captured graphs
        # read these buffers by pointer, so refilling them before each replay is
        # enough to change the routing.
        # ids/tw are filled per row count, created by Worker.bufs on first use.
        # Both the routing and the activation arrive per layer, so both are on the
        # critical path. The routing tensors are small enough that staging them
        # through pinned memory beats copying from the registered mapping (torch
        # re-queries pointer attributes on every copy: 21/18 us direct against
        # 18.5/13.9 staged). The payload is not: staging it copies 40 MiB on the
        # host that the H2D then copies again, so it goes straight from the slot.
        pin_i = torch.empty(rows, TOPK, dtype=torch.int32, pin_memory=True)
        pin_w = torch.empty(rows, TOPK, dtype=torch.float32, pin_memory=True)
        wire_out = slot_vec(OUT_BASE, 1)
        if not wire_out.is_pinned():
            raise SystemExit("out-slot is not device-accessible; cudaHostRegister failed")
        # The load's chunked staging tensors stay in the caching allocator; at 40
        # layers that gigabyte is the margin the graph capture needs.
        torch.cuda.empty_cache()
        log(f"pre-capture {mem_report()}")
        t0 = time.time()
        # Only the single-row decode graph is worth pre-capturing: a prefill
        # chunk's row count is not known until its first post, and capture is
        # lazy per (layer, rows) anyway.
        for L in range(layers):
            worker.capture(L, 1, slot_vec(OUT_BASE, 1))
        log(f"captured {layers} single-row graphs in {time.time() - t0:.1f}s")
        glayers = None
        if ingest == "graph":
            # In-graph ingestion: mapped routing/payload/out-slot, one per-layer
            # seq cell and out-flag each, all registered (see gx_ingest.py).
            import gx_ingest as gxi
            gbody = (gxi.cutlass_body_clamped(arena, swiglu_limit, K, TOPK)
                     if swiglu_limit > 0
                     else gxi.cutlass_body(arena, cm, MoEActivation, E, K, TOPK))
            if decode_kernel == "cuda-swz":
                gbody = gxi.swz_small_rows_body(gbody, arena, swiglu_limit)
                log(f"decode kernel: cuda-swz (v7, swizzled scales) at "
                    f"<= {gxi.GRAPH_MAX} rows (exact "
                    f"{sorted(gxi.EXACT_ROWS)} plus power-of-two buckets), "
                    f"clamped CUTLASS above")
            glayers = gxi.Layers(
                gbody,
                arena.cp,
                slab_addr, rout_addr, out_addr, flags_addr, FLAGS,
                cells_host, cells_addr, layers, fout_idx,
            )
            if round_graph:
                # One graph for all layers. A captured body has a fixed row
                # count (the ingest grid and the CUTLASS body are both baked at
                # capture), so the capture width is the row count the round may
                # serve: `--round-rows` sets it, and a round whose posted `n`
                # differs goes down the per-layer path instead of a graph that
                # would compute the wrong number of rows.
                rw = round_rows if round_rows > 0 else gxi.GRAPH_MAX
                t1 = time.perf_counter()
                glayers.capture_round(rw)
                glayers.round_rows = rw
                log(f"round graph: {layers} layers at exactly {rw} row(s), "
                    f"captured in {time.perf_counter() - t1:.1f}s")

        rounds = f"{iters + 1} rounds" if iters > 0 else "an unbounded schedule"
        log(f"moe mode: {layers} layers resident, ready for {rounds} of {layers}")
    else:
        rounds = f"{iters + 1} rounds" if iters > 0 else "an unbounded schedule"
        log(f"identity mode: ready for {rounds} of {layers}")
    out = torch.zeros(rows, K, dtype=torch.bfloat16, device="cuda:0")
    ev = torch.cuda.Event()
    # Warm the context and the copy kernels so the first served layer is not
    # charged ~27 ms of CUDA init.
    out.copy_(slot_x(IN_BASE, 1).to(torch.bfloat16))
    ev.record()
    ev.synchronize()
    if dirty:
        # The engine's epoch bump and this worker's start can happen in either
        # order. If the bump landed first, waiting for the next one waits
        # forever (a 530 s client stall); bound it and serve on the epoch that
        # is already there. runtime_seqs() then picks up whatever round the
        # client posts next, and serving a round the client has already skipped
        # past only rewrites flags the client never reads.
        deadline = time.monotonic() + 10.0
        while int(f32[SESSION_OFF // 4]) == epoch and time.monotonic() < deadline:
            time.sleep(0.002)
        now = int(f32[SESSION_OFF // 4])
        log(f"stale flags at start: epoch {epoch} -> {now}"
            + ("" if now != epoch else " (no bump within 10 s; serving anyway)"))
    # The arena, graphs and warm-up are done and no round has been started, so
    # this is the first quiescent point. A client that connected while the load
    # was running has already bumped the epoch and is waiting for this word.
    pub_ack(int(f32[SESSION_OFF // 4]))

    def runtime_seqs():
        """Unbounded schedule: the client's layer-0 in-flag IS the round number.

        Counting 1, 2, 3 here would require restarting the worker whenever the
        client or the wire engine restarts, because a new client begins again at
        1 while this counter is already past it (the flags get zeroed, so the
        worker would wait for a sequence the client never posts). Reading the
        client's number back costs one load per round. A wire-engine restart
        bumps the epoch word and zeroes the flags, so a new epoch resets the
        expectation to any sequence the new client chooses.

        One round is `layers` posts; a client that pipelines two rounds ahead
        would make this skip a number, which is why the runtime posts layer L+1
        only after fetching layer L.
        """
        ep = int(f32[SESSION_OFF // 4])
        last = 0
        # Do not assume the caller's baseline: a client that connected during
        # the load may already have bumped the epoch, and it waits for an ACK
        # carrying *that* value before it posts anything.
        pub_ack(ep)
        while True:
            e = int(f32[SESSION_OFF // 4])
            if e != ep:
                log(f"wire engine epoch {ep} -> {e}: resynchronizing the round sequence")
                ep = e
                last = 0
                pub_ack(e)
            v = int(f32[0])
            if v > last:
                last = v
                yield v

    seqs = (runtime_seqs() if iters <= 0
            else [SEQ_SER + r for r in range(iters)] + [SEQ_PIPE])
    seq_iter = iter(seqs)
    unbounded = iters <= 0
    calib_left = calib if worker is not None else 0
    calib_max = [0.0] * layers
    t_flag = 0.0
    t_work = 0.0
    n = 0
    isum = 0
    wsum = 0.0
    t_rout = t_h2d = t_launch = t_sync = 0.0

    def abort_round(seq: int, where: str, ep: int, round_ep: int, left: int) -> None:
        """Drop the round this worker was serving on the previous epoch.

        A client only bumps the epoch after it has cleared the flag plane and
        it does not post its first round until this worker acknowledges the new
        epoch, so no payload or routing written by the new client can be read
        here. Synchronize first: per-layer graph replays are asynchronous, and
        the ACK means "this worker is no longer reading round state". The old
        round's out-flags are cleared by the client after the ACK, so they are
        deliberately not published here.
        """
        nonlocal unbounded, seq_iter
        log(f"round {seq}: epoch {round_ep} -> {ep} at {where}; "
            f"discarding {left} layer(s) unread")
        torch.cuda.synchronize()
        pub_ack(ep)
        if not unbounded:
            # This worker was started for a fixed schedule, but a new client
            # owns the wire now and its first round is 1. Read the sequence it
            # posts instead of serving the rest of the fixed list.
            log(f"round {seq}: fixed schedule superseded by epoch {ep}; "
                "switching to the unbounded schedule")
            seq_iter = iter(runtime_seqs())
            unbounded = True

    while True:
        try:
            seq = next(seq_iter)
        except StopIteration:
            break
        round_ep = int(f32[SESSION_OFF // 4])
        aborted = False
        r_t0 = time.perf_counter()
        # t_flag accumulates over the whole run (the summary divides by n);
        # the per-round line must show this round only, so take its base.
        f_round = t_flag
        p = [0.0, 0.0, 0.0, 0.0]
        # The routing checksums are the cross-end proof that the plane carried
        # this round's ids/tw. Unbounded serving has no sentinel round, so it
        # accumulates every round and reports the sums on the round line.
        risum = 0
        risum_all = 0           # legacy path never fills these; keep the line total
        rbad = 0                # routing slots the ingest found corrupt this round
        rpad = 0                # routing slots that posted the -1 sentinel
        rwsum = 0.0
        if glayers is not None:
            glayers.isum.zero_()
            glayers.pad.zero_()
            glayers.invalid.zero_()
        in_round = False
        if glayers is not None and glayers.round_graph is not None:
            # The routing header is ONE region shared by every layer and is
            # rewritten by each post, so which path a round takes may only be
            # decided after this round's layer 0 has been posted: the client
            # release-stores layer 0's in-flag after it wrote the header, and
            # an acquire load of that flag orders the header read. Deciding
            # before the flag could read the previous round's n and hand a
            # wider round to a graph captured for one row.
            while int(f32[0]) < seq:
                ep = int(f32[SESSION_OFF // 4])
                if ep != round_ep:
                    aborted = True
                    abort_round(seq, "before layer 0", ep, round_ep, layers)
                    break
            if aborted:
                continue
            in_round = rout_count() == glayers.round_rows
            if not in_round:
                # A captured graph has one fixed row count; a round that posts
                # a different `n` would compute the wrong rows, so it is served
                # layer by layer. `--round-graph` with a capture width below
                # the engine's real batch therefore degrades to the per-layer
                # path, not to wrong numbers.
                if n % 100 == 0:
                    log(f"round {seq}: posted n={rout_count()} != capture rows "
                        f"{glayers.round_rows}; serving this round per layer")
        if in_round:
            # Round mode: one replay drives all `layers`. The graph's first spin
            # waits for this round's routing header, and the last layer's
            # out-flag is the completion proof the client is waiting on.
            # One header serves the whole round, so decide on it once, before
            # the graph can compute a row or release a single flag.
            nround = rout_count()
            routing_gate(seq, 0, nround, rout_slot(nround)[0])
            tg = time.perf_counter()
            glayers.replay_round(seq)
            last = (layers - 1) * FSTRIDE // 4
            while int(f32[last]) < seq:
                ep = int(f32[SESSION_OFF // 4])
                if ep != round_ep:
                    # A launched round graph cannot be interrupted: its device
                    # spin still runs to its bounded timeout before this
                    # synchronize returns. It reads only the old round's slots
                    # (the new client is waiting for the ACK), so nothing of
                    # the new client is consumed.
                    aborted = True
                    abort_round(seq, "inside the round graph", ep, round_ep, 0)
                    break
            if aborted:
                continue
            ep = int(f32[SESSION_OFF // 4])
            if ep != round_ep:
                abort_round(seq, "at the round graph's last layer", ep,
                            round_ep, 0)
                continue
            t_round = time.perf_counter() - tg
            t_work += t_round
            n += layers
            to = glayers.round_timeouts()
            iv = glayers.isum.cpu().tolist()
            risum, risum_all = iv[0], sum(iv)
            rbad = int(glayers.invalid.sum())
            rpad = int(glayers.pad.max())   # one shared header: same every layer
            if rbad:
                # The host gate cleared this post, so a count here means the
                # ingest saw routing the gate did not: never serve on that.
                routing_fail(seq, f"the ingest counted {rbad} routing slot(s) "
                                  "naming no expert that the host gate did not see")
            # The routing checksum cannot be read back per layer here (the one
            # shared header is rewritten by every post), so the round line
            # carries the work time and the spin evidence, and the client's own
            # isum is the routing proof for this round.
            log(f"round {seq}: {layers} layers in graph replay, "
                f"{1e6 * t_round / layers:.1f} us/layer at n={glayers.round_rows}"
                + f"; rout_recv isum0={risum} isum={risum_all}"
                + (f"; pad_ids={rpad}" if rpad else "")
                + (f"; {to} layer spin(s) timed out" if to else "; no spin timed out"))
            if to:
                glayers.stat.zero_()
            continue
        for L in range(layers):
            fi = L * FSTRIDE // 4
            t0 = time.perf_counter()
            if _NVTX:
                _nvtx.range_push("flag_wait")
            while True:
                ep = int(f32[SESSION_OFF // 4])
                if ep != round_ep:
                    aborted = True
                    abort_round(seq, f"layer {L}", ep, round_ep, layers - L)
                    break
                if int(f32[fi]) >= seq:
                    # The client clears the plane before it bumps the epoch,
                    # and both writes are ordered on its QP. Re-read anyway:
                    # a bump observed between the check above and this flag
                    # is still an abort, not a layer to serve.
                    ep = int(f32[SESSION_OFF // 4])
                    if ep != round_ep:
                        aborted = True
                        abort_round(seq, f"layer {L}", ep, round_ep, layers - L)
                    break
            if aborted:
                if _NVTX:
                    _nvtx.range_pop()
                break
            if _NVTX:
                _nvtx.range_pop()
            t1 = time.perf_counter()
            if worker is not None:
                nr = rout_count()
                if not 1 <= nr <= rows:
                    # The edge clears routing and flags before it publishes a
                    # new epoch. A worker that passed the flag wait just
                    # before that clear can read row count 0 while the epoch
                    # is still the old one; the recheck above cannot see the
                    # bump yet. If it has landed by now, this is that abort,
                    # not a client protocol error.
                    ep = int(f32[SESSION_OFF // 4])
                    if ep != round_ep:
                        aborted = True
                        abort_round(seq, f"layer {L}", ep, round_ep, layers - L)
                        break
                    # With the epoch handshake a raised in-flag belongs to a
                    # client that waited for this worker's ACK before posting,
                    # so its routing header is already on the wire: a row count
                    # outside 1..rows is a protocol error, not a phantom round
                    # to serve from whatever happened to be left in the slab.
                    routing_fail(seq, f"layer {L}: routing header row count {nr} "
                                      f"outside 1..{rows}")
                if glayers is not None:
                    # In-graph path: the graph reads routing and payload from the
                    # mapped slots, writes the out-slot and release-stores seq into
                    # the out-flag. The host only publishes the seq cell and replays.
                    # One header serves every layer of the round, so the routing is
                    # judged once, at layer 0, before any layer can publish.
                    if L == 0:
                        routing_gate(seq, L, nr, rout_slot(nr)[0])
                    tg = time.perf_counter()
                    glayers.replay(L, nr, seq)
                    t2 = time.perf_counter()
                    t_flag += t1 - t0
                    t_work += t2 - tg
                    n += 1
                    continue
                if (L, nr) not in worker.graphs:
                    worker.capture(L, nr, slot_vec(OUT_BASE, nr))
                    log(f"captured L{L} rows={nr}")
                ids_np, tw_np = rout_slot(nr)
                if _NVTX:
                    _nvtx.range_push("rout_stage")
                ta = time.perf_counter()
                pin_i[:nr].copy_(torch.from_numpy(ids_np))
                pin_w[:nr].copy_(torch.from_numpy(tw_np))
                worker.ids[nr].copy_(pin_i[:nr], non_blocking=True)
                worker.tw[nr].copy_(pin_w[:nr], non_blocking=True)
                tb = time.perf_counter()
                if _NVTX:
                    _nvtx.range_pop()
                if seq == SEQ_PIPE or iters <= 0:
                    isum += int(ids_np.sum(dtype=np.int64))
                    wsum += float(tw_np.sum(dtype=np.float64))
                    risum += int(ids_np.sum(dtype=np.int64))
                    rwsum += float(tw_np.sum(dtype=np.float64))
                # The payload lands in the in-slot, which is mapped pinned host
                # memory: one H2D, no host-side staging copy, and the scale the
                # graph needs is computed on the device inside the graph. The
                # slot holds e4m3 rows, so the H2D is half the bytes and the
                # dequantize is the elementwise multiply that follows it.
                if _NVTX:
                    _nvtx.range_push("load_x_h2d")
                load_x(IN_BASE, nr, rout_scale())
                if _NVTX:
                    _nvtx.range_pop()
                tc = time.perf_counter()
                if _NVTX:
                    _nvtx.range_push("replay")
                worker.graphs[(L, nr)].replay()
                if _NVTX:
                    _nvtx.range_pop()
                td = time.perf_counter()
                # the captured D2H already wrote this layer's out-slot
                if _NVTX:
                    _nvtx.range_push("sync_d2h")
                ev.record()
                ev.synchronize()
                if _NVTX:
                    _nvtx.range_pop()
                te = time.perf_counter()
                p[0] += tb - ta
                p[1] += tc - tb
                p[2] += td - tc
                p[3] += te - td
            else:
                nid = rout_count()
                if not 1 <= nid <= rows:
                    ep = int(f32[SESSION_OFF // 4])
                    if ep != round_ep:
                        aborted = True
                        abort_round(seq, f"layer {L}", ep, round_ep, layers - L)
                        break
                    routing_fail(seq, f"layer {L}: identity routing header row count "
                                      f"{nid} outside 1..{rows}")
                out[:nid].copy_(slot_x(IN_BASE, nid).to(torch.bfloat16))
                out[:nid].mul_(rout_scale())
                ev.record()
                ev.synchronize()
                slot_vec(OUT_BASE, nid).copy_(out[:nid])
            t2 = time.perf_counter()
            if _NVTX:
                _nvtx.range_push("out_flag")
            f32[FOUT_OFF // 4 + L * FSTRIDE // 4] = seq
            if _NVTX:
                _nvtx.range_pop()
            if calib_left > 0:
                calib_max[L] = max(calib_max[L], worker.act_amax(L, nr))
            t_flag += t1 - t0
            t_work += t2 - t1
            n += 1
        if aborted:
            # The abort path already synchronized the device and published the
            # ACK; do not summarize a round that was only partly served and do
            # not read this round's ingest counters.
            continue
        if calib_left > 0:
            calib_left -= 1
            if calib_left == 0:
                for L in range(layers):
                    if calib_max[L] > 0.0:
                        worker.a.set_a2(L, 6.0 / (calib_max[L] * 1.05))
                log("swiglu amax measured: "
                    + " ".join(f"{v:.0f}" for v in calib_max))
                if a2_path:
                    worker.a.save_a2(a2_path)
        if glayers is not None:
            # ingest=graph keeps the Arena for its scale sidecar but does no
            # host staging, so the device buffer is the only checksum source.
            iv = glayers.isum.cpu().tolist()
            risum, risum_all = iv[0], sum(iv)
            rbad = int(glayers.invalid.sum())
            rpad = int(glayers.pad.max())
            if rbad:
                # Defense in depth: layer 0's gate cleared this post, so a count
                # here means ingest and gate disagree on what the wire carried.
                routing_fail(seq, f"the ingest counted {rbad} routing slot(s) "
                                  "naming no expert that the host gate did not see")
        log(f"round {seq}: {layers} layers in {(time.perf_counter() - r_t0) * 1e3:.2f} ms; "
            f"per layer flag {(t_flag - f_round) / layers * 1e6:.1f} rout {p[0] / layers * 1e6:.1f} "
            f"h2d {p[1] / layers * 1e6:.1f} launch {p[2] / layers * 1e6:.1f} "
            f"sync {p[3] / layers * 1e6:.1f} us"
            + (f"; rout_recv isum0={risum} isum={risum_all}" if iters <= 0 else "")
            + (f"; pad_ids={rpad}" if rpad else "")
            + (f" wsum={rwsum:.6f}" if iters <= 0 and worker is None else ""))
    log(
        f"served {n} layer activations: flag wait {t_flag / n * 1e6:.1f} us, "
        f"work {t_work / n * 1e6:.1f} us -> {t_work / (n / layers) * 1e3:.2f} ms/round"
    )
    if worker is not None:
        log(f"  phases: rout {t_rout / n * 1e6:.1f}  h2d {t_h2d / n * 1e6:.1f}  "
            f"launch {t_launch / n * 1e6:.1f}  sync {t_sync / n * 1e6:.1f} us")
        log(f"rout_recv layers={layers} rows={rows} n={layers * rows * TOPK} "
            f"isum={isum} wsum={wsum:.6f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="/p")
    p.add_argument("--layers", type=int, default=NLAYERS)
    p.add_argument("--buckets", default="1,6,8")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--shm", default="/dev/shm/qafd")
    p.add_argument("--calib", type=int, default=0,
                   help="rounds of real-activation SwiGLU amax measurement at start")
    p.add_argument("--a2", default=None, help="per-layer SwiGLU scale sidecar")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--mode", default="moe", choices=("moe", "identity"))
    p.add_argument("--ingest", default="legacy", choices=("legacy", "graph"),
                   help="moe per-layer path: legacy host staging, or in-graph mapped ingestion")
    p.add_argument("--swiglu-limit", type=float, default=10.0,
                   help="SwiGLU clamp (checkpoint swiglu_limit); 0 = fused, unclamped body")
    p.add_argument("--round-rows", type=int, default=0,
                   help="--round-graph capture width; a round posting another n "
                        "falls back to the per-layer path (0 = widest small bucket)")
    p.add_argument("--round-graph", action="store_true",
                   help="ingest=graph: one graph for a whole round, each layer "
                        "spinning on the mapped routing header")
    p.add_argument("--decode-kernel", default="clamped",
                   choices=("clamped", "cuda-swz"),
                   help="ingest=graph body at <= GRAPH_MAX rows (exact 6 plus "
                        "power-of-two buckets): the serving clamped CUTLASS "
                        "split, or the v7 swizzled decode kernel "
                        "(gx_decode_swz.py); rows above GRAPH_MAX always use "
                        "clamped CUTLASS")
    args = p.parse_args()
    log(f"device {torch.cuda.get_device_name(0)} {mem_report()}")
    if args.verify:
        verify(args.pack)
        return
    if args.bench:
        bench(args.pack, args.layers, [int(b) for b in args.buckets.split(",")], args.iters)
        return
    if args.serve:
        serve(args.shm, args.pack, args.layers, args.rows, args.iters, args.mode,
              args.calib, args.a2, args.ingest, args.swiglu_limit,
              args.round_graph, args.round_rows, args.decode_kernel)
        return
    p.error("nothing to do: pass --verify, --bench or --serve")
if __name__ == "__main__":
    main()
