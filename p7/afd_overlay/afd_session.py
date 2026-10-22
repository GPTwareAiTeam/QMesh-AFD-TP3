"""P7-side AFD-TP3 peer: one transport peer per GX rank, one layer at a time.

The wire engine (qmesh_afd_peer.cpp) owns the protocol; this module gives the
serving runtime the calls a MoE layer makes:

    s = PeerSession("mlx5_0", 3, "192.0.2.10", 47101, rows=R)
    s.post(L, x, ids, tw)      # routing + payload + in-flag, x.shape[0] rows
    s.wait(L)                  # out-flag for this layer
    part = s.fetch(L, t)       # RDMA READ of t out-rows -> CUDA bf16 [t, 5120]

`rows` is the wire's row capacity, fixed when the QP opens; the row count of a
post is not, so a decode step sends one row and a prefill chunk sends its whole
batch over the same slots. The worker learns the count from the header the wire
writes at the head of the layer's routing slot.

Cost structure per layer (measured on P7 -> GX1, R=1):
  post   stage x (R*10 KiB D2H, pinned) + routing (16 B + 48 B/row) + 3 WRs
  fetch  one READ of the out-slot, then a pinned H2D of the partial

`wait` and `fetch` split so the caller can put the shared expert between them:
the flag arrives when the worker finished, the payload READ then overlaps the
shared-expert GEMMs instead of adding its wire time to the layer.
"""

from __future__ import annotations

import os
import sys

import torch

# AFD_NVTX=1 brackets post/fetch phases in NVTX ranges for nsys. Unset, every branch is dead.
_NVTX = os.environ.get("AFD_NVTX") == "1"
if _NVTX:
    import torch.cuda.nvtx as _nvtx

_BIN = os.environ.get("QMESH_AFD_BIN", "/opt/qafd")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import _qmesh_afd  # noqa: E402  (built from src/qmesh_afd_peer.cpp)

K = _qmesh_afd.K_HIDDEN
TOPK = _qmesh_afd.TOPK
LAYERS = _qmesh_afd.LAYERS

# e4m3's largest finite magnitude is 448. The payload is a single fp8 tensor
# scaled by 448/amax, and the scale travels in the routing header so the worker
# multiplies the rows back before its per-tensor fp4 quantizer sees them.
FP8_MAX = 448.0

class PeerSession:
    """Transport to one GX rank, driven per (layer) by the P7 MoE seam."""

    def __init__(self, dev: str, gid_idx: int, mgmt_ip: str, mgmt_port: int,
                 layers: int = LAYERS, rows: int = 1, iters: int = 0,
                 device: int = 0) -> None:
        self.rows = rows
        self.layers = layers
        self.device = device
        self.peer = _qmesh_afd.Peer(dev, gid_idx, mgmt_ip, mgmt_port, layers, rows, iters)
        # Routing staging. The wire region per layer is a 16 B header written by
        # the client, then ids[n][6] int32, then tw[n][6] f32; the 48 B a row
        # costs match ROUT_ROW on every side.
        #
        # The engine connects on the first layer of the first forward, i.e.
        # inside ``torch.inference_mode``: allocating there would make these
        # inference tensors, and the posts and fetches that write them run on
        # wire threads, where such a write is illegal.
        with torch.inference_mode(False):
            self._rout = torch.empty(2 * rows * TOPK, dtype=torch.int32, device=device)
            self._part = torch.empty(rows, K, dtype=torch.bfloat16, device=device)
            # Payload staging: the wire carries e4m3 rows, so the cast happens
            # here on the caller's stream and the D2H moves half the bytes.
            self._xq = torch.empty(rows, K, dtype=torch.float8_e4m3fn, device=device)
        # The activation and the routing are produced by kernels on the caller's
        # stream; the staged D2H copies must run there or they race the producer.
        # Posts run in wire threads, where torch's current stream is the legacy
        # default one, so the stream is captured here and reused as the context
        # for the staging work those threads enqueue.
        self._stream = torch.cuda.current_stream(device)
        self.peer.set_stream(self._stream.cuda_stream)

    # -- per layer ---------------------------------------------------------
    def post(self, layer: int, x: torch.Tensor, ids: torch.Tensor | None = None,
             tw: torch.Tensor | None = None, n: int | None = None) -> None:
        """Stage one layer's rows and post routing, payload and flag.

        `n` defaults to the activation's row count; it is the only thing that
        distinguishes a decode post from a prefill post on the wire.

        The payload leaves as e4m3 under one per-post scale, which the worker
        reads back out of the routing header. `abs().amax()` costs one reduction
        pass over rows*K bf16 and the cast one fused pass, against half the
        bytes on a wire that is the layer's critical path.
        """
        if n is None:
            n = int(x.shape[0])
        if not 1 <= n <= self.rows:
            raise ValueError(f"post rows {n} outside 1..{self.rows}")
        rout_ptr = 0
        with torch.cuda.stream(self._stream):
            if _NVTX:
                _nvtx.range_push("amax_sync")
            # float() is a device sync: it waits for this layer's producers (attention) to finish.
            scale = FP8_MAX / float(x[:n].abs().amax().clamp_min(1e-6))
            if _NVTX:
                _nvtx.range_pop()
                _nvtx.range_push("cast")
            self._xq[:n].copy_((x[:n] * scale).clamp(-FP8_MAX, FP8_MAX))
            if _NVTX:
                _nvtx.range_pop()
            if ids is not None:
                if _NVTX:
                    _nvtx.range_push("rout_copy")
                k = n * TOPK
                self._rout[:k].copy_(ids[:n].reshape(-1).to(torch.int32))
                self._rout[k:2 * k].copy_(tw[:n].reshape(-1).view(torch.int32))
                rout_ptr = self._rout.data_ptr()
                if _NVTX:
                    _nvtx.range_pop()
        if _NVTX:
            _nvtx.range_push("so_post")
        self.peer.post(layer, self._xq.data_ptr(), rout_ptr, n, 1.0 / scale)
        if _NVTX:
            _nvtx.range_pop()

    def begin_round(self) -> int:
        """Advance the sequence for one decode step.

        The GX worker serves a round as `layers` sequential layers under one
        sequence number and compares each layer's in-flag against it, so the
        client must bump exactly once per step, before the first post.

        A step may run under a different stream (CUDA graphs, a side stream), so
        re-bind the staging stream before the round's first post.
        """
        self._stream = torch.cuda.current_stream(self.device)
        self.peer.set_stream(self._stream.cuda_stream)
        return self.peer.begin_round()

    def wait(self, layer: int) -> int:
        return self.peer.wait(layer)

    def fetch(self, layer: int, n: int | None = None) -> torch.Tensor:
        """READ `n` out-rows and return them as a CUDA tensor (pinned H2D)."""
        if n is None:
            n = self.rows
        if _NVTX:
            _nvtx.range_push("rdma_read")
        wr = self.peer.read(layer, n)
        self.peer.collect(wr)
        if _NVTX:
            _nvtx.range_pop()
            _nvtx.range_push("h2d")
        host = torch.frombuffer(self.peer.out_mv(layer), dtype=torch.uint16)
        self._part[:n].view(torch.uint16).copy_(host.view(self.rows, K)[:n],
                                               non_blocking=True)
        if _NVTX:
            _nvtx.range_pop()
        return self._part[:n]

    def host_view(self, layer: int, n: int | None = None) -> torch.Tensor:
        if n is None:
            n = self.rows
        mv = self.peer.out_mv(layer)
        return torch.frombuffer(mv, dtype=torch.uint16).view(self.rows, K)[:n].view(torch.bfloat16)

    def close(self) -> None:
        self.peer.close()
