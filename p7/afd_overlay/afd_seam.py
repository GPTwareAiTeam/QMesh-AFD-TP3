"""Graph-safe P7 seam (AFD_SEAM=1): the MoE hooks drive `_qmesh_afd.Seam`.

Every call on the forward path is a kernel launch on the caller's current
stream: no host sync, no `.item()`, no thread pool, no Python futures. The
peers connect in `load()`, before any capture, so the hot path never connects.

Capture safety: the MoE forward is an opaque `torch.ops.vllm.moe_forward`, so
these hooks run only while the graph is recorded; replays execute the
recorded kernels. Warmup and capture passes are full 40-layer rounds (the
seam posts every layer), so the GX workers see only complete rounds.
"""

from __future__ import annotations

import json
import os
import sys

import torch

_BIN = os.environ.get("QMESH_AFD_BIN", "/opt/qafd")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import _qmesh_afd  # noqa: E402

if not hasattr(_qmesh_afd, "Seam"):
    raise ImportError(
        "AFD_SEAM=1 needs _qmesh_afd.Seam; the .so at " + _BIN + " has none. "
        "Build the seam .so or unset AFD_SEAM."
    )

K = _qmesh_afd.K_HIDDEN
TOPK = _qmesh_afd.TOPK


class SeamBridge:
    """Owns one `_qmesh_afd.Seam` over the configured GX peers."""

    def __init__(self) -> None:
        with open(os.environ.get("AFD_CONFIG", "/opt/afd/afd.json")) as f:
            cfg = json.load(f)
        self.layers = int(cfg.get("layers", 40))
        self.rows = int(os.environ.get("AFD_ROWS", cfg.get("rows", 1)))
        self.peer_cfg = cfg["peers"]
        self.peers: list = []
        self.seam = None
        self.device = 0

    def load(self, device: int) -> None:
        """Connect the peers and build the seam. Called once, before capture."""
        if self.seam is not None:
            return
        self.device = device
        for p in self.peer_cfg:
            self.peers.append(
                _qmesh_afd.Peer(p["dev"], int(p["gid"]), p["ip"], int(p["port"]),
                                self.layers, self.rows, 0)
            )
        self.seam = _qmesh_afd.Seam(self.peers, self.layers, self.rows, device)
        self._start_monitors()

    def _start_monitors(self) -> None:
        """Env-gated profiler/status helpers, started once from load().

        `load()` is reached either from the first post or from the load_model
        hook; starting here keeps monitoring independent of that choice.
        """
        if os.environ.get("AFD_PROF_CTL") and os.environ.get("AFD_NVTX") == "1":
            import afd_bridge
            afd_bridge._start_prof_thread()
        _log = os.environ.get("AFD_SEAM_STATUS_LOG")
        if not _log:
            return
        import threading
        import time as _time

        def _dump() -> None:
            with open(_log, "a", buffering=1) as f:
                while True:
                    try:
                        f.write(json.dumps({"t": _time.time(), **self.status()}) + "\n")
                    except Exception:
                        pass
                    _time.sleep(0.5)

        threading.Thread(target=_dump, daemon=True, name="seam-status").start()

    def post(self, layer: int, x: torch.Tensor, ids: torch.Tensor,
             tw: torch.Tensor) -> None:
        """Kernel-only: amax/cast/header/doorbell on the current stream."""
        if self.seam is None:
            # First forward is the profile pass, before any graph capture: connect here.
            self.load(x.device.index or 0)
        n = x.shape[0]
        self.seam.post(layer, x.data_ptr(), x.stride(0), ids.data_ptr(),
                       ids.dtype == torch.int64, tw.data_ptr(), n,
                       torch.cuda.current_stream(self.device).cuda_stream)

    def collect(self, layer: int, out: torch.Tensor) -> None:
        """Kernel-only: spin until both peers' rx words match, then combine."""
        n = out.shape[0]
        self.seam.collect(layer, out.data_ptr(), out.stride(0),
                          out.dtype == torch.float32, n,
                          torch.cuda.current_stream(self.device).cuda_stream)

    def status(self) -> dict:
        return self.seam.status() if self.seam is not None else {}

    def close(self) -> None:
        if self.seam is not None:
            self.seam.close()
            self.seam = None
        for p in self.peers:
            p.close()
        self.peers.clear()


_seam: SeamBridge | None = None


def seam() -> SeamBridge:
    global _seam
    if _seam is None:
        _seam = SeamBridge()
    return _seam
