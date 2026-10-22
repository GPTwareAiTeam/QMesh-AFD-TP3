"""Compare one layer's out-slot on the wire against an in-process recompute.

Run after a serve session: the slab holds the payload the client last sent and the
hidden state the worker last wrote, and the routing plane holds the ids/tw the
worker applied. Recomputing from those three with this layer's real weights must
reproduce the out-slot bit-for-bit (same kernels, same calibration, same routing).
"""
import mmap
import os
import sys

import numpy as np
import torch

try:                                   # repo checkout
    import gx_expert_worker as gxw
except ModuleNotFoundError:            # deployed in the container as /w/w.py
    import w as gxw

K, ROUT, ROUT_SLOT, SLOT, SLAB, TOPK, TRANSPORT_LAYERS = (
    gxw.K,
    gxw.ROUT,
    gxw.ROUT_SLOT,
    gxw.SLOT,
    gxw.SLAB,
    gxw.TOPK,
    gxw.TRANSPORT_LAYERS,
)
Arena, Worker, log = gxw.Arena, gxw.Worker, gxw.log


def main() -> None:
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    shm = sys.argv[3] if len(sys.argv) > 3 else "/dev/shm/qafd"
    pack = sys.argv[4] if len(sys.argv) > 4 else "/p"
    dev = torch.device("cuda:0")

    arena = Arena(dev, layers=L + 1)
    arena.load_layer(L, pack)
    arena.calibrate_a2()
    worker = Worker(arena, max_rows=rows)

    slab = mmap.mmap(os.open(f"{shm}.slab", os.O_RDWR), SLAB)
    rout = mmap.mmap(os.open(f"{shm}.rout", os.O_RDWR), ROUT)

    def slot(base: int) -> torch.Tensor:
        a = np.ndarray((rows, K), dtype=np.uint16, buffer=slab, offset=base)
        return torch.from_numpy(a).view(torch.bfloat16)

    rb = 0
    ids = torch.from_numpy(np.ndarray((rows, TOPK), np.int32, buffer=rout, offset=rb + 16).copy())
    tw = torch.from_numpy(
        np.ndarray((rows, TOPK), np.float32, buffer=rout, offset=rb + 16 + rows * TOPK * 4).copy()
    )

    x_in = slot(0).clone()
    out_wire = slot(SLOT).clone()
    out_ref = worker.step(L, x_in.to(dev), ids.to(dev), tw.to(dev)).float().cpu()
    wire = out_wire.float()

    cos = torch.nn.functional.cosine_similarity(out_ref.flatten(), wire.flatten(), dim=0).item()
    rel = ((out_ref - wire).norm() / out_ref.norm().clamp_min(1e-12)).item()
    log(
        f"layer {L} rows {rows}: cos={cos:.8f} rel_l2={rel:.6f} "
        f"max|d|={(out_ref - wire).abs().max().item():.5f} "
        f"|ref|max={out_ref.abs().max().item():.4f} ids={ids[0].tolist()}"
    )


if __name__ == "__main__":
    main()
