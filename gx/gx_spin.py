"""Device-side spin for the GX worker: one CUDA graph per round of 40 layers.

Instead of launching one graph per layer from the host (each launch costs a
host wake-up plus a graph-launch handshake), a single graph walks all layers
in order. Before each layer's body, a 1-CTA Triton kernel spins on that
layer's mapped routing header until the NIC-written `seq` matches the round's
base sequence and `n != 0`. The host therefore touches the wire exactly once
per round (publishing `base`), not once per layer.

Reads are system-scope acquire atomics on mapped host memory: the same
mapping, and the same scope, as the release store the out-flag kernel already
uses. Whether a NIC DMA write becomes visible to a spinning SMM kernel on
GB10 is exactly what `spin_test.py` measures; keep `--round-graph` off until
that passes.

Predicate: hdr[0] == base + li and hdr[1] != 0, where hdr is the 16-byte
routing header of layer li in the mapped slab.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


FLAG_STRIDE = 16    # in-flags are FSTRIDE=64 B apart -> 16 int32 words


@triton.jit
def _spin_wait(flag, base, cells, stat, li, spin_max, BLK: tl.constexpr):
    """Block until this layer's in-flag carries the round's sequence.

    The routing header is ONE region shared by every layer of a round and the
    client rewrites it per post, so the per-layer in-flag -- release-stored by
    the client after that write -- is what marks readiness. Same predicate the
    host loop used, moved onto the device.

    On success publishes the round sequence into cells[li]; the out-flag kernel
    reads that cell, so one graph serves a whole round with no host involvement.
    """
    want = tl.load(base)
    v = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
    hit = (want > 0) & (v >= want)
    i = tl.zeros((), dtype=tl.int64)
    while (i < spin_max) & (hit == 0):
        v = tl.atomic_add(flag, 0, sem="acquire", scope="sys")
        hit = (want > 0) & (v >= want)
        i += 1
    if hit:
        tl.store(cells + li, want)   # the out-flag kernel reads this cell
    else:
        tl.atomic_add(stat, 1)       # gave up; host reports it, keeps serving


def spin_wait_launch(flags, cells, stat, li: int, base_dev,
                     spin_max: int = 100_000_000) -> None:
    """`flags` and `cells` are int32 tensor views of the mapped regions."""
    _spin_wait[(1,)](flags[li * FLAG_STRIDE:], base_dev, cells, stat, li,
                     spin_max, BLK=1)




def wait_fn(flags, cells, stat, li: int, base_dev, spin_max: int):
    """Closure matching the `wait(li)` shape a round graph needs."""
    def wait():
        spin_wait_launch(flags, cells, stat, li, base_dev, spin_max)
    return wait
