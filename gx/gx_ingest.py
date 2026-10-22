"""In-graph ingestion for the GX expert worker (`--ingest graph`).

Per layer, one CUDA graph per row count runs, with no host copy and no host
event sync:

  ingest   mapped routing header + fp8 payload  -> device x (bf16, times header
           inv), ids, tw. Rows n..B-1 are padding: x=0, ids=0, tw=0.
           Id -1 is vLLM's unrouted/padded sentinel (a producer warm-up posts
           it): served as expert 0 with weight 0, counted in `pad`, and still
           included in `isum` because it is what the client summed. Any other
           id outside [0, E) is corruption: it is counted in `invalid` and the
           worker refuses the round (`rout_gate`) instead of serving a partial
           result, since letting one reach the body leaves a routing-map slot
           uninitialised for shuffle_rows to gather with.
  body     the MoE body (injected callable; the worker's cutlass call).
  copy_out device out rows -> mapped out-slot.
  flag     release-store (system scope) of the layer's sequence cell into the
           mapped out-flag, after copy_out in the same stream.

Row counts: the decode post's six rows get their own graph at exactly 6
(EXACT_ROWS). Every other count keeps the power-of-two buckets
1,2,4,...,GRAPH_MAX (16); all of those share ONE buffer set sized at GRAPH_MAX,
so retention is bounded at (row counts x layers) graphs and GRAPH_MAX rows of
buffers regardless of how many distinct chunk tails arrive. Without the exact
six-row graph a 6-row post replayed the 8-row bucket, and both kernel grids are
R*TOPK, so the two padding rows streamed weight the token does not have. Rows
above GRAPH_MAX run the same kernels eagerly over ONE buffer set on a doubling
ladder capped at the wire's row capacity (ROW_CAP), so a prefill can cost at
most one such set. No per-count state.

The routing header is ONE mapped region shared by every layer; `_ingest` reads
it inside the graph, so no host copy and no host checksum parse remains. It also
accumulates each layer's ids into `isum`, which is the cross-end check that the
plane carried the round.

Wire layout (mirrors gx_expert_worker.py): routing header 16 B
{int32 seq, int32 n, float32 inv, int32 fmt}; ids int32[n,6] at int32 index 4;
tw float32[n,6] at int32 index 4 + 6n (bit patterns). Payload e4m3[n,5120] at
slab offset 0.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

K = 5120
TOPK = 6
E = 384                 # routed experts (the body's id space)
BLK = 1024
GRAPH_MAX = 16
ROW_CAP = 4096          # the wire's own row capacity (routing slab offsets)
# Counts that get a graph at exactly that width instead of the next power of two.
# Only the decode post's six rows: 1,2,4,8,16 are already exact buckets, and a
# count nothing posts would add a graph per layer for a width that never runs.
EXACT_ROWS = frozenset({6})


def bucket_of(n: int) -> int:
    """Smallest power of two >= n (n >= 1)."""
    return 1 << max(0, (n - 1).bit_length())


def graph_rows(n: int) -> int:
    """Rows the captured graph for a post of `n` valid rows computes.

    Exact for EXACT_ROWS (the decode's six), the power-of-two bucket otherwise.
    Read on every replay, so a running worker must not change EXACT_ROWS: a
    graph's ingest and body grids are baked at capture from the width it was
    captured with, and the two must agree for the life of the process.
    """
    return n if n in EXACT_ROWS else bucket_of(n)


def classify_ids(ids: np.ndarray) -> tuple[int, int]:
    """(padding slots, corrupt slots) in one post's routing window.

    -1 is vLLM's unrouted/padded sentinel, so it is normal traffic (a producer
    warm-up posts nothing but -1s) and is served as a zero contribution. Any
    other id outside [0, E) names no expert at all: it must fail the request,
    not silently drop that slot's contribution.
    """
    return int((ids == -1).sum()), int(((ids < -1) | (ids >= E)).sum())


class RoutingError(RuntimeError):
    """A post's routing names an expert that does not exist: the round is refused."""


class _CudaAlias:
    """__cuda_array_interface__ over a raw device address of mapped host memory."""

    def __init__(self, addr: int, shape: tuple[int, ...], typestr: str):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (addr, False),
            "version": 3,
        }


def alias(addr: int, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
    """1-D device tensor over `nbytes` of mapped host memory at `addr`."""
    t = torch.as_tensor(_CudaAlias(addr, (nbytes,), "|u1"), device="cuda")
    return t.view(dtype)


@triton.jit
def _ingest(slab, rout, x, ids, tw, isum, pad, bad, li,
            K_: tl.constexpr, BLK_: tl.constexpr, TOPK_: tl.constexpr,
            TOPKP: tl.constexpr, E_: tl.constexpr):
    r = tl.program_id(0)
    n = tl.load(rout + 1)
    fmt = tl.load(rout + 3)
    inv = tl.load(rout + 2).to(tl.float32, bitcast=True)
    inv = tl.where(fmt == 1, inv, 1.0)
    valid = r < n
    j = tl.arange(0, TOPKP)
    lane = valid & (j < TOPK_)
    idv = tl.load(rout + 4 + r * TOPK_ + j, mask=lane, other=0)
    twf = tl.load(rout + 4 + n * TOPK_ + r * TOPK_ + j,
                  mask=lane, other=0).to(tl.float32, bitcast=True)
    # The wire carries client bytes. Two cases, kept apart because they mean
    # different things:
    #   -1        vLLM's unrouted/padded sentinel. Normal traffic (a producer
    #             warm-up posts nothing else); served as expert 0 with tw=0, so
    #             the row's contribution is exactly its routed lanes.
    #   anything  names no expert. get_cutlass_moe_mm_data DROPS such a slot,
    #   else      which is in no expert's segment, so its a_map entry is never
    #             written and shuffle_rows gathers x at whatever integer the
    #             allocator left there (measured: 6 of 96 slots for one -1 row,
    #             and a gather at one of them is an illegal address). Counted in
    #             `bad`; the worker refuses the round on a non-zero count.
    # Either way the stored id is in [0, E), so every map slot is written.
    pad_lane = lane & (idv == -1)
    bad_lane = lane & ((idv < -1) | (idv >= E_))
    tl.store(ids + r * TOPK_ + j, tl.where(pad_lane | bad_lane, 0, idv),
             mask=j < TOPK_)
    tl.store(tw + r * TOPK_ + j, tl.where(pad_lane | bad_lane, 0.0, twf),
             mask=j < TOPK_)
    # Cross-end routing proof: the client reports the ids it sent for the round,
    # so a per-layer total of what the wire carried is the readback that catches
    # a header lost by the plane. Padding is included -- it is what the client
    # summed, so a well-formed round matches exactly; a corrupt id is excluded
    # here and would fail the round before this total is consulted.
    tl.atomic_add(isum + li, tl.sum(tl.where(bad_lane, 0, idv), axis=0))
    tl.atomic_add(pad + li, tl.sum(tl.where(pad_lane, 1, 0), axis=0))
    tl.atomic_add(bad + li, tl.sum(tl.where(bad_lane, 1, 0), axis=0))
    for k0 in range(0, K_, BLK_):
        k = k0 + tl.arange(0, BLK_)
        v = tl.load(slab + r * K_ + k, mask=valid, other=0.0).to(tl.float32)
        tl.store(x + r * K_ + k, (v * inv).to(tl.bfloat16))


@triton.jit
def _copy_out(src, dst, K_: tl.constexpr, BLK_: tl.constexpr):
    r = tl.program_id(0)
    for k0 in range(0, K_, BLK_):
        k = k0 + tl.arange(0, BLK_)
        tl.store(dst + r * K_ + k, tl.load(src + r * K_ + k))


@triton.jit
def _flag(cell, flags, idx):
    s = tl.load(cell)
    tl.atomic_xchg(flags + idx, s, sem="release", scope="sys")

def cutlass_body(arena, cm, activation, E: int, K: int, TOPK: int):
    """MoE body over `r` valid rows: the same call as Worker._launch (old path).

    `a1g` is recomputed in-graph from x, so the activation scale tracks each
    replay's payload. Signature: body(li, r, x, ids, tw, out, ws13, ws2, a1g).
    """
    def body(li, r, x, ids, tw, out, ws13, ws2, a1g):
        a1g.fill_(6.0 / x.abs().amax().clamp_min(1e-6))
        cm.run_cutlass_moe_fp4(
            out, x, a1g,
            arena.w13[li], arena.s13[li], arena.g13[li] / a1g,
            arena.a2[li],
            arena.w2[li], arena.s2[li], arena.g2[li] / arena.a2[li],
            tw, ids, activation.SILU, ws13, ws2, r, arena.cp, K, E, arena.dev,
        )
    return body



def cutlass_body_clamped(arena, swiglu_limit: float, K: int, TOPK: int):
    """MoE body with the checkpoint's SwiGLU clamp (swiglu_limit, e.g. 10.0).

    The fused fp4 kernel (`silu_and_mul_scaled_fp4_experts_quant`) cannot clamp,
    so this is the split form the image already uses for non-SILU activations:
    GEMM1 -> clamped silu*up to bf16 -> fp4 quant -> GEMM2. Same routing,
    scales and combine as `run_cutlass_moe_fp4`. Graph-capturable: no host sync.
    Signature: body(li, r, x, ids, tw, out, ws13, ws2, a1g).
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import silu_and_mul_with_clamp
    from vllm.model_executor.layers.fused_moe.experts import cutlass_moe as cm



    def body(li, r, x, ids, tw, out, ws13, ws2, a1g):
        dev = arena.dev
        cp = arena.cp
        a2g = arena.a2[li]
        # routing metadata, as in run_cutlass_moe_fp4
        expert_offsets = torch.empty((E + 1,), dtype=torch.int32, device=dev)
        blockscale_offsets = torch.empty((E + 1,), dtype=torch.int32, device=dev)
        problem_sizes1 = torch.empty((E, 3), dtype=torch.int32, device=dev)
        problem_sizes2 = torch.empty((E, 3), dtype=torch.int32, device=dev)
        a_map = torch.empty((ids.numel(),), dtype=torch.int32, device=dev)
        c_map = torch.empty((ids.numel(),), dtype=torch.int32, device=dev)
        ops.get_cutlass_moe_mm_data(
            ids, expert_offsets, problem_sizes1, problem_sizes2, a_map, c_map,
            E, cp, K, blockscale_offsets, is_gated=True,
        )
        a1g.fill_(6.0 / x.abs().amax().clamp_min(1e-6))
        a = ops.shuffle_rows(x, a_map)
        rep_a_fp4, rep_a_bs = ops.scaled_fp4_experts_quant(
            a, a1g, expert_offsets, blockscale_offsets, TOPK,
        )
        c1 = ws13[: r * TOPK * 2 * cp].view(r * TOPK, 2 * cp)
        c2 = ws2[: r * TOPK * cp].view(r * TOPK, cp)
        c3 = ws13[: r * TOPK * K].view(r * TOPK, K)
        ops.cutlass_fp4_moe_mm(
            c1, rep_a_fp4, arena.w13[li], rep_a_bs, arena.s13[li],
            arena.g13[li] / a1g, problem_sizes1, expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
        # 3-arg form: this image's wrapper is (output, input, clamp_limit,
        # topk_ids=None, expert_map=None), so extra positional args would select
        # its grouped path instead of the fused clamp kernel.
        silu_and_mul_with_clamp(c2, c1, swiglu_limit)
        int_fp4, int_bs = ops.scaled_fp4_experts_quant(
            c2, a2g, expert_offsets, blockscale_offsets, TOPK,
        )
        ops.cutlass_fp4_moe_mm(
            c3, int_fp4, arena.w2[li], int_bs, arena.s2[li],
            arena.g2[li] / a2g, problem_sizes2, expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
        c3 = ops.shuffle_rows(c3, c_map)
        out.copy_((c3.view(r, TOPK, K) * tw.view(r, TOPK, 1).to(out.dtype)).sum(dim=1))
    return body


def swz_small_rows_body(base, arena, swiglu_limit: float,
                        limit: int = GRAPH_MAX):
    """`base` above `limit` rows; the v7 swizzled decode kernel at or below it.

    The decode-row kernel (`gx_decode_swz.make_body`) measured 212.7 us at one
    row against 271.8 us for the clamped split body (KernelFinish's bench on an
    arena-shaped pool), and it is only defined for the small buckets it was
    measured on: it reads the arena's swizzled scales in place and needs the
    whole per-row activation in registers, so prefill rows keep the CUTLASS path
    and its exact numerics.

    Sentinel policy is the ingest's, unchanged: `-1` padding and corrupt ids are
    stored as expert 0 with weight 0.0, and `g2[0] * 0.0` is an exact zero, so a
    padding lane adds nothing (and stays finite). No corrupt id can reach the
    kernel: `gx_expert_worker.routing_gate` refuses the post before any replay,
    and `_ingest` clamps whatever it stores into [0, E), so this body cannot
    silently zero a malformed row -- it never sees one.

    `r` is baked into a captured graph, so the dispatch is resolved once per
    capture; a replay runs exactly the body captured for its bucket.
    """
    import gx_decode_swz

    fast = gx_decode_swz.make_body(arena, swiglu_limit)

    def body(li, r, x, ids, tw, out, ws13, ws2, a1g):
        if r <= limit:
            fast(li, r, x, ids, tw, out, ws13, ws2, a1g)
        else:
            base(li, r, x, ids, tw, out, ws13, ws2, a1g)
    return body


class Layers:
    """Per-row-count steps: ingest -> body -> copy_out -> flag.

    One graph per `graph_rows(n)` width (exact 6, power-of-two buckets else),
    all over the same GRAPH_MAX-row buffer set, so a new width costs a graph
    per layer and no extra buffers.

    `body(li, r, x, ids, tw, out, ws13, ws2, a1g)` is the MoE body over views
    of exactly `r` rows; it must be capture-safe (no host sync).
    Addresses are host virtual addresses of registered (mapped) memory, which
    are also the device addresses on GB10 (unified addressing).
    """

    def __init__(self, body, cp: int, slab_addr: int, rout_addr: int,
                 out_addr: int, flags_addr: int, flags_bytes: int,
                 cells_host: np.ndarray, cells_addr: int, layers: int,
                 fout_idx: list[int]):
        self.body = body
        self.cp = cp
        self.layers = layers
        self.fout_idx = fout_idx
        self.slab = alias(slab_addr, 2 * K * 4096, torch.float8_e4m3fn)
        # One routing slot per round (the client overwrites it per post).
        self.rout = alias(rout_addr, 16 + 4096 * TOPK * 8, torch.int32)
        self.rout_addr = rout_addr
        self.cells_addr = cells_addr
        self.round_graph = None
        self.round_rows = 0
        self.stat = torch.zeros(1, dtype=torch.int32, device="cuda:0")
        # Compile-only spin uses its own budget/counter: a one-iteration
        # warm-up launch always "times out" and must not be reported.
        self.warm_stat = torch.zeros(1, dtype=torch.int32, device="cuda:0")
        self.out_slot = alias(out_addr, K * 2 * 4096, torch.bfloat16).view(4096, K)
        self.flags = alias(flags_addr, flags_bytes, torch.int32)
        # Per-layer seq cells: a host array (written by the host with a plain
        # store) over the same registered mapping the GPU reads at run time.
        self.cells_host = cells_host
        self.cells = alias(cells_addr, 4 * layers, torch.int32)
        # Device-side per-layer routing totals, zeroed once per round:
        #   isum     the ids the wire carried (the client's own ids.sum())
        #   pad      lanes that posted the -1 sentinel (normal warm-up/padding)
        #   invalid  lanes naming no expert (corruption; the worker fails closed)
        self.isum = torch.zeros(layers, dtype=torch.int32, device="cuda:0")
        self.pad = torch.zeros(layers, dtype=torch.int32, device="cuda:0")
        self.invalid = torch.zeros(layers, dtype=torch.int32, device="cuda:0")
        # Warm-up sums: a compile pass must not land in a served total.
        self.warm_sums = tuple(
            torch.zeros(layers, dtype=torch.int32, device="cuda:0")
            for _ in range(3))
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        dev = torch.device("cuda:0")
        # fixed small set: graph buckets
        self._small = self._alloc(GRAPH_MAX, dev)
        self._big: tuple[int, dict] | None = None   # grow-only eager set
        self.a1g = torch.full((E,), 6.0, dtype=torch.float32, device=dev)

    def _alloc(self, cap: int, dev) -> dict:
        cp = self.cp
        return {
            "x": torch.zeros(cap, K, dtype=torch.bfloat16, device=dev),
            "ids": torch.zeros(cap, TOPK, dtype=torch.int32, device=dev),
            "tw": torch.zeros(cap, TOPK, dtype=torch.float32, device=dev),
            "out": torch.zeros(cap, K, dtype=torch.bfloat16, device=dev),
            "ws13": torch.zeros(cap * TOPK * 2 * K, dtype=torch.bfloat16, device=dev),
            "ws2": torch.zeros(cap * TOPK * cp, dtype=torch.bfloat16, device=dev),
        }

    def _views(self, buf: dict, r: int):
        cp = self.cp
        return (buf["x"][:r], buf["ids"][:r], buf["tw"][:r], buf["out"][:r],
                buf["ws13"][: r * TOPK * 2 * K], buf["ws2"][: r * TOPK * cp])

    def _run(self, li: int, r: int, buf: dict, seq_cell_idx: int | None,
             out_idx: int | None, ingest: bool,
             sums: tuple[torch.Tensor, ...] | None = None) -> None:
        x, ids, tw, out, ws13, ws2 = self._views(buf, r)
        if ingest:
            isum, pad, bad = sums if sums is not None else (
                self.isum, self.pad, self.invalid)
            _ingest[(r,)](self.slab, self.rout, x, ids, tw, isum, pad, bad, li,
                          K_=K, BLK_=BLK, TOPK_=TOPK, TOPKP=8, E_=E)
        self.body(li, r, x, ids, tw, out, ws13, ws2, self.a1g)
        if out_idx is not None:
            _copy_out[(r,)](out, self.out_slot, K_=K, BLK_=BLK)
            _flag[(1,)](self.cells[seq_cell_idx:seq_cell_idx + 1], self.flags, out_idx)

    def capture(self, li: int, r: int) -> None:
        """Graph for (layer, r <= GRAPH_MAX) rows. Warm-up runs the body only."""
        if r > GRAPH_MAX:
            raise ValueError(f"graph width {r} > {GRAPH_MAX}")
        # The warm-up skips the ingest, so the body would read whatever routing
        # the previous bucket left in the small buffers -- and that history is
        # unchecked client bytes. A live round posted ids = -1, which the ingest
        # stored into row 0; the next capture's warm-up then fed the body one
        # invalid id, and get_cutlass_moe_mm_data left that slot's a_map entry
        # uninitialised for shuffle_rows to gather with. Warm up on neutral
        # routing so a capture cannot depend on what the buffers happened to
        # hold (the ingest rejects bad ids on the replayed path).
        self._small["ids"][:r].zero_()
        self._small["tw"][:r].zero_()
        self._run(li, r, self._small, None, None, ingest=False)  # compile/warm, no flags
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self._run(li, r, self._small, li, self.fout_idx[li], ingest=True)
        self.graphs[(li, r)] = g

    def ensure(self, li: int, r: int) -> None:
        if (li, r) not in self.graphs:
            self.capture(li, r)

    def replay(self, li: int, n: int, seq: int) -> None:
        """Serve layer `li` for a post of `n` valid rows. Host publishes `seq` in
        the layer's cell before launch; the graph stores it into the out-flag.

        The width is `graph_rows(n)`: exactly 6 for the decode post, the
        power-of-two bucket otherwise, so a 6-row post no longer replays an
        8-row graph whose grid computes two rows of padding.
        """
        self.cells_host[li] = seq  # plain host store into the mapped cell
        r = graph_rows(n)
        if r <= GRAPH_MAX:
            self.ensure(li, r)
            self.graphs[(li, r)].replay()
        else:
            self._eager(li, r, n, seq)

    def _eager(self, li: int, r: int, n: int, seq: int) -> None:
        """Prefill rows beyond GRAPH_MAX: one workspace set on a doubling ladder.

        A per-count allocation would retain 154,160 bytes for every distinct
        chunk tail the scheduler produces; doubling keeps at most one set alive
        and caps it at the wire's own row capacity, so retention is bounded by
        the protocol rather than by request history.
        """
        cap = 1 << max(0, (r - 1).bit_length())
        cap = min(cap, ROW_CAP)
        if self._big is None or self._big[0] < cap:
            old = 0 if self._big is None else self._big[0]
            self._big = (cap, self._alloc(cap, torch.device("cuda:0")))
            print(f"ingest: eager workspace {old} -> {cap} rows "
                  f"({cap * 154160 / 2**20:.0f} MiB)", flush=True)
        self._run(li, r, self._big[1], li, self.fout_idx[li], ingest=True)

    def capture_round(self, rows: int, spin_max: int = 300_000) -> None:
        """One graph for a full round: per layer, device spin then ingest/body/out.

        The layers run sequentially in one stream, so a single buffer set is
        enough. The host publishes the round's base sequence in cells[0] once
        per round; every layer's spin kernel derives its own seq as base+li and
        stores it into cells[li] for that layer's out-flag.
        """
        import gx_spin
        r = graph_rows(rows)
        if r > GRAPH_MAX:
            raise ValueError(f"round graph rows {rows} > {GRAPH_MAX}")
        self.round_rows = r
        # Compile every piece outside capture: the spin kernel (one iteration)
        # and each layer's ingest+body closures, which are per-layer objects.
        gx_spin.spin_wait_launch(self.flags, self.cells, self.warm_stat, 0,
                                 self.cells, 1)
        for li in range(self.layers):
            # This pass ingests the live header only to compile; counting it in
            # `isum` would double every round's routing proof.
            self._run(li, r, self._small, None, None, ingest=True,
                      sums=self.warm_sums)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            for li in range(self.layers):
                gx_spin.spin_wait_launch(self.flags, self.cells, self.stat, li,
                                         self.cells, spin_max)
                self._run(li, r, self._small, li, self.fout_idx[li], ingest=True)
        self.round_graph = g

    def replay_round(self, base: int) -> None:
        """Publish this round's base sequence and launch the whole round."""
        assert self.round_graph is not None, "capture_round() first"
        self.cells_host[0] = base
        self.round_graph.replay()

    def round_timeouts(self) -> int:
        """Spins that gave up; non-zero means at least one lost layer."""
        return int(self.stat.item())
