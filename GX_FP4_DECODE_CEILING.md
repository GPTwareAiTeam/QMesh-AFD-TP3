# GX expert compute (W4): kernel contract and decode ceiling

Kernel and microbenchmark numbers are measured on GX2 (GB10, sm_121, 48 SMs,
24 MB L2) inside the project's NCCL 2.3.12 GX image (torch 2.11.0+cu130,
vLLM 0.24.0), unless a section names GX1. Engine e2e numbers are referenced
from PLAN W7/W7.2/W8.

Scope of the tok/s figures: unless a section says otherwise, these are
**per-rank expert-chain ceilings** for one GX slice across 40 layers, with
attention mocked or zero. They are not end-to-end decode. The earlier
pre-ingest-fix full graph + `AFD_SEAM=1` decode was **41.0–41.4 tok/s
(24.19–24.39 ms/token)** against a ~6.5 tok/s eager baseline. The previous
campaign's final e2e (historical) is **53.94 / 53.89 / 53.73 tok/s** over
3 × 256 tokens (wall **4.75 / 4.75 / 4.76 s**), about **31% above** the earlier
result; its **>75 tok/s target was not met** (PLAN.md W7). The 2026-10-21
campaign starts from **53.46 / 54.02 / 54.03 tok/s**; the cooperative-staging
and Seam de-latency changes are installed and bit-exact in isolation. The
combined engine measured **55.62 / 55.57 / 55.79 / 55.34 / 54.75 tok/s** over five
256-token requests, **median 55.57 tok/s** versus the **54.02 tok/s** baseline
median (**+2.87%**), with both GX workers on the staged body. It is a combined
result; no separate e2e attribution is inferred. **>75 tok/s remains unmet.**

The 2026-10-22 pre-DSpark round (DSpark off; complete) started at **55.23 /
55.74 / 55.67 tok/s, median 55.67** over three measured runs. The listed
strategies ran together; the final integrated actual `bg179` is **PASS** and
engine healthy: five 256-token median **56.86 tok/s**, **+2.1376%** vs **55.67**,
TP1/GPU0. The 55.57 tok/s campaign above is historical. Exact-6 dispatch is
accepted and deployed via the `bg145` reload; readiness is confirmed on both
hosts and live proof is complete. Client timings are observation only; no causal
exact-6 e2e gain is claimed. The retention prerequisite was then CPU-proven only
(PLAN W7.2). The mHC/P7 fused consumer proof passes finite/NaN/Inf/mixed-block
stock-bit-exactness, with no isolated attribution. That round is historical; the
next round enabled DSpark and its measured results are in PLAN W8.

**Current DSpark round (2026-10-22; complete; PLAN W8).** DSpark was measured on
the served engine at k0/k1/k3/k5, **64 forced timed rows** per configuration.
Pair medians, `1024p_128c / 128p_1024c` tok/s: k0 **57.37 / 57.43**, k1 **72.69
/ 75.32**, k3 **82.13 / 96.70**, k5 early **78.49 / 102.82**. Best retained
per-layer k5 is **76.10 / 97.38** (paired); RG6 is **76.31 / 98.81**, shows no
paired gain (95% CI crosses zero) and is rejected. **120 tok/s was not
reached.** The drafter is loaded (**88.93 GiB** weights), and retention is
CPU-proven plus runtime-functional, not a full suffix bitwise proof. Per-type
medians and quality probes are in PLAN W8 and
the DSpark final-results artifact.

Prefill, SessionFix and transport evidence are in PLAN W5/W7; the campaign
ledgers are PLAN W7.1 (2026-10-21), W7.2 (pre-DSpark) and W8 (DSpark). The p7.2
and p7.3 pre-change baseline traces are in §5; p7.1 remains the pre-ingest
baseline.
The W4A16 accuracy caveat in §1 remains.

Current deployment deltas versus the plan: Engram runs through vLLM's
**pinned-host MXFP4 offload** (94.42 GiB per table, ~188.8 GiB total), not the
planned Gen5 NVMe FP8 per-token reads; **DSpark is on at k5 per-layer**
(`SPEC_TOKENS=5`; k1/k3 are measured points). Exact-6 GX dispatch is accepted
and deployed via `bg145`; readiness is confirmed on both hosts and live proof is
complete. Client timings are observation only; no causal exact-6 e2e gain is
claimed. The DSpark retention overlay is mounted only when `SPEC_TOKENS>0`; it
is CPU-proven and runtime-functional, with no full suffix bitwise proof (PLAN
W8). `MHC_QUANT=1` is the default launch and mounts 4 RO; `MHC_QUANT=0` is only
the experimental A/B config. The baseline full-graph `Spec=0` path is unchanged.
Historical planning text otherwise remains unchanged.

2026-10-21 campaign validation: both 40-layer GX workers were READY
(`--ingest graph --swiglu-limit10 --decode-kernel cuda-swz`); that
campaign's engine was healthy from the packaged
`scripts/launch_engine_graphs.sh` candidate (default full graph + Seam,
`cuda-swz` workers). Runtime smoke/bench was verified; no
permanent regression suite was added. The 2026-10-21 evening engine relaunch
was that campaign's run; its five-request measurement is in §8. The 2026-10-22
round is W7.2 and §8 below.

## 1. Activation format is decided by fidelity, not by preference

`PLAN.md` §6 targeted FP8 activations (W4A8) to match DeepSeek production. No
shipped path reaches that with a bit-exact pack:

| Candidate | Weight format | Available op | Verdict |
|---|---|---|---|
| W4A8 | int4 (`cutlass_encode_and_reorder_int4b`) | `cutlass_w4a8_mm`, `cutlass_w4a8_moe_mm` | **Rejected.** Requires re-encoding E2M1 values into signed int4 with per-group + per-channel scales. E2M1 spacing is non-uniform; the re-encode is lossy, so the "no re-quantisation" constraint is violated. |
| W4A4 | NVFP4 E2M1 + block-16 E4M3 scales | `cutlass_scaled_fp4_mm`, `cutlass_fp4_moe_mm` | **CUTLASS fallback.** Exactly the bytes in the pack. A operand comes from `scaled_fp4_quant`. Used for rows above the 16-row fast-dispatch threshold. |
| W4A16 | NVFP4 E2M1 + block-16 E4M3 scales; BF16 activations | `fp4_decode_moe_swz.cu` (v7) | **Decode path.** Dequantizes FP4 weights in registers, keeps the SwiGLU intermediate in FP32, and reads the arena's swizzled scales in place, so the 13.05 GB row-major duplicate v6 needs is absent. Used for ≤16 rows. |

`cutlass_scaled_mm_supports_fp4(121) == True`; the realised kernel is
`cutlass_scaled_fp4_mm_sm120a`.

The v7 decode path is therefore **W4A16**, not the W4A8 target above: it does no
activation quant. On the real pack, one layer at n=1/6/16 gives relL2
**1.65–1.67e-3** against an FP32 dequant reference; v7 and v6 are **bit-identical**
at every tested size. v7's W4A16 arithmetic differs from the W4A4 CUTLASS path by
**~0.15 relL2**; that is informational, not a bit-exactness claim, and quality
evaluation is not complete.

Accuracy recovery for the CUTLASS W4A4 path, if W6 shows FP4 activations cost too much: quantise A twice
(`A1 = fp4(A)`, `A2 = fp4(A - dequant(A1))`) and stack the rows into one
`[2M, K]` call, then add the two outputs. The B tile is still read once per call,
and M is 6→12 on a kernel that is bandwidth-bound on B, so the cost is a pair of
adds, not a second weight pass. Stack factor stays configurable (1 or 2).

## 2. Kernel contract, verified

```python
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

b_sc = swizzle_blockscale(linear_scales)        # [N, K/16] -> [N_pad, S_pad]
xc, xsf = scaled_fp4_quant(x, a_gs, True)       # True = swizzled output layout
out = cutlass_scaled_fp4_mm(xc, w_codes, xsf, b_sc, alpha, torch.bfloat16)
```
- `alpha = weight_global_scale / activation_global_scale`, i.e. `g13 / a_gs` for
  the first GEMM. Measured on GX1 with a self-built e2m1 codebook over six
  (g, a_gs) combinations in {1, 1e-4} × {1, 100, 85}: the empirical alpha is
  `g/a_gs` to five digits every time (`g=1, a_gs=100` → 9.95965e-3 vs 1e-2;
  `g=1e-4, a_gs=85` → 1.16912e-6 vs 1.17647e-6), and all six runs land at
  cos ≈ 0.99 against a bf16 reference. `1/(a_gs·g)` is wrong by `g²`, which for
  this checkpoint's `g ≈ 1.2e-4` is a 1e8 error. The same runs independently
  confirm nibble order (low nibble = even k), block-16 scale alignment,
  multiply-by-g, and the swizzle.
- `input_global_scale` must be **float32** and **one-dimensional of length E**.
  A bf16 tensor derived from activations (`6.0 / x.abs().max()` inherits the
  dtype of `x`) is rejected by an empty-message `STD_TORCH_CHECK` inside
  `scaled_fp4_experts_quant`; no zeros either, so calibrate all E entries.
- `swizzle_blockscale` reproduces vLLM's own swizzled layout byte-exactly
  (`torch.equal` True), pads N to a multiple of 128 and K/16 to a multiple of 4,
  and returns `[B, M_padded, K_padded]`. N=1728 → 1792.
- The per-expert kernel **requires** that padding: passing raw `[1728, 320]` fails
  with `scale_b must be padded and swizzled to a shape (1792x320)`.
- A must be FP4. A bf16 A fails with `Inconsistency of torch::stable::Tensor type:a`.
- Dataflow is two-stage: `w13` contracts over K=5120, `w2` contracts over C=864.
  Treating `w2` as K=5120 fails (`a and b shapes cannot be multiplied`).
- Activation halves are **gate first**: `silu(h[:, :C]) * h[:, C:]`, confirmed
  against a bf16 reference (cos 0.974 vs 0.904 for the swapped convention).
- Nibble order (only needed when building a dequant reference): low nibble is the
  even-K element.

## 3. Grouped MoE: the C%128 requirement, not an API limitation

The earlier conclusion that only expert 0 is ever computed was wrong, and the
cause is a hard kernel requirement rather than missing vLLM plumbing.
`cutlass_fp4_moe_mm` (grouped) needs the **contraction** dimension of each GEMM
to be a multiple of 128. If it is not, every expert after the first silently
returns exactly zero — no error, no warning, no NaN.

| Stage | Output | Contraction | Legal at C=864? |
|---|---|---|---|
| 1 (w13) | `2C` | `K = 5120` | yes |
| 2 (w2) | `K = 5120` | `C = 864` | **no** |

The fused helper `run_cutlass_moe_fp4` derives both GEMM shapes from a single
`n`, so `n = 864` is illegal on stage 2 and `n = 896` forces stage 1's output
width to `2·CP = 1792`. That is why the pack is loaded padded to C=896.

Two consequences that are easy to get wrong:

1. **Gate and up are split at CP, not appended.** Arena row `r` maps to pack row
   `r` for `r < C` (gate) and to pack row `r - (CP-C)` for `CP ≤ r < CP+C` (up).
   The 32 padding rows sit *between* the halves. The same rule applies to the
   block scales, which are padded `[E, 1792, 320]` and swizzled after padding.
2. **`a2_gscale` is `[E]` and must exist before the intermediate does.** The
   fused call takes the SwiGLU-intermediate global scale as an input, so it
   cannot be derived from the intermediate it is about to quantize. It is a
   per-layer calibrated `[E]` constant; calibrate an upper bound on
   `max|silu(gate)·up|` and divide by 1.5. A larger assumed max (smaller
   gscale) only costs code resolution, a smaller one clips.

Verified end to end on real pack bytes, layer 0, R=1, topk=6, CP=896, with the
arena loader (padded and swizzled at load) against a per-expert reference built
straight from the pack:

```
cos = 0.999997   rel = 3.00e-03   in_norm 72.07   out_norm 39.60
```

`rel ≈ 3e-3` is the FP4 quantization of the intermediate, not a layout error. Do
not try to pin the last factor of 1e-3 with a hand-written dequant reference: no
self-consistency test can fix the scale pairing. The model-level ladder is the
gate.

That number is the CUTLASS W4A4 path; the v7 W4A16 decode path measures
**1.65–1.67e-3 relL2** at n=1/6/16 against an FP32 dequant reference (§1).

## 3b. Graph pipeline: the wait cannot live inside the graph

A captured graph bakes a `cuStreamWaitValue32` immediate in at capture time, so
a per-step sequence number cannot be a node of a long-lived graph. The working
structure is 40 per-layer graphs per row bucket with the wait on the *stream* in
front of each replay:

```python
for L in range(40):                                # host, once per token, non-blocking
    cuStreamWaitValue32(stream, in_flag[L], seq)   # GEQ, seq in device memory
    graph[(L, R)].replay()
    cuStreamWriteValue32(stream, out_flag[L], seq) # stream-ordered after the kernels
```

The compute stream stalls on the semaphore while the host stays ahead, and other
streams (the RDMA progress thread, the peer's read-back) keep running: a
semaphore wait parks only its own stream, not the SMs. Host cost is ~5 µs per
layer, overlapped with an ~11 ms per-rank expert-chain token (275 µs/layer × 40).
A single 40-iteration graph would need a custom in-kernel spin, which this
structure makes unnecessary.

Measured on GB10 (2026-10-20): `cuStreamWaitValue32` returns 801, so no
stream-semaphore gate exists there. The deployed worker polls the mapped in-flag
from the host instead and synchronises on an event; the measured cost of that
gate is in §7.

## 4. GB10 bandwidth ceiling

| Measurement | Result |
|---|---|
| bf16 sum, 1 GB | 257 GB/s |
| copy | 229 GB/s |
| `cutlass_scaled_fp4_mm` N=1728 K=5120, M=6 / 32 / 128 | 25.7 / 22.6 / 22.6 µs (194 / 220 / 221 GB/s) |
| w13, 40 distinct experts (cold) | 28.7 µs, 173 GB/s |
| w2, N=5120 K=864 (L2-hot) | 18.2 µs, 137 GB/s |

w2's 137 GB/s with a hot L2 is the tell: at 2.49 MB per expert the kernel is
dominated by launch/short-K latency, not by DRAM. N=3456/6912 figures
(286–450 GB/s) are L2-inflated and not usable for planning.

## 5. Full-layer budget and the consequence

Realistic layer (6 experts × w13 + SwiGLU + quant + w2), grouped kernel, one
layer, 40 distinct experts, measured on GX1:

| Mode | µs/layer | GB/s | ms/token | tok/s ceiling |
|---|---|---|---|---|
| eager | 359.5 | — | 14.4 | 69 |
| CUDA-graph captured | **275.0** | 150 | 11.0 | **91** |

These are single-GX expert-compute ceilings for a token's 40-layer chain, not
end-to-end decode. The earlier pre-ingest-fix full graph + Seam result was
41.0–41.4 tok/s; the previous campaign's final e2e (historical) is
**53.94 / 53.89 / 53.73 tok/s** at 3 × 256 tokens (PLAN.md W7). Both results
are end-to-end; **>75 tok/s was not met**. The 2026-10-21 campaign's combined
e2e is **median 55.57 tok/s** (range 54.75–55.79) in PLAN W7.1, still below 75;
the 2026-10-22 round starts at **median 55.67 tok/s** with strategies still in
progress (PLAN W7.2).

Full ingestion graph, same layer, measured on GX1:

| rows | `cuda-swz` v7 | clamped CUTLASS |
|---|---|---|
| 1 | ~225 µs | 305–308 µs |
| 8 | 1232 µs | 1358 µs |
| 16 | 2808 µs | 2978 µs |

Fast `cuda-swz` dispatch is used only at ≤16 rows; larger buckets stay on
CUTLASS. These are per-layer expert-chain times, not end-to-end decode.

Isolated body comparison (GX1, GPU exclusive, serving worker stopped, real
pack layer 0, both bodies built from identical glue and timed in one process):
the staged v7 body (`GX_STG=8`, block-scale sectors staged in 4 KB of shared
memory per block) runs **209.2 µs/layer at rows=1 vs 213.5 µs** for the
pre-staging body, −4.3 µs/−2.0 % of the graph replay, and is **bit-identical**
to it (int16 diff 0, max|Δ| 0) on every row tested (1/2/6/16 real pack,
1/2/4/6/8 random arena). The gain is −2…−8 µs at rows 2–4 and nil at rows 6–16:
the removed `LDG.E.U16` scale reads (10→2 in stage 1, 6→0 in stage 2) only
shorten a latency tail that larger row counts hide anyway. The 2026-10-21
combined campaign e2e is **55.57 tok/s median** (PLAN W7.1, historical); the
2026-10-22 round baseline is **55.67 tok/s median**, and the final integrated
actual is **56.86 tok/s median** (`bg179` PASS), so this isolated kernel
comparison is not an e2e result and its share is not separable.

In the earlier valid node-traced pre-ingest-fix run (`p7.1.nsys-rep`), merged
GPU busy is 24.36 ms/token and idle is 1.26 ms/token: 14.31 ms/token is
`seam_collect_wait_kernel` (40 posts/token), the remote round trip carrying this
per-rank expert chain, and ~10 ms/token is P7's own arithmetic. The expert-chain
ceiling is therefore not an end-to-end tok/s figure (the two legs are serialized,
so their times add, not their tok/s), and the kernel-class rows overlap: do not
sum them (PLAN.md W7).
This trace predates graph ingestion/v7 and does not cover the 2026-10-21
campaign e2e (PLAN W7.1).
The final p7.2 trace (`p7run/p7.2.nsys-rep`, exported `optimized.sqlite`) covers
the v7 path. Its full 64-token window includes first-step cold sampling/JIT:
2560 = 64 × 40 seam waits, totaling 509.586012 ms = 7.9623 ms/token (p7.1:
915.643 ms = 14.307 ms/token); merged busy 18.440666 ms/token, idle
6.224742 ms/token from the cold sampling gap. The steady estimate, excluding the
first 40 waits/first step, spans 63 steps at 18.263686 ms/step, with busy
17.935739, idle 0.327948 and seam wait 7.624559 ms/step. That window runs from
wait #41.start to the last wait.end and slightly excludes layer-0 local work
before the first wait. Do not sum overlapping categories. At 75 tok/s the budget
is 13.333 ms/token; ~18.55 ms/token leaves ~5.2 ms/token. The GX wait remains
dominant and reducible, but not all of it is necessarily transport.

The fresh p7.3 trace (`p7run/p7.3.trace.1.nsys-rep`) gives a steady
**18.32 ms/token**, seam **7.77 ms (42.4%)**, zero overlap; the first **318 ms**
of sampler/JIT is excluded and the **854 µs** head sampler remains outside the
graph. `profile_window.json` reports **50.87 tok/s**, but that run has the
profiler attached: it is **not a speed result**. The profiler's **1.16 ms**
layer-0 inter-step doorbell includes head sampling, so it is not necessarily an
avoidable communication bubble.

Traffic is 41.3 MB/layer at C=864 (44.8 MB unpadded plus the 1792-row w13
padding and its scales): 1.65 GB/token per GX. P7's C=576 slice is 29.9 MB/layer.

Resident arena at CP=896 is 2.77 GiB/layer, 110.7 GiB for 40 layers against a
121.6 GiB unified pool. The padding itself costs 2.4 GiB; dropping it means
calling `get_cutlass_moe_mm_data` for a stage-1 width of 1728 while stage 2
contracts over 896, which requires reimplementing the tile-alignment and
`a_map`/`c_map` bookkeeping — not worth 2 GiB while the arena fits.

Capacity blocks any rebalancing onto P7: the pack costs 0.1327 GB per unit C, P7
has 97.9 GB total and already uses 71.19 GiB (C ≤ ~650), the GX ranks hold
114.7 GB of 121.6 (C ≤ ~830). The 576/864/864 split is at the VRAM limit. In the
2026-10-22 pre-DSpark load, GPU0 reported **89,318 MiB used / 7,932 MiB (7.744
GiB) free**; that is historical. The **7.388 GiB drafter is now loaded**, and
per-layer DSpark k5 reports **93,506 MiB used / 3,744 MiB free** (RG6 **94,262 /
2,988**; observed peak **94,312 MiB used with 2,938 MiB reported free**, earlier
k5 snapshots 3,744 MiB), so measured headroom is **~2.9–3.7 GiB** and rebalancing
is **ruled out**. Free memory is measured, not total-minus-used. (`nvidia-smi`
total includes the display reserve.) Closing the
expert-chain part of the gap to Mia's 74.9 tok/s total decode must come from
kernel efficiency (launch count, w2 short-K), not from moving columns.

## 6. Model config, from `text_config`

40 layers, `n_routed_experts 384`, `num_experts_per_tok 6`, `n_shared_experts 1`,
hidden 5120, `moe_intermediate_size 2304`, vocab 129280,
`num_nextn_predict_layers 3`.

## 7. AFD-TP3 wire path, measured

§3b's stream-semaphore plan does not run on GB10: `cuStreamWaitValue32` returns
801 there, so the deployed gate is host-side polling of the mapped flag word.
The cost of that polling is in the per-layer table below.

### Protocol

One RC QP per direction pair, three registered regions in `/dev/shm` on the
worker host (both sides map them; the client registers its own copies for RDMA):

| Region | Size | Role |
|---|---|---|
| `.slab` | 40×81920 in + 40×81920 out | activation payload in, hidden state out, one slot per layer |
| `.flags` | 5120 | in-flag per layer (stride 64) at 0, out-flag per layer at 2560 |
| `.rout` | 40×384 | router ids `[rows,6] i32` then tw `[rows,6] f32`, one slot per layer |

Per layer the client posts two unsignaled RDMA WRITEs on the data QP, routing
first, payload second, so the worker can never observe the in-flag ahead of the
routing it applies to. The worker polls the in-flag from the host, reads ids/tw
from `.rout` and the activation from the in-slot, replays this layer's graph,
and raises the out-flag only after the graph's D2H into the out-slot has
completed (`event.synchronize()`), after which the client reads the out-slot.

Flag byte 4 is a session epoch. A worker that starts against a region holding a
previous session's nonzero flags first waits for the wire engine's epoch bump
and then serves; without it the worker either serves stale flags or deadlocks.
Verified live on every session (`epoch 1 -> 2`, `2 -> 3`, `3 -> 4`).

### Per-layer cost, R=1, 30 layers resident

| Run | Worker serialized | Client serialized | Phases (µs/layer) |
|---|---|---|---|
| first, routing staged pageable | 412.7 mean / 388.4 p50 round trip | — | rout 48.8, h2d 21.4, launch 9.3, sync 295.6 |
| pinned routing + pin_x staging | **377** | 404.4 mean / 379.5 p50 / 420.8 p95 | rout 18.6, h2d 17.8, launch 5.6, sync 303.5, out-copy 23 |
| + captured D2H into the out-slot | **352** | 360.6 mean / 353.7 p50 (rerun p50 355.7) | rout 18.8, h2d 14.1, launch 8.1, sync 294.5, out-copy 0 |

The out-slot D2H is captured inside each per-layer graph
(`dst.copy_(self.out[r], non_blocking=True)` where `dst` is a bf16 view of the
registered `.slab` mapping), which removes the 10 KB device→host staging copy
plus its blocking pageable write: 377 → 352 µs/layer. Registering the mapping
is what makes the graph capturable (`cudaHostRegister(addr, SLAB, 0)` →
`cudaError.success`, `is_pinned() == True`, D2H replays correctly).

Copying *directly* from the registered `.rout`/`.slab` mappings into the device
buffers instead of staging through pinned buffers is slower and was reverted:
rout 21.0 vs 18.8 µs, h2d 17.8 vs 14.1, with heavy outliers (client mean 459.5
vs 360.6 µs) because torch re-queries the pointer attributes on every copy.

### Saturating the link, and why it does not matter

Posting a whole token's 30 layers up front (the stress round) costs 415–701
µs/layer and stretches the host-side `h2d` phase to 85–370 µs while
`rout`/`launch`/`sync` stay at 19/9/295: host staging out of the RDMA-target
mapping stalls once the inbound WRITEs saturate the link. Attention serializes
layers on P7, so the peer can never post a token's stack ahead; the serialized
numbers above are the deployment path.

### Wire fidelity

`gx/wirecheck.py` reloads one layer from the pack, reads the payload, the
routing and the out-slot back out of the slab, and replays the same kernel:

| Layer | cos | rel_l2 | max abs diff |
|---|---|---|---|
| 5 | 1.00000238 | 0.000000 | 0.00000 |
| 29 | 1.00000286 | 0.000000 | 0.00000 |

The out-slot is bit-identical to an in-process recompute from the received
payload, routing and that layer's real weights. Routing checksums match at both
ends for every round (`isum=34056 wsum=30.000001` at 30 layers).

### Consequence

40 layers at 352 µs (worker) = 14.1 ms/token = 71 tok/s of one GX's expert
chain; at the client's p50 of 355.7 µs (attention mocked at 0) = 70 tok/s.
Kernel time including the D2H is 294.5 µs/layer, i.e. 11.8 ms = 85 tok/s if all
host overhead vanished. Of the remaining 58 µs/layer, rout staging is 19,
activation staging 14, graph replay 8, and the in-flag poll plus out-flag set and
read ~8; the rest is end-of-layer poll quantization (93–100 flag polls per layer).
Mia's 74.9 tok/s at 40 layers is 334 µs/layer of *total* decode. In the earlier
verified pre-ingest-fix end-to-end graph the P7 work and this chain are
serialized: 24.36 ms/token of GPU busy is ~10 ms of P7 arithmetic plus
14.31 ms of seam wait, so the expert-chain ceiling above is not an end-to-end
figure and parity is not limited by P7 attention alone.

Graph-ingest v7 reduces the captured per-layer expert body at n=1 to ~225 µs
(§5); the 352 µs worker and 71 tok/s above are the earlier CUTLASS-captured
budget. The previous campaign's final e2e with v7 was 53.73–53.94 tok/s
(PLAN W7); the 2026-10-21 campaign measures 55.57 tok/s median (PLAN W7.1),
while the 2026-10-22 round baseline is 55.67 tok/s median with strategies in
progress (PLAN W7.2). The p7.2 node trace in §5 covers the earlier v7 path;
p7.1 remains the pre-ingest baseline. Its
steady 63-step window is 18.263686 ms/step with 7.624559 ms/step seam wait. At
75 tok/s the budget is 13.333 ms/token, so ~18.55 ms/token leaves ~5.2 ms/token.
The GX wait is still dominant and reducible, but not all of it is necessarily
transport.

### 40 layers resident: it fits, with ~1.5 GiB to spare

The arena is 2.7675 GiB/layer at C=864 / CP=896, i.e. 110.7 GiB for all 40
layers, against a 121.6 GiB unified pool (128 GB nominal minus firmware and OS).
That 10.9 GiB of nominal slack is mostly *not* available without work:

| Consumer | GiB |
|---|---|
| system + container | ~3 |
| page cache of the pack, resident before the run | 22 (drops to 0 on `drop_caches`) |
| torch caching allocator, load staging | ~1 |
| private CUDA graph pool per (layer, rows) | ~0.05–0.1 each |

Measured: with a warm page cache the load reached 110.7 GiB with 1.0 GiB free
and the container was OOM-killed during graph capture. With the page cache
dropped before start (`echo 3 > /proc/sys/vm/drop_caches`: host free 118 GiB,
CUDA free 115.7 GiB), `torch.cuda.empty_cache()` before capture, and one
`torch.cuda.graph_pool_handle()` shared by all 40 graphs, the same 40 layers
load in 69.7 s, capture 40 graphs in 1.2 s, and serve with 1.4 GiB free. No
streaming, no expert eviction.

Serving 40 resident layers, R=1, 5 serialized rounds:

| | µs/layer | ms/token | tok/s |
|---|---|---|---|
| worker round | 358.5 (rout 18.5, h2d 14.1, launch 8.2, sync 301.1) | 14.34 | 69.7 |
| client round trip p50 / mean / p95 | 361.9 / 402.6 / 488.0 | 14.48 | 69.1 |

`sync` grows 6.5 µs against the 30-layer run (shared graph pool, slightly worse
instruction/constant locality), which is the cost of fitting all 40. Routing
checksums at 40 layers: `isum=42576 wsum=40.000001` on both ends.


### The Python seam: what the P7 runtime actually drives

The serving runtime is Python, so the transport is also driven from Python:
`p7/afd_session.py` wraps the same wire engine (now a CPython extension,
`_qmesh_afd`, built from `qmesh_afd_peer.cpp` with `-libverbs -lcuda -lcudart`
and no torch headers) and `p7/afd_smoke.py` exercises it layer by layer. The
extension exposes the registered slab as `memoryview`s, so the device copy is
`torch.frombuffer(...).copy_(...)` into a pinned-mapped host region — no torch
C++ API, no pybind11.

One correction to the protocol above came out of this: the sequence number is
per *round*, not per layer. The worker serves a round as `layers` sequential
layers under a single sequence and compares each layer's in-flag against it, so
the client must bump once per decode step (`Peer.begin_round()`) and post all
40 layers with that number. A per-layer increment desynchronizes the two ends
after the first layer and stalls the round.

Identity mode (worker copies in-slot to out-slot, no weights, 40 layers × 2
measured rounds after 1 warmup):

| | µs/layer |
|---|---|
| post (routing + payload staging, 3 WRs) | 27.3 (p50 26.8) |
| wait (out-flag poll) | 41.0 (p50 40.5) |
| fetch (out-slot READ + pinned H2D) | 14.0 (p50 13.9) |
| round trip | **82.3** (p50 81.4, p95 87.2) |

`RESULT identity mismatches=0 of 80 layers`: every fetched partial is
bit-identical to the activation posted for that layer, so slot addressing,
flag ordering and the Python-side copies are all correct. 11.4 flag polls per
layer. This is the seam's floor — no expert compute, no attention.

40 layers resident, `--mode moe`, R=1, 4 rounds (3 measured):

| | µs/layer |
|---|---|
| post | 26.6 (p50 26.0) |
| wait | 369.8 (p50 369.7) |
| fetch | 13.7 (p50 13.4) |
| round trip | **410.1** (p50 409.0, p95 422.6) |
| worker round | 362.3 (rout 18.7, h2d 23.4, launch 8.2, sync 312.0) |

119.7 flag polls per layer. Against the C++ client on the same arena
(402.6 mean / 361.9 p50) the Python path costs **+7.5 µs/layer at the mean**,
i.e. 1.8%: three cross-language calls, a `torch.frombuffer`, and the routing
`copy_` per layer. The worker's `h2d` reads 23.4 µs here against 14.1 in the
40-layer C++ session because the page cache is cold in this run (1.2 GiB free,
the pack cannot stay cached); that difference alone is most of the 10 µs
between the two worker rounds.

Cross-end routing is now checked per round rather than only at the pipeline
sentinel, since unbounded serving has no sentinel: the worker's round line
carries `rout_recv isum/wsum` for the 40 layers it just consumed, and it equals
the client's `rout_sent` for that round times 40 —
`718→28720`, `1325→53000`, `741→29640`, `wsum 1.000000→40.000001`.

### Both GX ranks in one session: 2 × TP-slice peers

The AFD-TP3 split puts the routed experts on two GX hosts, so the client drives
two peers per layer. The extension and the smoke script were built for N peers
(`--peer DEV:GID:MGMT_IP:PORT`, repeatable): per layer it posts to every peer,
waits on all of them, then fetches from each. The two GX hosts sit on the two P7
NICs, one rail per GX host, each on its own P7 RDMA device.

Identity mode, 40 layers, 3 measured rounds after 1 warmup, both peers live:

| | µs/layer |
|---|---|
| post (per peer, 2 peers) | 27.2 |
| wait (both peers) | 22.8 |
| fetch (per peer, 2 peers) | 13.4 |
| round trip | **104.1** (p50 102.9, p95 117.2) |

`RESULT identity mismatches=0 of 160 layer-fetches` — each of the two peers
returned the exact activation posted to it for all 80 layer-fetches. 9.7 flag
polls per layer. The pair costs two posts and two fetches serialized on the
client's single loop while the waits overlap; single-peer GX2 alone is faster
than GX1 (63.2 vs 82.3 µs/layer, wait 21.8 vs 41.0), so in the pair the slower
wait sets the round.

Moe mode, 40 layers resident on both 110.7 GiB arenas, R=1, 4 rounds (3
measured):

| | µs/layer |
|---|---|
| post (per peer) | 27.3 (p50 26.9) |
| wait (both) | 353.7 (p50 349.3) |
| fetch (per peer) | 13.7 (p50 13.7) |
| round trip | **435.8** (p50 430.1, p95 450.9, max 575.5) |
| worker round (GX1 / GX2) | 18.69 / 18.66 ms per 40 layers |

335 flag polls per layer, 2295 layers/s. The two worker rounds are within
0.2% of each other, so the pair is balanced: each rank runs its own column
slice (864 experts) and the client's round is `post×2 + worker + fetch×2`.
Per-layer worker phases after the warmup round: GX1 `rout 18.6, h2d 23.3,
launch 8.5, sync 312.3`; GX2 `rout 8.1, h2d 9.6, launch 4.5, sync 326.2` — the
difference is host-side (the routing slab and the activation H2D), not kernel
time.

Both workers received identical routing per round — `isum` 53360, 28720, 53000,
29640 with `wsum 40.000000` — because the client posts the full 2304-expert
routing to every peer and each worker slices its own columns; the four rounds
sum to the client's 4118 sent tokens × 40 layers. `sum |partials|` per round is
printed from the summed host views, the value the P7 attention would consume.

The sequence number is derived by the worker, not counted: the client's
layer-0 in-flag *is* the round number, and an epoch word in the same slab
(written by the edge at startup) resets the expectation. Restarting the client
and both edge processes — workers untouched — reproduced the pair at 103.9
µs/layer in identity mode with 0/160 mismatches, so a serving restart no longer
requires reloading the 110 GiB arenas. The documented cost: a client that
pipelines two rounds ahead would make the worker skip a number, i.e. it may not
have more than one round in flight per peer.

## 8. Campaigns (2026-10-21 and 2026-10-22)

### 2026-10-21 campaign (measured)

Two installed bit-exact changes target the expert body and the P7 seam. The
combined engine measured **55.62 / 55.57 / 55.79 / 55.34 / 54.75 tok/s** over
five 256-token requests (wall **4.60 / 4.61 / 4.59 / 4.63 / 4.68 s**), **median
55.57 tok/s** versus the **54.02 tok/s** baseline median (**+2.87%**). Both GX
workers now run the staged body and the engine is healthy. It is a combined
result; no separate e2e attribution is inferred. The prior 53.94 / 53.89 /
53.73 tok/s and 53.39 tok/s restoration remain historical, and **>75 tok/s
remains unmet**.

**GX cooperative scale staging** (`GX_STG=8`; block-scale sectors in 4 KiB of
shared memory per block). Real pack rows 1/2/6/16 and random arena rows
1/2/4/6/8 are **bit-identical** to the pre-staging body (int16 diff 0,
max|Δ| 0). Rows=1 graph replay is **213.5 → 209.2 µs/layer** (kernel
211.5 → 207.1 µs, −4.4 µs/−2.1 %); rows 2–4 gain 1.6–8 µs; rows 6–16 tie.
`GX_STG=10` gains another 1.4 µs at rows=1 but costs a block of occupancy, so
8 is deployed.

**P7 Seam de-latency.** Fused amax+finish (last-block `fin` release/acquire,
amax re-armed in publish) removes the memset and finish launches: post
**5 → 3**, collect **2** (wait, combine). The single-post GPU ordering
invariant is preserved: one amax/finish → pack → publish chain per layer,
`ATOMG.E.MAX.STRONG.GPU` → `MEMBAR.SC.GPU` → `ATOMG.E.ADD.STRONG.GPU` →
`MEMBAR.SC.GPU` → volatile `LDG.E.STRONG.SYS`, and the first combine store after
`BAR.SYNC`. The installed candidate is source-matched to the byte-identical A/B
build and adds only a `fin` reset; arithmetic and that ordering are unchanged,
so there is no new quality or arithmetic change versus the starting baseline.
Old/new
harness kernel dump is **byte-identical at 697,713,779 bytes**; `quant_test` is
**byte-identical over 60 arrays**. Median-of-3 post rows=1 **9.48 → 7.42 µs**,
rows=6 10.26 → 9.17, rows=4096 **1470.5 → 1479–1483 µs (+0.7 %)** tradeoff;
combine rows=4096 np4 bf16 **~5.18 → ~4.95 ms** (−4 %), np2 and f32 np3
neutral.

**Serving validation.** A 64-token run produced exactly **2560 posts and 2560
completed reads per GX rank** (final **64,280 posts**, rx
**[64,280, 64,280]**), with **0 failed, timeouts or nonfinite**. Coherence:
12 prompt + 16 completion in **0.412 s** returned `Paris`; **3512 + 16** in
**1.801 s** and **6662 + 16** in **2.040 s** also returned `Paris`. Effective
prompt rates **1950** and **3266 tok/s** include decode; they are not pure
prefill and imply **no prefill speedup claim**.

The five-run result above is end-to-end, but no table row in §5 is an
end-to-end tok/s result. Current deployment still uses vLLM's **pinned-host
MXFP4 Engram** (~188.8 GiB), not the planned Gen5 NVMe FP8 reads. **DSpark was
off** (`SPEC_TOKENS=0`) in the pre-DSpark round below; measured DSpark results
follow and are in PLAN W8.

### 2026-10-22 pre-DSpark round (COMPLETE; final integrated PASS)

DSpark stayed off (`SPEC_TOKENS=0`); TP1/GPU0 unchanged. Round baseline:
**55.23 / 55.74 / 55.67 tok/s, median 55.67** over three runs. Final integrated
actual `bg179` is **PASS** and engine healthy: five 256-token runs
**56.89 / 56.67 / 56.86 / 56.86 / 56.87 tok/s**, median **56.86**, wall
**4.50 / 4.52 / 4.50 / 4.50 / 4.50 s**. Against **55.67**, that is **+2.1376%**;
the 2026-10-21 **55.57** is historical and not the gain baseline. The next
round tested the **120 tok/s with DSpark across the eight context types** goal;
that target was not a forecast, and it was not reached (W8).

A 64-token status check added **2,560 posts** and **2,560 completed reads** on
both ranks, final **64,280 posts** / rx **[64,280, 64,280]**, with **0**
timeouts, failed or nonfinite. Coherence: **12+16** in **0.359 s**, **3512+16**
in **1.689 s**, **6662+16** in **1.986 s**, all `Paris`-coherent. Effective
prompt/request rates **2079/3354 tok/s** include decode, not pure prefill.

Exact-6 dispatch is **accepted and deployed** via the `bg145` reload, with
readiness confirmed on both hosts and live proof complete. `gx_ingest.py` adds
`EXACT_ROWS={6}` and `graph_rows(n)=n if n in EXACT_ROWS else bucket_of(n)`,
used by `Layers.replay` and `capture_round`; `gx_expert_worker.py` changes
docstrings/log text only
(2 files, +51/-19; no kernel, CUDA, arithmetic, CLI or
`GX_WORKER_FLAGS` change). Two independent real-pack 2-layer benches give
1282 → 1251 µs/layer (+2.4%) and 1275 → 1246 (+2.3%), ~31 µs/layer or ~1.2 ms
per 40-layer round; the n=1 control is 0.6% apart. Exact-6 and bucket-8 are
bitwise identical on valid and sentinel (`ids=-1`) posts, twice; ingest x/ids/
tw are byte-exact vs wire; Triton cos=1.000000, rel 2.5e-6..7.2e-5; the
4096→1→6→16→6 transition and flag/isum/pad/invalid/finite checks are clean.

Live proof: five 40-layer rounds both rails PASS; repeat B6 == A6 bitwise
**80/80 layer-peer**, C8[:6] == A6 **80/80** with the same amax, E8 == C8
**80/80**; `ids=-1`/`tw=0` exact zero **all 40 on both**; fatal/routing
failures **0**. Client steady B6 **~1554.5 µs/layer** and E8
**~1917.5 µs/layer** are observation only, not a causal exact-6 gain. Accepted
gain evidence remains the isolated **+2.4%/+2.3%**.

Synthetic uniform ids over 384 overstate real-pack cost ~1.2x. GXReuse's
real-id, real-pack stage-1 A/B (committed `stage1_kernel`, `GX_GROUP=0`,
graph-captured, 8 draws x 50 iters x 7 reps, median; modal-expert padding,
random `tw`) gives L6 median **+3.20%** (mean +5.79%, range +0.3..+21.3%) and
L0 median **+3.11%** (mean +2.12%, range -0.2..+4.3%) of stage 1. This supports
the direction but does not establish the layer gain; do not convert stage-1
share into a layer percentage. The accepted six-row layer gain remains the
**1282 → 1251 (+2.4%)** and **1275 → 1246 (+2.3%)**. No 25% claim or projection
beyond 1.2 ms/40 layers. Exact-6 kernel files remain untouched. The width-6
graph was captured on demand at the first 6-row post and has now run the five
40-layer proof rounds; see above.

The 32-capture 8-type trace is eager-vs-graph exact and gives mean duplicate-slot
fractions of **17.8 / 32.7 / 40.6 / 45.7%** for consecutive M2/M4/M6/M8 and
**16.0 / 22.4 / 26.7%** for independent M4/M6/M8; the deployed body is at
**94.3%** of the in-situ stage-1 bandwidth ceiling, but GXReuse **REJECTS** all
real-trace aggregate variants: they are **1-4% slower** in the high-duplicate
case, while baseline L2 already dedups to **1.04-1.14x the perfect-dedup
floor**. Synthetic identical-group runs (**+15-22%**) did not transfer. The
kernel branch is fully reverted to md5 HEAD **29bf66**; no bench remains. The
p7.3 pre-change baseline profile (no final/optimized capture this round) shows
steady **18.32 ms/token**, seam **7.77 ms (42.4%)**, zero overlap, first **318
ms** excluded, **854 µs** head sampler outside the graph; the **50.87 tok/s** in
that run has the profiler attached and is not a speed result.

P7 seam candidate: vector combine **12.588 → 3.598 µs** at rows=1; automatic
pack vector at >=128 rows **43.45 → 29.46 µs** at rows=128 and **1454 → 814 µs**
at rows=4096. Isolated kernel times, no e2e.

mHC/P7 fused quant: final consumer proof passes finite, NaN/Inf and mixed-block
stock bit-exactness plus graph capture+replay on the actual served image; fixed
kernel frozen **c7c1d226** (PTX same **121e**). Mechanism estimate **0.69
µs/boundary × 40 = 27.7 µs/step (0.15%)** is below run noise; prior 5.3 µs/layer
and 1.2 µs estimates are superseded. There is **no isolated fusion-gain
attribution**. Repro: `p7/mhc_quant/test_mhc_quant_fused.py`,
`p7/mhc_quant/smoke_mhc_quant_consumer.py`, and
`p7/mhc_quant/patch_vllm_mhc_quant_sm120.py` (regeneration patcher). The source
composite includes 3 full upstream overlay files, but the patcher reproduces the
actual image. `MHC_QUANT=1` default launch mounts 4 RO; `MHC_QUANT=0` only the
experimental A/B. Baseline full-graph `Spec=0` unchanged; the DSpark retention
overlay is mounted only when `SPEC_TOKENS>0`. Integrated actual **bg179 PASS**;
five 256-token median **56.86 tok/s**.

### 2026-10-22 DSpark round (COMPLETE; measured)

DSpark ran on the served engine: four primary configurations (k0/k1/k3/k5) x
**64 forced timed rows** each (8 types x 2 prompts x 2 pairs x 2 repeats), plus
k5 RG6 and paired-k5 conditions, for **384 timed rows**. Rows are
performance-only. Pair medians, `1024p_128c / 128p_1024c` tok/s: k0 **57.37 /
57.43**, k1 **72.69 / 75.32**, k3 **82.13 / 96.70**, k5 early **78.49 /
102.82**; k5 paired **76.10 / 97.38**; k5 RG6 **76.31 / 98.81**. **120 tok/s
was not reached**: no aggregate pair median reaches it, and only **4 of 64**
k5-early timed requests reach it (1 short, 3 long), not per-type or per-context
medians. The k3 arithmetic ceiling is **114.1 tok/s** (4 tokens/step at 35.07
ms/step).

RG6 was executed for real (**8,742 n=6 graph-replay rounds per host**, 0 n=6
fallbacks, 0 spin timeouts) but is **REJECTED**: paired Δ +0.67 tok/s short
(95% CI [-2.65, +5.15]) and +0.19 long ([-5.83, +9.79]), 8 wins / 8 losses in
each pair. Rollback is clean and **per-layer k5 is retained**.

The drafter is **loaded** (`97 params`; **88.93 GiB** weights including the
7.388 GiB drafter; boot **88.92 GiB**), with draft layers 40-42 verified local
on P7 and all 40 main layers GX-only. k5 runs `SPEC_TOKENS=5`,
`GPU_MEM_UTIL=0.97`, `DSPARK_EXTRA_GIB=1`, captures `1,5,6`, retention extra 6,
KV capacity **20,227 tokens**, prefix caching off. Quality probes (official
chat, thinking off, 16 greedy at `temperature=0` + 8 sampled, twice per
configuration) are k0 15/16 twice, k3 15/16 twice, k5 15/16 then 14/16, sampled
8/8; there is **no formal quality or lossless claim**, and greedy outputs are
not bit-identical across passes even at k0. Retention is **CPU-proven plus
runtime-functional** (no null/evicted-block crash, no OOM, zero request
errors); **no full suffix bitwise proof** was recorded. Per-type medians,
efficiency, RG statistics and quality detail are in **PLAN W8** and
the DSpark final-results artifact.

Final DSpark repro: packaged `scripts/launch_engine_graphs.sh` with default
`MHC_QUANT=1` (4 RO mounts), vector-combine plus automatic pack at `>=128` rows,
exact-6 GX workers (`--ingest graph --swiglu-limit10 --decode-kernel cuda-swz`),
`SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1 CAPTURE_SIZES=1,5,6
MAX_CAPTURE=6`, TP1/GPU0. Source paths:
`p7/mhc_quant/patch_vllm_mhc_quant_sm120.py` (regenerates the actual image),
`p7/mhc_quant/test_mhc_quant_fused.py`,
`p7/mhc_quant/smoke_mhc_quant_consumer.py`, `p7/afd_overlay/`,
`gx/gx_ingest.py`, `gx/gx_expert_worker.py`, and
`scripts/launch_engine_graphs.sh`.

Bounded memory, pre-DSpark (historical): **81.34 / 82.16 / 1.75 / 0.06 / 1.56
GiB** (model loading / consumed weights+non-torch / peak activation / CUDA graph
/ KV in use), with **89,318 MiB used and 7,932 MiB free (7.744 GiB)**. DSpark
loads **88.93 GiB** weights (boot **88.92 GiB**) including the **7.388 GiB
drafter**; k5 reports **93,506 MiB used / 3,744 MiB free**, RG6 **94,262 /
2,988**, and observed peak **94,312 MiB used with 2,938 MiB reported free**
(earlier k5 snapshots 3,744 MiB), so measured headroom is **~2.9–3.7 GiB**.
There was no OOM across 384 timed + 144 quality rows. Rebalancing remains
ruled out. **>75 tok/s remains unmet; 120 tok/s was not reached.**
