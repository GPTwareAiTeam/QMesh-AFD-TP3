# AFD-TP3 implementation plan: DeepSeek-V4.1-Flash-1017 (NVFP4) on P7 GPU0 + GX1 + GX2

**Status:** plan plus measured results through 2026-10-22. Sizes come from the safetensors headers. Speeds are estimates unless a section reports a measurement; W5, W7, W7.2 and W8 distinguish current results from historical runs.

**2026-10-21 campaign (historical):** this campaign starts from
**53.46 / 54.02 / 54.03 tok/s** (three non-spec runs, same recipe). Two
bit-exact changes are installed: cooperative GX block-scale staging and Seam
post/collect de-latency. The combined served engine measured
**55.62 / 55.57 / 55.79 / 55.34 / 54.75 tok/s** over five 256-token requests
(wall **4.60 / 4.61 / 4.59 / 4.63 / 4.68 s**), **median 55.57 tok/s** versus
the **54.02 tok/s** baseline median, **+2.87%**. Both GX workers now run the
staged body. The result is **combined only**; do not attribute a separate e2e
gain to staging versus Seam. Neither change alters arithmetic or output
quality relative to the starting baseline. **>75 tok/s remains unmet.**
The prior campaign's 53.94 / 53.89 / 53.73 tok/s and 53.39 tok/s restoration
in W7 are historical. Deployment deltas from the plan: Engram is the vLLM
**pinned-host MXFP4 offload** (two tables, 94.42 GiB each, ~188.8 GiB), not the
planned Gen5 NVMe FP8 per-token read; **DSpark was off** in that campaign
(`SPEC_TOKENS=0`). Planning text in §1/§3/W5 remains complete; W8 records the
later measured DSpark results and the retention boundary.

**2026-10-22 pre-DSpark round (historical; complete):** the round baseline was
**55.23 / 55.74 / 55.67 tok/s**, **median 55.67** over three measured runs; the
listed non-spec strategies ran together, and the final integrated actual `bg179`
was **PASS** at five 256-token median **56.86 tok/s**, **+2.1376%** vs **55.67**.
Exact-6 remains accepted/deployed and GX-reuse remains rejected; non-spec **>75
tok/s was not reached**. DSpark was off; the retention prerequisite was then
CPU-proven only. Exact-6 readiness was confirmed on both hosts and live proof
complete. The mHC/P7 fused consumer proof passes
finite/NaN/Inf/mixed-block stock-bit-exactness, with no isolated gain attribution.
Loaded memory was **81.34 GiB model / 82.16 GiB consumed** with **7,932 MiB
(7.744 GiB) free**; rebalancing was ruled out. Those memory figures are
historical. The pre-DSpark round is W7.2. The next round tested the **120 tok/s
across the eight context types** target; it was then a goal rather than a
forecast, and W8 is its ledger. The 2026-10-21 campaign's 55.57 tok/s median
above is historical.

**Current round (2026-10-22, DSpark; complete; W8):** measured on the served
engine over four primary configurations (k0/k1/k3/k5) x **64 forced timed rows
each** (8 types x 2 prompts x 2 pairs x 2 repeats), plus k5 RG6 and paired
comparison conditions. Pair medians, `1024p_128c / 128p_1024c` tok/s: k0
**57.37 / 57.43**, k1 **72.69 / 75.32**, k3 **82.13 / 96.70**, k5 early
**78.49 / 102.82**. Best retained per-layer k5 is **76.10 / 97.38** under the
paired conditions; RG6 is **76.31 / 98.81** with no paired gain (95% CI crosses
zero) and is rejected. **120 tok/s is not reached**: no aggregate median reaches
it, and only **4 of 64** early-k5 timed requests do (1 short, 3 long). The
drafter is **loaded** (`DSpark draft model loaded: 97 params`; **88.93 GiB**
model weights including
the 7.388 GiB drafter). Retention is **CPU-proven plus runtime-functional**, not
a full suffix bitwise proof. Quality probes (official chat, thinking off, 16
greedy + 8 sampled, twice per configuration) are k0 **15/16, 15/16**, k3
**15/16, 15/16**, k5 **15/16, 14/16**, sampled **8/8** throughout; there is no
formal quality or lossless claim. W8 is this round's ledger.

**DSpark k5 RG6 arm: REJECT (2026-10-22).** The paired same-condition
comparison showed no benefit: short **76.31 RG6 vs 76.10 per-layer** tok/s,
long **98.81 vs 97.38**, with paired median deltas **+0.67 [-2.65, +5.15]** and
**+0.19 [-5.83, +9.79]** tok/s (95% CI crosses zero; 8 wins / 8 losses per
pair). RG executed for real: **8,742 n=6 graph-replay rounds per host** across
all 40 layers, with **0 n=6 fallbacks** and **0 spin timeouts**. The rollback is
clean and per-layer k5 is the retained/best configuration. The GX production
worker remains the baseline `6c365353048d6d75115d7f1265cafde5` with no round-graph
flags; `gx_worker.sh` is restored to baseline
`74162f5d0981d89f144e85b5c2432cb6`. The rejected gate, arm helper and CPU test
were removed (`gx/gx_round_arm.py`, `scripts/arm_round_graph.sh`,
`scripts/test_gx_round_arm.py`), and the `gx_worker.sh` staging requirement for
`gx_round_arm.py` was reverted. Exact-6 remains accepted and deployed; no
arithmetic, spin timeout or 3 s host guard change is retained from this
candidate.

**Scope:** text-only, DSpark speculative decoding on the target deployment (k5 per-layer retained; k1/k3 measured points), thinking and tools on. Vision comes later.
**Hard constraints:**
- P7 GPU1 is never used. GPU2 drives the display.
- No FP2 and no re-quantization: routed experts use the bit-exact NVFP4 transcode; everything else keeps its shipped FP8/BF16/MXFP4.

---

## 1. Topology (AFD-TP3 = P7 attention + weighted MoE tensor parallelism)

| Device | Owns |
|---|---|
| **P7 GPU0** (RTX PRO 6000 Max-Q, sm_120, 97,887 MiB total, 97,248 MiB free when idle) | Embedding, all 40 attention blocks (window + compressor + indexer + candidate + sparse attention), Hyper-Connections, router, shared experts, head, sampling, DSpark drafter (`mtp.*`), the whole KV cache, scheduler, Engram compute. Also routed-expert columns **[0:576)** of every expert. **P7 GPU1 and GPU2 are never used**: every P7 container gets GPU0 only (`--gpus device=0`, TP=1). |
| **P7 host RAM (pinned MXFP4 Engram)** | Engram tables are loaded through vLLM's **pinned-host MXFP4 offload**: two tables (`384,006,168 × 256` and `384,016,682 × 256`), **94.42 GiB each / ~188.8 GiB total**, resident in pinned host RAM. The source tensors are the original FP8 shards 47–48 on the Gen5 NVMe, read once at boot; they are **not** read from the SSD per token and are **not** GPU memory. The planned per-token Gen5 NVMe FP8 path was not used. Per token the engine performs 48 row reads (2 layers × 3 n-grams × 8 heads). |
| **GX1** (GB10, sm_121, budget 0.92 × 121.6 = 111.9 GiB) | Routed-expert columns **[576:1440)** of every expert, every layer. Stateless. |
| **GX2** (GB10, sm_121, same budget) | Routed-expert columns **[1440:2304)**. Stateless. |

Per MoE layer, for each decode or verify step (R ≤ 48 rows; R = 6 for single-stream DSpark):

1. P7 runs the router and sends `x_norm` as FP8 e4m3 rows under one per-post
   scale, plus top-6 ids and routing weights, to GX1 and GX2. That is about 31 KB
   at R = 6.
2. All three ranks run SwiGLU (with clamp 10) on their own columns of the selected experts:
   - gate/up: rows = the rank's columns
   - down: K = the rank's columns
   - The routing weight is applied once, in the intermediate step, as DeepSeek's reference `Expert.forward` does.
3. GX1 and GX2 return BF16 partials (R × 5120, about 61 KB each).
4. P7 sums the partials in FP32, in a fixed order: P7 slice, then GX1, then GX2, then the shared expert. It then applies the Hyper-Connections post step and continues.

Two ranks never talk to each other. The GX1–GX2 link is unused. The only data paths are the two 100 Gb/s P7 rails.

## 2. Memory ledger (NVFP4 pack, measured header bytes)

The NVFP4 routed experts are 284.77 GiB, 6% larger than native MXFP4 (268.95 GiB) because of the 1×16 scales. Everything is computed per expert column (2304 columns per expert).

| | P7 GPU0 | GX1 | GX2 |
|---|---|---|---|
| Routed-expert slice | 576 cols → 71.2 GiB | 864 cols → 106.8 GiB | 106.8 GiB |
| Attention, shared, router, HC, Engram projection, embedding, head, DSpark (`mtp.*`) | 16.6 GiB | — | — |
| CUDA context and library workspaces | ~1.5 GiB | ~1.0 GiB | ~1.0 GiB |
| KV (~0.9 KB per token), activations, graph pools | ~5.0 GiB | ~1.0 GiB (row and partial buffers, graphs) | ~1.0 GiB |
| **Total / budget** | **~94.2 / 95.6** | **~108.8 / 111.9** | **~108.8 / 111.9** |

This is tight but closes. Levers if P7 is short after the Gate A measurement:

1. Move the embedding (1.23 GiB BF16, a pure gather) into pinned host memory.
2. Cap `max-num-seqs` and context to shrink the KV and activation reserve.
3. Last resort: P7 takes 512 columns, so GX takes 896 each. GX then sits at 112.7 GiB, about 0.8 GiB over 0.92, so this needs about 0.93.

The 128-column 4/7/7 split does not fit (GX would be −0.8 GiB at 0.92).

**Kernel granularity:**
- 576 = 9 × 64 and 864 = 27 × 32. NVFP4 blocks are 16 along K, so every cut is block-aligned and the global and input scales are per projection.
- The FP4 tensor-core MMA has K = 64, so the GX down-projection (K = 864) needs either a K tail of 32 in the kernel or zero-padding to 896 in the pack. Padding costs about 1.3 GiB per GX and still fits at about 110.1 GiB.
- The gate/up output columns (N = 864) only need N tiles of 32.

## 3. Weights (download started 2026-10-20 10:00 EDT; moved to a dedicated model disk and resumed 10:46 EDT)

The checkpoints live outside the repo in a dedicated model directory on a Gen5 NVMe mount. `tools/download.sh` fetches the pinned revisions there, and revisions are never floating. The artifacts are never committed: they are a separate download.

| Local dir | Source | Use |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-NVFP4/` | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4@5be99cd1…`, 399.5 GiB | All non-Engram weights, including **DSpark** as `mtp.0-2.*`: unchanged MXFP4 experts plus FP8 attention, 7.39 GiB. Routed experts are NVFP4 (bit-exact transcode of DeepSeek's MXFP4). Its own Engram tables are quantized relative to the original FP8 reference and are **not used**; the pinned-host offload is fed from the original shards below. |
| `DeepSeek-V4.1-Flash-1017-orig/` | `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e10…`: shards 47–48 only (188.8 GiB Engram FP8), plus config, tokenizer, index, `inference/`, `encoding/` | Source of the served Engram tables, mapped as `engram-L01`/`engram-L14`; reference implementation and official prompt encoder. |

Total about 632 GB, all on the Gen5 NVMe. `download.sh` originally passed explicit filenames together with `--include`, so `hf` ignored the include list and fetched only shard 48. Shard 47 was fetched separately afterwards, and the script has been fixed. Integrity check: `tools/verify_shards.sh` compares each shard's sha256 with the Hugging Face LFS ids (`tools/nvfp4.sha256`, `tools/orig.sha256`).

## 4. Workstreams

### W0. Read-only inventory (Gate A, part 1)
- P7: `nvidia-smi --query-gpu=index,uuid,memory.total,memory.used` to get the real free memory on GPU0 by UUID. Also collect `ibdev2netdev`, `rdma link`, GIDs on the two P7 RoCE devices, `nvidia-peermem` status, and the NIC firmware version.
- GX1/GX2: `MemAvailable`, driver/CUDA, the GX RoCE device GID, link rate. Confirm they are idle.
- Output: `INVENTORY.md` with exact numbers. The memory ledger in §2 is re-checked against it.

### W1. Pack builder (CPU only, on P7)
- Input: the NVFP4 pack. Output: three content-addressed rank packs plus `MANIFEST.json` (sha256 per tensor, slice bounds, pack version).
  - `p7.pack`: all non-expert tensors, `mtp.*`, and expert slices `w1/w3[0:576,:]` and `w2[:, 0:576]`, including the matching scale rows/cols, `weight_scale_2`, and `input_scale`.
  - `gx1.pack`: experts `[576:1440)`. `gx2.pack`: experts `[1440:2304)`. Laid out for the GX kernel: expert-major, contiguous per layer, so one arena per rank. The down K-padding decision from §2 is recorded here.
- **Bit-exact check:** for every expert, concatenating the three slices reproduces the source bytes, so there are no gaps or overlaps.
- **Transcode check:** spot-check, by range reads, that the NVFP4 codes and scales decode to the same values as the original MXFP4 for sampled experts. This uses the repo's `verify_transcode.py` method.

### W2. Expert kernel (one source, built for sm_120 and sm_121)
- **Contract:** grouped SwiGLU expert slice for R ≤ 48 rows and up to R × 6 (row, expert) pairs.
  - Inputs: `x` BF16 [R, 5120], ids/weights [R, 6], the rank's slice arena.
  - Output: BF16 [R, 5120] partial; the P7 host sums partials in FP32.
  - Pipeline: gate/up GEMM, clamp (up to ±10, gate ≤ 10), SiLU × up, × weight, down GEMM.
- **Activation semantics, design:** DeepSeek production runs FP4 weights against **FP8 activations** (block 32, UE8M0 scales,
  `act_quant`). That is the accuracy target; LibertAI's `input_scale` calibration is for vLLM's W4A4 path and is **not used**;
  the CUTLASS fallback's W4A4 path is separate and uses its own activation quantization.
- **Activation semantics, as implemented:** the v7 `cuda-swz` decode body is **W4A16**: NVFP4 weights dequantized in registers,
  BF16 activations, FP32 intermediate and accumulation, with no activation quant. It reads the arena's swizzled block scales directly,
  so the 13.05 GB row-major scale duplicate that v6 needs does not exist. Rows above the 16-row fast-dispatch threshold stay on the
  CUTLASS `clamped` body, which quantizes the SwiGLU intermediate to FP4 (**W4A4**).
  - On the real pack, one layer at n=1/6/16 gives v7 relL2 **1.65–1.67e-3** against an FP32 dequant reference; v7 and v6 are
    **bit-identical** at every tested size. v7's W4A16 arithmetic differs from the old W4A4 path by **~0.15 relL2**; that is
    informational, with no bit-exact claim against the old path and no completed quality evaluation.
  - Decode is bandwidth-bound, so W4A16 dequantization costs nothing here.
  - A native FP4 × FP4 MMA (`mxf4nvf4`) path is an opt-in prefill speed lever, only after a quality A/B.
- **Starting points** (all generic, not DeepSeek-specific):
  - TensorFold upstream `nvfp4_experts.cu` / `mma4.cuh` (Apache-2.0).
  - FlashInfer SM12x unified MoE (MXFP4/NVFP4, W4A16/W4A8).
  - QMesh's `fp2_fusion_pack` arena, route-map and descriptor design. The FP2 math is not used.
- **Prefill:** larger M. Rows above 16 use the CUTLASS `clamped` body; the v7
  `cuda-swz` body is decode-only by dispatch. For prefill, keep CUTLASS W4A4 or
  dequantize per expert to BF16 plus cuBLASLt grouped GEMM; decide by measurement.
- **Acceptance:** W1 checks the three-slice join bit-exactly. On real pack rows, v7's one-layer relL2 is 1.65–1.67e-3 against an
  FP32 dequant reference at n=1/6/16, and v7/v6 are bit-identical at every tested size; the CUTLASS W4A4 fallback is ≈3e-3
  (GX_FP4_DECODE_CEILING.md §3). Bit-identical across repeated runs, with no tactic/autotune nondeterminism (the GLM-5.3 lesson).
  The W4A16-vs-W4A4 gap (~0.15 relL2) is informational; model-level quality evaluation remains W6.5.

### W3. Transport (Gate A, part 2: the go/no-go)
- Two independent links: P7↔GX1 on one dedicated P7 NIC/RDMA device (its own /30 RoCE subnet), and P7↔GX2 on a second, independent P7 NIC/RDMA device (a second /30).
- **Bring-up:** NCCL 2.31.2, one 2-rank communicator per GX link, graph-captured `send`/`recv`.
- **Target:** a one-sided RDMA-write mesh: the payload plus a sequence flag written into the peer's registered buffer, with the consumer kernel spinning on the flag (`ld.acquire.sys`). This is the design that bertholomus measured at about 23 µs per exchange.
  - P7 side: GPUDirect into VRAM if `nvidia-peermem` and the NIC driver allow it. Otherwise use `cuMemHostRegister`ed pinned host buffers: at about 61 KB per BF16 return over PCIe5 the zero-copy read is ≤ 5 µs.
  - GX side: unified memory, host-registered.
- **Gate:** a 2 × 40-exchange-per-step loop inside one CUDA graph, at R = 1, 6 and 48.
  - Pass: ≤ 60 µs per layer round trip, about 2.4 ms per step, excluding compute.
  - Fail if > 150 µs per layer. Then AFD-TP3 cannot beat Mia's TP2, so stop and re-plan.
- Per-hop measurements from history, for calibration:
  - NCCL P2P 4 KB: 354 µs p50 (PP3 gate; ungraphed, host-driven).
  - 3-rank AllReduce: 57 µs/op ordinary, 104 µs/op graphed.

### W4. GX expert worker (no vLLM)
- One persistent process per GX that loads its pack into a single arena. QMesh carry-over:
  - arena, release staging and fingerprint handshake from `canonical_gx_worker.py`
  - TCP control channel with length-prefixed, CRC'd, sequenced envelopes
  - current-boot readiness markers
- **Per step:** P7 posts a step header (row bucket, epoch, step sequence number). The GX host replays **one CUDA graph per row bucket**. That graph contains 40 iterations of: wait on the layer flag, run the W2 kernel, write the partial, raise the flag.
  - No Python or host work per layer. This is the GLM-5.2/5.3 lesson: host-driven per-layer work was 81% of wall time.
- **Stateless:** no KV, no request ids, no scheduler mirror.
  - Ownership handshake: the client reads the flag plane, preserves the epoch/ACK words, clears every in/out flag, bumps the epoch, then waits for the GX to publish `ACK == new epoch`. The worker publishes the ACK only after `torch.cuda.synchronize()`, i.e. after it has stopped reading the old round. The client clears the plane once more and posts its first round as sequence 1. On an epoch change the worker discards the old round without reading payload, routing or writing out-flags, so a resume can never be served from the previous round's slab. The edge's own startup clear preserves the epoch and ACK words before bumping the epoch, so neither the worker nor a client can observe a transient epoch 0. A missing ACK is a fail-fast, not a timeout to suppress.
  - CUDA/RDMA errors are fatal. Restart is cheap: the pack reloads from local NVMe.
- **Memory guard:** refuse to start below a configured `MemAvailable` floor, and watch it during serving (Mia's `memguard` pattern).

### W5. P7 runtime: vLLM, TP = 1, DeepSeek-V4.1
- **Base:** vLLM main with FlashInfer ≥ 0.7.1, plus:
  - #60762 (SM120 64-token kernel blocks; required, since v0.32.0 serves nothing on SM120 without it)
  - #62535 (DSpark/FULL-graph prefill padding fix; required for 1017 chat/thinking)
  - the `kacper-daftcode/vLLM-Moet` `tools/dsv41_sm120` patch set: DeepGEMM paged-MQA page sizes, MXFP4 indexer cache, MoE glue, mHC overlap, MXFP8 GEMV
- **Model changes**, kept small and in one module:
  - `AFDMoE` replaces `FusedMoE` in `deepseek_v41`. It runs the router locally, launches the transport send, runs the P7 W2 slice plus the shared expert, waits on the two return flags, and does the ordered sum.
  - It also runs for the DSpark verify rows. DSpark's own MoE (128 experts, top-3, MXFP4) stays entirely on P7 in vLLM's normal path.
  - **Engram source (deployed):** the loader takes only `p7` pack tensors plus
    the original FP8 table shards 47–48, mapped as `engram-L01`/`engram-L14`;
    the NVFP4 checkpoint's `engram_embed_quant: mxfp4` tables are not used. At
    boot vLLM places the tables in pinned host RAM through its offload path, so
    they are not read from the SSD per token.
  - **Planned, not deployed:** Engram lookup from the Gen5 SSD. A P7-side row
    fetcher hashes the n-grams on the host (or reads the hash ids back from the
    GPU), issues 48 O_DIRECT reads per token into a pinned staging buffer with
    Linux AIO or io_uring, and copies them to the GPU before layer 1 (and again
    before layer 14). The layer-14 fetch overlaps layers 1–13.
    - Budget: a Gen5 random 4 KiB read is about 60–90 µs at QD1 and far less at QD48. One batch of 48 reads must finish within the time layers 0–1 take (about 0.3 ms), so the read queue must be deep: all 48 issued at once.
    - Measure in W0/W7 (`fio` randread 4 KiB at QD48 on the model disk).
- **Flags (initial):**
  - `--tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice --language-model-only`
  - `--block-size 64 --kv-cache-dtype fp8 --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"fp8"}'`
  - `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`
  - `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`, with graph sizes in multiples of 6
- `CUDA_VISIBLE_DEVICES=<GPU0 UUID>` only.
- **Status (2026-10-21):** the engine served earlier as `dsv41-p7` on
  `127.0.0.1:8001` (`deepseek-ai/DeepSeek-V4.1-Flash`, `/health` ready 261–337 s
  after launch; FlashInfer autotune dominates), with both GX workers holding 40
  layers. That run was `--enforce-eager --no-enable-prefix-caching`, `SPEC_TOKENS=0`
  (no DSpark draft), `MAX_MODEL_LEN=8192`, `MAX_NUM_BATCHED_TOKENS=4096`; row
  counts seen were `[1]`, `[128]`, `[2828]`, `[4096]`. Prefill 3.1–3.5 k tok/s at
  2538 prompt tokens, 3.6–3.8 k at 6924; eager decode 6.2–8.6 tok/s (earlier
  measurement). The earlier full graph + `AFD_SEAM=1` decode (pre-ingest-fix) was
  41.0–41.4 tok/s; the >75 tok/s target is not met (W7). The previous campaign's
  final e2e is in W7; the 2026-10-21 campaign is W7.1 and the 2026-10-22 round is W7.2.
  In that pre-ingest deployment, each `(layer, rows)` pair captured its CUDA graph
  on first use, so the first request of a session was 3–4× slower than the second.
  The pre-fix ingestion deployment did not serve end-to-end: the launch took an
  illegal access on GX1 in `glayers.capture` after
  the 4096-row profile captured, then failed on the small rows. Production dummy
  routing's sentinel `-1` padding reached the CUTLASS maps; the dropped slots left
  `a_map` uninitialised, and the subsequent small-bucket capture inherited that
  stale routing and gathered at a recycled index. The fix initialises the capture
  ids/tw and normalises sentinel `-1` padding to expert 0 with weight 0, counted in
  `pad`; `ids < -1` or `ids >= 384` are counted invalid, suppress ready, and fail
  the worker closed, not silently zeroed. Pre-fix, the harness reproduces the
  illegal access; fixed capture passes `n=4096,1,16,1,6,16` with `max|diff|=0`
  against the exact-row reference. With graph ingestion, per-layer expert time is
  ~225 µs at n=1 vs 305–308 µs for clamped CUTLASS, 1232 vs 1358 at n=8, and
  2808 vs 2978 at n=16; fast `cuda-swz` dispatch is used only at ≤16 rows
  (GX_FP4_DECODE_CEILING.md §5). **Current:** both 40-layer GX workers are READY,
  restarted onto the staged `cuda-swz` body, with
  `--ingest graph --swiglu-limit10 --decode-kernel cuda-swz`; the engine is
  healthy from the packaged `scripts/launch_engine_graphs.sh` candidate (default
  full graph + Seam, `cuda-swz` workers; runtime smoke/bench verified; no
  permanent regression suite added). SessionFix handshake is proven on both
  hosts; the old session's counters show 5 of 40 incomplete and 35 old-round
  aborts, and the epoch ACK arrives before the first post. Combined campaign
  e2e: five 256-token runs at **55.62 / 55.57 / 55.79 / 55.34 / 54.75 tok/s**,
  **median 55.57**, **+2.87%** over the 54.02 baseline median and still below
  75; counts coherent; `Paris` coherent (W7.1). The W4A16 quality caveat from
  W2 remains.

### W6. Correctness ladder (one loaded setup; keep fixtures as regressions)
1. **Kernel:** W2 acceptance (above).
2. **Layer:** a CPU reference harness runs DeepSeek's `inference/model.py` one block at a time with streamed weights (about 7 GiB per layer, so it fits in host RAM). It compares, keyed by request, step, layer and op, the `Block` output against AFD-TP3 captures on the same input. Record n_diff, max/mean abs, relL2 and cosine — numbers, not digests.
3. **Model:** teacher-forced logits on 8 short prompts vs the streamed CPU reference: top-1 agreement and KL.
4. **Serving:**
   - greedy repeatability, fresh vs warmed
   - DSpark on vs off: token agreement is tracked, but bit-exactness is not required on SM120 (known vLLM numerics difference)
   - chat, tool calls, thinking
   - needles at 32K, 128K and 512K
   - cancel and abort mid-step, then confirm the next request is clean
5. **Quality:** GSM8K-100 and MMLU-200 (bertholomus' seeded subsets, so results compare directly), plus the 69-scenario tool eval used by the 2× and 4× Spark lanes.

### W7. Performance
- **Ledger per step:** P7 attention and indexer, P7 W2 slice, transport out/back per layer, GX kernel per layer, DSpark draft, sampling. All from one correlated trace, without summing overlapping spans.
- **Bench:** Mia's and tonyd2wild's prompt sets at C1/C2/C4; prefill at 8K–128K.
- **Reference points on GB10:**

  | Setup | C1 prose | C1 code | C4 aggregate | Prefill |
  |---|---|---|---|---|
  | Mia, 2× GB10, EXL3 2.9 bpw | 74.9 tok/s | 109.9 tok/s | 130 tok/s | ~2,150 tok/s |
  | tonyd2wild, 4× GB10, EXL3 3.5 bpw | 98.6 tok/s (C1 mean of 8 categories) | — | 198 tok/s | ~2,215 tok/s at 93K |
- **Levers, in order:**
  1. transport (RDMA flags, GPUDirect)
  2. GX kernel bandwidth efficiency
  3. P7 attention-phase overlap: the GX can prefetch the next layer's expert metadata
  4. split re-balance

- **Measured, in-slot as e4m3 (2026-10-21):** the post payload is now fp8 with one
  per-post scale in the routing header (`hdr[2]=1/s`, `hdr[3]=1`; `hdr[3]==0` still
  means bf16, so the two protocols coexist). The out-slot stays bf16: the
  attention host sums those partials in fp32.

  | Bench | bf16 in-slot | e4m3 in-slot |
  |---|---|---|
  | 1024 → 2538 prompt tok | 2.168 s, 1171 tok/s | 0.744–0.817 s, **3108–3492 tok/s** (8 runs, 3 engine sessions) |
  | 3072 → 6924 prompt tok | — | 1.830–1.972 s, **3592–3783 tok/s** (5 runs) |

  The first request after a (re)launch of either the engine or a worker is
  capture-polluted: every `(layer, rows)` pair captures its CUDA graph in band,
  25–30 ms per layer, so it reads 828–904 tok/s where the steady state reads
  ~3400. The short bench is noisy at the ~10% level (3108 vs 3410 within one
  session, same code); the long one sits at 3.6–3.8 k.

  Per layer at 2538 rows: `post 4.2 ms` (was 5.9, of which ~2.1 ms is now the
  50 Gb/s floor for 13 MiB; the rest is the amax reduction, the cast and the
  D2H), `r0 wait 7.9 + fetch 2.5`, `r1 wait 7.2 + fetch 1.3`,
  `local_between 0.47`; at 2828 rows `post 5.0 + local 0.46 + r0 7.9/2.7 + r1 7.2/1.4`.
  The GX round is 723–767 ms per 40 layers (18–19 ms/layer, `sync` 8.3 ms/layer
  idle): the wire is no longer the longer leg, so further cuts to the post alone
  now buy little. `kChunk` (4 MiB) and the worker's `shuffleInputRowsKernel`
  (4.3 ms of 27.1 at 4096 rows) are the remaining single variables.
- **Wire correctness of the scale path:** a deterministic pattern
  (`row*1000 + col`) posted over all 40 layers to both peers comes back
  bit-identical to the e4m3 round trip, `mismatches=0` of 160 layer-fetches
  (`afd_smoke.py --mode identity`, 3 rounds × 40 layers × 2 peers at 128 rows;
  it needs the GX worker started with `--serve --mode identity`). End-to-end, the
  engine answers `The capital of France is` → ` Paris.`
- **Previous campaign's final actual e2e, packaged v7 graph ingestion
  (2026-10-21; historical):** engine healthy from
  `scripts/launch_engine_graphs.sh` (default full graph + Seam,
  `cuda-swz` workers); 3 × 256-token requests. Wall **4.75 / 4.75 / 4.76 s** =
  **53.94 / 53.89 / 53.73 tok/s**, about **31% above** the 41 tok/s pre-ingest
  result; the **>75 tok/s target is not met**. Token counts are coherent; the
  `Paris` prompt returns 32 tokens, correct. Prefill: **3512 prompt tokens in
  1.646 s plus 16 completion tokens**, and **6662 prompt tokens in 2.050 s plus
  16 completion tokens**; both are `Paris`-coherent. These prompts are **not
  comparable** to the earlier 2538/6924-token prefill run. A request containing
  1500 repetitions of a seven-token sentence (prompt >8192 model limit) was
  rejected HTTP 400 (normal). After the bench, transport status is **posts
  41240**, `rx [41240, 41240]`, failed/timeouts/nonfinite **0**. A separate measured
  64-token request moved posts **43840 → 46400**, exactly **2560 = 64 × 40**;
  `rx` on both hosts moved **+2560**; failed/timeouts/nonfinite **0**.
  After the profiling run, the ordinary non-profiled engine was restored and
  passed another 256-token counting request: **4.80 s, 53.39 tok/s**.
- **Decode, earlier eager path (2026-10-21, historical):** 256 forced tokens
  (`min_tokens=max_tokens`, `ignore_eos`) at 136–161 ms/token, i.e. **6.2–7.3
  tok/s** (8.6 on a lucky run; a 128-token first request paid 19.5 s of graph
  capture). Per layer at `rows=[1]`: `post 0.52`, `local_between 0.65`,
  `r0 wait 0.03 + fetch 0.19`, `r1 wait 0.04 + fetch 0.17` — **1.6 ms per layer
  in total**, i.e. about 64 ms/token across 40 layers, not 1.6 ms/token. The
  remaining ~72–97 ms/token is P7-side work (local MoE slice, attention, indexer,
  router, sampling) plus per-step Python; these counters do not isolate attention.
  `local_before` in the step line reads 3.05 ms/layer but includes inter-request
  idle, so it must not be read as attention time.
- **Decode, earlier full graph + `AFD_SEAM=1` (historical; pre-ingest-fix,
  2026-10-21):** 2 GX peers, 256 tokens, greedy. Three runs each:
  **24.69 / 24.49 / 24.66 ms/token = 40.5 / 40.8 / 40.6 tok/s**, and a plain
  relaunch of the same config at
  **24.30 / 24.19 / 24.39 ms/token = 41.2 / 41.4 / 41.0 tok/s**, against the
  ~140 ms/token (6.5 tok/s) eager baseline. This was the last verified result
  before the ingestion/v7 deployment; that previous-campaign e2e supersedes it.
  The **>75 tok/s target is not met**. Coherence:
  `The capital of France is` → ` Paris.` (T=0).
- **Decode attribution, node-traced Nsight p7.1 (valid; historical
  pre-ingest-fix):** 64-token window,
  `p7.1.nsys-rep` captured with `--cuda-graph-trace=node`. Merged GPU busy is
  **24.36 ms/token** and idle **1.26 ms/token**; the sum of kernel durations is
  29.12 ms/token, so computation overlaps across streams by 1.20 within each
  replay. `seam_collect_wait_kernel` is **14.31 ms/token (40 posts/token)**: the
  seam status log moved 2840 → 5400 posts over 64 tokens = 2560 = 64 × 40, `rx`
  [2560, 2560] on both peers, `timeouts=0`, `nonfinite=0`, so every layer of every
  token really executed remotely. P7's own arithmetic is the other ~10 ms/token,
  matching the P7-only graph floor of 9.9 ms/token. The kernel-class rows
  overlap (the name classifier is not a partition) and the kernel durations
  overlap across streams: **do not sum the class rows**. Only the merged busy and
  idle totals are additive. The earlier graph-level trace (`p7.1.graphonly`:
  1.39 ms/token busy, 24.3 ms/token idle) is invalid and must not be used.
  This node trace predates the ingestion/v7 deployment and does not cover the
  2026-10-21 campaign e2e (W7.1).
- **Final decode attribution, node-traced Nsight p7.2 (2026-10-21):** artifacts
  `p7run/p7.2.nsys-rep` and exported `optimized.sqlite`; the 64-token window
  includes first-step cold sampling/JIT. Full window: 2560 = 64 × 40 seam
  waits, totaling **509.586012 ms = 7.9623 ms/token** (p7.1 baseline:
  **915.643 ms = 14.307 ms/token**); merged busy **18.440666 ms/token**, idle
  **6.224742 ms/token** from the cold sampling gap. Steady estimate excluding
  the first 40 waits/first step: 63 steps span **18.263686 ms/step**, busy
  **17.935739**, idle **0.327948**, seam wait **7.624559 ms/step**. This steady
  window runs from wait #41.start to the last wait.end and slightly excludes
  layer-0 local work before the first wait. Do not sum overlapping categories.
  At 75 tok/s the budget is **13.333 ms/token**; ~18.55 ms/token leaves
  **~5.2 ms/token** to remove. The GX wait remains dominant and reducible, but
  not all of it is necessarily transport.

### W7.1 Campaign 2026-10-21 evening: cooperative GX staging and Seam de-latency

**Status: both changes installed and measured together.** The previous
campaign's 53.94 / 53.89 / 53.73 tok/s (and 53.39 tok/s restoration) above are
historical. This campaign's starting baseline is **53.46 / 54.02 / 54.03
tok/s** on the same `SPEC_TOKENS=0` recipe. Five 256-token runs measured
**55.62 / 55.57 / 55.79 / 55.34 / 54.75 tok/s** (wall
**4.60 / 4.61 / 4.59 / 4.63 / 4.68 s**), **median 55.57 tok/s**, which is
**+2.87%** over the **54.02 tok/s** baseline median. Both GX workers now run
the staged body. The result is **combined only**; no individual staging or
Seam e2e gain is claimed. **>75 tok/s remains unmet.**

**Cooperative GX block-scale staging (v7 `cuda-swz`, `GX_STG=8` default).** The
staged body keeps the block-scale sectors in 4 KiB of shared memory per block.
Against the pre-staging body, on the real pack at rows **1/2/6/16** and the
random arena at rows **1/2/4/6/8**, it is **bit-identical** (int16 diff 0,
max|Δ| 0) on every row. Graph replay at rows=1 falls **213.5 → 209.2 µs/layer**
(kernel **211.5 → 207.1 µs**, −4.4 µs/−2.1 %); rows 2–4 gain 1.6–8 µs; rows
6–16 tie. `GX_STG=10` is another 1.4 µs faster at rows=1 but costs a block of
occupancy, so 8 is the deployed default. Rollback is the pre-staging `.so` and
previous kernels tree from the site's backup directory.

**Seam post/collect de-latency (P7 extension).** `amax` and `finish` are fused
through a last-block `fin` release/acquire handshake, and `publish` re-arms
`amax`; the separate `cudaMemsetAsync(amax)` and `finish` launches are removed.
The post path is **5 → 3 launches** (fused amax/finish, pack, publish); collect
stays **2** (wait, combine), with `combine` loading local + `pv0..pv3` before
the failed gate and every store after it. The arithmetic is unchanged: same
scale double-math, same FP8 clamp/cast, same ordered FP32 adds, one BF16 round.
**The single-post GPU ordering invariant is preserved:** one amax/finish →
pack → publish chain per layer, SASS `ATOMG.E.MAX.STRONG.GPU` →
`MEMBAR.SC.GPU` → `ATOMG.E.ADD.STRONG.GPU` → `MEMBAR.SC.GPU` → volatile
`LDG.E.STRONG.SYS`, and the first combine `STG` after `BAR.SYNC`.
The installed candidate differs from the byte-identical A/B build only by an
added `fin` reset: arithmetic and the GPU ordering above are unchanged, so
there is no new quality or arithmetic change versus this campaign's starting
baseline.

**Proof and timing.** The old/new harness kernel dump is **byte-identical at
697,713,779 bytes**; the `quant_test` dump is **byte-identical over 60
arrays**. Coverage: post rows 1/6/128/4096 × patterns 0–4 eager, capture at
rows 1/6/128/4096 finite+nonfinite, combine rows 1/6/128/4096 × np 1/2/4 with
f32 and padded strides, plus the eager failed gate and sticky-failure replay.
Median-of-3 device times: post rows=1 **9.48 → 7.42 µs**, rows=6 10.26 → 9.17,
rows=128 49.16 → 48.7; rows=4096 **1470.5 → 1479–1483 µs**, a **+~10 µs
(+0.7 %) tradeoff** because the 1024-block fin handshake costs more than the
two removed launches at max grid. Combine rows=4096 np4 bf16 improves
**~5.18 → ~4.95 ms** (−4 %); np2 and f32 np3 are neutral.

**Combined e2e (measured):** the five 256-token runs above are the campaign
result: **median 55.57 tok/s**, **+2.87%** over the 54.02 tok/s baseline median,
and **still below 75 tok/s**. It is a combined measurement; do not infer a
separate e2e contribution for GX staging versus Seam. A 64-token run produced
exactly **2560 posts** and **2560 completed reads per GX rank** (final
**64,280 posts**, rx
**[64,280, 64,280]**), with **0 failed, timeouts or nonfinite**. Coherence:
12 prompt + 16 completion in **0.412 s** returned `Paris`; **3512 + 16** in
**1.801 s** and **6662 + 16** in **2.040 s** also returned `Paris`. Effective
prompt rates **1950** and **3266 tok/s** include decode; they are not pure
prefill and imply **no prefill speedup claim**.

### W7.2 Campaign 2026-10-22: pre-DSpark strategy round (complete; final integrated PASS)

**Status: COMPLETE; report ready for commit.** DSpark stayed off
(`SPEC_TOKENS=0`); the listed non-spec strategies ran together on TP1/GPU0. The
round baseline is **55.23 / 55.74 / 55.67 tok/s**, **median 55.67** over three
measured runs. The final integrated actual run (`bg179`, engine healthy) is
**PASS**: five 256-token runs **56.89 / 56.67 / 56.86 / 56.86 / 56.87 tok/s**,
**median 56.86**, wall **4.50 / 4.52 / 4.50 / 4.50 / 4.50 s**. Against the
round starting median **55.67**, that is **+2.1376%**. The 2026-10-21
**55.57 tok/s** is historical and is **not** the gain baseline. **>75 tok/s
remains unmet.** The next round tested the **120 tok/s with DSpark across the
eight context types** target; it was a goal, not a projection, and was not
reached (W8).

| Track | State |
|---|---|
| Integrated engine (bg179) | **PASS**; 5×256 median **56.86** tok/s; **+2.1376%** vs 55.67 |
| Exact-6 GX dispatch | Accepted; deployed; five 40-layer rounds + **80/80** checks PASS |
| GX reuse / stratified aggregate | **REJECT**; real traces 1-4% slower |
| P7 profile / harness | **COMPLETE**; p7.3 pre-change baseline only |
| mHC/P7 fused quant | Stock-bit-exact PASS; integrated PASS; no isolated attribution |
| DSpark retention boundary | CPU proof PASS; runtime-functional in the DSpark round (W8) |

**Round closure:** the follow-up DSpark patch, runtime and eight-type evals are
complete in W8; no other W7.2 item is open.

**Final integrated actual (bg179 PASS; engine healthy).** The default launch
used `MHC_QUANT=1`, vector-combine plus automatic pack at `>=128` rows, and
exact-6 GX workers, with DSpark off and TP1/GPU0 unchanged. A 64-token status
check added **2,560 posts** and **2,560 completed reads** on both ranks, final
**64,280 posts** and rx **[64,280, 64,280]**; timeouts, failed and nonfinite
**0**. Coherence: **12+16** in **0.359 s**, **3512+16** in **1.689 s**,
**6662+16** in **1.986 s**, all `Paris`-coherent. Effective prompt/request
rates **2079/3354 tok/s** include decode; they are not pure prefill. The
**+2.1376%** is the bundled strategy result, not a fusion attribution. The
exact-6 live proof remains the recorded **80/80** layer-peer checks.

**Pre-DSpark final repro.** Launch the packaged `scripts/launch_engine_graphs.sh` with
default `MHC_QUANT=1` (4 RO mounts), vector-combine plus automatic pack at
`>=128` rows, exact-6 GX workers
(`--ingest graph --swiglu-limit10 --decode-kernel cuda-swz`), `SPEC_TOKENS=0`
(DSpark off), and TP1 on GPU0. Source paths:
`p7/mhc_quant/patch_vllm_mhc_quant_sm120.py` (regenerates the actual image),
`p7/mhc_quant/test_mhc_quant_fused.py`,
`p7/mhc_quant/smoke_mhc_quant_consumer.py`, `p7/afd_overlay/`,
`gx/gx_ingest.py`, `gx/gx_expert_worker.py`, and
`scripts/launch_engine_graphs.sh`.

**Exact-6 GX dispatch (accepted; deployed via reload).** The 6-row graph
removes the bucket-8 padding for `k=5` (`1+k=6`). The accepted change is
`EXACT_ROWS={6}` and `graph_rows(n)=n if n in EXACT_ROWS else bucket_of(n)` in
`gx_ingest.py`, used by `Layers.replay` and `capture_round`; `gx_expert_worker.py`
changes docstrings/log text only. It touches 2 files, +51/-19, with no kernel,
CUDA, arithmetic, CLI or `GX_WORKER_FLAGS` change. The shared 16-row buffer set,
`GRAPH_MAX` retention bound, >16 eager clamped-CUTLASS ladder, and sentinel,
epoch, isum and pad semantics are unchanged.

Two independent 11-iteration real-pack 2-layer runs on GX1 GPU0 exclusive,
with the worker stopped and the harness driving `Layers.replay`, measure
**1282 → 1251 µs/layer (+2.4%)** and **1275 → 1246 (+2.3%)**, about
**31 µs/layer** or **1.2 ms per 40-layer round**. The n=1 exact-vs-bucket
control is 0.465 vs 0.462 ms/post (0.6% apart).
Exact-6 vs bucket-8 is bitwise identical on a valid post and a sentinel
(`ids=-1`) post, twice; ingest x/ids/tw are byte-exact vs wire; flag/isum/pad/
invalid/finite checks are clean; the independent Triton kernel is cos=1.000000,
rel 2.5e-6..7.2e-5 on both layers; the 4096→1→6→16→6 transition through the
one shared buffer set keeps both 6-row results bit-identical to the
pre-transition snapshot. Context in the same run: n=8 1635, n=16 2970, n=1
233 µs/layer. Padding rows carry `ids=0/tw=0`, so all 12 padding tiles cost
about 30 µs total; bucket-8 was already nearly free at 6 rows, and there is no
25% claim.

**Live exact-6 proof (five 40-layer rounds, both rails).** Repeat B6 == A6
bitwise **80/80 layer-peer**; C8[:6] == A6 **80/80** with the same amax;
E8 == C8 **80/80**; `ids=-1`/`tw=0` exact zero **all 40 on both**; fatal and
routing failures **0**. Client steady B6 **~1554.5 µs/layer** and E8
**~1917.5 µs/layer** are observation only, not a causal exact-6 gain. The
accepted gain evidence remains the isolated two-repeat **+2.4%/+2.3%**.

An exploratory >=10% gate initially rejected this as too small; the Coordinator
accepted it because the dispatch change is minimal (one extra 6-row graph) and
the DSpark path was expected to save about **1.2 ms per 40 layers** `[INFERENCE]`.

GXReuse's real-id, real-pack stage-1 A/B (committed `stage1_kernel`,
`GX_GROUP=0`, graph-captured, 8 draws x 50 iters x 7 reps, median) tested
padded-r8 vs r6 with padding rows set to the layer's modal expert and random
`tw`. Stage-1 padding costs L6 median **+3.20%** of stage 1 (mean +5.79%,
range +0.3..+21.3%) and L0 median **+3.11%** (mean +2.12%, range -0.2..+4.3%).
This supports the direction of the exact-6 gain but does not establish the layer
gain; stage-1 share of the step is not a layer conversion. Per draw the cost is
+1.8..3.6% when the padded expert is already present, +3.1% when it adds one
distinct expert, and +10..21% only in the sparse case (uniq 9→10 on L6 d6);
6-row posts live in that higher sparse-union regime. The accepted layer gain
remains the six-row own real-pack two-repeat result, **1282 → 1251 µs/layer
(+2.4%)** and **1275 → 1246 (+2.3%)**; the earlier **0-3%, centred near 2%**
real-id estimate is replaced by the measured **3.1-3.2% of stage 1**, not a
layer percentage. The original A/B's synthetic uniform ids overstate absolute
cost about 1.2x and used `tw=0`; GXReuse's random-`tw` padding costing +3.2% of
stage 1 shows no evidence that `tw=0` hid work. No projection beyond 1.2 ms/40
layers is made.

Repo state: `gx/gx_ingest.py` b7a10c16, `gx/gx_expert_worker.py` 6c365353 (HEAD
38ea225b / c67886a); the worker's `/w` view is the staged tree copied to each
host, and both hosts carry the accepted pair. The `bg145` reload left both
resident at 110.7 GiB, both logging the `cuda-swz` exact-row path
and 40 resident layers; 40 single-row graphs were captured on each host, with
no errors or tracebacks. Exact-6 readiness is confirmed, and the dispatch is
deployed. The width-6 graph was captured on demand at the first 6-row post and
has now run the five 40-layer proof rounds; see above. Exact-6 touches no kernel
files; GXReuse's kernel branch is separately reverted.

**8-type routing trace (complete; exact).** 32 npz captures cover prose, math,
codegen, codeedit, json, tool, extract and longqa: consecutive M1 and independent
M4/M6/M8, each 64 steps x 40 layers x 6 slots, shapes `[40,64,M,6]`. Eager and
graphed outputs are exact (sha256 `657d7dfc3299...` matching the reference; 256
completion tokens on `Count: 1, 2, 3,`). Duplicate expert slots, mean over the
eight types, are **17.8 / 32.7 / 40.6 / 45.7%** for consecutive M2/M4/M6/M8,
versus **16.0 / 22.4 / 26.7%** for independent M4/M6/M8. The distinction
matters: DSpark verify rows are consecutive positions of one sequence, so the
consecutive-window overlap is the relevant one. The deployed body is at
**94.3%** of the in-situ stage-1 ceiling (224.1 of 237.6 GB/s; the 17.995 ms
step contains an 8.280 ms R=1 chain), but `GXReuse` **REJECTS** all real-trace
aggregate variants: they are **1-4% slower** in the high-duplicate case, while
baseline L2 already dedups to **1.04-1.14x the perfect-dedup floor**. Synthetic
identical-group runs (**+15-22%**) did not transfer. The kernel branch is fully
reverted to md5 HEAD **29bf66**; no bench remains. Stratified aggregation is
closed and makes no speed claim.

**p7.3 pre-change baseline profile (attribution only; no final/optimized
capture this round).** `p7run/p7.3.trace.1.nsys-rep` gives a
steady **18.32 ms/token**, seam **7.77 ms (42.4%)**, zero overlap; the first
**318 ms** of sampler/JIT is excluded and the **854 µs** head sampler remains
outside the graph. `profile_window.json` reports **50.87 tok/s**, but that run
has the profiler attached: it is **not a speed result**. The profiler's
**1.16 ms** layer-0 inter-step doorbell includes head sampling, so it is not
necessarily an avoidable communication bubble.

**P7 seam candidate (isolated kernels).** Vector combine measures **12.588 →
3.598 µs** at rows=1; automatic pack vector at >=128 rows measures **43.45 →
29.46 µs** at rows=128 and **1454 → 814 µs** at rows=4096. These are isolated
kernel times, not an e2e speed result.

**mHC/P7 fused quant (final consumer proof PASS; no isolated gain attribution).**
The actual served image passes finite, NaN/Inf and mixed-block stock bit-exactness,
graph capture+replay; the fixed kernel is frozen at **c7c1d226** (PTX same
**121e**). Mechanism estimate **0.69 µs/boundary × 40 = 27.7 µs/step (0.15%)**
is below run noise; the prior 5.3 µs/layer and 1.2 µs estimates are superseded.
There is **no isolated fusion-gain attribution**. Repro:
`p7/mhc_quant/test_mhc_quant_fused.py`,
`p7/mhc_quant/smoke_mhc_quant_consumer.py`, and
`p7/mhc_quant/patch_vllm_mhc_quant_sm120.py` (regeneration patcher). The source
composite includes 3 full upstream overlay files, but the patcher reproduces
the actual image. `MHC_QUANT=1` is the default launch and mounts 4 RO;
`MHC_QUANT=0` is only the experimental A/B config. The baseline full-graph
`Spec=0` path is unchanged; the DSpark retention overlay is mounted only when
`SPEC_TOKENS>0`. Integrated
actual **bg179 PASS**; five 256-token median **56.86 tok/s**.

**Memory.** The pre-DSpark round measured **81.34 GiB model / 82.16 GiB
consumed**, peak activation **1.75 GiB**, CUDA graph **0.06 GiB**, KV **1.56
GiB**, and **89,318 MiB used / 7,932 MiB (7.744 GiB) free**; that is historical.
The DSpark runs load **88.93 GiB** of weights (boot log **88.92 GiB**), including
the **7.388 GiB drafter**, which is **loaded**. k5
per-layer launch: **93,506 MiB used / 3,744 MiB free** at `GPU_MEM_UTIL=0.97`
plus `DSPARK_EXTRA_GIB=1`; RG6: **94,262 / 2,988**. Observed peak was **94,312
MiB used with 2,938 MiB reported free** (earlier k5 snapshots reported **3,744
MiB free**), so measured headroom is **~2.9–3.7 GiB**; 100% util, **184.4-184.6
W**, **2295 MHz**, **78 C**. GPU0's **97,887 MiB** total includes the
driver/display reserve; free memory comes from the measured query, not
total-minus-used. There was **no OOM or failure** across 384 timed + 144 quality
rows. Rebalancing onto P7 remains **ruled out**; the loaded drafter plus runtime
must stay reserved.

**DSpark retention boundary (CPU-proven plus runtime-functional).** The
required retention is **134 = 128 + 6**; **140** is DeepInfer's conservative
example, not the minimum, and neither value is a recommendation to widen
replay. The DSpark-only
`extra_retained_tokens = max(existing, num_speculative_tokens+1)` and the
null-block guard are vllm-moet commits `c64b2f5` (patcher) and `3fa14b9` (CPU
test):
`tools/dsv41_sm120/patch_vllm_dspark_retention.py` in the external vllm-moet tree.
CPU proof at `C=255`, `W=128`, block 128: E=0 frees and nulls table entry 0;
E=6 skips 122 tokens, keeps entry 0, and maps position 127 to real slot 255;
the guard masks null blocks to -1. Runtime: the overlay is mounted for every
`SPEC_TOKENS>0` run, and k5 sets extra retained **6**. All DSpark runs
completed with no null/evicted-block crash, no OOM, and zero request errors.
That is a **runtime functional boundary**, not a full suffix bitwise proof; no
bit-exact verification of the whole retained suffix was recorded. The non-spec
path remains byte-for-byte unchanged. Regenerate with
`--vllm-dir <vllm source tree> --overlay-dir <overlay dir>` against the external
vllm-moet checkout; the manifest sits beside the overlay directory;
overlay sha256 `ff922a17...` and `50445556...`.

### W8. DSpark speculative decoding: measured results (2026-10-22)

**Status.** COMPLETE for k0/k1/k3/k5; k7 (`SPEC_TOKENS=7`) was measured and
**rejected**. Four primary configurations x **64 forced
timed rows** each (8 types x 2 prompts x 2 pairs x 2 repeats = 64 rows per
configuration), plus k5 RG6 and paired-k5 comparison conditions (64 rows
each), for **384 timed rows**; the matched k5/k7 A/B adds **128 timed rows** (64
per configuration) and **144 quality rows**. Forced budgets (`ignore_eos`,
`max_tokens == `min_tokens`) make every timed row performance-only and invalid
for quality. TTFT includes request, template and network overhead and is not
pure prefill; decode is the streaming decode-window rate from SSE
`usage.completion_tokens`, never chunk count. The **120 tok/s goal was not
reached**, and no aggregate 120 tok/s claim is made.

**Deployment.** The drafter is loaded: `DSpark draft model loaded: 97 params`,
boot `Model loading took 88.92 GiB`, **88.93 GiB** weights including the
**7.388 GiB** drafter. Draft layers **40-42 were verified local on P7**; all 40
main layers are GX-only. k5 per-layer: `SPEC_TOKENS=5`, `GPU_MEM_UTIL=0.97`,
`DSPARK_EXTRA_GIB=1`, captures `1,5,6`, `MAX_CAPTURE=6`, extra retained **6**,
KV capacity **20,227** tokens, prefix caching off, no DBO, adaptive off; GPU1
and GPU2 were never used. k1/k3 use captures `1,2`/`1,3,4` and their defaults.
Engram remains pinned-host MXFP4 (~188.8 GiB), not GPU memory.

**Pair medians.** `short = 1024p_128c`, `long = 128p_1024c`.

| condition | short decode tok/s | short TTFT s | short E2E s | long decode tok/s | long TTFT s | long E2E s |
|---|---:|---:|---:|---:|---:|---:|
| k0 | 57.37 | 0.549 | 2.77 | 57.43 | 0.296 | 18.11 |
| k1 | 72.69 | 0.553 | 2.30 | 75.32 | 0.298 | 13.88 |
| k3 | 82.13 | 0.561 | 2.12 | 96.70 | 0.306 | 10.88 |
| k5 early | 78.49 | 0.570 | 2.17 | 102.82 | 0.321 | 10.27 |
| k5 RG6 | 76.31 | 0.573 | 2.24 | 98.81 | 0.319 | 10.67 |
| k5 paired | 76.10 | 0.571 | 2.24 | 97.38 | 0.318 | 10.82 |

**Per-context medians, 1024 prompt / 128 completion.** Each cell is decode
tok/s / TTFT s / E2E s, median of n=4.

| context type | k0 | k1 | k3 | k5 early |
|---|---:|---:|---:|---:|
| code_editing_debugging | 57.36 / 0.568 / 2.78 | 72.03 / 0.572 / 2.34 | 78.32 / 0.584 / 2.21 | 73.60 / 0.591 / 2.31 |
| code_generation | 57.32 / 0.545 / 2.76 | 76.77 / 0.548 / 2.21 | 101.26 / 0.554 / 1.81 | 109.71 / 0.561 / 1.72 |
| extraction_short_answers | 57.34 / 0.559 / 2.77 | 71.57 / 0.562 / 2.34 | 86.14 / 0.567 / 2.04 | 80.05 / 0.587 / 2.17 |
| json_schema_output | 57.37 / 0.558 / 2.77 | 72.68 / 0.560 / 2.31 | 86.84 / 0.564 / 2.03 | 95.05 / 0.578 / 1.90 |
| long_document_qa | 57.35 / 0.544 / 2.76 | 72.50 / 0.546 / 2.30 | 93.86 / 0.555 / 1.91 | 82.65 / 0.544 / 2.08 |
| math_reasoning | 57.38 / 0.544 / 2.76 | 73.14 / 0.546 / 2.28 | 80.88 / 0.557 / 2.13 | 78.11 / 0.568 / 2.19 |
| prose | 57.43 / 0.541 / 2.75 | 69.23 / 0.542 / 2.38 | 64.72 / 0.557 / 2.51 | 69.57 / 0.566 / 2.40 |
| tool_agent_exchange | 57.35 / 0.563 / 2.78 | 73.38 / 0.567 / 2.30 | 80.79 / 0.574 / 2.15 | 67.74 / 0.583 / 2.46 |

**Per-context medians, 128 prompt / 1024 completion.** Same cell format.

| context type | k0 | k1 | k3 | k5 early |
|---|---:|---:|---:|---:|
| code_editing_debugging | 57.39 / 0.297 / 18.12 | 75.01 / 0.300 / 13.94 | 100.26 / 0.310 / 10.51 | 100.16 / 0.323 / 10.70 |
| code_generation | 57.47 / 0.291 / 18.09 | 77.57 / 0.295 / 13.48 | 97.42 / 0.305 / 10.81 | 114.05 / 0.317 / 9.29 |
| extraction_short_answers | 57.43 / 0.300 / 18.11 | 76.45 / 0.305 / 13.69 | 99.88 / 0.309 / 10.56 | 99.96 / 0.325 / 10.64 |
| json_schema_output | 57.43 / 0.298 / 18.11 | 75.86 / 0.307 / 13.79 | 98.50 / 0.310 / 10.70 | 104.30 / 0.329 / 10.13 |
| long_document_qa | 57.43 / 0.291 / 18.11 | 75.99 / 0.294 / 13.76 | 87.91 / 0.298 / 11.95 | 97.09 / 0.315 / 10.89 |
| math_reasoning | 57.48 / 0.289 / 18.09 | 76.39 / 0.292 / 13.69 | 96.69 / 0.300 / 10.90 | 100.31 / 0.317 / 11.22 |
| prose | 57.35 / 0.297 / 18.14 | 71.10 / 0.298 / 14.68 | 83.23 / 0.305 / 12.60 | 83.10 / 0.322 / 12.66 |
| tool_agent_exchange | 57.39 / 0.312 / 18.14 | 72.97 / 0.317 / 14.34 | 97.28 / 0.326 / 10.86 | 101.75 / 0.339 / 10.63 |

**Speculative efficiency.** Acceptance = accepted drafted tokens / draft slots;
emitted tokens/spec step = accepted drafted tokens + one bonus token, divided
by steps (not accepted drafted tokens alone).

| condition | acceptance short / long | emitted/step short / long | aggregate acceptance | aggregate emitted/step |
|---|---:|---:|---:|---:|
| k1 | 0.8364 / 0.9088 | 1.8364 / 1.9088 | n/a | n/a |
| k3 | 0.6625 / 0.7991 | 2.9876 / 3.3973 | 0.7821 | 3.346 |
| k5 early | 0.5370 / 0.6852 | 3.6851 / 4.4262 | 0.66577 | 4.32884 |

k5 RG6 aggregate is **0.65210 / 4.26049**; paired k5 is **0.65084 / 4.25420**.
Short responses have materially lower acceptance than long responses.

**120 tok/s verdict.** No condition reaches 120 tok/s in an aggregate pair
median. Isolated timed requests at or above 120 exist only as **4 of 64**
k5-early requests (1 short, 3 long), **5 of 64** RG6 (0 short, 5 long), and
**1 of 64** paired k5 (0 short, 1 long); k0/k1/k3 have none. These are individual
forced requests, not per-type or per-prompt cell medians; no per-context median
above reaches 120.

The k3 ceiling is arithmetic: **4 tokens/step** at its best **35.07 ms/step**
is **114.1 tok/s**, below 120. k5 emits **3.53 tokens/step short / 4.37 long**,
so 120 tok/s needs at least **4.21** emitted tokens/step at 35.07 ms; observed
median step gaps are **~45.8 / 46.4 ms** (paired), **45.7 / 45.7 ms** (RG6), and
**46.3 / 46.3 ms** (early). At 120 tok/s the required step time is **~29.4 ms**
short and **~36.4 ms** long. The remaining gap is step latency, not acceptance
alone. "120 tok/s across eight types" is not demonstrated.

**RG6 comparison and rejection.** RG6 ran on the same k5 engine: 16 prompts per
pair, two repeats per prompt, repeat medians; `Δ` is the median of 16 per-cell
paired differences (RG6 minus per-layer k5), not a difference of aggregate
medians.

| pair | RG6 median tok/s | per-layer k5 median tok/s | paired Δ tok/s | 95% CI | RG6 wins / losses |
|---|---:|---:|---:|---:|---:|
| 1024p_128c | 76.31 | 76.10 | +0.67 | [-2.65, +5.15] | 8 / 8 |
| 128p_1024c | 98.81 | 97.38 | +0.19 | [-5.83, +9.79] | 8 / 8 |

The CIs include zero and each pair splits 8/8. RG activation was real:
**8,742 n=6 graph-replay rounds per host** across all 40 layers, **14**
prefill-width fallbacks, **0 n=6 fallbacks**, **0 real spin timeouts**, no
FATAL after arming. RG6 is **REJECTED**; per-layer k5 is retained as the best
configuration. Rollback is clean: production worker
`6c365353048d6d75115d7f1265cafde5`, `gx_worker.sh` baseline
`74162f5d0981d89f144e85b5c2432cb6`, and `gx/gx_round_arm.py`,
`scripts/arm_round_graph.sh`, `scripts/test_gx_round_arm.py` removed. `k5 early`
and `k5 paired` are separate conditions under different host conditions and are
not a controlled A/B; the earlier **102.82** is not a direct RG comparison.

**k7 trial (`SPEC_TOKENS=7`; rejected, 2026-10-22).** Same budget as k5:
`GPU_MEM_UTIL=0.97`, `DSPARK_EXTRA_GIB=1`, captures `1,7,8`, `MAX_CAPTURE=8`,
retention extra **8** (`SPEC_TOKENS + 1`), `MAX_MODEL_LEN=8192`,
`MAX_NUM_SEQS=1`, `MAX_NUM_BATCHED_TOKENS=4096`, prefix caching off. Two
configurations under identical prompts and budgets, **64 forced rows** each,
64/64 valid, `prompt_exact` and `budget_exact` in every row:

| condition | short decode tok/s (1024p/128c) | long decode tok/s (128p/1024c) | short E2E s | long E2E s |
|---|---:|---:|---:|---:|
| k5 (k7 A/B baseline) | 77.84 | 105.07 | 2.219 | 10.06 |
| k7 | 67.67 | 95.97 | 2.465 | 10.99 |
| delta | **-13.07%** | **-8.65%** | +11.1% | +9.26% |

TTFT is flat (0.568 / 0.319 s at k5, 0.578 / 0.324 s at k7), so the loss is in
the decode loop. Spec counters: acceptance **0.6918 -> 0.5486**, accepted
tokens 28,656 -> 29,332, drafted tokens 41,420 -> 53,466 over 8,284 -> 7,638
draft steps. `mean_accepted_length` rises 4.459 -> 4.840, but that counter
includes the bonus token, so it is *emitted tokens per speculative step*:
3.459 + 1 accepted at k5 against 3.840 + 1 at k7. Acceptance by position is
0.896/0.788/0.687/0.589/0.499 at k5 and
0.873/0.750/0.643/0.536/0.448/**0.351**/**0.239** at k7: the two new positions
are the weakest of the seven, so the wider verify step buys fractionally more
emitted tokens per step at a large cost in per-step work and 13.07% short-decode
rate. The target is still not met: k7 reaches 120 tok/s in **4 of 64** individual
requests (all long, max 134.35) against 3 of 64 in the k5 baseline, and no
per-context median does.

**Confound.** GPU0's SM clock mean is **2294 MHz** in both windows, GPU0 power
178.3 W (k7) vs 181.0 W (k5), GPU0 util 99.1% vs 98.6%, so clock and thermal
throttling do not explain the gap. However GPU1 carried an external workload
during part of the k7 window (util mean **23.6%**, GPU1 SM mean 1879 MHz) that
is outside our control, and the k5 counter window is the prior run's, not the
paired run's. The drop is therefore **not proven engine-only** and is not
attributed to clocks or heat. Verdict: **k5 per-layer retained, k7 rejected,
no follow-up k7 campaign**; k5 was restored and served healthily on port 8001.

Artifacts (eval results directory, untracked): `round_k5_k7pair_20261022T122131Z/`,
`round_k5_k7pair_chatq1/`, `round_k5_k7pair_chatq2/`,
`round_k7_20261022T124251Z/` (with `k7_pair_analysis.json`), `round_k7_chatq1/`,
`round_k7_chatq2/`, `round_k7_boot_20261022T123333Z/`,
`round_k5_restore_20261022T125508Z/`, `compare_k7_20261022T125136Z.txt`,
`gpu_sampler_1017.csv`, `k7_result_20261022T130500Z.json`. Harness SHA256
`c379d3f6a6fccb61fe3bdde6757bd9f560ae7ff36fe08bf005a9d623f1f21045`; fixtures
unchanged (`2433c0eb...`).

**Quality.** Official chat endpoint, thinking disabled, `max_tokens=1024`,
natural stop: 3 configurations x 2 passes x 24 rows = **144 quality rows**
(16 greedy + 8 sampled per pass).

| config | pass 1 greedy | pass 2 greedy | sampled (both passes) |
|---|---:|---:|---:|
| k0 | 15/16 | 15/16 | 8/8 |
| k3 | 15/16 | 15/16 | 8/8 |
| k5 | 15/16 | 14/16 | 8/8 |

`codeedit-b` fails `page_bounds` in all six passes, including k0: the generated
module defines `total_pages()` and omits the required symbol, so the requested
public surface is incomplete (the code parses; the contract still fails). k5
pass 2 also fails `codegen-b` (private `_failure_count` vs public `failures`):
one k5-only discordance, zero reverse,
exact McNemar **p=1.0**. Greedy outputs are not bit-identical across passes even
at k0 (k0 **6/16** exact repeats, k3 **5/16**, k5 **7/16**). There is **no
formal quality or lossless claim**.

The k7 A/B added three natural-stop passes per configuration on the same
endpoint: k5 **15/16, 15/16, 15/16**, k7 **15/16, 15/16, 14/16**, sampled
**8/8** in all six. k7's third pass misses `codegen-b`: the generated module
exposes `failure_count` where the probe requires the consumer-visible
`failures` attribute (it parses, 1567 chars, `finish_reason=stop`), so a caller
reading `failures` breaks. That is a public-API requirement of the probe, not a
harness or token-check artifact, and it is k7-only. k7 shows **no quality
gain**.

**Memory.** Weights **88.93 GiB**; k0 launch **86,622 MiB used / 10,628 MiB
free** (util 0.90); k3 **93,498 / 3,752**; k5 per-layer **93,506 / 3,744**; RG6
**94,262 / 2,988**. Observed peak **94,312 MiB used with 2,938 MiB reported
free** (earlier k5 snapshots 3,744 MiB free), so measured headroom is
**~2.9–3.7 GiB**; 100% util, **184.4-184.6 W**, **2295 MHz**, **78 C**. GPU0's
**97,887 MiB** total includes the driver/display reserve; do not derive free
memory from total-minus-used.

The k7 trial kept the same headroom: KV capacity **20,134** tokens (k5 20,227),
CUDA graph pool **0.09 GiB** (k5 0.08), and one main plus one dspark graph
captured, the same graph count as k5. After the k7 timed run GPU0 read
**95,398 MiB used / 1,852 MiB free**; the restored k5 read **93,536 / 3,714
MiB** at idle. Boot to healthy was **502 s** for k7 and **461 s** for the
restored k5. Measured snapshots only; never derive free from total-minus-used.
There was no OOM or failure across the campaign's 384 timed + 144 quality rows,
nor in the k7 trial's 128 timed + 144 quality rows. Rebalancing onto P7 remains
**ruled out**.

**Retention.** Minimum is **134 = 128 + 6**; **140** is DeepInfer's conservative
example, not the minimum. Patches: vllm-moet `c64b2f5` (patcher) and `3fa14b9`
(CPU test),
`tools/dsv41_sm120/patch_vllm_dspark_retention.py` in the external vllm-moet tree.
CPU proof at `C=255`, `W=128`, block 128: E=0 frees and nulls entry 0; E=6
skips 122 tokens, keeps entry 0, maps position 127 to real slot 255; the guard
masks null blocks to -1. Runtime: the overlay is mounted for every
`SPEC_TOKENS>0` run; k5 retains **6** extra tokens, and the rejected k7 trial
retained **8** (`SPEC_TOKENS + 1`). All runs completed with no
null/evicted-block crash, no OOM, and zero request errors, so the retention
boundary holds at `E=8` as it does at `E=6`. This is a **runtime functional
boundary; no full suffix bitwise proof was recorded**.

**Guards.** Use exact capture sizes only; padded SM120 sparse-MLA rows are a
known hang. Keep `AFD_LAYERS=40` and `AFD_NUM_EXPERTS=384`; never route draft
layers 40-42 to GX. Do not touch the transport. Do not adopt RG6 or k7, and do
not quote an aggregate 120 tok/s or quality/lossless result.

**Repro.** Canonical k5 per-layer launch:

```bash
SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1 \
CAPTURE_SIZES=1,5,6 MAX_CAPTURE=6 NAME=dsv41-dspark-k5 \
bash scripts/launch_engine_graphs.sh
```

```bash
python3 eval/dspark_workload_eval.py --execute --label k5 \
  --out-dir <results-dir>/round_k5_perlayer_ab \
  --repeats 2 --warmup 2 --quality-sampled --overwrite \
  --engine-container dsv41-dspark-k5
```

Rejected k7 trial:

```bash
SPEC_TOKENS=7 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1 \
CAPTURE_SIZES=1,7,8 MAX_CAPTURE=8 NAME=dsv41-dspark-k7 \
bash scripts/launch_engine_graphs.sh
```

```bash
python3 eval/dspark_workload_eval.py --execute --label k7 \
  --out-dir <results-dir>/round_k7 --repeats 2 --warmup 2 \
  --quality-sampled --overwrite --engine-container dsv41-dspark-k7
```

Harness SHA256 `7ffd626c9c2fa168f1369e40a225575ebf24742e79c185a996e0a95b286a1806`
(k5 campaign; the k7 A/B used `c379d3f6...`);
fixtures SHA256 `2433c0eb06cf766301979895ab6599f580162518bb18a834982710cee377179c`
(earlier k0/k3 fixtures `b06dc480...`). Raw rounds:
the eval results directory. Full per-type/quality tables: the DSpark final-results artifact.

## 5. Gates and order

| Gate | Depends on | Pass criterion |
|---|---|---|
| A | W0, W3 | Memory ledger re-closed on measured numbers. Graphed 80-exchange loop ≤ 60 µs per layer round trip. |
| B | W1, W5 (experts stubbed to zero; diagnostic only, never reported as a result) | P7 serves the non-MoE path with FULL graphs, the planned Gen5 Engram fetcher (not deployed) and DSpark. No #62417 symptoms. |
| C | W1, W2 | Three-slice sum matches FP64 on real rows. Deterministic on both architectures. |
| D | A, B, C, W4 | First real request end to end. Layer captures match the CPU reference (W6.2). |
| E | D | W6.3–W6.5 pass. Cancel and recovery OK. |
| F | E | W7 benchmark vs Mia at equal quality or better. |

W1 and W2 can start on CPU as soon as the download finishes, with no GPU. W0 and W3 need P7 GPU0, GX1 and GX2, which are dedicated to AFD-TP3 once execution starts.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Transport latency is too high | Gate A is first. Fallback: NCCL graphed, then accept less speed. |
| SM120 vLLM DeepSeek-V4.1 stack churn (FlashInfer page sizes, DeepGEMM pins) | Pin vLLM, FlashInfer and DeepGEMM commits in one immutable image. Follow #60762 / #62535 / vLLM-Moet. |
| DSpark acceptance is lower on our traffic than 0909-lineage reports | Measured acceptance **0.84/0.91** (k1), **0.66/0.80** (k3), **0.54/0.69** (k5) short/long. k3/k5 improve decode; k5 per-layer is retained and RG6 is rejected (W8). |
| P7 memory margin at DSpark k5 is **~2.9–3.7 GiB free** with the 7.388 GiB drafter loaded | Rebalancing is ruled out; measured peak **94,312 MiB** with **2,938 MiB reported free** (earlier k5 snapshots 3,744 MiB), so headroom is **~2.9–3.7 GiB**. Do not raise capture sizes or GPU util without measuring headroom. |
| GB10 unified-memory pressure | Stateless worker, memory floor guard, no compile or caches at load. |
| Activation-format mismatch (W4A16/W4A4 vs DeepSeek W4A8) | Current kernels are not the FP8-activation target: decode is W4A16 (`cuda-swz`), rows >16 are CUTLASS W4A4. The two paths differ by ~0.15 relL2 (informational); no bit-exact claim against the old W4A4 path, and quality evaluation is not complete. |
| DSpark target reads an evicted/null KV block at the SWA boundary | DSpark-only retained suffix of 134 total plus a null-block guard are **CPU-proven and runtime-functional** (W8): no null/evicted-block crash, no OOM, zero request errors. Retain the guard; no full suffix bitwise proof. |

## 7. DeepInfer (deepseek-ai/DeepInfer @ 774ce244, Apache-2.0, released 2026-10-20): what we take

DeepInfer cannot run on our hardware. The only hardware it accepts is SM100 or Ascend 950 (`deep_cluster/hardware/device.py: is_sm_major_10()` → `capability[0] == 10`, otherwise "unknown GPU type"). Its pinned kernel libraries have no sm_120 or sm_121 code: DeepGEMM `1fc9c669` has 31 `sm100*` and 16 `sm90*` files and none for sm12x; FlashMLA `2e5429fc` has 107 `sm100` paths. Its production design also assumes Wide-EP (EP64–128) for decode, with DP attention. The P7 runtime therefore stays on vLLM. The following parts of DeepInfer are adopted as references, ports or tooling:

| Item | Where (in DeepInfer) | How we use it |
|---|---|---|
| **Expert numerics, confirmed** | `mlp.py` `_grouped_gemm`: `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(recipe_a=(1, 32), recipe_b=(1, 32))`. `linear.py` `get_fp8_casting_granularity() = 32`. `swiglu_forward_and_per_token_cast(..., topk_weights, clamp_value)` | Settles production's activation contract: FP8 activations, per-token groups of 32, UE8M0. SwiGLU (with clamp) and the routing weight are fused **before** the down projection. That is still the accuracy target, but the shipped kernels do not implement it: decode is W4A16 (`cuda-swz`), and rows >16 use CUTLASS W4A4. The doc confirms production experts run MXFP4 (per-32, ue8m0). Our NVFP4 pack is a lossless re-encoding of those values. |
| **MoE seam** | `DeepseekMoE.forward`: `gate_forward` → `dispatch` → `routed_experts_forward` → `shared_forward` → `combine` | The structure AFDMoE copies: dispatch becomes the send to the GX, `routed_experts_forward` becomes the local P7 slice, combine becomes the wait plus a fixed-order sum. The shared expert overlaps the wait, as DeepInfer overlaps it with combine. |
| **Expert-load counters** | `eplb/__init__.py`: `expert_count [layers, experts]` per step, `logical_count`, `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | The same counters in AFDMoE (a device-side `bincount` of top-k ids) drive W7's expert-hotness profiling and any later split rebalancing. |
| **DSpark confidence-scheduled verification** | `tilelang_ops/cuda/schedule_mtp_varlen.py` (cumulative product of per-position confidence, global rank, budget cut); `csrc/engine/mtp_varlen.hpp` (runtime estimators that pick the verify budget from a fitted step-time curve) | Port to P7 (TileLang is portable; this kernel disables TMA and warp specialization, so it should compile for sm_120 [INFERENCE]). DeepInfer's data: verified tokens drop from 5.00 to 3.2–4.4 per step with accepted length about the same (3.29 vs 3.20), worth +4–7% at high concurrency but −1.5% at 16 sequences/rank. **For us, the gain is fewer expert bytes per verify step:** each verified row adds distinct experts that every GX must read. So it is worth evaluating at C1–C4 even though DeepInfer's batch gain is small there. W7 lever. |
| **SWA bounded replay (YOCO)** | Doc §"KV cache hits and recomputation"; `v4_modules/yoco.py`, `input_metadata/yoco.py`; `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` | Decoder layers (20–39) recompute only the last 128 (+6 speculative) tokens of the prompt, which roughly halves prefill compute. It is model-level logic. vLLM has it upstream (#58132, `--swa-bounded-replay`, default on). **AFD-TP3 benefit:** decoder-layer MoE traffic to the GX is also bounded during prefill. Enable it, and use DeepInfer as the reference for edge cases (cap the hit length at L−134). |
| **Async scheduling** | Doc §"Asynchronous scheduling"; `csrc/engine/zos.cpp` | The CPU runs `schedule(N+1)` while the GPU runs `forward(N)`. vLLM has an async scheduler; W5 enables it. Confidence-length recomputation happens on the GPU (above). |
| **Batch-invariant mode** | `DEEPINFER_BATCH_INVARIANT=1` | The design reference for W6 repeatability: fixed reduction orders, independent of batch composition. Our fixed-order partial sum follows the same rule. |
| **Prompt and protocol layer** | `deepseek-ai/deepseek-recipe` (MIT, Rust with Python bindings) | Official encoder and decoder aligned with the DeepSeek API: special-token injection protection, `</think>` and DSML tool calls parsed by token ID, Chat/Responses/Messages. Candidate to replace vLLM's `deepseek_v41` tokenizer mode and parsers in front of P7; that layer has known tier and tool-result encoding bugs (vLLM #62199/#62196/#62200; vLLM-Moet effort-tier patch). W5 item. |
| **Eval harness** | `eval/` (AIME 2026 ×32, GPQA Diamond ×8, HLE; Pass@1; checkpoint `encoding/encoding.py`) | Added to W6.5 so our scores are directly comparable to DeepSeek's. Runs against our OpenAI endpoint with a thin client adapter. |
| **Engram** | `v4_modules/engram.py` (tables sharded across ranks, gathered with DeepEP `EngramBuffer`) | Not used. Our tables are the **pinned-host MXFP4 offload in P7 RAM**, not the planned Gen5 fetcher. Reference only for hashing parity tests (`CompressedTokenizer`, `NgramHashMapping`). |
| **Weight loader** | `hugging_face/weights.py` (per-layer coverage checks; routed experts concatenated to a 3D packed layout) | The pattern for W1's pack builder and its coverage checks (every expert's w1/w3/w2 plus scale present before packing). Its slicing is by expert (EP), not by column, so the code itself is not reused. |

**Retention values.** In the YOCO row, `140` is DeepInfer's conservative
example, not the minimum. The hard bound here is **134 = 128 window + 6
speculative tokens**; neither value is a recommendation to widen replay.

Not used: DeepGEMM mega-MoE, mega-gate, FlashMLA and DeepEP (SM100/TMEM/tcgen05 and NVLink `symm_mem`); EPLB expert shuffling (we split by columns, not by expert, so load is always balanced); DP attention and Wide-EP.
