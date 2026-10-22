# QMesh AFD-TP3

QMesh AFD-TP3 is a heterogeneous inference backend for one
DeepSeek-V4.1-Flash-1017 checkpoint. It is not a general-purpose mesh router
or a model-agnostic serving stack: it splits the routed experts by output
column across three fixed machines. One RTX PRO 6000 Blackwell (`P7`) keeps
the whole attention stack, router, shared experts, head, KV cache, scheduler
and DSpark drafter; two GB10 systems (`GX1`, `GX2`) hold stateless slices of
every routed expert and return partial FFN outputs over RDMA.

## Credits and upstream foundations

This work builds on substantial upstream contributions:

- **[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** — the model and DSpark draft weights; **[LibertAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4)** — the NVFP4 checkpoint/transcode.
- **[vLLM](https://github.com/vllm-project/vllm)** and **[kacper-daftcode / vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)** — the serving foundation, SM120 integration, and upstream code. Our mHC kernels are **modified vLLM-Moet derivatives**, not independent QMesh implementations; those kernels and the copied runtime overlays remain **Apache-2.0**.
- **[TensorFold](https://github.com/ashhart/TensorFold)** — a confirmed design starting reference for the GX NVFP4 decode kernel; the current implementation is structurally different, no preserved TensorFold source copy has been identified, and the initial prototype's provenance is not established. **[DeepInfer](https://github.com/deepseek-ai/DeepInfer)** — investigated scheduling and retention design references, not a deployed replacement engine.
- **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)**, **[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)**, **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)**, and the other runtime dependencies listed in **[CREDITS.md](CREDITS.md)**.

QMesh contributes the distributed expert partitioning, native transport,
graph-safe synchronization, integration, and measured optimizations described
here. See **[CREDITS.md](CREDITS.md)** for the full attribution and license map,
and **[third_party/vllm/NOTICE](third_party/vllm/NOTICE)** for upstream notices.

## Measured status

This is a measured research deployment, not a production serving system. It is
specific to this model, these three devices and the column split below.

The non-spec round reached a median **56.86 tok/s** (5 × 256-token runs,
+2.1376% over the 55.67 tok/s round baseline). With DSpark, the best median was
**102.82 tok/s** in the long-decode condition (k5, 128 prompt / 1024 completion)
and **82.13 tok/s** in the short-decode condition (k3, 1024 prompt / 128
completion). The **120 tok/s target was not reached** in any aggregate, and no
lossless or formal quality claim is made.

A matched k5/k7 A/B then measured `SPEC_TOKENS=7` (captures 1,7,8, max 8)
against the retained k5 under the same engine budget: k5 **77.84 / 105.07
tok/s**, k7 **67.67 / 95.97 tok/s** (short / long decode), **−13.07% / −8.65%**,
acceptance **0.6918 vs 0.5486**, and no quality gain (**15/16** in all three k5
passes vs **15/16, 15/16, 14/16** at k7). k7 was rejected and k5 restored.

## Getting started

All commands below run from the repository root. The standalone repository
`GPTwareAiTeam/QMesh-AFD-TP3` is private for review; cloning requires access:

```bash
git clone https://github.com/GPTwareAiTeam/QMesh-AFD-TP3.git qmesh-afd-tp3
cd qmesh-afd-tp3
```

For a source archive, use the root of the extracted tree as the repository root.

## Architecture

| Device | Role |
|---|---|
| **P7 GPU0** — RTX PRO 6000 Blackwell Max-Q, sm_120, 97,887 MiB | Embedding, all 40 attention blocks (window, compressor, indexer, candidate, sparse attention), Hyper-Connections, router, shared experts, head, sampling, DSpark drafter, the entire KV cache, scheduler, Engram compute, and routed-expert columns **[0:576)**. |
| **P7 host RAM** | Engram tables, served by vLLM's **pinned-host MXFP4 offload**: two tables, 94.42 GiB each, **~188.8 GiB total**, sourced from the original FP8 shards. Resident in pinned host RAM, not read from the SSD per token and not in GPU memory. A planned Gen5 NVMe row-fetcher was not used. |
| **GX1** — GB10, sm_121, 121.6 GiB unified, 0.92 budget (111.9 GiB) | Routed-expert columns **[576:1440)** of every expert, every layer. Stateless. |
| **GX2** — GB10, same budget | Routed-expert columns **[1440:2304)**. Stateless. |

P7 GPU1 and GPU2 are never used: the engine container receives GPU0 only
(TP=1, `--gpus device=0`).

### Per-layer exchange

1. P7 normalizes the MoE input, runs the router, and selects top-6 experts.
2. It sends each GX the FP8 e4m3 normalized activation (~30 KB at 6 rows) plus
   the top-6 ids and weights, into that layer's slot in a shared-memory slab.
3. Each GX reads its columns, runs the SwiGLU expert slice (clamp 10), and
   returns a BF16 `[rows, 5120]` partial (~60 KB at 6 rows).
4. P7 sums the three partials and the shared expert in FP32 in a fixed order
   (P7, GX1, GX2, shared). Fixed order keeps results independent of arrival
   timing.

GX1 and GX2 do not talk to each other. They hold no KV cache, no scheduler
state, and no sampler; a worker is one persistent process per host with all 40
layers resident and one captured graph per (layer, row bucket). At decode the
deployed graph is the exact-6 graph, so a k=5 verify step (6 rows) does not pay
bucket-8 padding.

### Transport

- One RC queue pair per direction pair, over **two independent 100 Gb/s RoCE
  rails** (P7↔GX1 and P7↔GX2). The GX-to-GX path is unused.
- **No GPUDirect RDMA**: the deployed path uses a pinned-host slab, not
  `nvidia_peermem` and not peer memory. `gdr/` records an optional IOMMU/rail
  enablement experiment; it is not part of the deployed path.
- Three registered regions per worker host in `/dev/shm`: a per-layer
  activation slab, per-layer in/out flag words, and per-layer router ids and
  weights. Both sides map them; the client registers its own copies for RDMA.
- Per layer the client posts routing then payload as two unsignaled RDMA
  WRITEs, so the worker cannot see the flag before the routing it applies. The
  worker polls the in-flag from the host, replays the layer graph, copies the
  output into the return slot, then raises the out-flag. The client reads a
  4-byte out-flag with an RDMA READ on a second QP before pulling the payload.
- Flag byte 4 carries a session epoch, and an epoch/ACK handshake quiesces
  buffers before a new session. A worker that sees a previous session's flags
  waits for the epoch bump instead of serving stale data or deadlocking.
- A stream semaphore (`cuStreamWaitValue32`) does not work on GB10 (error 801),
  so the deployed gate is host-side polling of the mapped flag word.

Measured per-layer edge overhead is **352–362 µs** at one row on a 30-layer
resident worker, including both transports and the worker replay; the identity
seam itself is 82.3 µs. These are per-layer transport costs, not end-to-end
decode rates.

### Why three machines

- The routed experts alone are 2304 columns × 384 experts × 40 layers. The
  three rank packs are **71.19 GiB (P7), 106.79 GiB (GX1), 106.79 GiB (GX2)**;
  no single device has room for all of them with the rest of the model.
- P7 runs at **88.93 GiB** of weights including the 7.388 GiB drafter and keeps
  only **~2.9–3.7 GiB** free at k=5. Moving expert columns back to P7 was
  measured and ruled out; there is no headroom to rebalance.
- The expert kernels are SM120/SM121-specific (`cutlass_scaled_fp4_mm_sm120a`
  and the `cuda-swz` decode path), the GB10 unified-memory budget is 0.92 ×
  121.6 = 111.9 GiB, each GX worker is resident at ~110.7 GiB, and there is no
  NVLink between hosts. The split is sized to those exact budgets, not chosen
  for portability.

## Requirements

- **P7**: RTX PRO 6000 Blackwell Max-Q (sm_120) with 97,887 MiB; GPU0 only.
  Enough host RAM for the engine plus the ~188.8 GiB pinned Engram tables.
  A CUDA 13.x driver/toolkit and the vLLM build used below.
- **GX1/GX2**: NVIDIA GB10 (sm_121), 121.6 GiB unified memory, a working
  container runtime, root access for `/dev/shm` and RDMA device permissions,
  and enough host storage for a ~107 GiB rank pack.
- **Network**: two RoCE-capable 100 Gb/s links, one per GX host, each on its own
  subnet, with matching GID indices. No GPUDirect support is required.
- **Software**: Linux, Python 3 with development headers, rdma-core/libibverbs
  development headers, a CUDA 13.x toolkit for the native transport, and a vLLM
  checkout/image with SM120/SM121 NVFP4 and FlashInfer kernels. The engine image
  used here is `vllm-moet:dsv41-213d5fa`; the vLLM-MoE patches behind it are
  experimental and are **not upstream**. The launcher expects that image's
  `docker/sm120/run-dsv41.sh` in `VLLM_SRC`.

## Weights

The checkpoints are not in this repository and are not redistributed. Both are
MIT-licensed on Hugging Face and are downloaded separately.

| Artifact | Pinned source | Contents |
|---|---|---|
| NVFP4 model | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4@5be99cd15675589fb6693d0a8651e413bcb9a396` | 399.5 GiB, 48 shards. Routed experts are a **bit-exact MXFP4→NVFP4 transcode**. Includes the DSpark `mtp.0`–`mtp.2` weights (7.39 GiB). Its own MXFP4 Engram tables are quantized relative to the original FP8 reference and are **not used**. |
| Original FP8 Engram + metadata | `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170` | Only shards 47–48 (188.8 GiB FP8 Engram) plus config, tokenizer, index, `inference/` and `encoding/`. The Engram shards are mapped into the model directory as `engram-L01`/`engram-L14` and served from pinned host RAM; the planned per-token SSD fetch was not used. License file is MIT (Copyright (c) 2023 DeepSeek). |

Total download is about **632 GB**. `MODEL_ROOT` selects the download root;
`tools/download.sh` fetches the pinned revisions there, and
`tools/verify_shards.sh` compares every shard's SHA-256 against the
Hugging Face LFS ids in `tools/nvfp4.sha256` and `tools/orig.sha256`. The
recorded runs verified **48/48** NVFP4 shards and **2/2** original shards.

### Rank packs

`pack/build_packs.py` reads the NVFP4 checkpoint and writes three rank
directories plus `MANIFEST.json` (per-file SHA-256 and byte count). It slices
routed experts by output column while keeping every NVFP4 code and scale block
intact, and it verifies that concatenating the three slices reproduces each
source tensor byte-for-byte.

```
p7:  columns [0:576)      71.19 GiB
gx1: columns [576:1440)  106.79 GiB
gx2: columns [1440:2304) 106.79 GiB
```

The builder takes the checkpoint and pack root explicitly. Both default to the
layout created by `tools/download.sh`; `MODEL_ROOT` configures the download and
verification scripts, not these builder paths.

```bash
python3 pack/build_packs.py \
  --src <nvfp4-checkpoint> --out <pack-root> --layers all --ranks all
```

The tracked `pack/MANIFEST.p7.json`, `MANIFEST.gx1.json` and `MANIFEST.gx2.json`
are the manifests from the recorded build.

To assemble a loadable P7 model directory (symlinked pack shards, generated
`model.safetensors.index.json`, config and tokenizer links):

```bash
python3 p7/make_p7_model_dir.py \
  --pack <pack-root>/p7 --src <nvfp4-checkpoint> --dest <p7-model-dir>
```

It needs `safetensors` at run time; `--help` works without it. It does **not**
add the Engram tables. The recorded model directory also maps the two original
FP8 Engram shards as `engram-L01.safetensors` and `engram-L14.safetensors` and
extends the index to point `layers.1.engram.embed.*` and
`layers.14.engram.embed.*` at them; that mapping feeds vLLM's pinned-host
Engram offload and is not automated by this script. Rerunning the script
regenerates the index from pack shards only, so reapply the Engram mapping
afterwards.

## Build the native transport

`native/` is the current transport source (peer extension, edge binary, CUDA
seam). `native/build.sh` builds it. The older `wire/qmesh_afd_edge.cpp` snapshot
(8-row capacity) is not part of the current tree or build.

```bash
native/build.sh all     # default: edge + CPython extension
native/build.sh edge
native/build.sh ext
```

Environment knobs:

| Variable | Default |
|---|---|
| `AFD_NATIVE_DIR` | directory containing `build.sh` |
| `AFD_BIN_DIR` | `$AFD_NATIVE_DIR/bin` (gitignored) |
| `CUDA_HOME`, `NVCC` | detected toolkit, else `PATH` |
| `CUDA_ARCH` | `sm_120` on x86_64, `sm_121` on aarch64 |
| `PYTHON` | `python3` (headers and `EXT_SUFFIX` for the extension) |

Artifacts are `$AFD_BIN_DIR/qmesh_afd_edge.$(uname -m)` and
`$AFD_BIN_DIR/_qmesh_afd$(python3 sysconfig EXT_SUFFIX)`. The build links the
system `libibverbs` (rdma-core) and CUDA runtime; nothing is downloaded.
`QMESH_AFD_BIN` must point at `AFD_BIN_DIR` so the Python overlay can import
`_qmesh_afd`. Source hashes and the origin commit are in
`native/PROVENANCE.md`.

## Configure the peers

The peer file is JSON with the layer count, wire row capacity, and one entry per
peer. In each entry, `dev` and `gid` are the **P7-side** RDMA device and GID
index; `ip` and `port` are the management address the GX edge binds.
`AFD_CONFIG` (inside the engine container) defaults to `/opt/afd/afd.json`.
Addresses below are RFC 5737 documentation addresses and must be replaced with
the site's RoCE addresses.

```json
{
  "_comment": "RFC 5737 placeholder; set public_sample to false when editing a real copy.",
  "public_sample": true,
  "layers": 40,
  "rows": 4096,
  "peers": [
    {"dev": "rocep1s0f0", "gid": 3, "ip": "192.0.2.11", "port": 47101},
    {"dev": "rocep1s0f1", "gid": 3, "ip": "198.51.100.12", "port": 47120}
  ]
}
```

The public sample `p7/afd_overlay/afd.example.json` is a placeholder with
`"public_sample": true`; the launcher refuses to start a real engine against
it. Copy it to the gitignored `p7/afd_overlay/afd.local.json` (the launcher's
default when that file exists) or pass `AFD_CONFIG_FILE`:

```bash
cp p7/afd_overlay/afd.example.json p7/afd_overlay/afd.local.json
```

Engine-side variables used by the overlay:

| Variable | Meaning |
|---|---|
| `AFD_ENABLE` | enable the AFD model overlay |
| `AFD_CONFIG` | peer JSON path inside the container |
| `AFD_TRANSPORT` | `1` uses the real edge transport; `0` runs the P7-only floor |
| `AFD_SEAM` | `1` uses the graph-safe seam |
| `AFD_ROWS` | wire row capacity (4096 from the engine; `1` for smoke tests) |
| `AFD_LAYERS` | decoder layers with routed experts (40) |
| `AFD_NUM_EXPERTS` | routed experts per layer (384) |
| `AFD_ROUTED_INTERMEDIATE` | per-rank routed column width (576 on P7) |
| `QMESH_AFD_BIN` | directory containing `_qmesh_afd*.so` |
| `AFD_TIMING`, `AFD_DUMP`, `AFD_NVTX`, `AFD_PROF_CTL` | optional diagnostics |

## Run

The engine and edges are driven by `scripts/launch_engine_graphs.sh`; site
values come from `scripts/local.env` (copy `scripts/local.env.example`) or the
environment. Site inputs `VLLM_SRC` (which must contain
`docker/sm120/run-dsv41.sh`) and `MODEL_DIR` (absolute, with `config.json`) are
required; `AFD_BIN_DIR` must contain the built extension. `IMAGE` defaults to
`vllm-moet:dsv41-213d5fa`. `EDGES` defaults to the included
`scripts/restart_afd_edges.sh`. The launcher also accepts
`AFD_CONFIG_FILE`, `GX1_HOST`, `GX2_HOST`, `AFD_NATIVE_DIR`, `MODEL_MOUNT`,
`NAME`, `PORT`, `CACHE_DIR` and `EXTRA_DOCKER_ARGS`.

1. Build the packs and P7 model directory, and start the GX workers. The worker
   script expects a long-lived container (`gxw1` by default) on each host, an
   SSH alias, and a staging tree under `/tmp/gxpy` containing
   `gx_expert_worker.py`, `gx_ingest.py`, `gx_spin.py`, `gx_decode_swz.py` and
   `kernels/`:

   ```bash
   scripts/gx_worker.sh gx1 up
   scripts/gx_worker.sh gx2 up
   scripts/gx_worker.sh gx1 wait
   scripts/gx_worker.sh gx2 wait
   ```

   `up` restarts a worker and costs a ~106 GiB arena reload; leave an unchanged
   host alone. The deployed serving flags are `GX_WORKER_FLAGS` defaults:
   `--ingest graph --swiglu-limit 10`.

2. Validate the launch without starting or restarting anything:

   ```bash
   DRY_RUN=1 SKIP_EDGES=1 bash scripts/launch_engine_graphs.sh
   ```

   `DRY_RUN=1` checks the peer file, the extension and the model directory and
   prints the container environment and argv. It starts nothing (no container,
   edge or completion request), though it may query Docker and `nvidia-smi` for
   validation. `SKIP_EDGES=1` is required because the dry run does not start or
   validate the edges.

3. Start the engine (this restarts the edges unless `SKIP_EDGES=1`):

   ```bash
   bash scripts/launch_engine_graphs.sh
   ```

   `/health` becomes ready after roughly 261–337 s (FlashInfer autotune). The
   engine serves on `127.0.0.1:8001` with `MAX_MODEL_LEN=8192`,
   `MAX_NUM_SEQS=1` and `MAX_NUM_BATCHED_TOKENS=4096`; prefix caching is off.

   The included `scripts/restart_afd_edges.sh` starts one edge process per GX
   host before the engine. It reads the peer addresses from `AFD_CONFIG_FILE`,
   copies `AFD_EDGE_BIN` (default `$AFD_BIN_DIR/qmesh_afd_edge.aarch64`) to
   each host, and takes the GX-side device/GID from `GX1_DEV`/`GX1_GID` and
   `GX2_DEV`/`GX2_GID`. Build that binary on an aarch64 host with
   `native/build.sh edge`, or set `AFD_EDGE_BIN` to a cross-built copy.

For the DSpark k=5 configuration used in the results below:

```bash
SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1 \
CAPTURE_SIZES=1,5,6 MAX_CAPTURE=6 DSPARK_MEM_ACK=1 \
bash scripts/launch_engine_graphs.sh
```

The rejected k=7 trial (captures `1,7,8`, verify width 8, retention extra 8)
used the same budget and acknowledged it explicitly:

```bash
SPEC_TOKENS=7 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1 \
CAPTURE_SIZES=1,7,8 MAX_CAPTURE=8 DSPARK_MEM_ACK=1 NAME=dsv41-dspark-k7 \
bash scripts/launch_engine_graphs.sh
```

Non-spec launches default to `SPEC_TOKENS=0`, `GPU_MEM_UTIL=0.90` and
`MHC_QUANT=1`. DSpark requires an explicit memory budget; the launcher has no
implicit 0.98. `AFD_TRANSPORT=0` is a diagnostic P7-only mode with no wire and
no edges.

## Results

### Non-spec round

Five 256-token runs measured **56.89 / 56.67 / 56.86 / 56.86 / 56.87 tok/s**,
median **56.86**, against a round baseline of 55.23 / 55.74 / 55.67 tok/s
(median 55.67): **+2.1376%**. The gain is the bundled strategy result, not an
isolated attribution to one optimization. Non-spec **>75 tok/s was not
reached**. This round ran with DSpark off (`SPEC_TOKENS=0`), exact-6 GX
dispatch, `MHC_QUANT=1`, and TP1/GPU0.

### DSpark round

Forced timed rows (`ignore_eos`, `max_tokens == min_tokens`): 64 rows per
configuration (8 context types × 2 prompts × 2 pairs × 2 repeats): **384 timed
rows** for the original six conditions plus **128** for the k5/k7 A/B
(two conditions). Medians of per-prompt repeat medians:

| condition | short decode tok/s (1024p/128c) | long decode tok/s (128p/1024c) | long E2E s |
|---|---:|---:|---:|
| k0 | 57.37 | 57.43 | 18.11 |
| k1 | 72.69 | 75.32 | 13.88 |
| k3 | 82.13 | 96.70 | 10.88 |
| k5 (early) | 78.49 | **102.82** | 10.27 |
| k5 RG6 (rejected) | 76.31 | 98.81 | 10.67 |
| k5 paired | 76.10 | 97.38 | 10.82 |
| k5 (k7 A/B baseline) | 77.84 | 105.07 | 10.06 |
| k7 (rejected) | 67.67 | 95.97 | 10.99 |

- Acceptance is 0.84/0.91 for k1, 0.66/0.80 for k3 and 0.54/0.69 for k5
  (short/long). Short responses accept materially less than long ones.
- The k3 arithmetic ceiling is **114.1 tok/s** (4 tokens/step at its best
  35.07 ms/step), already below 120.
- **No condition reaches 120 tok/s in an aggregate pair median.** Only 4 of 64
  k5-early requests, 5 of 64 RG6 and 1 of 64 paired-k5 requests reach 120
  individually; no per-context median does.
- RG6 was rejected: its paired difference is within the 95% CI of zero and
  splits 8/8 wins/losses. The retained configuration is per-layer k5. The
  "early" and "paired" k5 conditions ran under different host conditions and
  are not a controlled A/B, so 102.82 is not a direct RG comparison.
- `k5 early` is the first k5 campaign; `k5 paired` is the matched comparison
  used for the RG6 verdict.
- The matched **k5/k7 A/B** ran 64 forced rows per configuration (8 context
  types × 2 prompts × 2 pairs × 2 repeats) with identical prompts and budgets;
  all 64 rows are valid in each configuration and `prompt_exact` and
  `budget_exact` hold in every row. **k7 (captures 1,7,8, max 8) is rejected**:
  −13.07% short and −8.65% long decode. It reaches 120 tok/s in 4 of 64
  individual requests (3 of 64 for the k5 baseline), all long, with no
  per-context median at 120.
- k7 accepts **0.5486** of its draft slots against k5's **0.6918**, while its
  counter `mean_accepted_length` — which includes the bonus token, i.e. emitted
  tokens per speculative step — is **4.840** against **4.459**. The extra width
  is not free: positions 5 and 6 accept only **0.351** and **0.239**, so the
  wider verify step costs more than the extra emitted token returns.
- GPU0's SM clock mean was **2294 MHz** in both windows and GPU0 power slightly
  lower at k7 (178.3 W vs 181.0 W), but GPU1 carried an external workload
  through part of the k7 window (util mean 23.6%) that is outside our control,
  and the k5 counter window is the prior run's, so the A/B is confounded. The
  gap is not proven to be engine-only; it is not attributed to clocks or heat.
- k5 was restored after the trial and served healthily on port 8001.

Raw rounds and the paired verdict live under the eval results directory (not
tracked): `round_k5_k7pair_20261022T122131Z/` (k5 baseline matrix),
`round_k5_k7pair_chatq1/` and `round_k5_k7pair_chatq2/` (k5 quality passes),
`round_k7_20261022T124251Z/` (k7 matrix plus `k7_pair_analysis.json`),
`round_k7_chatq1/` and `round_k7_chatq2/` (k7 quality passes),
`round_k7_boot_20261022T123333Z/` (k7 boot and shutdown snapshots),
`round_k5_restore_20261022T125508Z/` (restored k5),
`compare_k7_20261022T125136Z.txt` and `k7_result_20261022T130500Z.json`.

**TTFT is not pure prefill.** It includes request, chat-template and network
overhead. Prefill figures elsewhere in `PLAN.md` are "effective" rates that
include decode and are not comparable to pure prefill kernels. Decode rates are
the streaming decode-window rate derived from the endpoint's completion-token
usage, never chunk counts.

### Quality

Quality was measured separately on the official chat endpoint (thinking
disabled, `max_tokens=1024`, natural stop): three configurations × two passes ×
24 rows (16 greedy + 8 sampled) = 144 rows.

| config | pass 1 greedy | pass 2 greedy | sampled (both) |
|---|---:|---:|---:|
| k0 | 15/16 | 15/16 | 8/8 |
| k3 | 15/16 | 15/16 | 8/8 |
| k5 | 15/16 | 14/16 | 8/8 |

`codeedit-b` fails in every pass, including k0: the generated module defines
`total_pages()` and omits the required `page_bounds` symbol, so the requested
public surface is incomplete (the code parses; the contract still fails). k5
pass 2 adds one `codegen-b` discordance
(McNemar p=1.0). Greedy outputs are not bit-identical across passes even at k0
(6/16 exact repeats at k0, 5/16 at k3, 7/16 at k5). **There is no lossless or
formal quality claim.**

The matched k5/k7 A/B added three natural-stop quality passes per configuration
on the same official chat endpoint (thinking disabled, `max_tokens=1024`), 16
greedy plus 8 sampled rows per pass:

| config | pass 1 | pass 2 | pass 3 | sampled (all three) |
|---|---:|---:|---:|---:|
| k5 | 15/16 | 15/16 | 15/16 | 8/8 |
| k7 (rejected) | 15/16 | 15/16 | 14/16 | 8/8 |

k7 gains nothing. Its third pass misses `codegen-b`: the generated module
exposes `failure_count` instead of the required consumer-visible `failures`
attribute (it parses, 1567 chars, `finish_reason=stop`), so a caller that reads
`failures` breaks. That is a public-surface requirement of the probe, not a
harness or token-check artifact, and it is k7-only across the six passes.

### Memory

P7 weights are **88.93 GiB** including the drafter. At k5, GPU0 reports
93,506 MiB used / 3,744 MiB free; the observed peak was 94,312 MiB used /
2,938 MiB free, so measured headroom is **~2.9–3.7 GiB**. The 97,887 MiB total
includes driver/display reserve; free memory was measured, not derived. No run
in the campaign OOMed.

The k7 trial booted and served with no OOM. KV capacity **20,134 tokens**
against k5's **20,227**, CUDA graph pool **0.09 GiB** against **0.08 GiB**, and
one main plus one DSpark graph captured — the same count as k5. Boot to healthy
was **502 s** for k7 and **461 s** for the restored k5. After the k7 timed run
GPU0 read **95,398 MiB used / 1,852 MiB free**; the restored k5 read **93,536
MiB used / 3,714 MiB free** at idle. Free memory is what the driver reported in
each snapshot, never total-minus-used.

## Reproduce the measurements

The Python-side transport proof runs without loading model weights:

# identity mode: every fetched partial must be bit-identical to its post-scale
# source value
python3 p7/afd_smoke.py --mode identity --device 0 \
  --peer <dev>:<gid>:<ip>:<port> --peer <dev>:<gid>:<ip>:<port> \
  --layers 40 --rows 1 --rounds 3 --warmup 1

# moe mode: the real 40-layer arena answers and routing is checked
python3 p7/afd_smoke.py --mode moe --device 0 \
  --peer <dev>:<gid>:<ip>:<port> --peer <dev>:<gid>:<ip>:<port>
```

`p7/afd_repro.py --config <peers.json> --rounds 8 --rows 1` repeats the
session-level transport check (`AFD_CONFIG` is used when `--config` is
omitted); `--split` additionally checks the split path.

The DSpark workload harness drives the live OpenAI endpoint:

```bash
# tokenizer-only dry run; no completions
python3 eval/dspark_workload_eval.py --check

# write the benchmark matrix without sending requests
python3 eval/dspark_workload_eval.py --plan-only --out-dir <results-dir>/round1

# timed replay plus quality probes
python3 eval/dspark_workload_eval.py --execute --label k5 \
  --out-dir <results-dir>/round_k5 --repeats 2 --warmup 2 \
  --quality-sampled --overwrite

# matched comparison of two completed runs
python3 eval/dspark_workload_eval.py --compare <dir-a> <dir-b> --out-dir <cmp-dir>

# matched k7 A/B: same harness, same prompts and budgets, one configuration
# per run; compare needs the completed k5 and k7 directories
python3 eval/dspark_workload_eval.py --execute --label k7 \
  --out-dir <results-dir>/round_k7 --repeats 2 --warmup 2 \
  --quality-sampled --overwrite --engine-container dsv41-dspark-k7
python3 eval/dspark_workload_eval.py --compare <k5-dir> <k7-dir> --out-dir <cmp-dir>
```

`--base-url` defaults to `http://127.0.0.1:8001`; `--seed 1017` and
`--timeout` are configurable. `--quality-only --quality-mode chat` runs the
quality probes without timed replay. `--engine-container` additionally samples
container/GPU counters when available; `--no-metrics` disables that.

## Repository map

| Path | Contents |
|---|---|
| `PLAN.md` | Full implementation plan: topology, memory ledger, workstreams W0–W8, gates, risks, DeepInfer review, DSpark results |
| `GX_FP4_DECODE_CEILING.md` | Kernel and transport ceiling measurements behind the decode-rate limits |
| `INVENTORY.md` | Measured host, network, memory, NVMe and transport numbers, with site identifiers removed |
| `gateA/` | Early transport benchmark, GX memory probe and summarized results |
| `gdr/` | Optional P7 GPUDirect/IOMMU enablement experiment; not deployed |
| `pack/` | Rank-pack builder and the three tracked manifests |
| `p7/` | P7 runtime: overlay (`afd_overlay`, `afd_bridge`, `afd_session`, `afd_seam`), smoke/repro drivers, model-directory builder, mHC overlay |
| `gx/` | GX worker and ingestion/graph code, worker-side kernels |
| `third_party/vllm/` | Apache-2.0 license text and third-party notice for the vLLM-derived files |
| `native/` | Current transport sources and offline build script (vendored; see `PROVENANCE.md`) |
| `eval/` | OpenAI-endpoint workload and quality harness |
| `tools/` | Download and shard-verification scripts |
| `scripts/` | Engine launcher, GX worker control, local site template |

## Provenance and license

- **Original QMesh code** is MIT-licensed; see the root `LICENSE`
  (`Copyright (c) 2026 QMesh contributors`). That collective line covers the
  QMesh distribution and does not name individual authors. The root MIT text
  does not cover the Apache-2.0 files below.
- **vLLM-derived files** remain **Apache-2.0**: the modified copies under
  `p7/mhc_quant/overlay/models/deepseek_v41/` and
  `p7/dspark_retention/overlay/`, the unmodified extracts under
  `p7/dspark_retention/src/`, and the mHC kernels
  `p7/mhc_quant/kernel/mhc_post_norm_sm120.cu` and `.py`. The mHC kernels are
  a modified derivative of vLLM-Moet's
  `tools/dsv41_sm120/mhc_overlap/mhc_post_norm_sm120.cu` and `.py`
  (Apache-2.0); QMesh removed the optional P2P all-reduce consumer and added
  the MXFP8/UE8M0 quantization epilogue. They retain vLLM-Moet and vLLM
  attribution and are not relicensed by the root MIT text. The exact file
  list, modification notices, upstream blob IDs and attribution are in
  `third_party/vllm/NOTICE`; the Apache-2.0 text is
  `third_party/vllm/LICENSE`. The repository redistributes only these derived
  files, not the full vLLM or vLLM-Moet source trees; the engine image built
  from vLLM-Moet is an external dependency.
- **GX kernels** `gx/kernels/ext_variant.cu` and
  `gx/kernels/fp4_decode_moe_swz.cu` are QMesh code under the root MIT
  license. TensorFold's `nvfp4_experts.cu` and `mma4.cuh` were a confirmed
  design starting reference and are acknowledged in `CREDITS.md`; the current
  implementation is structurally different, and no preserved TensorFold source
  copy has been identified in it. The preserved record does not establish the
  provenance of the initial prototype. The decode kernel's block-scale layout
  follows vLLM's documented `nvfp4_utils.swizzle_blockscale`; no vLLM source
  is copied into it.
- **Native transport sources** in `native/` are first-party QMesh code from
  the transport tree at commit `b88ec90fd2465d647dd250f7282c7a3716584c26`.
  The four seam files are byte-for-byte copies; `qmesh_afd_edge.cpp` and
  `qmesh_afd_peer.cpp` were sanitized (generic device/address examples,
  explicit arguments). The origin tree has no separate license or copyright
  header; these files are offered under the root MIT text with the collective
  `Copyright (c) 2026 QMesh contributors` line. Per-file hashes are in
  `native/PROVENANCE.md` and `native/NOTICE`.
- **Model weights** are MIT-licensed at both pinned Hugging Face revisions
  (`Copyright (c) 2023 DeepSeek`): the original
  `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`
  and the NVFP4 transcode
  `LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4@5be99cd15675589fb6693d0a8651e413bcb9a396`.
  They are downloaded separately and never committed.
- **DeepInfer** (`deepseek-ai/DeepInfer@774ce244`, Apache-2.0, released
  2026-10-20) is a design reference only. It cannot run on SM120/SM121: its
  device gate accepts only SM100 or Ascend 950, and its pinned DeepGEMM and
  FlashMLA trees contain no sm12x kernels. No DeepInfer code is vendored or
  required here. Its MoE seam structure (route → dispatch → local experts →
  shared expert → combine) informed `AFDMoE`, and the plan records which of its
  ideas were adopted as references versus not used; see `PLAN.md` §7.
- **vLLM-Moet** and its DSpark patches are an external experimental
  dependency, not part of this repository and not upstream vLLM. The engine
  image built from them is required to reproduce the DSpark numbers.
- **Full credits and the license map** are in `CREDITS.md`; the retained
  upstream notices are in `third_party/vllm/NOTICE`.
