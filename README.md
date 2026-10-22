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
completion). The **120 tok/s target was not reached by any per-request or
aggregate-pair median**, and no lossless or formal quality claim is made.

A matched k5/k7 A/B then measured `SPEC_TOKENS=7` (captures 1,7,8, max 8)
against the retained k5 under the same engine budget: k5 **77.84 / 105.07
tok/s**, k7 **67.67 / 95.97 tok/s** (short / long decode), **−13.07% / −8.65%**,
acceptance **0.6918 vs 0.5486**, and no quality gain (**15/16** in all three k5
passes vs **15/16, 15/16, 14/16** at k7). k7 was rejected and k5 restored.

The final 2026-10-22 campaign replaced the earlier reusable fixtures with four
authored, exactly tokenized tasks (prose, structured JSON, Python code and a
CPU-oracle reasoning problem). One boot with a **2 GiB explicit KV pool** served
**64/64 groups and 240/240 measured requests** at C1/2/4/8 with zero invalid
rows, zero preemptions and healthy post-run state. The overall per-request decode
median was **88.27 tok/s at C1** and **22.05 at C8**; the best per-type C1 median
was **111.18 tok/s** (code, peak 119.78). The C8 group aggregate was **163.09
tok/s**, but that is eight concurrent requests, not a single-request result.
Group prefill stayed flat (2,679.00 → 2,716.73 tok/s, +1.4%). These distinct
fixtures are not directly comparable to the earlier reusable-fixture campaign,
and this campaign measured only the 2 GiB pool; no 3 GiB fit is claimed.

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
   launcher's defaults are `MAX_MODEL_LEN=8192`, `MAX_NUM_SEQS=1` and
   `MAX_NUM_BATCHED_TOKENS=4096`; prefix caching is off. These are defaults,
   not the concurrency recipe below.

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

For the concurrency / 2 GiB KV campaign:

```bash
SPEC_TOKENS=5 GPU_MEM_UTIL=0.97 DSPARK_EXTRA_GIB=1 \
CAPTURE_SIZES=1,5,6,12,24,48 MAX_CAPTURE=48 \
MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=8 MAX_NUM_BATCHED_TOKENS=2048 \
KV_CACHE_BYTES=2147483648 NATIVE_CTX_MEM=1 \
EXTRA_DOCKER_ARGS="-e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64" \
DSPARK_ENGINE_EVENT_FILE=/tmp/afd_dump/cwb-engine-events.jsonl \
NAME=dsv41-dspark-c8-kv2g bash scripts/launch_engine_graphs.sh
```

`NATIVE_CTX_MEM=1` mounts the exact-bound indexer overlay, and
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` is passed through the container
boundary as the campaign's explicit logits bound. The engine event file is
opt-in and must live under `/tmp/afd_dump`, which is shared with the container.

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

**Prefill here is request-level, not a kernel.** TTFT includes request,
chat-template and network overhead. The concurrency campaign's engine prefill
is `prompt_tokens / (first_token_ts - scheduled_ts)`, so it includes the first
generated token and scheduler interleaving. Neither figure is comparable to a
pure prefill kernel. Decode rates are the streaming decode-window rate derived
from the endpoint's completion-token usage, never chunk counts.

### Distinct-fixture concurrency campaign (2026-10-22; current)

The final campaign used four authored fixtures rather than the earlier reusable
prompt sets: a **prose** story with fixed station/family facts, a **structured**
shipment-records-to-typed-JSON task, a **code** task for a thread-safe
capacity-bounded Python 3.12 LRU cache and tests, and a **reasoning** production
allocation problem with a frozen, CPU-enumerated oracle. Each fixture is
tokenized to exactly **2048** tokens for the prefill prompt; decode uses **512**
prompt tokens from the same fixture, and the structured decode prompt carries
both complete shipment records (SHP-3101 and SHP-3102). The task text and
expected-shape guidance are in `eval/fixtures_distinct.py` (SHA256
`0584df50…`); the approved task list is SHA256 `ae4e6090…`.

One boot, not restarted for this run, served the matrix with the retained k5
recipe: `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=8`,
`MAX_NUM_BATCHED_TOKENS=2048`, captures `1,5,6,12,24,48`, native-context
overlays, prefix caching off, and an explicit **2 GiB** KV pool. It completed
**64/64 groups, 240/240 measured requests and 32 excluded warmups**, with zero
invalid rows, zero preemptions, **272 unique engine events** and no foreign
request IDs, then exited zero under `--strict --require-engine-events` and
`--require-exclusive`. Run
`20261022T215606Z-a83313` ran **21:56:03–22:39:58Z**; the 64 measured groups
took **2,174.71 s (36m15s)**, and the full harness wall including startup and
warmup was **2,631.831 s (43m52s)**; final HTTP health was **200** with
running/waiting/KV **0/0/0** and no post-run counter movement.

Prefill is **2048 prompt / 1 completion**; decode is **512 prompt / 2048
completion** with `ignore_eos` and `max_tokens == min_tokens`, so the decode
window measures **2047** tokens after the first chunk. Engine prefill is
`prompt_tokens / (first_token_ts - scheduled_ts)`; stream decode removes the
whole first multi-token emission, `(usage.completion_tokens -
len(first_chunk_token_ids)) / (last_token_arrival - first_token_arrival)`.
Group prefill is `sum(prompt_tokens) / (last first_token - gate release)`;
group decode is `sum(decode_tokens_after_first) / (last token - first token)`.
Medians and peaks below are per valid request; peak is the single fastest
request, never a sustained rate. `n` is 2/4/8/16 measured requests per cell
(two groups), so the cells are descriptive, not statistically strong. Each type
has one authored fixture and two groups per cell; these are fixture-level
observations, not category-level generalizations.

| type | C | n | prefill med | prefill peak | decode med | decode peak | weighted accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| prose | 1 | 2 | 2,712.79 | 2,714.31 | 50.56 | 52.36 | 0.2336 |
| prose | 2 | 4 | 1,862.37 | 2,166.44 | 32.49 | 35.00 | 0.2464 |
| prose | 4 | 8 | 1,459.87 | 2,170.48 | 18.49 | 19.01 | 0.2240 |
| prose | 8 | 16 | 1,362.44 | 2,169.66 | 13.02 | 15.31 | 0.2377 |
| structured | 1 | 2 | 2,670.86 | 2,672.67 | 98.39 | 106.76 | 0.7162 |
| structured | 2 | 4 | 1,830.00 | 2,123.23 | 59.73 | 66.62 | 0.6958 |
| structured | 4 | 8 | 1,439.81 | 2,123.77 | 36.78 | 42.07 | 0.8088 |
| structured | 8 | 16 | 1,341.59 | 2,124.36 | 23.61 | 29.20 | 0.7196 |
| code | 1 | 2 | 2,717.73 | 2,719.09 | 111.18 | 119.78 | 0.8332 |
| code | 2 | 4 | 1,868.90 | 2,180.21 | 66.18 | 72.93 | 0.8100 |
| code | 4 | 8 | 1,459.88 | 2,178.43 | 35.74 | 41.97 | 0.7822 |
| code | 8 | 16 | 1,364.21 | 2,174.85 | 26.02 | 29.65 | 0.8080 |
| reasoning | 1 | 2 | 2,727.36 | 2,727.83 | 81.73 | 86.51 | 0.5237 |
| reasoning | 2 | 4 | 1,874.28 | 2,183.51 | 54.60 | 55.83 | 0.5691 |
| reasoning | 4 | 8 | 1,467.35 | 2,183.00 | 32.66 | 36.79 | 0.5768 |
| reasoning | 8 | 16 | 1,369.70 | 2,186.63 | 20.89 | 30.02 | 0.5909 |

Prefill columns use only prefill-shape 2048/1 rows; decode columns use
decode-shape 512/2048 rows. Decode-shape engine prefill is secondary and never
pooled. The weighted acceptance column is `sum(accepted draft tokens) /
sum(proposed draft tokens)` for that cell, not a mean of percentages.

| shape | C | requests | per-request med | per-request peak | group med | group peak | group wall med (s) | TTFT med (s) | active max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill | 1 | 8 | 2,715.34 | 2,727.83 | 2,679.00 | 2,691.69 | 0.768 | 0.767 | 1 |
| prefill | 2 | 16 | 1,843.53 | 2,183.51 | 2,702.13 | 2,716.10 | 1.519 | 1.427 | 2 |
| prefill | 4 | 32 | 1,454.45 | 2,183.00 | 2,712.25 | 2,724.38 | 3.024 | 2.464 | 2 |
| prefill | 8 | 64 | 1,364.21 | 2,186.63 | 2,716.73 | 2,729.98 | 6.034 | 3.976 | 2 |
| decode | 1 | 8 | 88.27 | 119.78 | 88.27 | 119.78 | 23.630 | 0.424 | 1 |
| decode | 2 | 16 | 55.81 | 72.93 | 107.91 | 132.08 | 38.693 | 0.817 | 2 |
| decode | 4 | 32 | 33.74 | 42.07 | 125.23 | 147.58 | 66.494 | 1.141 | 4 |
| decode | 8 | 64 | 22.05 | 30.02 | 163.09 | 200.24 | 101.482 | 1.905 | 8 |

From C1 to C8, group prefill was flat at **2,679.00 → 2,716.73 tok/s (+1.4%)**
and group decode rose **88.27 → 163.09 tok/s (1.85×)** for 8× offered
concurrency. Per-request medians fell **2,715.34 → 1,364.21** (prefill) and
**88.27 → 22.05** (decode); group walls rose **0.768 → 6.034 s** and
**23.630 → 101.482 s**. C1 is the latency-optimal point.

**The 120 tok/s target is a single-stream, per-request target.** The overall C1
per-request decode median is **88.27**; the best per-type C1 median is
**111.18** with a single-request peak of **119.78** (code). No per-request or
per-type median reaches 120. The **163.09 tok/s C8 value is a group aggregate
across eight concurrent requests**, not a single-request result. The earlier
old-fixture tool-agent peak of **135.0 tok/s** remains historical evidence for
that fixture set; these distinct fixtures neither reproduce nor replace it, and
the two sets are not directly comparable.

**DSpark acceptance (decode only).** Across all 32 decode groups, **69,404
steps** proposed **347,020** draft tokens and accepted **176,466**, weighted
**0.5085** (50.85%). Per-position unconditional acceptance was **0.7853 /
0.6022 / 0.4689 / 0.3758 / 0.3104** at positions 0–4. The derived
**1 + accepted/steps = 3.5426** tokens/step is theoretical and includes the
bonus token; the actual generation-counter rate was **3.5410** tokens/step,
which also includes the first generated token and final output-limit
truncation. Neither is a lossless, quality or batch-invariance proof. Prefill
groups proposed zero draft tokens, so their acceptance is **N/A**, not zero.

**Residency, queues, memory and source pins.** Decode request-scoped
`[scheduled, last]` overlap is exactly **1/2/4/8** at C1/2/4/8 for every type
and repeat; all eight C8 requests receive tokens. Prefill `active_max` is only
**2** at C4/C8 because one 2048-token prompt fills the entire
`MAX_NUM_BATCHED_TOKENS=2048` budget, so offered C4/C8 are not fully resident
for prefill: queue-time maxima were **1.50 s at C4 and 4.51 s at C8** (decode
C8: 0.766 s). C8 prefill is a queued control, not eight parallel prefills.
Every group's DSpark counters settled in two polls (maximum 215–217 ms);
`/metrics` running gauges were not used as residency proof.

The headerless GPU CSV has **8,938 rows**, of which **8,937 are GPU0 samples**
at 250 ms, with a **572 MiB** free floor in every type×C cell, **96,678 MiB**
maximum used, 100% peak utilization and **196.08 W** peak power. This is an
observed floor, not a reserve or future-headroom estimate; no 3 GiB fit is
claimed. Source pins: harness `25e35671…`, fixtures `0584df50…`, contexts and
prompt index `2433c0eb…` / `b12a6fa5…`, prep manifest `67b09d35…`. Artifacts
are named relative to the concurrency results directory: `campaign/summary.json`
(SHA256 `96b70046…`), `campaign/summary.md` (`8c1301ad…`), `rows.jsonl`
(`ed6aaf3d…`), `groups.jsonl` (`a6615379…`), `campaign_engine_events.jsonl`
(`066486d1…`), `gpu_campaign.csv` (`8e694b02…`) and
`analysis/artifact_summary.json` (`f8c309b3…`), with 128 per-group
before/after metric files under `campaign/metrics/`.

### Earlier concurrency campaign (old reusable fixtures, 2026-10-22; historical)

**Historical evidence with the old reusable fixtures and old type labels.**
These tables and figures are preserved exactly but are not directly comparable
to the current distinct-fixture campaign. The **135.0 tok/s** single-request
peak is a historical old-tool-agent observation, not a current hardware limit.

#### Old-fixture measurement table (historical)

One boot configured for native 1,048,576 tokens (no 1M request was sent) with
an explicit 2 GiB KV pool served the full matrix: **64/64 groups, 240/240
measured requests, plus 32 warmup requests with zero invalids**, zero
preemptions. Warmups are excluded from the 240.
Prefill is **2048 prompt / 1 completion**; decode is **512 prompt / 2048
completion**. Recipe: k5, `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=8`,
`MAX_NUM_BATCHED_TOKENS=2048`, captures `1,5,6,12,24,48`,
`NATIVE_CTX_MEM=1`, prefix caching off.

Engine prefill is `prompt_tokens / (first_token_ts - scheduled_ts)`; stream
decode is `(usage.completion_tokens - len(first_chunk_token_ids)) /
(last_token_arrival - first_token_arrival)`. Group prefill is
`sum(prompt_tokens) / (last first_token - gate release)`; group decode is
`sum(decode_tokens_after_first) / (last token - first token)`. Median and peak
are over valid requests; peak is the single fastest request, not a sustained
rate. The prefill columns use only prefill-shape **2048/1** rows and the decode
columns use decode-shape **512/2048** rows; decode-shape engine prefill is
secondary and never pooled. `n` is measured requests per type×C (2/4/8/16),
with two groups per cell, so these medians are descriptive rather than
statistically strong.

| type | C | n | prefill med | prefill peak | decode med | decode peak |
|---|---:|---:|---:|---:|---:|---:|
| code | 1 | 2 | 2,651.6 | 2,652.1 | 95.8 | 99.7 |
| code | 2 | 4 | 1,815.3 | 2,089.4 | 73.9 | 75.2 |
| code | 4 | 8 | 1,438.1 | 2,088.6 | 42.1 | 43.1 |
| code | 8 | 16 | 1,335.7 | 2,089.1 | 28.4 | 32.0 |
| prose | 1 | 2 | 2,673.2 | 2,676.6 | 88.9 | 95.0 |
| prose | 2 | 4 | 1,828.1 | 2,103.7 | 65.6 | 74.6 |
| prose | 4 | 8 | 1,448.7 | 2,100.9 | 32.7 | 39.8 |
| prose | 8 | 16 | 1,342.9 | 2,099.0 | 23.7 | 28.9 |
| json | 1 | 2 | 2,651.8 | 2,652.8 | 78.7 | 81.4 |
| json | 2 | 4 | 1,818.8 | 2,096.5 | 49.7 | 57.5 |
| json | 4 | 8 | 1,438.9 | 2,101.2 | 30.5 | 34.3 |
| json | 8 | 16 | 1,336.3 | 2,098.6 | 20.8 | 25.3 |
| toolagent | 1 | 2 | 2,615.7 | 2,615.8 | 119.4 | 135.0 |
| toolagent | 2 | 4 | 1,786.0 | 2,055.7 | 69.3 | 74.0 |
| toolagent | 4 | 8 | 1,418.3 | 2,059.7 | 44.0 | 45.9 |
| toolagent | 8 | 16 | 1,314.7 | 2,058.4 | 31.1 | 31.9 |

C1 is the latency-optimal point. From C1 to C8, group prefill was flat
(**2,633.4 → 2,664.6 tok/s, +1.2%**) while group decode rose **1.90×**
(**93.5 → 178.0**) for 8× offered concurrency. Per-request medians fell from
**2,651.6 → 1,336.4** (prefill; includes the first token and interleaving) and
**93.5 → 25.0** (decode). Group walls rose **0.781 → 6.152 s** (prefill) and
**22.328 → 93.107 s** (decode).

The **120 tok/s target is a single-stream, per-request target**, not a
multi-request throughput target: the 178.0 tok/s C8 value is a group aggregate
across eight concurrent requests, not a per-request result.
The overall per-request decode median is **93.5 at C1 and 25.0 at C8**, and the
best per-type C1 median is **119.4** (tool-agent), so this campaign does not
establish a 120 tok/s single-request result.

Decode residency is real: request-scoped `[scheduled, last]` overlap is exactly
**1/2/4/8** at C1/2/4/8 for every type and repeat, and all C requests receive
tokens. Prefill cannot be: one 2048-token prompt fills
`MAX_NUM_BATCHED_TOKENS=2048`, so scheduled-to-last overlap peaks at **2** at
C2/4/8; resident `[queued, finished]` overlap is **1/2/4/8** with median queue
wait **0 / 0 / 385 / 1,920 ms**. The C8 prefill group is therefore a queued
control, not eight parallel prefills. `/metrics` group summaries are sampled
after completion; an independent 250 ms watcher over the last 20 groups
recorded decode running **1/2/4/8** and C8 prefill running **1** / waiting
**6**. Its cache-usage gauge reported 0.0 throughout and is unusable.

The 2 GiB pool and the 572 MiB campaign minimum are in [Memory](#memory). The
2,073,385-token and 1.98× figures there are admission arithmetic, not a measured
1M-token request.

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

The final distinct-fixture campaign used an explicit **2 GiB KV pool** on the
same k5 stack, not restarted for the run. The engine reported **18,320 blocks ×
117,216 B = 2,147,397,120 B** (86,528 B spare), block size **128**,
`nvfp4_ds_mla`, **2,073,385 tokens**, and **1.98×** maximum native concurrency
at the configured 1,048,576-token limit. The headerless GPU CSV has **8,938
rows**, of which **8,937 are GPU0 samples** at **250 ms**; the free floor was
**572 MiB** in every type×C cell and every sampled phase, used reached
**96,678 MiB**, utilization peaked at **100%** and power at **196.08 W**. There
was no OOM, no preemption and no measured leak. This is an observed campaign
floor, not a reserve or a forecast of future headroom.

At that floor, a **512 MiB** reserve allows only **+60 MiB** of cache
(**2.06 GiB**, +537 blocks / +68,703 tokens); a 1 GiB reserve leaves no room.
Only the 2 GiB pool was measured; **no 3 GiB fit is claimed.**

The earlier old-fixture campaign recorded **7,612** samples with the same
**572 MiB** floor and **96,678 MiB** watermark, plus a **1,222 MiB** boot/load
minimum and **1,656 MiB** post-boot idle snapshot. Those numbers are retained
as historical context for that run.

The token and concurrency figures are pool/admission arithmetic, not a measured
1M-token request. `MAX_MODEL_LEN` was **1,048,576**, but the largest prompt
tested is **2048 tokens** and the largest total is **2560**; no long-context or
1M-request claim follows.

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

The distinct-fixture concurrency / KV campaign uses
`eval/fixtures_distinct.py` through its own harness against the same endpoint:

```bash
python3 -u eval/concurrency_workload_eval.py \
  --base-url http://127.0.0.1:8001 \
  --engine-event-file /tmp/afd_dump/cwb-engine-events.jsonl \
  --metrics --metrics-interval 0.2 --metrics-settle-timeout 12 --metrics-settle-poll 0.2 \
  --spec-k 5 --spec-strict --require-exclusive --strict --require-engine-events --overwrite \
  --out <results-dir>/campaign --label distinct-full \
  --types prose,structured,code,reasoning --shapes prefill,decode \
  --concurrency 1,2,4,8 --repeats 2 --warmup 1 \
  --prefill-tokens 2048 --prefill-output 1 --decode-tokens 512 --decode-output 2048 \
  --gpu-csv <results-dir>/gpu_campaign.csv \
  --cache-capacity-json <results-dir>/cache_capacity_c8_kv2g.json
```

Defaults used were `--seed 20261022`, `--timeout 600`,
`--engine-event-timeout 30` and `--idle-timeout 120`. The harness sends the
exact fixture prompt to raw `/v1/completions` with temperature 0,
`ignore_eos=true` and `max_tokens == min_tokens`, so this is a forced-performance
measurement, not a chat-quality evaluation. It writes `campaign/summary.json`
and `summary.md` directly from the measured rows, with counts, hashes and GPU
aggregates in `analysis/artifact_summary.json`; no API rerun or offline
reprocessing step is needed for those summaries.

`--cache-capacity-json` is the block-arithmetic input used for the capacity
cross-check, not an engine measurement. `--require-engine-events` rejects rows
without a request-scoped engine event, and `--require-exclusive` rejects
non-exclusive groups.

`--base-url` defaults to `http://127.0.0.1:8001`; the concurrency harness seed
defaults to `20261022`, and `--timeout` is configurable. `--quality-only
--quality-mode chat` (DSpark harness) runs the quality probes without timed
replay. `--engine-container` additionally samples
container/GPU counters when available; `--no-metrics` disables that.

## Repository map

| Path | Contents |
|---|---|
| `PLAN.md` | Full implementation plan: topology, memory ledger, workstreams W0–W10, gates, risks, DeepInfer review, DSpark and concurrency results |
| `GX_FP4_DECODE_CEILING.md` | Kernel and transport ceiling measurements behind the decode-rate limits |
| `INVENTORY.md` | Measured host, network, memory, NVMe and transport numbers, with site identifiers removed |
| `gateA/` | Early transport benchmark, GX memory probe and summarized results |
| `gdr/` | Optional P7 GPUDirect/IOMMU enablement experiment; not deployed |
| `pack/` | Rank-pack builder and the three tracked manifests |
| `p7/` | P7 runtime: overlay (`afd_overlay`, `afd_bridge`, `afd_session`, `afd_seam`), smoke/repro drivers, model-directory builder, mHC overlay |
| `gx/` | GX worker and ingestion/graph code, worker-side kernels |
| `third_party/vllm/` | Apache-2.0 license text and third-party notice for the vLLM-derived files |
| `native/` | Current transport sources and offline build script (vendored; see `PROVENANCE.md`) |
| `eval/` | OpenAI-endpoint workload and quality harnesses, including `fixtures_distinct.py` for the four authored contexts |
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
