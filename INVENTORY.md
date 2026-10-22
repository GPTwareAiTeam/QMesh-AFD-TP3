# AFD-TP3 inventory and Gate A measurements (2026-10-20)

Every number below was measured on the three experiment hosts on 2026-10-20; none are estimates. Host addresses are omitted (see Network). The model root, image tags and device names used here are the ones that were measured and are examples for a reproduction, not defaults that any script applies. P7 GPU1 and GPU2 were out of scope and never touched: every P7 container was restricted to GPU0.

## Hosts (measured 2026-10-20)

| Host | GPU | Memory | Driver / CUDA | Disk (at measurement) |
|---|---|---|---|---|
| P7 (x86_64) | GPU0 RTX PRO 6000 Blackwell **Max-Q**, sm_120, bus 52:00.0, 97,887 MiB total, 2 MiB used / 97,248 MiB free when idle | 503 GB RAM | 595.99.02, nvcc 13.2 | model root `dsv41` = Gen5 NVMe `nvme1n1p1` 3.7 TB ext4 |
| GX1 (aarch64) | GB10, sm_121, 121.63 GiB unified | MemTotal 121.6 GiB, MemAvailable 117.4 GiB at idle, swap 16 GiB | 580.173.02, nvcc 13.0 | **`/` is 100% full (5.5 GB free)** |
| GX2 (aarch64) | GB10, sm_121 | MemAvailable 117.3 GiB at idle | 580.173.02, nvcc 13.0 | `/` 95% (47 GB free) |

- P7 GPU1 and GPU2 were not used. GPU2 drives the desktop. Both are out of scope.
- **The Max-Q variant** has a lower power limit than the 600 W card, which caps P7's compute; its bandwidth still needs to be measured.

## Network (direct /30 links; NCCL `NET/IB` RoCE v2, GDR off)

Addresses are omitted. Each rail is a dedicated point-to-point /30 between the two named interfaces; use the site's own addressing when reproducing.

| Edge | P7 side | GX side | Rate | RoCE v2 IPv4 GID |
|---|---|---|---|---|
| P7–GX1 | `enp142s0f0np0` / `rocep142s0f0` | `enp1s0f0np0` / `rocep1s0f0` | 100 Gb/s | P7 idx 5, GX idx 3 |
| P7–GX2 | `enp142s0f1np1` / `rocep142s0f1` | `enp1s0f0np0` / `rocep1s0f0` | 100 Gb/s | P7 idx 5, GX idx 3 |
| GX1–GX2 | — | `roceP2p1s0f1` | 200 Gb/s | idx 3 (unused by AFD-TP3) |

- P7 HCA: bnxt_en firmware 223.0.208.0. `nvidia_peermem` is not loaded. NCCL reports "GPU Direct RDMA Disabled" on both sides and "DMA-BUF is available on GPU device 0" on P7. Its DMA-BUF MR path is unavailable because the container's `libmlx5`/rdma-core lacks `mlx5dv_reg_dmabuf_mr`; bnxt uses `ibv_reg_dmabuf_mr` [INFERENCE].
- `perftest` (`ib_write_lat`) is installed on the GX10s but not on P7. Installing it needs sudo.

## Images (NCCL 2.31.2 via `/opt/nccl-2.31.2/lib`, torch 2.11.0+cu130)
- P7: `qmesh-nccl2312-official-p7:20260814-2`. GX: `qmesh-nccl2312-official-gx10:20260814-3`.
- There is no vLLM DSv4.1 image yet. `vllm-moet-sm120:v024*` exist locally but predate DSv4.1.

## Gate A.1: transport (NCCL send/recv, graph-captured, `gateA/xfer_bench.py`)

Each link is its own 2-rank world, with bootstrap and data both on its /30. Per layer, P7 sends `R×(5120+24)` bf16 and the GX returns `R×5120` fp32. 40 layers make one step. Data integrity was checked (`data_ok=true`) in every run, including graph replay with changed inputs.

| Run | R | Eager µs/layer | Graph µs/layer | Step (eager) |
|---|---:|---:|---:|---:|
| Link 1 alone (r3) | 1 | 95.4 | 65.8 | 3.82 ms |
| Link 1 alone (r3) | 6 | 90.2 | 97.3 | 3.61 ms |
| Link 1 alone (r3) | 48 | 192.1 | 223.5 | 7.68 ms |
| **Both links concurrent** (r4) | 1 | 75–89 | 120–124 | 3.0–3.6 ms |
| **Both links concurrent** (r4) | 6 | 119–121 | 161–164 | 4.75–4.83 ms |
| **Both links concurrent** (r4) | 48 | 375–400 | 404–433 | 15.0–16.0 ms |

**Verdict against the plan's gate** (pass ≤ 60 µs per layer round trip; fail > 150):
- **NCCL as the transport fails** at the serving shape: R = 6 with both links active is about 120 µs eager and 160 µs graphed.
- Graph capture makes it *slower*, not faster. This matches the earlier QMesh finding that a graphed 3-rank AllReduce costs 104 µs per op versus 57 eager.
- The cost does not come from bandwidth: 184 KB per layer at 100 Gb/s is about 15 µs of wire time.
- The dominant terms are host staging, because GDR is off, so NCCL bounces through host buffers; NCCL's proxy thread; and per-op kernel launches. The two links also interfere with each other on P7: R = 6 goes from 90 to 120 µs.

The decision this forces is in §Next.

## Gate A.2: GX memory (`gateA/mem_probe.py`, 1 GiB chunks, every page touched)

| Host | Reached | Host MemAvailable at full | CUDA free at full | 1 GiB read+write bandwidth |
|---|---:|---:|---:|---:|
| GX1 | 107.8 GiB (106.8 arena + 1.0 buffers) | 7.30 GiB | 4.35 GiB | 222.5 GB/s |
| GX2 | 107.8 GiB | 7.23 GiB | 4.26 GiB | 224.3 GB/s |

**Pass.** The 864-column NVFP4 slice (106.79 GiB from the pack) plus 1 GiB of worker buffers fits, leaving about 7.3 GiB host headroom (above the 5 GiB floor), and holds steady. The 0.92 budget (111.9 GiB) has a further 4 GiB of margin.

## Gen5 NVMe Engram reads (historical measurement, not a deployed configuration)

Measured 2026-10-20 on the Gen5 device, O_DIRECT 4 KiB random, on the 101.5 GB shard 48. This established the storage-path ceiling; the current tree does not ship the Engram reader and the measured model root is only an example location.

| Pattern | p50 | p99 |
|---|---:|---:|
| QD1, single thread | 37 µs | 99 µs |
| QD48 (48 processes; device view) | 48 µs per IO | 81 µs per IO; 780k IOPS aggregate |
| 48 reads per token through a Python thread pool | 1,155 µs | 2,329 µs (dominated by Python overhead) |

The device can serve one token's 48 Engram rows in about one IO latency, roughly 50–80 µs. A native submit path is needed: AIO or io_uring from C/C++, as in the TensorFold dsv41 `engram_aio.zig`. The layer-14 fetch can overlap layers 1–13. The layer-1 fetch, for the next token, can start as soon as the token is sampled.

## Weights (sha256 verified against HF LFS ids; paths as measured)

- `DeepSeek-V4.1-Flash-1017-NVFP4/`: 48/48 shards OK (`logs/verify_nvfp4.log`).
- `DeepSeek-V4.1-Flash-1017-orig/`: shards 47 and 48 OK (`logs/verify_orig.log`), plus config, tokenizer, index, `inference/`, `encoding/`.
- NVFP4 expert format, checked: `w1/w3.weight` U8 [2304, 2560], `weight_scale` F8_E4M3 [2304, 320] (block 16), `weight_scale_2` F32 scalar per expert per projection. In layer 0, w1 and w3 share 3 distinct global scales across 384 experts, and w2 has 4 distinct values. The `input_scale` W4A4 values are not used.

## Next (decision from Gate A.1)

NCCL cannot carry AFD-TP3's per-layer exchange fast enough. The transport has to be replaced before the model is built. Options, cheapest first:

1. **A custom RDMA-write mesh with device-polled flags** (W3 target design, the bertholomus `rdma_gather.cu` pattern: about 23 µs per exchange over RoCE on GB10):
   - P7 writes `x`, and each GX writes its partial, straight into the peer's registered buffer with one `IBV_WR_RDMA_WRITE_WITH_IMM`.
   - A CUDA kernel on each side spins on a sequence flag (`ld.acquire.sys`).
   - No NCCL proxy, no per-op launch: the whole step is one graph, with flag waits as kernels.
   - Without GDR on P7, the buffers are `cuMemHostRegister`ed host memory, which the GPU reads and writes over PCIe5: about 2–5 µs for 184 KB.
   - On GB10, unified memory is coherent, so host-registered memory is fine.
2. **Enable GDR on P7** (load `nvidia_peermem`, or a DMA-BUF MR path with a newer rdma-core). This removes the host bounce for option 1, and for NCCL too. It needs sudo and must be checked against the bnxt driver.
3. Re-measure: target ≤ 60 µs per layer round trip at R = 6 with both links concurrent.
