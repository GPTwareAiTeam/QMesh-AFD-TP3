# Credits, licenses, and provenance

QMesh AFD-TP3 is a publication snapshot of an experimental
attention-FFN-disaggregation runtime for DeepSeek-V4.1-Flash-1017. This file
records the copyright and license status of what is in the repository, the
upstream work that informed it, and what is deliberately not redistributed.
It is not legal advice.

## License map

| Material | License | Copyright / notice |
| --- | --- | --- |
| Original QMesh code, scripts, and documentation, except the carve-outs below | MIT | `Copyright (c) 2026 QMesh contributors`; root [`LICENSE`](LICENSE) |
| `p7/mhc_quant/kernel/mhc_post_norm_sm120.cu` and `.py` | Apache-2.0 | vLLM-Moet and vLLM upstream attribution plus QMesh modifications; see below |
| `gx/kernels/ext_variant.cu` and `gx/kernels/fp4_decode_moe_swz.cu` | MIT | `Copyright (c) 2026 QMesh contributors`; TensorFold and vLLM design-reference acknowledgment below |
| Seven copied or modified vLLM/vLLM-Moet files (list below) | Apache-2.0 | Upstream vLLM/vLLM-Moet notices; [`third_party/vllm/LICENSE`](third_party/vllm/LICENSE), [`third_party/vllm/NOTICE`](third_party/vllm/NOTICE) |
| `native/` transport sources | MIT | `Copyright (c) 2026 QMesh contributors`; root [`LICENSE`](LICENSE); origin commit recorded in [`native/PROVENANCE.md`](native/PROVENANCE.md) |
| Model weights and rank packs | MIT (upstream) | `Copyright (c) 2023 DeepSeek`; not included in this repository |

The root MIT license applies to the original QMesh material. It does not
relicense the Apache-2.0 files.

## vLLM and vLLM-Moet

The runtime is built on vLLM and on the experimental
[`kacper-daftcode/vLLM-Moet`](https://github.com/kacper-daftcode/vLLM-Moet)
SM120 patch set. The base is vLLM v0.24.0. The experiment used the image
`vllm-moet:dsv41-213d5fa`, recorded in the source tree as vLLM-Moet build
commit `49cb2bbf71cfd36ee5aa33d1e184b1cddff55294`, project HEAD
`19befd2f38783fed5033fb260d82e5c4acb12b36`. Those two image/HEAD commits are
not present in the public vLLM-Moet repository API; the public main commit
used for the direct source comparison below is
`5fcc1991fc12a5dfeb0a9ec0e3b741a10e282dab` (2026-10-22).

Copied or modified files retained in this repository:

- `p7/mhc_quant/overlay/models/deepseek_v41/nvidia/ops/mhc.py` (modified)
- `p7/mhc_quant/overlay/models/deepseek_v41/nvidia/model.py` (modified)
- `p7/mhc_quant/overlay/models/deepseek_v41/attention.py` (modified)
- `p7/dspark_retention/overlay/vllm/v1/core/kv_cache_planning.py` (modified)
- `p7/dspark_retention/overlay/vllm/v1/attention/backends/mla/sparse_swa.py`
  (modified)
- `p7/dspark_retention/src/vllm/v1/core/kv_cache_planning.py` (unmodified copy)
- `p7/dspark_retention/src/vllm/v1/attention/backends/mla/sparse_swa.py`
  (unmodified copy)

The five modified files carry QMesh modification notices. The two unmodified
extracts carry their upstream headers and are pinned by SHA-256 in
`p7/dspark_retention/manifest.txt` and `package.sh`. The full upstream NOTICE
is retained in `third_party/vllm/NOTICE`; the Apache-2.0 text is
`third_party/vllm/LICENSE`. Those files remain Apache-2.0 and are not
relicensed by the root MIT text. The full vLLM and vLLM-Moet source trees, the
engine image, and any prebuilt wheel are external dependencies and are not
redistributed here.

## Derived mHC kernel

`p7/mhc_quant/kernel/mhc_post_norm_sm120.cu` and
`p7/mhc_quant/kernel/mhc_post_norm_sm120.py` are **modified derivatives** of
vLLM-Moet, not independent implementations. The public upstream files are:

- `tools/dsv41_sm120/mhc_overlap/mhc_post_norm_sm120.cu`, blob
  `13ec2c97631ce020cc194354ef8e008af696a21d`
- `tools/dsv41_sm120/mhc_overlap/mhc_post_norm_sm120.py`, blob
  `e2c58328b1c32c3e6b7524b87c51714ae5f7b820`

The direct line comparison found 478 matching lines out of 579 upstream lines
in the `.cu` file (difflib ratio 0.811) and 139 matching lines out of 161
upstream lines in the `.py` file (ratio 0.781). The local file keeps the
upstream kernel structure and most of its comments. QMesh changes are:

- removed the optional peer-to-peer all-reduce consumer (`kAR` path, `ar_*`
  arguments, and the `p2p_allreduce_sm120` call);
- added the MXFP8/UE8M0 quantization epilogue (`mhc_post_norm_quant`) and its
  swizzled scale output;
- adapted the Python wrapper and the `mhc_post_norm` signature.

The upstream `.cu` file implements the behavior of vLLM's TileLang mHC kernels
(`mhc_fused_tilelang`, `mhc_post_tilelang`, `mhc_pre_big_fuse_with_norm`) and
references vLLM-Moet's `mhc_pre_norm_sm120.cu`; the latter was compared at a
lower line-similarity ratio (0.210) and is a behavioral reference rather than
the primary copied source.

Both QMesh mHC files carry:

```text
SPDX-FileCopyrightText: Copyright 2026 the vLLM-Moet authors
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
SPDX-FileCopyrightText: 2026 QMesh contributors
SPDX-License-Identifier: Apache-2.0
```

They are Apache-2.0, are not under the root MIT license, and include a
prominent modification notice as required by Apache-2.0 section 4(b). The
upstream NOTICE supplies the project-level holder `Copyright 2026 the
vLLM-Moet authors`; the specific upstream file has no per-file copyright line.

## TensorFold and the GX kernel

The shipped GX decode kernel (`gx/kernels/fp4_decode_moe_swz.cu`) is an
independent QMesh NVFP4 implementation. TensorFold
([`ashhart/TensorFold`](https://github.com/ashhart/TensorFold), commit
`bd264368ae7382672faef6cc99c2add8b24c5ec0`, Apache-2.0, `Copyright 2026
TensorFold contributors`) was consulted as a design starting point. The files
examined were `zig/kernels/cuda/nvfp4_experts.cu` (blob
`afbaac8925b42bbce21adc98c64d317fda740f15`) and `mma4.cuh` (blob
`6c905f55df09d0b91522b9e68e48daceb442d5b8`), together with TensorFold's
`NOTICE` (blob `50e45b9ac90ea00ca45ab7afbdf3590b696e53d1`) and `LICENSE` (blob
`d645695673349e3947e8e5ae42332d0ac3164cd7`).

After stripping comments and tokenizing, the shipped v7 kernel has a
difflib sequence ratio of 0.069 against `nvfp4_experts.cu`. The longest
matching blocks are 14 generic tokens (PTX register constraints, an unrolled
loop header, standard includes); there are no shared QMesh/TensorFold function
or type names. The excluded v6 kernel, which is not shipped in this
repository, gives 0.070 and the same generic-only matches. On that evidence,
TensorFold is a design reference, not copied source; no TensorFold source code
is redistributed. The GX kernel remains MIT-licensed QMesh code, with the
TensorFold acknowledgment in its header.

The GX decode kernel's block-scale layout follows vLLM's documented
`nvfp4_utils.swizzle_blockscale` convention (Apache-2.0, `Copyright
contributors to the vLLM project`). No vLLM source text is copied into the GX
kernel. The convention is an interface/layout reference; the GX implementation
is QMesh's.

## Native transport

`native/` contains six first-party QMesh transport sources from the private
transport tree at branch `qmesh-hybrid-hotpath`, commit
`b88ec90fd2465d647dd250f7282c7a3716584c26`. The four `qmesh_afd_seam*` files
are byte-for-byte copies; `qmesh_afd_edge.cpp` and `qmesh_afd_peer.cpp` were
public-sanitized (generic RDMA device names, loopback examples, explicit
arguments) with executable code otherwise unchanged.

The origin tree has no `LICENSE`, `COPYING`, or `NOTICE` file and the sources
carry no per-file copyright or SPDX header. The repository owner publishes
them under the root MIT license text with the collective line `Copyright (c)
2026 QMesh contributors`. That line covers the QMesh distribution and does not
name individual authors. Per-file SHA-256 values and the include closure are
recorded in `native/PROVENANCE.md`; they are unchanged by this notice work.

## Model weights and DSpark

The experiment serves the DeepSeek-V4.1-Flash-1017 checkpoint. Both pinned
Hugging Face revisions contain an MIT license file (`Copyright (c) 2023
DeepSeek`):

- Original FP8 Engram and metadata:
  [`deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
- NVFP4 transcode, including DSpark `mtp.0`-`mtp.2`:
  [`LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4@5be99cd15675589fb6693d0a8651e413bcb9a396`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4/tree/5be99cd15675589fb6693d0a8651e413bcb9a396)

The weights, rank packs, and checkpoints are not committed to this
repository. They are downloaded separately. DSpark is part of the model
checkpoint (`mtp.0`-`mtp.2`), not a vendored library. The transcode's own
license file is the upstream MIT text; retain it if the weights or packs are
redistributed.

## Other acknowledgments

These projects informed the design, are runtime dependencies of the external
engine image, or are system libraries. Except where noted, no source code,
license text, or binary from them is redistributed in this repository.

- **DeepGEMM** ([`deepseek-ai/DeepGEMM`](https://github.com/deepseek-ai/DeepGEMM),
  MIT, `Copyright (c) 2025 DeepSeek`), runtime pin `a6b593d2`. Its SM120 backend
  was authored by the NVIDIA DevTech APAC team
  ([DeepGEMM PR #324](https://github.com/deepseek-ai/DeepGEMM/pull/324)). The
  engine image builds it from source; no DeepGEMM source or prebuilt wheel is
  in this repository.
- **DeepInfer** ([`deepseek-ai/DeepInfer`](https://github.com/deepseek-ai/DeepInfer),
  Apache-2.0, commit `774ce244`), design reference only. It cannot run on
  SM120/SM121 (its device gate accepts SM100 or Ascend 950), and its pinned
  DeepGEMM and FlashMLA trees contain no sm12x kernels. No DeepInfer code is
  vendored or required.
- **NVIDIA CUTLASS** ([`NVIDIA/cutlass`](https://github.com/NVIDIA/cutlass),
  BSD-3-Clause, `Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES`),
  used through vLLM custom ops. No CUTLASS source or license text is in this
  repository; review the upstream license before publishing a binary that
  contains it.
- **FlashInfer** ([`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer),
  Apache-2.0), runtime/reference for SM120/SM121 NVFP4 and MXFP8 kernels. No
  FlashInfer source is in this repository.
- **TileLang** ([`tile-ai/tilelang`](https://github.com/tile-ai/tilelang), MIT
  with additional Microsoft collaboration terms for 2024-12-01 through
  2025-03-14), behavior reference for vLLM's mHC TileLang kernels. No TileLang
  source is in this repository.
- **FlashMLA** ([`deepseek-ai/FlashMLA`](https://github.com/deepseek-ai/FlashMLA),
  MIT), runtime dependency of the copied vLLM MLA backend. No FlashMLA source
  is in this repository.
- **bertholomus**, research reference for the one-sided RDMA gather pattern and
  the seeded GSM8K/MMLU evaluation subsets. No canonical URL or source is
  recorded in this repository; nothing from that work is bundled.
- **rdma-core / libibverbs** ([`linux-rdma/rdma-core`](https://github.com/linux-rdma/rdma-core)),
  **NVIDIA CUDA runtime** ([CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)),
  and **NVIDIA NCCL** ([`NVIDIA/nccl`](https://github.com/NVIDIA/nccl)) are
  external system/runtime dependencies. The licenses of the installed
  packages govern; no source or binaries from them are redistributed here.

## Provenance notes

- The root MIT boilerplate was inherited from a llama.cpp/ggml-derived
  template and previously carried `Copyright (c) 2023-2026 The ggml authors`.
  No ggml or llama.cpp source is present in this repository, and that line was
  not a statement of QMesh authorship. It has been replaced with the
  collective line `Copyright (c) 2026 QMesh contributors`.
- This standalone snapshot has no `.git` directory. Commit and image metadata
  above come from the source tree's notices and manifests, not from a local
  Git history.
- The prior `third_party/vllm/NOTICE` text described the mHC kernel as having
  unresolved derivation and as copying no upstream source file. Direct
  comparison against public vLLM-Moet source shows that was wrong; the notice
  and kernel headers now record the derivative and the Apache-2.0 treatment.
- All live overlay headers, patcher strings, and package-script references now
  cite the standalone path `third_party/vllm/NOTICE`; no live reference keeps
  the original `experiments/dsv41-afd-tp3/...` prefix. The DSpark overlay files
  remain SHA-256 pinned in `p7/dspark_retention/manifest.txt` and `package.sh`,
  and those overlay pins were regenerated after the header change. Source
  hashes are unchanged.
- No static expected-hash pin covers the mHC kernel files or the GX kernel
  sources. Changing their headers changes JIT/build cache keys and will force a
  rebuild, but does not invalidate a committed manifest hash.

## Open legal/attribution items

This section records what is not fully settled. It is not a claim of legal
clearance.

1. **mHC attribution specificity.** The shipped file is a modified copy on the
   direct code comparison above (478 of 579 upstream lines match); on that
   evidence QMesh treats it as a derivative and applies the Apache-2.0
   obligations (license text, NOTICE, modification notices, upstream
   attribution). The remaining question is that the specific upstream mHC file
   has no per-file copyright line; the header uses the project-level holder
   `Copyright 2026 the vLLM-Moet authors` from the upstream NOTICE. If upstream
   identifies a different specific copyright holder for that file, update the
   header.
2. **Image revision provenance.** The experiment image records build commit
   `49cb2bbf71cfd36ee5aa33d1e184b1cddff55294` and project HEAD
   `19befd2f38783fed5033fb260d82e5c4acb12b36`, which are not in the public
   vLLM-Moet API. The mHC comparison used public main
   `5fcc1991fc12a5dfeb0a9ec0e3b741a10e282dab`. If the image's mHC source
   differs, re-run the comparison and update the blob IDs and change summary.
3. **Native authorization.** The native origin tree supplies no license file or
   copyright header. Publishing those first-party files under the root MIT
   text depends on the owner's authority to license the work under the
   collective QMesh contributors line. This is an ownership/authorization
   item, not a third-party license.
4. **Model weights.** Both pinned model revisions contain MIT license files.
   The weights and rank packs are not in this repository. If they are published
   separately, include the upstream MIT notice and verify the transcode's
   provenance.
5. **TensorFold attribution treatment.** The token comparison above found no
   copied TensorFold source text in the shipped or excluded QMesh kernels. On
   that evidence the GX kernel remains MIT-licensed QMesh code with a
   design-reference acknowledgment, and no TensorFold NOTICE is added. A
   reviewer may confirm this treatment.
6. **Not legal advice.** A reviewer should confirm the collective copyright
   line, the mHC attribution, and the native authorization before a formal
   release.
