#!/usr/bin/env python3
"""Patch vLLM's sparse-indexer prefill bound to the exact per-chunk size.

`vllm/v1/attention/backends/mla/indexer.py::get_max_prefill_buffer_size` used
to return `max_model_len * (576 * 2 // 132) * factor` (8.7x `max_model_len` at
factor 2). That multiplier exists only to fill the flashmla_sparse prefill
workspace envelope. The DSV4.1 AFD-TP3 recipe selects
FLASHINFER_MLA_SPARSE_DSV41, which never allocates that workspace, so the extra
rows bought no chunking capacity and cost 1.06 GiB of persistent K-gather
workspace at max_model_len=1M.

The returned value is consumed in two places that must agree:

* `DeepseekV32IndexerMetadataBuilder.max_prefill_buffer_size`
  (= value // compress_ratio) is the chunker's per-chunk
  `total_seq_lens` bound (N constraint);
* `DeepseekV41Indexer.max_total_seq_len` (= value // compress_ratio) is the
  K-gather workspace row count, whose per-chunk view is
  `k_quant_full[: chunk.max_local_total_seq_lens]`.

`factor * max_model_len` rows (factor >= 2) is an exact bound for both: the
chunker never exceeds `workspace_size` except for its single-request fallback,
and that fallback is one request's compressed length, which is at most
`ceil(max_model_len / compress_ratio)`.

This patcher is deterministic: it rewrites one function body and inserts the
Apache-2.0 section 4(b) modification notice after the SPDX header. Everything
else is byte-identical to the extracted image source. `check_bounds.py` proves
the bound and the byte-identity of the rest of the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "Modified by QMesh (2026) for the DSV4.1 AFD-TP3 experiment."

REL_PATH = "vllm/v1/attention/backends/mla/indexer.py"

NOTICE = f"""# {MARKER}
# Native 1M memory: size the sparse indexer's prefill K workspace to the exact
# per-chunk bound (factor * max_model_len rows) instead of the 8.7x
# flashmla_sparse envelope that the served backend never allocates.
# Provenance and license: third_party/vllm/NOTICE
"""

OLD = '''def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    factor = vllm_config.attention_config.sparse_mla_prefill_workspace_factor
    assert factor is not None
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of factor * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * factor (40 at factor 5) to maximize this workspace
    # size while still fitting within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * (576 * 2 // 132) * factor
'''

NEW = '''def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    factor = vllm_config.attention_config.sparse_mla_prefill_workspace_factor
    assert factor is not None
    # One prefill chunk gathers at most one request's compressed context:
    # `_split_indexer_prefill_chunks` caps a multi-request chunk's
    # total_seq_lens at this value, and its single-request fallback uses that
    # request's own compressed length, which is <= max_model_len for every
    # compress_ratio. `factor * max_model_len` rows (factor >= 2) is therefore
    # an exact bound for every chunk this indexer can emit, in the same units
    # as the chunker and the K-gather workspace.
    #
    # The previous `max_model_len * (576 * 2 // 132) * factor` (8.7x larger)
    # filled the flashmla_sparse prefill workspace envelope. This recipe uses
    # FLASHINFER_MLA_SPARSE_DSV41, which allocates no such workspace, so those
    # rows only inflated the persistent K-gather buffer (1.06 GiB at
    # max_model_len=1M with MXFP4 indexer K) without allowing larger chunks.
    #
    # Modified by QMesh (2026) for the DSV4.1 AFD-TP3 experiment: exact
    # per-chunk bound instead of the unused flashmla_sparse envelope.
    # Provenance and license: third_party/vllm/NOTICE
    return max_model_len * factor
'''


def insert_notice(text: str) -> str:
    if MARKER in text:
        return text
    lines = text.splitlines(keepends=True)
    at = 0
    while at < len(lines) and lines[at].startswith("# SPDX-"):
        at += 1
    lines.insert(at, "\n" + NOTICE)
    return "".join(lines)


def patch(text: str) -> str:
    if text.count(OLD) != 1:
        raise SystemExit(
            "patch: expected exactly one original get_max_prefill_buffer_size "
            f"body, found {text.count(OLD)}"
        )
    return insert_notice(text.replace(OLD, NEW))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path,
                        help="image source tree root (contains vllm/...)")
    parser.add_argument("--out", required=True, type=Path,
                        help="overlay output tree root (contains vllm/...)")
    args = parser.parse_args()

    src = args.src / REL_PATH
    dst = args.out / REL_PATH
    if not src.is_file():
        raise SystemExit(f"patch: missing source {src}")
    patched = patch(src.read_text())
    compile(patched, str(dst), "exec")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(patched)
    print(f"patch: wrote {dst}")


if __name__ == "__main__":
    main()
