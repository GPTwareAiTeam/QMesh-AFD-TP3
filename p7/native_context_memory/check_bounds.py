#!/usr/bin/env python3
"""CPU-only proof for the sparse-indexer prefill bound overlay.

No torch, no GPU, no vLLM import: the two functions that matter are extracted
from the *overlay* source with `ast`, annotations are stripped, and they are
executed against fake metadata. Proves:

1. byte-exactness: the overlay is exactly `patch(src)`, no other drift;
2. the chunker's N invariant holds for every scenario at the new bound:
   every emitted chunk's total_seq_lens <= workspace_size, including the
   single-request fallback (whose compressed length can be up to
   ceil(max_model_len / compress_ratio));
3. the K-gather workspace row count (same value) is >= every chunk's N, so
   `k_quant_full[: chunk.max_local_total_seq_lens]` never over-runs;
4. the old and new workspace/block-table footprints at 1M context.

Exit status is nonzero on any failure.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from patch_indexer_prefill_bound import (  # noqa: E402
    NEW,
    OLD,
    REL_PATH,
    patch,
)

SRC = HERE / "src" / REL_PATH
OVERLAY = HERE / "overlay" / REL_PATH
MXFP8_REL = "vllm/model_executor/layers/quantization/utils/mxfp8_utils.py"

HEAD_DIM = 128
MXFP4_BLOCK = 32
# `_gather_workspace_shapes` with use_fp4_cache=True:
#   values (T, head_dim // 2) uint8 + scales (T, head_dim // 32) uint8
FP4_BYTES_PER_ROW = HEAD_DIM // 2 + HEAD_DIM // MXFP4_BLOCK
FP8_BYTES_PER_ROW = HEAD_DIM + 4
GIB = 1024 ** 3
MIB = 1024 ** 2


class Config:
    class model_config:
        max_model_len = 0

    class attention_config:
        sparse_mla_prefill_workspace_factor = 2


def make_config(max_model_len: int, factor: int) -> Config:
    cfg = Config()
    cfg.model_config.max_model_len = max_model_len
    cfg.attention_config.sparse_mla_prefill_workspace_factor = factor
    return cfg


class Item:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class Col(list):
    """Minimal tensor stand-in: len(), indexing, [a:b], .item()."""

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(key, slice):
            return Col(value)
        return Item(value)


def strip_annotations(node: ast.AST) -> None:
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            child.returns = None
            for arg in (
                *child.args.args,
                *child.args.posonlyargs,
                *child.args.kwonlyargs,
            ):
                arg.annotation = None
            if child.args.vararg is not None:
                child.args.vararg.annotation = None
            if child.args.kwarg is not None:
                child.args.kwarg.annotation = None
            child.decorator_list = []


def load_functions(text: str, names: set[str]) -> dict[str, object]:
    tree = ast.parse(text)
    found: dict[str, ast.FunctionDef] = {}
    # `_split_indexer_prefill_chunks` is a staticmethod nested in the metadata
    # builder class, so search the whole tree, not just module top level.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in names
            and node.name not in found
        ):
            node = copy.deepcopy(node)
            strip_annotations(node)
            found[node.name] = node
    missing = names - set(found)
    if missing:
        raise SystemExit(f"check_bounds: missing functions {sorted(missing)}")
    module = ast.Module(body=list(found.values()), type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, "<overlay-extract>", "exec"), namespace)
    return namespace


def check_identity() -> str:
    src_text = SRC.read_text()
    overlay_text = OVERLAY.read_text()
    expected = patch(src_text)

    assert overlay_text == expected, (
        "overlay is not exactly patch(src): unexpected drift"
    )
    assert src_text.count(OLD) == 1, "src no longer holds the original function"
    assert OLD not in overlay_text, "old formula still present in overlay"
    assert overlay_text.count(NEW) == 1, "new function body count != 1"
    assert "return max_model_len * (576 * 2 // 132) * factor" not in overlay_text
    compile(overlay_text, str(OVERLAY), "exec")

    overlay_files = sorted(
        str(p.relative_to(HERE / "overlay"))
        for p in (HERE / "overlay").rglob("*")
        if p.is_file()
    )
    assert overlay_files == sorted([REL_PATH, MXFP8_REL]), (
        f"unexpected overlay files: {overlay_files}"
    )
    return overlay_text


def scenarios(max_model_len: int):
    """(label, seq_lens, query_lens) prefill batches; lens in tokens."""
    mm = max_model_len
    half = mm // 2
    return [
        ("8x full context", [mm] * 8, [mm] * 8),
        ("8x half context", [half] * 8, [half] * 8),
        ("8x 2048", [2048] * 8, [2048] * 8),
        (
            "C8 harness mix",
            [2048, 1512, 2048, 2048, 1512, 2048, 2048, 1512],
            [2048, 1512, 2048, 2048, 1512, 2048, 2048, 1512],
        ),
        ("8x single token", [1] * 8, [1] * 8),
        ("single full context", [mm], [mm]),
    ]


def run_chunks(split, compressed, query_lens, workspace_size, logits_mb):
    """Invoke the image chunker and assert every emitted chunk's N bound."""
    chunks = split(
        Col(compressed), Col(query_lens), workspace_size, logits_mb * MIB
    )
    assert chunks, ("no chunks", compressed, query_lens)
    # One request-level chunk can be emitted as several query sub-chunks with
    # the same req_slice; group them so M/N are counted once per chunk.
    groups: dict[tuple[int, int], list[slice]] = {}
    for req_slice, query_slice in chunks:
        groups.setdefault((req_slice.start, req_slice.stop), []).append(query_slice)
    m_total = 0
    n_total = 0
    for (start, stop), query_slices in groups.items():
        chunk_n = sum(compressed[start:stop])
        chunk_m = sum(query_lens[start:stop])
        # N constraint enforced by the chunker for multi-request chunks.
        assert chunk_n <= workspace_size, (
            "N invariant",
            compressed,
            query_lens,
            chunk_n,
            workspace_size,
            logits_mb,
        )
        # Query sub-chunks must tile this chunk's query rows exactly: the
        # metadata builder treats (req_slice, query_slice) as covering those
        # rows once, and the op's K slice is bounded by chunk_n.
        expect = 0
        for query_slice in query_slices:
            assert query_slice.start == expect, (query_slices, chunk_m)
            assert query_slice.stop > query_slice.start
            expect = query_slice.stop
        assert expect == chunk_m, (expect, chunk_m, query_slices)
        m_total += chunk_m
        n_total += chunk_n
    assert m_total == sum(query_lens), (m_total, query_lens)
    assert n_total == sum(compressed), (n_total, compressed)
    return len(chunks)


def prove() -> None:
    overlay_text = check_identity()
    fn_ns = load_functions(
        overlay_text, {"get_max_prefill_buffer_size", "_split_indexer_prefill_chunks"}
    )
    get_bound = fn_ns["get_max_prefill_buffer_size"]
    split = fn_ns["_split_indexer_prefill_chunks"]

    print(f"overlay is byte-exact patch(src); {len(overlay_text)} bytes, compiles")
    print(f"overlay sha256: {hashlib.sha256(overlay_text.encode()).hexdigest()}")
    print(
        f"K-gather bytes/row: fp4={FP4_BYTES_PER_ROW} "
        f"(head_dim={HEAD_DIM}, block={MXFP4_BLOCK}); fp8={FP8_BYTES_PER_ROW}"
    )

    checked_chunks = 0
    for mm in (8192, 131072, 1048576):
        for factor in (2, 5):
            bound = get_bound(make_config(mm, factor))
            assert bound == mm * factor, (mm, factor, bound)
            for ratio in (1, 2, 8):
                workspace_size = bound // ratio
                # The builder floors compressed lengths, and the chunker's
                # single-request fallback (which bypasses the workspace check)
                # can only see a length this large. It must fit the workspace.
                max_single = mm // ratio
                assert max_single <= workspace_size, (
                    mm, ratio, max_single, workspace_size
                )
                # Boundary: a request whose compressed length is exactly the
                # workspace size must be accepted without sub-chunk overflow.
                checked_chunks += run_chunks(
                    split, [workspace_size], [1], workspace_size, 512
                )

                for label, seq_lens, query_lens in scenarios(mm):
                    # Real builder: floor division for ratio > 1.
                    compressed = [s // ratio if ratio > 1 else s for s in seq_lens]
                    logits_budgets = (512,)
                    # 64 MB is only a documented tuning knob; skip its huge
                    # sub-chunk expansions at 1M full-context batches.
                    if mm <= 131072:
                        logits_budgets = (512, 64)
                    for logits_mb in logits_budgets:
                        checked_chunks += run_chunks(
                            split,
                            compressed,
                            query_lens,
                            workspace_size,
                            logits_mb,
                        )

    # Footprints at the 1M target.
    mm = 1048576
    new_rows = mm * 2
    old_rows = mm * (576 * 2 // 132) * 2
    print("\n1M context, factor=2, MXFP4 indexer K (68 B/row):")
    for ratio in (1, 2):
        old_bytes = old_rows // ratio * FP4_BYTES_PER_ROW
        new_bytes = new_rows // ratio * FP4_BYTES_PER_ROW
        print(
            f"  ratio={ratio}: rows {old_rows // ratio:,} -> {new_rows // ratio:,} ; "
            f"{old_bytes / MIB:8.1f} MiB -> {new_bytes / MIB:8.1f} MiB "
            f"(-{(old_bytes - new_bytes) / GIB:.3f} GiB)"
        )
    peak_old = old_rows * FP4_BYTES_PER_ROW
    peak_new = new_rows * FP4_BYTES_PER_ROW
    print(
        f"  peak over layers: {peak_old / GIB:.4f} GiB -> {peak_new / GIB:.4f} GiB "
        f"(-{(peak_old - peak_new) / GIB:.4f} GiB)"
    )

    for rows in (4096, 2048):
        width = -(-mm // 128)  # cdiv(max_model_len, block_size=128)
        assert width == 8192, width
        total = 2 * rows * width * 4  # two metadata builders, int32 rows
        print(
            f"  block tables @ rows={rows}, width={width}: "
            f"{total / MIB:.0f} MiB (2 tables)"
        )

    print(
        f"\nchecked {checked_chunks} emitted chunks across mm "
        f"{{8192,131072,1048576}}, factor {{2,5}}, ratio {{1,2,8}}; "
        "all N <= workspace and <= factor*max_model_len//ratio"
    )


if __name__ == "__main__":
    prove()
