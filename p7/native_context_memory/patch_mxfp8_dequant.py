#!/usr/bin/env python3
"""Patch vLLM's MXFP8 dequantization to reuse its float32 weight copy.

`vllm/model_executor/layers/quantization/utils/mxfp8_utils.py::
dequant_mxfp8_to_bf16` materializes `x_float = x.to(torch.float32)` and then
`dequantized = x_blocked * descale.unsqueeze(-1)`, which allocates a second
float32 tensor as large as the weight. At the native-1M / 3 GiB KV / C=8 boot,
`wo_a.weight` is `[8192, 4096]` FP8, so each float32 temporary is 128 MiB. The
first copy succeeds and the second request fails with only 75.88 MiB free.

`x_float` is a fresh copy of a float8 tensor, so the element-wise broadcast
multiply can run in place. The only remaining allocation is the bf16 result
(64 MiB for `wo_a`), which fits. The float32 multiply is unchanged.

This patcher is deterministic: it rewrites one line and inserts the
Apache-2.0 section 4(b) modification notice after the SPDX header. Everything
else is byte-identical to the extracted image source. The CPU torch checker
proves bit-identical float32/bf16 results, including NaN, infinities and
negative zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "Modified by QMesh (2026) for the DSV4.1 AFD-TP3 experiment."

REL_PATH = "vllm/model_executor/layers/quantization/utils/mxfp8_utils.py"

NOTICE = f"""# {MARKER}
# Native 1M memory: dequantize MXFP8 weights with an in-place float32
# broadcast multiply, reusing the fresh `x.to(torch.float32)` copy instead of
# allocating a second weight-sized float32 temporary (128 MiB for wo_a at
# [8192, 4096]) that cannot fit in the remaining boot headroom.
# Provenance and license: third_party/vllm/NOTICE
"""

OLD = '''    dequantized = x_blocked * descale.unsqueeze(-1)
'''

NEW = '''    # Modified by QMesh (2026) for the DSV4.1 AFD-TP3 experiment: x_float is
    # a fresh float32 copy of x, so reuse it instead of allocating a second
    # weight-sized float32 temporary. The float32 multiply is unchanged.
    dequantized = x_blocked.mul_(descale.unsqueeze(-1))
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
            "patch: expected exactly one original dequant multiply, "
            f"found {text.count(OLD)}"
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
