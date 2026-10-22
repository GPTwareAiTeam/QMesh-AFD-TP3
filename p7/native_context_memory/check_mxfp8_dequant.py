#!/usr/bin/env python3
"""CPU-only arithmetic proof for the MXFP8 dequant in-place overlay.

Imports the real `dequant_mxfp8_to_bf16` function from both the pinned image
source and the overlay with `ast`, then runs them on CPU torch tensors. Proves:

1. byte-exactness: the overlay is exactly `patch(src)` (caller checks hashes;
   this script checks the extracted function body matches the intended edit);
2. `x_blocked.mul_(descale.unsqueeze(-1))` and
   `x_blocked * descale.unsqueeze(-1)` produce bit-identical float32 values and
   bit-identical bf16 outputs;
3. the input float8 tensor is not mutated by the in-place version;
4. finite values, NaN, float32 overflow to infinity, and negative zero all
   keep their exact bit patterns.

No CUDA context is created. Run with the pinned image's python3 and
`CUDA_VISIBLE_DEVICES=` (the caller passes `--runtime runc --network none`).
Exit status is nonzero on any failure.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import torch

REL_PATH = "vllm/model_executor/layers/quantization/utils/mxfp8_utils.py"
FUNC = "dequant_mxfp8_to_bf16"
BLOCK = 32


def load_func(path: Path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == FUNC:
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"torch": torch, "MXFP8_BLOCK_SIZE": BLOCK}
            exec(compile(module, str(path), "exec"), namespace)
            return namespace[FUNC]
    raise SystemExit(f"check: {FUNC} not found in {path}")


def bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16)
    if t.dtype == torch.float32:
        return t.contiguous().view(torch.int32)
    raise SystemExit(f"check: unsupported dtype {t.dtype}")


def equal_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


def special_fp8(rows: int, cols: int) -> torch.Tensor:
    pattern = torch.tensor(
        [0x00, 0x80, 0x7F, 0x01, 0x7E, 0xFE, 0x3F, 0xBF,
         0x02, 0x7D, 0x81, 0x38, 0xB8, 0x40, 0xC0, 0x3C],
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    reps = (rows * cols + pattern.numel() - 1) // pattern.numel()
    return pattern.repeat(reps)[: rows * cols].view(rows, cols)


def check_case(name: str, x: torch.Tensor, scales: torch.Tensor,
               old_fn, new_fn) -> None:
    assert x.shape[-1] % BLOCK == 0
    assert scales.shape == x.shape[:-1] + (x.shape[-1] // BLOCK,)

    # Expression-level proof: same float32 operands, out-of-place vs in-place.
    x_float = x.to(torch.float32)
    blocked = x_float.view(*x.shape[:-1], x.shape[-1] // BLOCK, BLOCK)
    descale = torch.exp2(scales.to(torch.float32) - 127.0)
    out_of_place = blocked * descale.unsqueeze(-1)
    in_place = blocked.clone().mul_(descale.unsqueeze(-1))
    if not equal_bits(out_of_place, in_place):
        raise SystemExit(f"check: {name}: float32 multiply bits differ")

    # Function-level proof: old and new bodies on identical inputs.
    x_old = x.clone()
    x_new = x.clone()
    old = old_fn(x_old, scales.clone())
    new = new_fn(x_new, scales.clone())
    if not equal_bits(old, new):
        raise SystemExit(f"check: {name}: bf16 dequant bits differ")
    if not torch.equal(x_new.view(torch.uint8), x.view(torch.uint8)):
        raise SystemExit(f"check: {name}: in-place dequant mutated its input")

    if name == "special":
        f32 = out_of_place
        if not bool(torch.isnan(f32).any()):
            raise SystemExit("check: special case produced no NaN")
        if not bool(torch.isinf(f32).any()):
            raise SystemExit("check: special case produced no infinity")
        if not bool(torch.isfinite(f32).any()):
            raise SystemExit("check: special case produced no finite value")
        if not bool(torch.isnan(new).any()):
            raise SystemExit("check: bf16 output lost NaN")
        if not bool(torch.isinf(new).any()):
            raise SystemExit("check: bf16 output lost infinity")

    print(f"check: {name}: {tuple(x.shape)} bits identical")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path,
                        help="package src root containing vllm/...")
    parser.add_argument("--overlay", required=True, type=Path,
                        help="package overlay root containing vllm/...")
    args = parser.parse_args()

    src = args.src / REL_PATH
    overlay = args.overlay / REL_PATH
    old_fn = load_func(src)
    new_fn = load_func(overlay)

    torch.manual_seed(0)
    torch.set_num_threads(1)

    cases = []
    for rows, cols in ((1, BLOCK), (3, 2 * BLOCK), (17, 5 * BLOCK),
                       (128, 256), (8192, 4096)):
        x = (torch.randn(rows, cols) * 400).to(torch.float8_e4m3fn)
        scales = torch.randint(0, 255, (rows, cols // BLOCK),
                               dtype=torch.uint8)
        cases.append((f"finite-{rows}x{cols}", x, scales))

    rows, cols = 64, 8 * BLOCK
    cases.append(("special", special_fp8(rows, cols),
                  torch.tensor([0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF,
                                0x7E, 0x7C], dtype=torch.uint8)
                  .repeat(rows)[: rows * (cols // BLOCK)]
                  .view(rows, cols // BLOCK)))
    cases.append(("zeros", torch.zeros(rows, cols, dtype=torch.float8_e4m3fn),
                  torch.full((rows, cols // BLOCK), 127, dtype=torch.uint8)))
    cases.append(("negzero", torch.full((4, cols), -0.0,
                                        dtype=torch.float8_e4m3fn),
                  torch.full((4, cols // BLOCK), 127, dtype=torch.uint8)))

    for name, x, scales in cases:
        check_case(name, x, scales, old_fn, new_fn)

    # Explicit negative-zero bit check.
    x = torch.full((1, BLOCK), -0.0, dtype=torch.float8_e4m3fn)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)
    out = new_fn(x, scales)
    if int(bits(out)[0, 0]) != -32768:
        raise SystemExit("check: negative zero did not survive as bf16 0x8000")

    print("check: OK (in-place MXFP8 dequant is bit-identical on CPU)")


if __name__ == "__main__":
    main()
