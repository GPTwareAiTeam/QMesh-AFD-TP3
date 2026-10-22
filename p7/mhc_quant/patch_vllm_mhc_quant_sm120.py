#!/usr/bin/env python3
"""Feed the attention projection the MXFP8 bytes mHC already has (DeepSeek-V4.1-Flash, sm_120).

The sm_120 mHC overlap kernel (`mhc_post_norm`) is the only writer of the decoder
layer's attention input and the attention boundary's `fused_wqa_wkv` is its only
consumer. The projection's FlashInfer MXFP8 kernel quantizes that same bf16
`layer_input` again (`mxfp8_e4m3_quantize(x.view(-1, K), is_sf_swizzled_layout=True)`,
~9.3 us of the decode graph per boundary). `mhc_post_norm_quant` (this experiment's
kernel, tools/dsv41_sm120/mhc_overlap/) computes the payload and the F8_128x4 scales
in the CTA that already holds the norm result, bit-identical to that op.

This patch wires the two together without a second layout or arithmetic change:
  * nvidia/ops/mhc.py: `mhc_shifted_post_pre(..., prequant=True)` makes the
    attention boundary run `mhc_post_norm_quant` (cached q/sf buffers per
    (n, hidden, device) on the side stream) and carries the result on the returned
    bf16 `layer_input` as `_moet_qa = QuantizedActivation(...kMxfp8Dynamic)`.
    Without the fused entry point in the kernel module it stays on `mhc_post_norm`.
  * nvidia/model.py: attention boundary of `forward` only. The FFN boundary and
    `write_kv` keep the default (False): their consumers are not this projection.
  * attention.py: `_fused_wqa_wkv_gemm` consumes `_moet_qa` when the fused
    projection's kernel is one of the FlashInfer MXFP8 kernels that declare
    kMxfp8Dynamic input (the same restriction query_quant's can_fuse_query_quant
    makes for the F8_128x4 scale layout); every other case falls through to the
    served path unchanged.

Idempotent, anchor-based. Targets image vllm-moet:dsv41-213d5fa (vLLM 21d93d0d
plus the sm_120 mHC overlap patch). Usage:
    python3 patch_vllm_mhc_quant_sm120.py [--vllm-root DIR] [--out DIR] [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARKER = "# [vllm-moet] sm_120 mHC fused MXFP8 quant"

NOTICE_MARKER = "Modified by QMesh (2026) for the DSV4.1 AFD-TP3 experiment."
NOTICE_HEADERS = {
    "mhc.py": (
        "\n"
        "# Modified by QMesh (2026) for the DSV4.1 AFD-TP3 experiment.\n"
        "# mHC fused MXFP8 quant: adds a prequant boundary path that runs\n"
        "# mhc_post_norm_quant and carries the F8_128x4 activation as _moet_qa, with\n"
        "# per-(n, hidden, device) q/sf buffers; falls back to mhc_post_norm.\n"
        "# Provenance and license: third_party/vllm/NOTICE\n"
    ),
    "model.py": (
        "\n"
        "# Modified by QMesh (2026) for the DSV4.1 AFD-TP3 experiment.\n"
        "# mHC fused MXFP8 quant: enables prequant at the attention boundary only; the\n"
        "# FFN boundary and write_kv keep the default path.\n"
        "# Provenance and license: third_party/vllm/NOTICE\n"
    ),
    "attention.py": (
        "\n"
        "# Modified by QMesh (2026) for the DSV4.1 AFD-TP3 experiment.\n"
        "# mHC fused MXFP8 quant: _fused_wqa_wkv_gemm consumes _moet_qa when the\n"
        "# FlashInfer MXFP8 kernel declares the matching dynamic input; every other\n"
        "# kernel uses the served path.\n"
        "# Provenance and license: third_party/vllm/NOTICE\n"
    ),
}


def add_notice(src: str, name: str) -> str:
    if NOTICE_MARKER in src:
        return src
    lines = src.splitlines(keepends=True)
    at = 0
    while at < len(lines) and lines[at].startswith("# SPDX-"):
        at += 1
    lines.insert(at, NOTICE_HEADERS[name])
    return "".join(lines)

MHC = Path("models/deepseek_v41/nvidia/ops/mhc.py")
MODEL = Path("models/deepseek_v41/nvidia/model.py")
ATTENTION = Path("models/deepseek_v41/attention.py")

HELPERS = '''# [vllm-moet] sm_120 mHC fused MXFP8 quant: layer_input leaves mhc_post_norm already in
# FlashInfer's F8_128x4 activation ABI, carried on the bf16 tensor as `_moet_qa`, so the
# attention projection consumes it instead of quantizing the same bytes a second time.
def _moet_boundary_quant_buffers(stream, n: int, hidden: int, device: torch.device):
    cache = getattr(stream, "_moet_quant_buf", None)
    if cache is None:
        cache = {}
        stream._moet_quant_buf = cache
    key = (n, hidden, device.index)
    buf = cache.get(key)
    if buf is None:
        buf = (
            torch.empty((n, hidden), dtype=torch.uint8, device=device),
            torch.empty(
                ((n + 127) // 128) * 128 * (hidden // 32),
                dtype=torch.uint8,
                device=device,
            ),
        )
        cache[key] = buf
    return buf


def _moet_boundary_post_norm(
    kernel_module,
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    pre_mix,
    norm_weight,
    norm_eps,
    capture_aux,
    prequant,
    stream,
):
    """mhc_post_norm, plus the attention consumer's MXFP8 payload and scales when asked."""
    if prequant and hasattr(kernel_module, "mhc_post_norm_quant"):
        from vllm.model_executor.layers.fusion.quant_activation import (
            QuantizedActivation,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kMxfp8Dynamic,
        )

        n, _, hidden = residual.shape
        q, sf = _moet_boundary_quant_buffers(stream, n, hidden, residual.device)
        residual_out, layer_input, aux, q, sf = kernel_module.mhc_post_norm_quant(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            pre_mix,
            norm_weight,
            norm_eps,
            capture_aux=capture_aux,
            q_out=q,
            sf_out=sf,
            swizzled=True,
        )
        # The payload is consumed by this boundary's attention projection, on the
        # same stream, before the next boundary reuses the buffers.
        layer_input._moet_qa = QuantizedActivation(
            q.view(torch.float8_e4m3fn),
            sf,
            layer_input.dtype,
            layer_input.shape,
            kMxfp8Dynamic,
        )
        return residual_out, layer_input, aux
    return kernel_module.mhc_post_norm(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        pre_mix,
        norm_weight,
        norm_eps,
        capture_aux=capture_aux,
    )


'''

MHC_SIG_OLD = """    capture_aux: bool,
    stream: torch.cuda.Stream,
) -> tuple["""
MHC_SIG_NEW = """    capture_aux: bool,
    stream: torch.cuda.Stream,
    prequant: bool = False,
) -> tuple["""

MHC_CALL_OLD = """    residual_out, layer_input, aux = kernel_module.mhc_post_norm(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        pre_mix,
        norm_weight,
        norm_eps,
        capture_aux=capture_aux,
    )"""
MHC_CALL_NEW = """    residual_out, layer_input, aux = _moet_boundary_post_norm(
        kernel_module,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        pre_mix,
        norm_weight,
        norm_eps,
        capture_aux,
        prequant,
        stream,
    )"""

MHC_DISPATCH_SIG_OLD = """    capture_aux: bool = False,
    *,
    stream: torch.cuda.Stream | None = None,
    reduce_results: bool = False,
) -> tuple["""
MHC_DISPATCH_SIG_NEW = """    capture_aux: bool = False,
    *,
    stream: torch.cuda.Stream | None = None,
    reduce_results: bool = False,
    prequant: bool = False,
) -> tuple["""

MHC_DISPATCH_CALL_OLD = """                capture_aux,
                stream,
            )"""
MHC_DISPATCH_CALL_NEW = """                capture_aux,
                stream,
                prequant=prequant,
            )"""

MODEL_OLD = """                capture_aux=capture_previous_aux,
                stream=mhc_stream,
                reduce_results=self.fuse_mhc_all_reduce,
            )"""
MODEL_NEW = """                capture_aux=capture_previous_aux,
                stream=mhc_stream,
                reduce_results=self.fuse_mhc_all_reduce,
                prequant=True,
            )"""

ATTN_OLD = '''    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Override point: the ROCm layer preshuffles this weight in place, so
        # it cannot go through fused_wqa_wkv directly.
        num_tokens = hidden_states.shape[0]'''
ATTN_NEW = '''    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [vllm-moet] sm_120 mHC fused MXFP8 quant: the mHC boundary kernel already wrote
        # this activation's F8_128x4 payload/scales; only the FlashInfer MXFP8 kernels that
        # declare kMxfp8Dynamic take them (same restriction as can_fuse_query_quant).
        qa = getattr(hidden_states, "_moet_qa", None)
        if qa is not None:
            from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
                FlashInferCutedslMxfp8LinearKernel,
                FlashInferCutlassMxfp8LinearKernel,
            )

            kernel = getattr(
                getattr(self.fused_wqa_wkv, "quant_method", None), "kernel", None
            )
            if type(kernel) in (
                FlashInferCutlassMxfp8LinearKernel,
                FlashInferCutedslMxfp8LinearKernel,
            ) and kernel.input_quant_key() == qa.quant_key:
                qr_kv, _ = self.fused_wqa_wkv(qa)
                return qr_kv
        # Override point: the ROCm layer preshuffles this weight in place, so
        # it cannot go through fused_wqa_wkv directly.
        num_tokens = hidden_states.shape[0]'''


def _replace_once(src: str, old: str, new: str, what: str) -> str:
    count = src.count(old)
    if count != 1:
        raise SystemExit(f"{what}: anchor found {count} times (expected 1)")
    return src.replace(old, new, 1)


def patch_mhc(src: str) -> str:
    if MARKER in src:
        return src
    anchor = "def _moet_mhc_shifted_post_pre_overlap(\n"
    src = _replace_once(src, anchor, HELPERS + anchor, "mhc helpers")
    src = _replace_once(src, MHC_SIG_OLD, MHC_SIG_NEW, "mhc overlap signature")
    src = _replace_once(src, MHC_CALL_OLD, MHC_CALL_NEW, "mhc overlap call")
    src = _replace_once(src, MHC_DISPATCH_SIG_OLD, MHC_DISPATCH_SIG_NEW, "mhc dispatch signature")
    src = _replace_once(src, MHC_DISPATCH_CALL_OLD, MHC_DISPATCH_CALL_NEW, "mhc dispatch call")
    return src


def patch_model(src: str) -> str:
    if MARKER in src:
        return src
    return _replace_once(src, MODEL_OLD, MODEL_NEW, "model attention boundary")


def patch_attention(src: str) -> str:
    if MARKER in src:
        return src
    return _replace_once(src, ATTN_OLD, ATTN_NEW, "attention fused_wqa_wkv_gemm")


PATCHERS = {MHC: patch_mhc, MODEL: patch_model, ATTENTION: patch_attention}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", type=Path, default=DEFAULT_VLLM_ROOT)
    ap.add_argument("--out", type=Path, default=None,
                    help="write the patched files under this directory (same relative paths)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    for rel, patch in PATCHERS.items():
        path = args.vllm_root / rel
        src = add_notice(path.read_text(), rel.name)
        patched = patch(src)
        compile(patched, str(path), "exec")
        if patched == src:
            print(f"{path}: already patched")
            continue
        if args.check:
            print(f"{path}: patch applies cleanly (not written)")
            continue
        dst = args.out / rel if args.out is not None else path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(patched)
        print(f"{dst}: patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
