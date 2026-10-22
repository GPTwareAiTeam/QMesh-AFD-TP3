# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Modified by QMesh (2026) for the DSV4.1 AFD-TP3 experiment.
# mHC fused MXFP8 quant: adds a prequant boundary path that runs
# mhc_post_norm_quant and carries the F8_128x4 activation as _moet_qa, with
# per-(n, hidden, device) q/sf buffers; falls back to mhc_post_norm.
# Provenance and license: third_party/vllm/NOTICE
"""Dispatch DSV4.1 mHC operations and overlap coefficient generation."""

from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_fused_post_pre_delayed_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

from .mega_mhc import (
    can_use_mega_mhc,
    mhc_shifted_post_pre_deep_gemm,
)

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    from .cute_dsl import AllReduceMHC

logger = init_logger(__name__)

# GB200 TP4/FlashInfer improves through 16 tokens; larger screens tie or regress.
MHC_OVERLAP_MAX_TOKENS = 16


# [vllm-moet] sm_120 mHC overlap: the projection + Sinkhorn on the side stream behind the sublayer, the
# post-mix + collapse + RMSNorm in one kernel on the model stream (tools/dsv41_sm120/mhc_overlap).
from vllm.logger import init_logger as _moet_init_logger

_moet_logger = _moet_init_logger(__name__)
_MOET_MHC = None  # None = not resolved yet, False = off / unavailable, else the module


def _moet_mhc_post_norm():
    """The sm_120 critical-path kernel module (mhc_post_norm_sm120) or None."""
    global _MOET_MHC
    if _MOET_MHC is None:
        import os
        import sys

        _MOET_MHC = False
        if os.environ.get("VLLM_MOET_MHC_OVERLAP", "1") == "1" and (
            current_platform.is_cuda() and current_platform.is_device_capability_family(120)
        ):
            kernel_dir = os.environ.get("VLLM_MOET_MHC_OVERLAP_DIR", "/opt/vllm-moet/dsv41_sm120/mhc_overlap")
            if kernel_dir not in sys.path:
                sys.path.insert(0, kernel_dir)
            try:
                import mhc_post_norm_sm120 as _m

                _m._ext()  # build / load the extension now, not inside the first forward
                _MOET_MHC = _m
                _moet_logger.info_once(
                    "vllm-moet sm_120 mHC overlap: post-mix + collapse + RMSNorm in one kernel on "
                    "the model stream, projection + Sinkhorn on a side stream "
                    "(VLLM_MOET_MHC_OVERLAP=0 turns it off)"
                )
            except Exception as exc:  # noqa: BLE001
                _moet_logger.warning(
                    "vllm-moet sm_120 mHC overlap unavailable, using the TileLang pair: %r", exc
                )
    return _MOET_MHC or None


def _moet_mhc_overlap_supported(vllm_config: VllmConfig) -> bool:
    config = vllm_config.model_config.hf_config
    return (
        config.hc_mult == 4
        and config.hidden_size % 1024 == 0
        and 1024 <= config.hidden_size <= 5120
        and not vllm_config.parallel_config.use_ubatching
        and is_deep_gemm_supported()
        and _moet_mhc_post_norm() is not None
    )


def _moet_proj_kind() -> str:
    """Which projection runs on the side stream for <= 32 tokens (VLLM_MOET_MHC_PROJ):
    "ours"  - the served kernel's partials from one CTA per token x split, 64 registers per thread
              (default: bit-identical to the served step and the smallest footprint next to the sublayer),
    "fused" - the served TileLang kernel itself ([T, 12, 8] x 128 CTAs; bit-identical, but it takes
              SMs from the sublayer it runs behind),
    "tf32"  - DeepGEMM's TF32 prenorm GEMM on the bf16 streams (the served path above 32 tokens;
              not bit-identical below 33)."""
    import os

    kind = os.environ.get("VLLM_MOET_MHC_PROJ", "ours")
    return kind if kind in ("fused", "ours", "tf32") else "ours"


def _moet_overlap_max_tokens() -> int:
    """The token limit the decoder layer applies to the side stream: every captured decode batch
    on sm_120 (the kernel takes any count; the projection path switches at 32 like the served one).
    Only cheap checks here - this runs at import; the extension loads in supports_mhc_overlap."""
    import os

    try:
        if os.environ.get("VLLM_MOET_MHC_OVERLAP", "1") == "1" and (
            current_platform.is_cuda() and current_platform.is_device_capability_family(120)
        ):
            return int(os.environ.get("VLLM_MOET_MHC_OVERLAP_MAX_TOKENS", "64"))
    except Exception:  # noqa: BLE001
        pass
    return 16


# [vllm-moet] sm_120 mHC fused MXFP8 quant: layer_input leaves mhc_post_norm already in
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


def _moet_mhc_shifted_post_pre_overlap(
    kernel_module,
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    capture_aux: bool,
    stream: torch.cuda.Stream,
    prequant: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Shifted post/pre on sm_120: the critical path in one kernel, the coefficients on `stream`.

    Only the streams, the layer input and the aux are ready on the caller stream; join `stream`
    before reading post, comb or the next pre-mix (the decoder layer does, after the sublayer).

    The tensors the side stream reads are kept alive on the stream object until the next boundary
    instead of `record_stream`: under graph capture the allocator cannot free a recorded block
    before the capture ends, so every boundary's inputs would stay allocated for the whole graph
    (+0.15 GiB of graph memory per rank, taken from the KV cache); a Python reference released
    at the next boundary - by which point the decoder layer has joined the stream - costs nothing.
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_FUSED_TILELANG_KERNEL,
        mhc_fused_post_pre_split_config,
    )
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    n, hc, hidden = residual.shape
    mix_size = hc * (hc + 2)
    input_size = hc * hidden
    post = torch.empty((n, hc), device=residual.device, dtype=torch.float32)
    comb = torch.empty((n, hc * hc), device=residual.device, dtype=torch.float32)
    next_pre = torch.empty_like(post)
    main = torch.cuda.current_stream()
    # the previous boundary's side-stream inputs: joined by the caller, safe to release now
    stream._moet_pending = None
    fused_config = mhc_fused_post_pre_split_config(n, hidden, hc)
    proj = _moet_proj_kind()
    if proj == "tf32":
        fused_config = None
    # The critical path first, the fork after it. In the captured graph the kernel is then the
    # only child of the all-reduce that precedes it (the caller joined the side stream before the
    # all-reduce, so that node carries the join) and, launched with PDL, starts 0.2 us after it;
    # as a sibling of the side branch or with the join on it, 2.2 us (8 us in a micro-benchmark of
    # the captured pattern). The projection needs only this boundary's inputs, so starting it after
    # the kernel costs nothing.
    residual_out, layer_input, aux = _moet_boundary_post_norm(
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
    )
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        if fused_config is not None:
            # The projection of the fp32 post-mix recomputed from the same inputs (what the served
            # fused kernel does): ours (one CTA per split x 4 tokens, the same bits) or the served
            # TileLang kernel ([T, 12, 8] x 128 CTAs).
            if proj == "ours" and kernel_module.mhc_proj_applicable(x, residual):
                mixes, sqrsum = kernel_module.mhc_proj(x, residual, post_layer_mix, comb_res_mix, fn)
            else:
                mixes, sqrsum, _ = _MHC_FUSED_TILELANG_KERNEL(
                    comb_res_mix,
                    residual,
                    post_layer_mix.view(n, hc),
                    x,
                    fn.view(mix_size, hc, hidden),
                    hc,
                    hidden,
                    mix_size,
                )
        else:
            # Above the fused kernel's token range the served path projects the bf16 streams with
            # DeepGEMM's TF32 prenorm GEMM; here it reads the streams the kernel just wrote.
            n_splits = compute_mhc_pre_num_splits(input_size, n)
            mixes = torch.empty(
                (n_splits, n, mix_size), device=residual.device, dtype=torch.float32
            )
            sqrsum = torch.empty((n_splits, n), device=residual.device, dtype=torch.float32)
            tf32_hc_prenorm_gemm(residual_out.view(n, input_size), fn, mixes, sqrsum, n_splits)
        MHC_PRE_NORM_KERNEL(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            residual_out,
            post,
            comb,
            layer_input,
            norm_weight,
            pre_mix,
            next_pre,
            layer_input,  # unused aux slot in the stats split mode
            hidden_size=hidden,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            norm_eps=norm_eps,
            hc_mult=hc,
            use_pre_mix_in=True,
            save_pre_mix=True,
            rms_numel=input_size,
            split_mode="stats",
        )
    # everything the side stream reads or writes, alive until the next boundary (see the docstring)
    stream._moet_pending = (x, residual, post_layer_mix, comb_res_mix, residual_out, mixes, sqrsum, pre_mix, post, comb, next_pre)
    return residual_out, post.unsqueeze(-1), comb.view(n, hc, hc), layer_input, next_pre, aux


MHC_OVERLAP_MAX_TOKENS = _moet_overlap_max_tokens()

# The fused all-reduce + mHC kernel of the overlap path, built once by
# init_mhc_all_reduce.
_all_reduce_mhc: "AllReduceMHC | None" = None


def supports_mhc_overlap(vllm_config: VllmConfig) -> bool:
    """Check kernel requirements and safety of sharing the coefficient stream."""
    config = vllm_config.model_config.hf_config
    # DeepGEMM's prenorm kernel requires K % 64 == 0, N % 8 == 0, and N <= 32.
    mix_size = config.hc_mult * (config.hc_mult + 2)
    # [vllm-moet] sm_120 mHC overlap: the side stream on sm_120 too
    if _moet_mhc_overlap_supported(vllm_config):
        return True
    return (
        current_platform.is_device_capability_family(100)
        and is_deep_gemm_supported()
        and config.hidden_size % 64 == 0
        and 0 < mix_size <= 32
        and mix_size % 8 == 0
        and not vllm_config.parallel_config.use_ubatching
    )


def supports_mhc_all_reduce(vllm_config: VllmConfig) -> bool:
    parallel = vllm_config.parallel_config
    config = vllm_config.model_config.hf_config
    if (
        parallel.tensor_parallel_size != 4
        or parallel.enable_expert_parallel
        or config.hidden_size != 5120
        or config.hc_mult != 4
    ):
        return False
    # [vllm-moet] sm_120 mHC overlap: the sublayer's all-reduce moves into mhc_shifted_post_pre, in front of
    # the boundary kernel (a plain all-reduce, not the CuTe DSL kernel: init_mhc_all_reduce builds
    # nothing here; VLLM_MOET_MHC_FUSE_ALLREDUCE=0 keeps it in the linear layers)
    if _moet_mhc_overlap_supported(vllm_config):
        import os

        return os.environ.get("VLLM_MOET_MHC_FUSE_ALLREDUCE", "1") == "1"
    # The custom all-reduce's MNNVL buffers succeed exactly when NVLink
    # multicast does, which the fused kernel's own mailbox needs too.
    comm = cast("CudaCommunicator", get_tp_group().device_communicator).ca_comm
    return comm is not None and bool(comm.mnnvl_lamport_ag_multicast_ptr)


def init_mhc_all_reduce(vllm_config: VllmConfig) -> None:
    """Build the fused all-reduce + mHC kernel the overlap path reduces with.

    Collective over the TP group, so every rank must call it, and only when
    ``supports_mhc_all_reduce`` holds.
    """
    global _all_reduce_mhc
    # [vllm-moet] sm_120 mHC overlap: the fused path's all-reduce is a plain one in front of the boundary kernel
    if _moet_mhc_overlap_supported(vllm_config):
        _moet_logger.info_once(
            "vllm-moet sm_120 mHC overlap: the sublayer all-reduce runs right before the boundary "
            "kernel (VLLM_MOET_MHC_FUSE_ALLREDUCE=0 keeps it in the linear layers)"
        )
        return
    from .cute_dsl import AllReduceMHC

    config = vllm_config.model_config.hf_config
    _all_reduce_mhc = AllReduceMHC(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        max_num_tokens=MHC_OVERLAP_MAX_TOKENS,
        top_k=config.num_experts_per_tok,
        device=current_platform.current_device(),
    )
    logger.info_once(
        "DSV4.1 mHC: CuTe DSL Lamport all-reduce fused with mHC for up to %d tokens.",
        MHC_OVERLAP_MAX_TOKENS,
    )


def mhc_pre_delayed_overlap(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    *,
    stream: torch.cuda.Stream,
    layer_input: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the input on the caller stream and coefficients on another stream.

    Returns post mix, residual mix, normalized input, and next pre-mix. Only the
    input is ready on the caller stream; join stream before using coefficients.
    The caller must retain the weights and coefficient outputs until the join.
    """
    from vllm.model_executor.kernels.mhc.warmup import (
        MHC_PRE_NORM_KERNEL,
        compute_mhc_pre_num_splits,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

    n, hc, hidden = residual.shape
    assert residual.is_contiguous() and residual.dtype == torch.bfloat16
    assert norm_weight is not None
    if x is None:
        x = residual.view(n, hc * hidden)
    assert x.is_contiguous()
    post = torch.empty((n, hc), device=residual.device, dtype=torch.float32)
    comb = torch.empty((n, hc * hc), device=residual.device, dtype=torch.float32)
    next_pre = torch.empty_like(post)
    compute_input = layer_input is None
    if layer_input is None:
        layer_input = torch.empty(
            (n, hidden), device=residual.device, dtype=torch.bfloat16
        )
    outputs = post.unsqueeze(-1), comb.view(n, hc, hc), layer_input, next_pre
    if n == 0:
        return outputs
    splits = compute_mhc_pre_num_splits(x.shape[1], n)
    mix = torch.empty(
        (splits, n, hc * (hc + 2)), device=residual.device, dtype=torch.float32
    )
    sqr = torch.empty((splits, n), device=residual.device, dtype=torch.float32)
    epilogue = partial(
        MHC_PRE_NORM_KERNEL,
        mix,
        sqr,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        norm_weight,
        pre_mix if pre_mix is not None else post,
        next_pre,
        layer_input,  # Unused aux output in split modes.
        hidden_size=hidden,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=norm_eps,
        hc_mult=hc,
        use_pre_mix_in=pre_mix is not None,
        save_pre_mix=True,
        rms_numel=x.shape[1],
    )
    main = torch.cuda.current_stream()
    # Prioritize input readiness for small batches before releasing statistics.
    if compute_input and n <= 8:
        epilogue(split_mode="input")
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        tf32_hc_prenorm_gemm(x, fn, mix, sqr, splits)
        epilogue(split_mode="stats")
    for tensor in (x, mix, sqr):
        tensor.record_stream(stream)
    if compute_input and n > 8:
        epilogue(split_mode="input")
    return outputs


def mhc_shifted_post_pre(
    x: torch.Tensor | MoEOutput,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    capture_aux: bool = False,
    *,
    stream: torch.cuda.Stream | None = None,
    reduce_results: bool = False,
    prequant: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Dispatch shifted post/pre to overlap, Mega-mHC, or fused TileLang.

    A MoE output left unfinalized is finalized inside the fused all-reduce.
    When stream is supplied, join it before consuming the returned coefficients.
    """
    # [vllm-moet] sm_120 mHC overlap: one kernel on the critical path, the coefficients behind the sublayer.
    # With fuse_mhc_all_reduce the sublayer's all-reduce happens here, right before that kernel:
    # the join of the side stream then lands on the all-reduce node, and the kernel has one
    # parent (see _moet_mhc_shifted_post_pre_overlap).
    if stream is not None and (_moet_kernels := _moet_mhc_post_norm()) is not None:
        # the deferred MoE finalize needs TRTLLM-Gen experts (SM100): sm_120 MoEs finalize
        assert isinstance(x, torch.Tensor), "sm_120 mHC overlap: unexpected unfinalized MoE output"
        if not _moet_kernels.mhc_post_norm_applicable(x, residual, pre_mix, norm_weight):
            stream = None  # not the kernel's shape: the served path, coefficients on this stream
        else:
            assert pre_mix is not None and norm_weight is not None
            if reduce_results:
                x = get_tp_group().all_reduce(x)
            return _moet_mhc_shifted_post_pre_overlap(
                _moet_kernels,
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                pre_mix,
                norm_weight,
                norm_eps,
                capture_aux,
                stream,
                prequant=prequant,
            )
    layer_input = None
    if isinstance(x, MoEOutput):
        assert _all_reduce_mhc is not None, "init_mhc_all_reduce was not called"
        assert pre_mix is not None and norm_weight is not None
        routed = x.routed
        assert isinstance(routed, UnfinalizedMoEOutput)
        assert x.shared_output is not None
        residual, layer_input = _all_reduce_mhc.finalize(
            routed.gemm2_permuted,
            routed.expert_weights,
            routed.expanded_idx_to_permuted_idx,
            x.shared_output,
            residual,
            post_layer_mix,
            comb_res_mix,
            pre_mix,
            norm_weight,
            norm_eps,
        )
        if stream is None:
            # Outside FULL graphs, compute the next mixes inline.
            stream = torch.cuda.current_stream()
    elif reduce_results:
        if stream is not None and 0 < x.shape[0] <= MHC_OVERLAP_MAX_TOKENS:
            assert _all_reduce_mhc is not None, "init_mhc_all_reduce was not called"
            assert pre_mix is not None and norm_weight is not None
            residual, layer_input = _all_reduce_mhc(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                pre_mix,
                norm_weight,
                norm_eps,
            )
        else:
            x = get_tp_group().all_reduce(x)
    if stream is not None:
        if layer_input is None:
            assert isinstance(x, torch.Tensor)
            residual = mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix)
        aux = (
            residual.mean(dim=1)
            if capture_aux
            else residual.new_empty(0, residual.shape[-1])
        )
        pre_outputs = mhc_pre_delayed_overlap(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            stream=stream,
            layer_input=layer_input,
        )
        return residual, *pre_outputs, aux

    assert isinstance(x, torch.Tensor)
    if can_use_mega_mhc(x, residual, pre_mix, norm_weight, capture_aux):
        assert pre_mix is not None and norm_weight is not None
        outputs = mhc_shifted_post_pre_deep_gemm(
            x,
            residual,
            pre_mix,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_post_mult_value,
            hc_sinkhorn_eps,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )
        return *outputs, x.new_empty(0, x.shape[1])

    return mhc_fused_post_pre_delayed_tilelang(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix=pre_mix,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        capture_aux=capture_aux,
    )
