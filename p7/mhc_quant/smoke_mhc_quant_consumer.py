#!/usr/bin/env python3
"""Consumer-chain check of the sm_120 mHC fused MXFP8 quant on a served image.

Drives the patched decoder-layer boundary exactly as `DeepseekV4DecoderLayer.forward`
does -- `mhc_shifted_post_pre(..., prequant=True, stream=side)` followed by the
patched `DeepseekV4Attention._fused_wqa_wkv_gemm` -- against the served pair
(`prequant` off, stock projection quantizing `layer_input` itself), for the real
DeepSeek-V4.1-Flash shapes (HC=4, H=5120, fused_wqa_wkv N=1792 = q_lora_rank 1280 +
head_dim 512):

  1. every boundary output (residual, post/pre mixes, layer_input, aux) bit-identical
     with and without `prequant`;
  2. the carried MXFP8 payload and F8_128x4 scales bit-identical to the consumer's
     stock op `mxfp8_e4m3_quantize(layer_input.view(-1, H), is_sf_swizzled_layout=True)`
     and to a fresh allocation (the cached buffers hold no stale bytes);
  3. the real `FlashInferCutlassMxfp8LinearKernel.apply_weights` GEMV/CUTLASS output
     bit-identical from the carried QuantizedActivation and from the stock op, and
     the projection consumes it with zero `mxfp8_e4m3_quantize` calls (stock: one);
  4. the patched dispatch consumes the payload only for FlashInfer MXFP8 kernels that
     declare kMxfp8Dynamic, and falls through to the stock path (one quantization,
     no assertion) for anything else;
  5. the whole producer+consumer chain inside a CUDA graph (capture, two replays)
     bit-identical to eager, T=1 and T=6, with and without the draft aux, plus
     several boundaries sharing the cached q/sf buffers in one graph;
  6. non-finite rows (NaN, +/-Inf, one NaN element, 1e30, zeros): payload and
     per-block SF byte-identical to the *served* op mxfp8_e4m3_quantize — the
     ship gate is the consumer op, not the torch reference cast — and the
     projection output bit-identical, eager and inside a captured graph.

Run on one sm_120 GPU with the bind mounts in run_smoke.sh (patched
mhc.py/model.py/attention.py on site-packages, this kernel dir on
/opt/vllm-moet/dsv41_sm120/mhc_overlap):  python3 smoke_mhc_quant_consumer.py
"""

from __future__ import annotations

import os
import sys
import types

import torch

H, HC, MIX, N = 5120, 4, 24, 1792  # N: q_lora_rank 1280 + head_dim 512
RMS_EPS, HC_EPS, POST_MULT, SINKHORN, NORM_EPS = 1e-20, 1e-6, 2.0, 20, 1e-20
MARKER = "# [vllm-moet] sm_120 mHC fused MXFP8 quant"

FAILS: list[str] = []


def check(ok: bool, what: str) -> bool:
    print(f"[{'ok ' if ok else 'BAD'}] {what}")
    if not ok:
        FAILS.append(what)
    return ok


def sf_real_offsets(T: int, hidden: int) -> torch.Tensor:
    """Byte offsets of the 128x4 swizzled SF layout that belong to rows < T.

    The layout stores a 128-row tile as (col&3) + (col>>2)*512 + (row&31)*16 +
    ((row&127)>>5)*4 + (row>>7)*128*(hidden/32). Rows in [T, 128) of the tile are
    padding: the fused kernel writes only real rows, the stock op also writes the
    padding, so exactness is asserted on the real rows (the projection reads only
    those; proven separately by the bit-identical GEMV/CUTLASS outputs).
    """
    pc = hidden // 32
    rows = torch.arange(T, device="cuda")[:, None]
    cols = torch.arange(pc, device="cuda")[None, :]
    return ((cols & 3) + (cols >> 2) * 512 + (rows & 31) * 16
            + ((rows & 127) >> 5) * 4 + (rows >> 7) * (128 * pc)).reshape(-1)

def inputs(T, dev):
    x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
    residual = (torch.randn(T, HC, H, device=dev) * 0.8).to(torch.bfloat16)
    post_layer_mix = (torch.rand(T, HC, 1, device=dev) * 2).float()
    comb_res_mix = torch.softmax(torch.randn(T, HC, HC, device=dev), dim=-1).float().contiguous()
    fn = (torch.randn(MIX, HC * H, device=dev) * 0.01).float()
    scale = torch.tensor([0.02, 0.02, 0.05], device=dev)
    base = (torch.randn(MIX, device=dev) * 0.5).float()
    norm_w = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
    pre_mix = (torch.rand(T, HC, device=dev) * 0.9 + 0.05).float()
    return x, residual, post_layer_mix, comb_res_mix, fn, scale, base, norm_w, pre_mix


# ---------------------------------------------------------------- real consumer


def make_layer_config():
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )

    return Mxfp8LinearLayerConfig(weight_shape=(N, H))


class _QuantMethod:
    def __init__(self, kernel, layer):
        self.kernel = kernel
        self.layer = layer

    def apply(self, layer, x, bias=None):
        return self.kernel.apply_weights(self.layer, x, bias)


class FakeFusedLinear:
    """The shape `_fused_wqa_wkv_gemm` uses: `.quant_method.apply` -> MergedColumnParallelLinear."""

    def __init__(self, kernel, layer):
        self.quant_method = _QuantMethod(kernel, layer)

    def __call__(self, x):
        return self.quant_method.apply(self.quant_method.layer, x, None), None


def build_kernel(dev, quantize):
    import vllm.model_executor.kernels.linear.mxfp8.flashinfer as F

    kernel = F.FlashInferCutlassMxfp8LinearKernel(make_layer_config())
    w = (torch.randn(N, H, device=dev) * 0.02).to(torch.bfloat16)
    w_q, w_sf = quantize(w, is_sf_swizzled_layout=False)
    layer = types.SimpleNamespace(
        weight=torch.nn.Parameter(w_q), weight_scale=torch.nn.Parameter(w_sf, requires_grad=False)
    )
    kernel.process_weights_after_loading(layer)
    return kernel, layer


def attach_qa(layer_input, q, sf):
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic

    layer_input._moet_qa = QuantizedActivation(
        q.view(torch.float8_e4m3fn), sf, layer_input.dtype, layer_input.shape, kMxfp8Dynamic
    )
    return layer_input._moet_qa


def main() -> int:
    os.environ.setdefault("VLLM_MOET_MHC_OVERLAP", "1")
    os.environ.setdefault("VLLM_MOET_MHC_OVERLAP_DIR", "/opt/vllm-moet/dsv41_sm120/mhc_overlap")
    import vllm
    from vllm.models.deepseek_v41.nvidia.ops import mhc as ops
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    import vllm.model_executor.kernels.linear.mxfp8.flashinfer as F
    from vllm.models.deepseek_v41.attention import DeepseekV4Attention

    for name in (ops.__file__, sys.modules[DeepseekV4Attention.__module__].__file__):
        src = open(name).read()
        check(MARKER in src, f"patched: {name}")
    check(hasattr(ops, "_moet_boundary_post_norm"), "mhc.py carries _moet_boundary_post_norm")
    kernels = ops._moet_mhc_post_norm()
    check(kernels is not None, "sm_120 mHC kernel module loaded")
    check(hasattr(kernels, "mhc_post_norm_quant"), "kernel module exposes mhc_post_norm_quant")
    if FAILS:
        print("RESULT FAIL (deployment)")
        return 1

    dev = torch.device("cuda")
    torch.manual_seed(11)
    side = torch.cuda.Stream()
    print(f"vLLM {vllm.__version__}, {torch.cuda.get_device_name()}, "
          f"cap {torch.cuda.get_device_capability()}, MHC_OVERLAP_MAX_TOKENS {ops.MHC_OVERLAP_MAX_TOKENS}, "
          f"PDL {kernels.PDL_DEFAULT}")

    # Count the consumer's stock quantization. Warm the sm_120 GEMV probe first so
    # its own weight quantization is not counted.
    stock_quantize = F.mxfp8_e4m3_quantize
    calls = [0]

    def counting_quantize(*a, **k):
        calls[0] += 1
        return stock_quantize(*a, **k)

    kernel, layer = build_kernel(dev, stock_quantize)
    warm = (torch.randn(4, H, device=dev) * 0.5).to(torch.bfloat16)
    for _ in range(2):
        kernel.apply_weights(layer, warm)
    F.mxfp8_e4m3_quantize = counting_quantize

    def run(stream, T, capture_aux, ins, prequant):
        x, residual, post_layer_mix, comb_res_mix, fn, scale, base, norm_w, pre_mix = ins
        outs = ops.mhc_shifted_post_pre(
            x, residual, post_layer_mix, comb_res_mix, fn, scale, base,
            RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
            pre_mix=pre_mix, norm_weight=norm_w, norm_eps=NORM_EPS,
            capture_aux=capture_aux, stream=stream, prequant=prequant,
        )
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        return outs

    names = ("residual", "post_mix", "comb_mix", "layer_input", "next_pre_mix", "aux")
    cases = ((1, False), (1, True), (4, False), (6, True), (6, False), (7, False),
             (16, True), (64, False))
    eager_chain = {}
    for T, aux in cases:
        ins = inputs(T, dev)
        ref = run(None, T, aux, ins, prequant=False)
        eager = run(side, T, aux, ins, prequant=False)
        fused = run(side, T, aux, ins, prequant=True)
        li = fused[3]
        qa = getattr(li, "_moet_qa", None)
        ok = check(qa is not None, f"T={T:2d} aux={aux}: layer_input carries _moet_qa")
        if not ok:
            continue
        check(torch.equal(fused[3], ref[3]), f"T={T:2d} aux={aux}: layer_input unchanged by prequant")
        same = [n for n, r, f in zip(names, ref, fused) if not torch.equal(r, f)]
        check(not same, f"T={T:2d} aux={aux}: all boundary outputs bit-identical"
              + (f" (differ: {same})" if same else ""))
        check(torch.equal(li.view(torch.int16), eager[3].view(torch.int16)),
              f"T={T:2d} aux={aux}: eager side-stream boundary bit-identical")

        q_stock, sf_stock = stock_quantize(li.view(-1, H), is_sf_swizzled_layout=True)
        expect_sf = ((T + 127) // 128) * 128 * (H // 32)
        check(qa.scale.numel() == expect_sf,
              f"T={T:2d}: swizzled sf numel {qa.scale.numel()} == {expect_sf}")
        off = sf_real_offsets(T, H)
        qa_sf = qa.scale.view(torch.uint8).reshape(-1)
        st_sf = sf_stock.view(torch.uint8).reshape(-1)
        qe = torch.equal(qa.data.view(torch.uint8), q_stock.view(torch.uint8))
        se = torch.equal(qa_sf[off], st_sf[off])
        detail = ""
        if not qe:
            d = (qa.data.view(torch.uint8) != q_stock.view(torch.uint8)).nonzero()[:3].tolist()
            detail += f" payload first diffs {d}"
        if not se:
            d = off[(qa_sf[off] != st_sf[off]).nonzero().flatten()[:3]]
            detail += f" sf first diffs {d.tolist()}"
        check(qe and se, f"T={T:2d} aux={aux}: payload+SF bit-identical to stock op{detail}"
              + f" (sf real rows {off.numel()}/{qa_sf.numel()})")
        check(qa.orig_dtype == torch.bfloat16 and tuple(qa.orig_shape) == (T, H),
              f"T={T:2d}: QA orig_dtype/orig_shape bf16/{(T, H)}")

        # fresh buffers: every real-row byte is overwritten (no stale tail)
        q_fresh, sf_fresh = kernels.mhc_post_norm_quant(
            ins[0], ins[1], ins[2], ins[3], ins[8], ins[7], NORM_EPS,
            capture_aux=aux, swizzled=True,
        )[3:]
        check(torch.equal(q_fresh.view(torch.uint8), qa.data.view(torch.uint8))
              and torch.equal(sf_fresh.view(torch.uint8).reshape(-1)[off], qa_sf[off]),
              f"T={T:2d}: cached payload/SF == fresh allocation")

        # real projection: stock (quantizes itself) vs carried QA
        calls[0] = 0
        y_stock = kernel.apply_weights(layer, li)
        n_stock = calls[0]
        calls[0] = 0
        y_qa = kernel.apply_weights(layer, qa)
        n_qa = calls[0]
        check(n_stock == 1 and n_qa == 0,
              f"T={T:2d}: mxfp8_e4m3_quantize calls stock={n_stock} QA={n_qa}")
        check(torch.equal(y_stock.view(torch.int16), y_qa.view(torch.int16)),
              f"T={T:2d}: projection output bit-identical from QA and stock ({tuple(y_qa.shape)}, "
              f"{'GEMV' if T <= 16 else 'CUTLASS mm'})")

        # patched dispatch: fake `self` with the real kernel/quant_method shape
        attn = types.SimpleNamespace(shard_fused_wqa_wkv=False,
                                     fused_wqa_wkv=FakeFusedLinear(kernel, layer))
        calls[0] = 0
        y_dispatch = DeepseekV4Attention._fused_wqa_wkv_gemm(attn, li)
        n_dispatch = calls[0]
        check(n_dispatch == 0,
              f"T={T:2d}: patched dispatch consumed QA without re-quantizing (calls={n_dispatch})")
        check(torch.equal(y_dispatch.view(torch.int16), y_qa.view(torch.int16)),
              f"T={T:2d}: patched dispatch output bit-identical to QA projection")
        y_stock_dispatch = DeepseekV4Attention._fused_wqa_wkv_gemm(attn, li.detach())
        check(torch.equal(y_stock_dispatch.view(torch.int16), y_stock.view(torch.int16)),
              f"T={T:2d}: plain tensor still takes the stock path bit-identically")

        eager_chain[(T, aux)] = (ins, y_dispatch.clone())

    # Unsupported kernel type: the payload must be ignored, not asserted on.
    ins = inputs(1, dev)
    fused = run(side, 1, False, ins, prequant=True)
    li = fused[3]

    class _OtherKernel:
        def __init__(self, inner):
            self.inner = inner

        def input_quant_key(self):
            return li._moet_qa.quant_key

        def apply_weights(self, layer, x, bias=None):
            assert isinstance(x, torch.Tensor), "payload forwarded to a non-FlashInfer kernel"
            return self.inner.apply_weights(layer, x, bias)

    other = types.SimpleNamespace(
        shard_fused_wqa_wkv=False,
        fused_wqa_wkv=FakeFusedLinear(_OtherKernel(kernel), layer),
    )
    y_stock_other = kernel.apply_weights(layer, li)
    calls[0] = 0
    y_other = DeepseekV4Attention._fused_wqa_wkv_gemm(other, li)
    check(calls[0] == 1 and torch.equal(y_other.view(torch.int16), y_stock_other.view(torch.int16)),
          f"unsupported kernel falls back to the stock path (calls={calls[0]})")

    # CUDA graph: the whole producer + consumer chain, replayed twice.
    def graphed(fn, reps=2):
        warm_s = torch.cuda.Stream()
        warm_s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(warm_s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        outs = []
        for _ in range(reps):
            g.replay()
            outs.append(out.clone())
        torch.cuda.synchronize()
        return outs, g

    for T, aux in ((1, False), (1, True), (6, True), (6, False)):
        ins, y_ref = eager_chain[(T, aux)]
        attn.fused_wqa_wkv = FakeFusedLinear(kernel, layer)

        def chain():
            outs = run(side, T, aux, ins, prequant=True)
            return DeepseekV4Attention._fused_wqa_wkv_gemm(attn, outs[3])

        reps, g = graphed(chain)
        check(all(torch.equal(r.view(torch.int16), y_ref.view(torch.int16)) for r in reps),
              f"T={T:2d} aux={aux}: graph capture + 2 replays bit-identical to eager")

        def stock_chain():
            outs = run(side, T, aux, ins, prequant=False)
            return DeepseekV4Attention._fused_wqa_wkv_gemm(attn, outs[3])

        sreps, _sg = graphed(stock_chain)
        check(all(torch.equal(r.view(torch.int16), y_ref.view(torch.int16)) for r in sreps),
              f"T={T:2d} aux={aux}: stock chain graph bit-identical to fused eager")

    # Capture lifetime: several boundaries share the cached q/sf buffers inside
    # one graph. Every iteration must reproduce the eager reference, so a later
    # boundary must not overwrite a payload a pending consumer still reads.
    ins1, ref_gemm = eager_chain[(1, False)]

    def chain_multi(fused, n):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        got = []

        def body():
            for _ in range(n):
                outs = run(s, 1, False, ins1, prequant=fused)
                got.append(DeepseekV4Attention._fused_wqa_wkv_gemm(attn, outs[3]))

        with torch.cuda.stream(s):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        g.replay()
        torch.cuda.synchronize()
        return got

    for n in (2, 4):
        for fused in (False, True):
            got = chain_multi(fused, n)
            check(all(torch.equal(x.view(torch.int16), ref_gemm.view(torch.int16)) for x in got),
                  f"{n} boundaries in one graph, prequant={fused}: all {len(got)} projections "
                  "bit-identical to eager (cached q/sf not aliased across boundaries)")

    # Non-finite rows: the ship gate is the *served* consumer op, not the torch
    # reference cast. NaN/Inf must quantize exactly as mxfp8_e4m3_quantize does,
    # so the fused chain's projection is bit-identical to the stock chain.
    F.mxfp8_e4m3_quantize = stock_quantize
    Tn = 6
    x, residual, pm, cm, fn, scale, base, w, pre = inputs(Tn, dev)
    x, residual = x.clone(), residual.clone()
    residual[0] = float("nan")
    residual[1] = float("inf")
    residual[2] = float("-inf")
    x[3, 7] = float("nan")
    x[4] = 1e30
    x[5].zero_()
    residual[5].zero_()
    ins_n = (x, residual, pm, cm, fn, scale, base, w, pre)

    # the overlap path (stream is not None) is the only one that carries the payload
    ref = run(side, Tn, False, ins_n, prequant=False)
    fused = run(side, Tn, False, ins_n, prequant=True)
    li_s, li_f = ref[3], fused[3]
    print("  non-finite rows (nan, inf, -inf, one-nan, 1e30, zeros) -> li non-finite per row "
          f"{[bool(torch.isnan(li_s[r]).any() or torch.isinf(li_s[r]).any()) for r in range(Tn)]}")
    qa_n = getattr(li_f, "_moet_qa", None)
    if check(qa_n is not None, "non-finite: layer_input carries _moet_qa"):
        q_stock, sf_stock = stock_quantize(li_s.view(-1, H), is_sf_swizzled_layout=True)
        off = sf_real_offsets(Tn, H)
        qa_sf = qa_n.scale.view(torch.uint8).reshape(-1)
        st_sf = sf_stock.view(torch.uint8).reshape(-1)
        qa_q = qa_n.data.view(torch.uint8).reshape(Tn, H)
        st_q = q_stock.view(torch.uint8).reshape(Tn, H)
        tags = ("nan", "inf", "-inf", "one-nan", "1e30", "zeros")
        bad = [tags[r] for r in range(Tn) if not torch.equal(qa_q[r], st_q[r])]
        check(not bad, "non-finite: payload byte-identical to the served op on every row"
              + (f" (differ: {bad})" if bad else ""))
        pc = H // 32
        bad_sf = [tags[r] for r in range(Tn)
                  if not torch.equal(qa_sf[off][r * pc:(r + 1) * pc],
                                     st_sf[off][r * pc:(r + 1) * pc])]
        check(not bad_sf, "non-finite: per-block SF byte-identical to the served op"
              + (f" (differ: {bad_sf})" if bad_sf else ""))

        F.mxfp8_e4m3_quantize = counting_quantize
        calls[0] = 0
        y_s = kernel.apply_weights(layer, li_s)
        n_s = calls[0]
        calls[0] = 0
        y_f = kernel.apply_weights(layer, qa_n)
        n_f = calls[0]
        F.mxfp8_e4m3_quantize = stock_quantize
        check(n_s == 1 and n_f == 0,
              f"non-finite: mxfp8_e4m3_quantize calls stock={n_s} QA={n_f}")
        check(torch.equal(y_s.view(torch.int16), y_f.view(torch.int16)),
              "non-finite: projection output bit-identical from carried QA and stock")

        attn.fused_wqa_wkv = FakeFusedLinear(kernel, layer)

        def nchain():
            outs = run(side, Tn, False, ins_n, prequant=True)
            return DeepseekV4Attention._fused_wqa_wkv_gemm(attn, outs[3])

        eager_n = nchain().clone()
        reps_n, _g_n = graphed(nchain)
        check(all(torch.equal(r.view(torch.int16), eager_n.view(torch.int16)) for r in reps_n),
              "non-finite: graph capture + 2 replays bit-identical to eager")

        # Mixed blocks: a non-finite value in the inputs poisons the row-wide sumsq, so a
        # block holding both finite and non-finite elements needs the poison in the norm
        # weight, which localizes it to one column of an otherwise finite block.
        xm, resm, pmm, cmm, fnm, scm, bm, wm, prem = inputs(1, dev)
        wm = wm.clone()
        wm[7] = float("nan")
        wm[64] = float("inf")
        wm[100] = float("-inf")
        ins_m = (xm, resm, pmm, cmm, fnm, scm, bm, wm, prem)
        li_ms = run(side, 1, False, ins_m, prequant=False)[3]
        li_mf = run(side, 1, False, ins_m, prequant=True)[3]
        qa_m = getattr(li_mf, "_moet_qa", None)
        if check(qa_m is not None, "mixed blocks: layer_input carries _moet_qa"):
            check(int((~torch.isfinite(li_ms)).sum()) == 3,
                  "mixed blocks: exactly the 3 planted columns are non-finite in layer_input")
            q_st, sf_st = stock_quantize(li_ms.view(-1, H), is_sf_swizzled_layout=True)
            off_m = sf_real_offsets(1, H)
            qm = qa_m.data.view(torch.uint8).reshape(-1)
            stm = q_st.view(torch.uint8).reshape(-1)
            sfm = qa_m.scale.view(torch.uint8).reshape(-1)[off_m]
            sfs = sf_st.view(torch.uint8).reshape(-1)[off_m]
            cols = (7, 64, 100)
            print("  mixed-block bytes (nan, +inf, -inf columns) served="
                  f"{[int(stm[c]) for c in cols]} fused={[int(qm[c]) for c in cols]} "
                  f"served_sf={[int(sfs[c // 32]) for c in cols]} "
                  f"fused_sf={[int(sfm[c // 32]) for c in cols]}")
            check(torch.equal(qm, stm), "mixed blocks: payload byte-identical to the served op")
            check(torch.equal(sfm, sfs), "mixed blocks: per-block SF byte-identical to the served op")
            check(torch.equal(kernel.apply_weights(layer, li_ms).view(torch.int16),
                              kernel.apply_weights(layer, qa_m).view(torch.int16)),
                  "mixed blocks: projection output bit-identical from carried QA and stock")

    print("RESULT", "PASS" if not FAILS else f"FAIL ({len(FAILS)} checks)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
