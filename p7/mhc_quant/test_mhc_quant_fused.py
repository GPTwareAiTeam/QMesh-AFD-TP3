"""Bit-exactness of the fused MXFP8 quantization against the unfused pair:
stock mhc_post_norm -> the served consumer op torch.ops.vllm.mxfp8_quantize (GPU0).

The served op is flashinfer's cute-dsl mxfp8 kernel on sm120 (capability 12.0 satisfies
has_device_capability(100)), and its element rule is mul, min.f32 +448, max.f32 -448,
cvt.rn.satfinite.e4m3x2.f32, where min/max do not propagate nan. Its scale is the UE8M0
exponent of the block amax/448, taken from a max tree with no zero initial value: nan is
ignored while any numeric element shares the 32-element block, only an all-nan block makes
the scale nan -> 254, and a block holding an inf gets 254, whose inverse scale has
exponent field 0 and is therefore 0.0 - so inf * 0 and -inf * 0 are nan too and every
non-finite element lands on +448 = 0x7e, the sign of an inf never reaching the payload.
All of this is asserted here against the served op; vLLM's torch reference (which keeps
0x7f) is printed as a diagnostic only.
"""
import torch
from mhc_post_norm_sm120 import mhc_post_norm, mhc_post_norm_quant

H, HC = 5120, 4


def _u8(t):
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def ref_served(li, layout, align=32):
    """The op the engine actually calls (importing the module registers torch.ops.vllm.mxfp8_quantize)."""
    import vllm.model_executor.layers.quantization.utils.mxfp8_utils  # noqa: F401
    q, sf = torch.ops.vllm.mxfp8_quantize(li, layout, align)
    return _u8(q), _u8(sf)


def ref_torch(li, layout):
    """vLLM's own elementwise reference (torch cast). Diagnostic only: it keeps nan as 0x7f."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_torch
    q, sf = _mxfp8_e4m3_quantize_torch(li, layout)
    return _u8(q), _u8(sf)


def case(name, x, res, plant=None):
    """plant=(col, value) puts a non-finite value into the norm weight, the only way to build a
    genuinely mixed row: a non-finite x element poisons the row-wide sumsq and makes the entire
    row non-finite."""
    T = x.shape[0]
    pm = torch.randn(T, HC, device="cuda", dtype=torch.float32) * 0.5
    cm = torch.randn(T, HC, HC, device="cuda", dtype=torch.float32) * 0.5
    pre = torch.randn(T, HC, device="cuda", dtype=torch.float32) * 0.5
    w = torch.randn(H, device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
    if plant is not None:
        w[plant[0]] = plant[1]
    _, li_ref, _ = mhc_post_norm(x, res, pm, cm, pre, w, 1e-6, mode=1, pdl=False)
    _, li_new, _, q, sf = mhc_post_norm_quant(x, res, pm, cm, pre, w, 1e-6, mode=1, pdl=False)
    _, _, _, qs, sfs = mhc_post_norm_quant(x, res, pm, cm, pre, w, 1e-6, mode=1, pdl=False, swizzled=True)
    bad = []
    if not torch.equal(li_ref.view(torch.int16), li_new.view(torch.int16)):
        bad.append("layer_input moved")
    pc = H // 32
    rows = torch.arange(T, device="cuda")[:, None]
    cols = torch.arange(pc, device="cuda")[None, :]
    swz = ((cols & 3) + (cols >> 2) * 512 + (rows & 31) * 16 + ((rows & 127) >> 5) * 4 +
           (rows >> 7) * (128 * pc)).reshape(-1)
    for layout in (False, True):
        qr, sfr = ref_served(li_ref, layout)
        qrt, _ = ref_torch(li_ref, layout)
        q2, sf2 = (qs, sfs) if layout else (q, sf)
        q2u, sf2u = _u8(q2).reshape(-1), _u8(sf2).reshape(-1)
        qru, sfru = qr.reshape(-1), sfr.reshape(-1)
        qe = torch.equal(q2u, qru[: q2u.numel()])
        if layout:
            se = torch.equal(sf2u[swz], sfru[swz])
        else:
            se = torch.equal(sf2u, sfru[: sf2u.numel()])
        if not (qe and se):
            bad.append(f"layout={'swz' if layout else 'lin'} payload={qe} sf={se} "
                       f"first_diff_q={(q2u != qru[: q2u.numel()]).nonzero()[:3].tolist()} "
                       f"first_diff_sf={(sf2u != sfru[: sf2u.numel()]).nonzero()[:3].tolist()}")
        n_torch = int((q2u != qrt.reshape(-1)[: q2u.numel()]).sum())
        note = f" torch_ref_diff_bytes={n_torch}" if n_torch else ""
        if plant is not None and not layout:
            col = plant[0]
            lv = li_ref[0, col].float().item()
            b = int(q2u[col])
            s = int(sf2u[col // 32])
            note += f" planted_col={col} li={lv!r} sf={s} byte=0x{b:02x}"
            if not torch.isfinite(li_ref[0, col]):
                # Pin the mechanism, not only equality: nan, +inf and -inf all land on 0x7e
                # because an inf drives the block amax to inf, the scale to 254 and the
                # inverse scale to 0.0, so inf * 0 is nan like nan * any. A nan on its own
                # leaves the scale to the finite amax; an inf takes it to 254.
                if b != int(qru[col]) or b != 0x7e:
                    bad.append(f"planted byte 0x{b:02x} != served 0x{int(qru[col]):02x} / 0x7e")
                if (lv != lv) == (s == 254):
                    bad.append(f"planted scale {s} for li={lv!r}: a nan keeps the finite amax "
                               f"scale, an inf forces 254")
            else:
                bad.append("plant did not survive into layer_input")
        print(f"  {name:14s} layout={'swz' if layout else 'lin'} payload_exact={qe} sf_exact={se} "
              f"sf_head={sf2u[:4].tolist()} ref_sf_head={sfru[:4].tolist()}{note}")
    print(("OK   " if not bad else "FAIL ") + name + (" " + "; ".join(bad) if bad else ""))
    return not bad


torch.manual_seed(0)
NAN, PINF, NINF = float("nan"), float("inf"), float("-inf")
ok = True
for T in (1, 4, 7):
    base = (torch.randn(T, H, device="cuda") * 2.0).to(torch.bfloat16)
    res = (torch.randn(T, HC, H, device="cuda") * 1.5).to(torch.bfloat16)
    ok &= case(f"random T={T}", base, res)
    ok &= case(f"zeros T={T}", torch.zeros(T, H, device="cuda", dtype=torch.bfloat16), res)
    ok &= case(f"448 T={T}", torch.full((T, H), 448.0, device="cuda", dtype=torch.bfloat16), res)
    ok &= case(f"denorm T={T}", torch.full((T, H), 1e-40, device="cuda", dtype=torch.bfloat16), res)
    ok &= case(f"sat T={T}", (torch.randn(T, H, device="cuda") * 1e3).to(torch.bfloat16), res)
    # Whole-row non-finite: the row-wide sumsq goes nan, so every element of the row is nan,
    # every scale block is all-nan -> SF 254 and payload 0x7e everywhere.
    nanx = base.clone(); nanx[0, :64] = NAN
    ok &= case(f"row_nan T={T}", nanx, res)
    infx = base.clone(); infx[0, :64] = PINF; infx[0, 64:128] = NINF
    ok &= case(f"row_inf T={T}", infx, res)
    # Mixed rows: one non-finite element inside an otherwise finite block, byte 0x7e for
    # nan/+inf/-inf (the nan keeps the finite amax scale, the infs force it to 254).
    ok &= case(f"mix_nan T={T}", base, res, plant=(100, NAN))
    ok &= case(f"mix_pinf T={T}", base, res, plant=(100, PINF))
    ok &= case(f"mix_ninf T={T}", base, res, plant=(100, NINF))
    ok &= case(f"mix_nan2 T={T}", base, res, plant=(31, NAN))


def capture_case(T=1):
    """Graph-capture replay with preallocated q_out/sf_out and PDL on (the served launch)."""
    x = (torch.randn(T, H, device="cuda") * 2.0).to(torch.bfloat16)
    res = (torch.randn(T, HC, H, device="cuda") * 1.5).to(torch.bfloat16)
    pm = torch.randn(T, HC, device="cuda") * 0.5
    cm = torch.randn(T, HC, HC, device="cuda") * 0.5
    pre = torch.randn(T, HC, device="cuda") * 0.5
    w = torch.randn(H, device="cuda", dtype=torch.bfloat16) * 0.1 + 1
    ro = torch.empty_like(res)
    li = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
    ax = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
    q = torch.empty((T, H), dtype=torch.uint8, device="cuda")
    sf = torch.empty(T * (H // 32), dtype=torch.uint8, device="cuda")
    call = lambda: mhc_post_norm_quant(x, res, pm, cm, pre, w, 1e-6, mode=1, pdl=True,
                                       out=(ro, li, ax), q_out=q, sf_out=sf)
    call(); torch.cuda.synchronize()
    ref = (ro.view(torch.int16).clone(), li.view(torch.int16).clone(), q.clone(), sf.clone())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    ro.zero_(); li.zero_(); q.zero_(); sf.zero_()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    got = (ro.view(torch.int16), li.view(torch.int16), q, sf)
    ok = all(torch.equal(a, b) for a, b in zip(ref, got))
    print(("OK   " if ok else "FAIL ") + f"graph capture replay (pdl=True) T={T} bitwise exact={ok}")
    return ok


gok = capture_case(1) and capture_case(4)
ok = ok and gok
print("RESULT", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
