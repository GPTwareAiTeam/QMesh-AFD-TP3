"""AFD-TP3 overlay for vLLM's DeepSeek-V4.1 stack.

Two entry points, both installed from ``sitecustomize`` so that the spawned
EngineCore and worker processes inherit them (a preload in the launcher script
would not survive ``spawn``):

1. weight names -- the rank packs store routed experts as one fused tensor per
   (layer, projection, kind): ``L{l}.w13.weight`` etc.  Translate those into the
   checkpoint-native name ``layers.{l}.ffn.experts.w13_weight`` before the
   model's own ``WeightsMapper`` pipeline runs, which then re-roots them under
   ``language_model.model.`` like any other checkpoint key.
2. routed-expert width -- main-layer routed experts hold only this rank's
   intermediate columns (``afd.routed_intermediate_size``), while the shared
   expert and the DSpark draft MoEs keep ``text_config.moe_intermediate_size``.
   Every expert kernel is linear in the intermediate dimension, so the ranks'
   partial sums reconstruct the full expert output.

The transport that sums the remote partials lives in :mod:`afd_bridge`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re

_PACK_KEY = re.compile(
    r"^L(\d+)(?:\.e(\d+))?\.(w13|w2|w1|w3)\.(weight|weight_scale|weight_scale_2|input_scale)$"
)

_installed = False


def config() -> dict:
    path = os.environ.get("AFD_CONFIG", "/opt/afd/afd.json")
    with open(path) as f:
        return json.load(f)


def _translate(key: str) -> str | None:
    m = _PACK_KEY.match(key)
    if m is None:
        return None
    layer, expert, proj, kind = m.groups()
    if expert is not None:
        # Per-expert pack: the checkpoint-native form V41's load_weights maps
        # onto the fused w13_/w2_ parameters.
        return f"layers.{layer}.ffn.experts.{expert}.{proj}.{kind}"
    return f"layers.{layer}.ffn.experts.{proj}_{kind}"


def _install_weight_names() -> None:
    from vllm.model_executor.models.utils import WeightsMapper

    orig = WeightsMapper._map_name_with_shard

    def _map_name_with_shard(self, key):
        new = _translate(key)
        return orig(self, key if new is None else new)

    WeightsMapper._map_name_with_shard = _map_name_with_shard


def _with_routed_width(config_obj, width: int):
    """Shallow config copy whose routed intermediate width is ``width``."""
    c = copy.copy(config_obj)
    text = getattr(c, "text_config", None)
    if text is not None:
        c.text_config = copy.copy(text)
        c.text_config.moe_intermediate_size = width
    c.moe_intermediate_size = width
    return c


def _install_routed_width() -> None:
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.models.deepseek_v4.nvidia import model as dsv4
    from vllm.logger import init_logger

    logger = init_logger("vllm.afd")
    width = int(os.environ["AFD_ROUTED_INTERMEDIATE"])
    orig = dsv4.DeepseekV4MoE._init_fused_moe_experts

    def _init_fused_moe_experts(self, vllm_config, config_obj, quant_config, prefix):
        n_layers = getattr(config_obj, "num_hidden_layers", 0)
        if extract_layer_index(prefix) >= n_layers:
            return orig(self, vllm_config, config_obj, quant_config, prefix)
        out = orig(
            self,
            vllm_config,
            _with_routed_width(config_obj, width),
            quant_config,
            prefix,
        )
        routed = getattr(getattr(self, "experts", None), "routed_experts", None)
        if routed is not None:
            logger.warning_once(
                "AFD: routed intermediate %d (w13 %s w2 %s)",
                width,
                tuple(routed.w13_weight.shape),
                tuple(routed.w2_weight.shape),
            )
        return out

    dsv4.DeepseekV4MoE._init_fused_moe_experts = _init_fused_moe_experts


def _install_activation_scale_fix() -> None:
    """Repair the NVFP4 routed-expert activation quantizer.

    The checkpoint stores ``input_scale = amax / (6 * 448)``, so
    ``process_weights_after_loading`` derives the per-layer multiplier
    ``a2_gscale = 1 / max_e(input_scale)`` 448x too large.  The fp4 block scale
    ``amax_block * gscale / 6`` then overflows e4m3, saturates at 448, and the
    activation is clipped at ``2688 / gscale``: 8.9 .. 150 over the 40 layers
    against a real local activation amax of 26.5 at layer 3.  Replaying the
    engine chain with those scales reproduces the dumped routed output to
    cos 0.999966 while the unclipped result is 2.588x larger, so the clip is
    the whole numerical defect.

    The activation amax spans 3700x across layers (0.0074 .. 4.18 on the remote
    slices), so a single divisor is unsafe: a blanket 448 would push the fp8
    block scale down to 5e-5 at the quiet layers and quantize them to zero.
    ``a2_gscale.json`` instead carries one divisor per layer, derived from the
    GX workers' per-layer --calib activation amax measured on the same model,
    putting ``a2_gscale`` at ``2 / amax_gx``: clip = 1344 * amax_gx, i.e. 13x
    above the largest observed local/remote amax ratio, and sf_max = ratio/2.
    Scaling ``w2_input_global_scale`` by any factor leaves the GEMM product
    untouched (``a2_gscale * g2_alphas`` stays the pack weight_scale_2), so only
    the quantizer moves.  The a1/x quantizer is left alone: its clip level
    (1.10 .. 8.19) is above every observed hidden-state amax.
    """
    import torch
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4FusedMoE,
    )

    logger = init_logger("vllm.afd")
    path = os.environ.get(
        "AFD_A2_SCALE_JSON",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "a2_gscale.json"),
    )
    with open(path) as f:
        doc = json.load(f)
    table, meta = doc["div"], doc["layers"]
    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    # Draft DSpark layers reuse ModelOptNvFp4FusedMoE but must never receive a
    # main-layer activation scale: their names are layers.40..42, which parse
    # cleanly but are outside the 40-entry table.  Reject those by index before
    # the value-match fallback can consider them.
    main_layers = int(os.environ.get("AFD_LAYERS", "40"))
    bad_keys = sorted(
        k for k in table if not (str(k).isdigit() and int(k) < main_layers)
    )
    if bad_keys:
        raise RuntimeError(
            f"AFD a2 fix: {path} has non-main layer keys {bad_keys} "
            f"(AFD_LAYERS={main_layers})"
        )

    # ModelOptNvFp4FusedMoE is the live routed-expert quantizer here (the
    # compressed_tensors W4A4 method is never instantiated, and its
    # process_weights_after_loading is dead code in this build).  The stored
    # per-expert w2_input_scale enters convert_to_nvfp4_moe_kernel_format and
    # becomes a2_gscale = 1/a2_scale in the quant config, so multiplying it by
    # mu divides the quantizer multiplier by mu -- the direction table["div"]
    # was computed for.  The tensor is a per-expert vector; mu broadcasts.
    orig = ModelOptNvFp4FusedMoE.process_weights_after_loading
    seen = [0]
    draft_seen = [0]

    def _layer_index(name, gs_now):
        m = layer_re.search(name or "")
        if m is not None:
            if int(m.group(1)) >= main_layers:
                return None, "draft"
            return m.group(1), "name"
        # Fallback: the stored multipliers are distinct across the 40 layers,
        # so match the observed value against the table.  The table only has
        # main layers, and the index is re-checked so this can never land on a
        # draft layer.
        best, err = None, None
        for idx, info in meta.items():
            e = abs(math.log(gs_now / info["a2_gscale"]))
            if err is None or e < err:
                best, err = idx, e
        if err is not None and err < 0.02 and int(best) < main_layers:
            return best, "value"
        return None, "none"

    def process_weights_after_loading(self, layer):
        p = getattr(layer, "w2_input_scale", None)
        if isinstance(p, torch.Tensor) and p.numel():
            gs_now = float(1.0 / p.detach().float().reshape(-1)[0].item())
            if gs_now > 0.0:
                name = getattr(layer, "layer_name", "") or getattr(
                    layer, "prefix", ""
                )
                idx, how = _layer_index(name, gs_now)
                mu = table.get(idx) if idx is not None else None
                if mu is not None:
                    with torch.no_grad():
                        p.data.mul_(mu)
                    gs_new = float(1.0 / p.detach().float().reshape(-1)[0].item())
                    if seen[0] < 4 or idx in ("3", "30"):
                        logger.warning(
                            f"AFD a2 fix L{idx} ({how}): mu={mu:.4f} "
                            f"gscale {gs_now:.4f} -> {gs_new:.4f} "
                            f"clip {2688.0 / gs_now:.2f} -> {2688.0 / gs_new:.2f}"
                        )
                elif how == "draft":
                    if draft_seen[0] < 4:
                        logger.warning(
                            f"AFD a2 fix: draft layer excluded name={name!r} "
                            f"gscale={gs_now:.4f} (AFD_LAYERS={main_layers})"
                        )
                    draft_seen[0] += 1
                elif seen[0] < 4:
                    logger.warning(
                        f"AFD a2 fix: unmatched layer name={name!r} "
                        f"gscale={gs_now:.4f}"
                    )
                seen[0] += 1
        return orig(self, layer)

    ModelOptNvFp4FusedMoE.process_weights_after_loading = (
        process_weights_after_loading
    )
    logger.warning(
        f"AFD: routed NVFP4 a2 scale fix active, {len(table)} layers from {path}"
    )
    logger.warning(
        f"AFD: a2 fix draft exclusion: layers >= {main_layers} never receive a "
        f"main-layer scale (value fallback is main-only)"
    )


def _install_moe_seam() -> None:
    """Send the routed activation to the GX ranks and add their partials.

    The two hooks are separate on purpose: ``post`` runs between routing and
    the local kernel so the wire time hides under P7's own 576-wide GEMMs,
    while ``collect`` runs once the local routed output is final and before
    ``routed_scaling_factor`` is applied.
    """
    import torch
    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    import afd_bridge
    import afd_seam

    _SEAM = os.environ.get("AFD_SEAM") == "1"

    logger = init_logger("vllm.afd")
    main_layers = int(os.environ.get("AFD_LAYERS", "40"))
    n_experts = int(os.environ.get("AFD_NUM_EXPERTS", "384"))
    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

    def _main_layer(name):
        m = layer_re.search(name or "")
        if m is None:
            return None
        i = int(m.group(1))
        return i if i < main_layers else None

    orig_post = RoutedExperts.forward_modular
    orig_scale = MoERunner._maybe_apply_routed_scale_to_output

    def forward_modular(self, x, topk_weights, topk_ids,
                        shared_experts=None, shared_experts_input=None):
        layer = _main_layer(getattr(self, "layer_name", "")
                            or getattr(self, "prefix", ""))
        if (
            layer is None
            or self.global_num_experts != n_experts
            or self.quant_method.is_monolithic
        ):
            return orig_post(self, x, topk_weights, topk_ids,
                             shared_experts, shared_experts_input)
        if self.apply_router_weight_on_input:
            raise RuntimeError("AFD: apply_router_weight_on_input is unsupported")
        if not getattr(self, "_afd_scale_logged", False):
            self._afd_scale_logged = True
            qc = getattr(self.quant_method, "moe_quant_config", None)
            dump = os.environ.get("AFD_DUMP")

            def _vec(name):
                t = getattr(qc, name, None) if qc is not None else None
                if t is None:
                    return None
                return t.detach().float().reshape(-1).cpu().numpy()

            def _show(name):
                v = _vec(name)
                if v is None:
                    return "None"
                return "[" + " ".join(f"{x:.6g}" for x in v[:4].tolist()) + "]"

            logger.warning(
                f"AFD local scales L{layer}: "
                f"a1_gscale={_show('a1_gscale')} a2_gscale={_show('a2_gscale')} "
                f"g1_alphas={_show('g1_alphas')} g2_alphas={_show('g2_alphas')} "
                f"w13_s2={_show('w13_scale_2')} w2_s2={_show('w2_scale_2')} "
                f"w13_in={_show('w13_input_scale')} w2_in={_show('w2_input_scale')} "
                f"clamp={getattr(self, 'swiglu_limit', None)} "
                f"actcfg={getattr(getattr(self, 'quant_method', None), 'activation_config', None)} "
                f"runner_scale={getattr(getattr(self, 'runner', None), 'routed_scaling_factor', 'n/a')} "
                f"self_scale={getattr(self, 'routed_scaling_factor', 'n/a')}"
            )
            if dump:
                arrays = {}
                for name in ("a1_gscale", "a2_gscale", "g1_alphas",
                             "g2_alphas", "w13_scale_2", "w2_scale_2",
                             "w13_input_scale", "w2_input_scale"):
                    v = _vec(name)
                    if v is not None:
                        arrays[name] = v
                for name in ("w13_weight_global_scale", "w2_weight_global_scale"):
                    v = getattr(self, name, None)
                    if v is not None:
                        arrays[name] = v.detach().float().reshape(-1)[:64].cpu().numpy()
                try:
                    import numpy as np
                    os.makedirs(dump, exist_ok=True)
                    np.savez(os.path.join(dump, f"scales_L{layer}.npz"), **arrays)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"AFD scale dump failed: {exc!r}")
        if _SEAM:
            afd_seam.seam().post(layer, x, topk_ids, topk_weights)
        else:
            afd_bridge.bridge().post(layer, x, topk_weights, topk_ids)
        return orig_post(self, x, topk_weights, topk_ids,
                         shared_experts, shared_experts_input)

    def _maybe_apply_routed_scale_to_output(self, shared_output, fused_output):
        layer = _main_layer(self.layer_name)
        if layer is not None and isinstance(fused_output, torch.Tensor):
            if _SEAM:
                afd_seam.seam().collect(layer, fused_output)
            else:
                afd_bridge.bridge().collect(layer, fused_output)
        return orig_scale(self, shared_output, fused_output)

    RoutedExperts.forward_modular = forward_modular
    MoERunner._maybe_apply_routed_scale_to_output = (
        _maybe_apply_routed_scale_to_output
    )
    logger.warning_once(
        f"AFD: routed seam installed over {main_layers} main layers, "
        f"{n_experts} experts"
    )


def _install_seam_load() -> None:
    """Connect the GX peers and build `_qmesh_afd.Seam` once, after weights load.

    Runs before any CUDA graph capture or profile pass, so the hot path never
    connects. The import in `afd_seam` raises on a .so without Seam.
    """
    import torch
    from vllm.logger import init_logger
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    import afd_seam

    logger = init_logger("vllm.afd")
    orig = GPUModelRunner.load_model

    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        device = self.device.index if self.device.index is not None else 0
        afd_seam.seam().load(device)
        logger.warning("AFD: Seam loaded on device %d (AFD_SEAM=1)", device)
        return out

    GPUModelRunner.load_model = load_model



def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    _install_weight_names()
    if os.environ.get("AFD_ROUTED_INTERMEDIATE"):
        _install_routed_width()
    if os.environ.get("AFD_TRANSPORT") == "1":
        _install_moe_seam()
    if os.environ.get("AFD_TRANSPORT") == "1" and os.environ.get("AFD_SEAM") == "1":
        _install_seam_load()
    if os.environ.get("AFD_A2_SCALE_FIX", "1") not in ("0", "", "false"):
        _install_activation_scale_fix()
