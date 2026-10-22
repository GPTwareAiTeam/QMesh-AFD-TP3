#!/usr/bin/env python3
"""CPU proof of the AFD x DSpark layer boundary.

DSpark's three draft MoE layers are ``layers.40/41/42`` (128 experts, top-3).
They reuse the main model's MoE classes but must never be routed to GX and must
never receive a main-layer a2 activation scale.  This test stubs vLLM/torch,
loads ``afd_overlay.py`` as shipped, and exercises the installed hooks.

No GPU, no container, no network.  Run: python3 p7/test_afd_dspark_routing.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY_PATH = os.path.join(HERE, "afd_overlay", "afd_overlay.py")
A2_PATH = os.path.join(HERE, "afd_overlay", "a2_gscale.json")

failures: list[str] = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


# --- minimal torch -----------------------------------------------------------
class Tensor:
    def __init__(self, value):
        self.v = float(value)

    @property
    def data(self):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def reshape(self, *args):
        return self

    def numel(self):
        return 1

    def __getitem__(self, index):
        return self

    def item(self):
        return self.v

    def mul_(self, factor):
        self.v *= float(factor)
        return self

class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


torch = types.ModuleType("torch")
torch.Tensor = Tensor
torch.no_grad = _NoGrad
sys.modules["torch"] = torch


class Logger:
    def __init__(self):
        self.messages: list[str] = []

    def warning(self, msg, *args):
        self.messages.append(msg % args if args else msg)

    warning_once = warning


LOGGER = Logger()

# --- minimal vllm tree -------------------------------------------------------
def mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def link(parent, attr, child):
    setattr(parent, attr, child)
    return child


class WeightsMapper:
    def _map_name_with_shard(self, key):
        return key


class RoutedExperts:
    def forward_modular(self, x, topk_weights, topk_ids,
                        shared_experts=None, shared_experts_input=None):
        return ("orig", getattr(self, "layer_name", ""))


class MoERunner:
    def _maybe_apply_routed_scale_to_output(self, shared_output, fused_output):
        return ("orig", getattr(self, "layer_name", ""))


class ModelOptNvFp4FusedMoE:
    def process_weights_after_loading(self, layer):
        return ("orig", layer)


class DeepseekV4MoE:
    @staticmethod
    def _init_fused_moe_experts(self, vllm_config, config_obj, quant_config, prefix):
        return prefix


vllm = mod("vllm")
vllm.__path__ = []
me = link(vllm, "model_executor", mod("vllm.model_executor"))
memodels = link(me, "models", mod("vllm.model_executor.models"))
utils = link(memodels, "utils", mod("vllm.model_executor.models.utils"))
utils.WeightsMapper = WeightsMapper
utils.extract_layer_index = lambda prefix: int(
    prefix.rsplit("layers.", 1)[1].split(".", 1)[0]
)
layers = link(me, "layers", mod("vllm.model_executor.layers"))
quant = link(layers, "quantization",
             mod("vllm.model_executor.layers.quantization"))
modelopt = link(quant, "modelopt",
                mod("vllm.model_executor.layers.quantization.modelopt"))
modelopt.ModelOptNvFp4FusedMoE = ModelOptNvFp4FusedMoE
fm = link(layers, "fused_moe", mod("vllm.model_executor.layers.fused_moe"))
re_mod = link(fm, "routed_experts",
              mod("vllm.model_executor.layers.fused_moe.routed_experts"))
re_mod.RoutedExperts = RoutedExperts
runner = link(fm, "runner", mod("vllm.model_executor.layers.fused_moe.runner"))
mr = link(runner, "moe_runner",
          mod("vllm.model_executor.layers.fused_moe.runner.moe_runner"))
mr.MoERunner = MoERunner
models = link(vllm, "models", mod("vllm.models"))
v4 = link(models, "deepseek_v4", mod("vllm.models.deepseek_v4"))
v4n = link(v4, "nvidia", mod("vllm.models.deepseek_v4.nvidia"))
dsv4 = link(v4n, "model", mod("vllm.models.deepseek_v4.nvidia.model"))
dsv4.DeepseekV4MoE = DeepseekV4MoE
logger_mod = link(vllm, "logger", mod("vllm.logger"))
logger_mod.init_logger = lambda name: LOGGER


class Recorder:
    def __init__(self):
        self.posts = []
        self.collects = []


RECORDER = Recorder()
afd_seam = mod("afd_seam")
afd_seam.seam = lambda: types.SimpleNamespace(
    post=lambda layer, *a: RECORDER.posts.append(layer),
    collect=lambda layer, *a: RECORDER.collects.append(layer),
)
afd_bridge = mod("afd_bridge")
afd_bridge.bridge = lambda: types.SimpleNamespace(
    post=lambda layer, *a: RECORDER.posts.append(layer),
    collect=lambda layer, *a: RECORDER.collects.append(layer),
)

# --- load the shipped overlay ------------------------------------------------
os.environ["AFD_LAYERS"] = "40"
os.environ["AFD_NUM_EXPERTS"] = "384"
os.environ["AFD_A2_SCALE_JSON"] = A2_PATH
spec = importlib.util.spec_from_file_location("afd_overlay_under_test", OVERLAY_PATH)
ov = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ov
spec.loader.exec_module(ov)

with open(A2_PATH) as f:
    doc = json.load(f)
div, meta = doc["div"], doc["layers"]
check(len(div) == 40, f"a2 table must have 40 main layers, has {len(div)}")
check(set(div) == set(meta), "a2 div and layers keys differ")
for key in div:
    check(key.isdigit() and int(key) < 40, f"a2 table has non-main key {key!r}")

# --- a2 activation scale: main applies, draft never does ---------------------
ov._install_activation_scale_fix()
hook = ModelOptNvFp4FusedMoE.process_weights_after_loading


class A2Layer:
    def __init__(self, name, gscale):
        self.layer_name = name
        self.prefix = ""
        self.w2_input_scale = Tensor(1.0 / gscale)


nameless = A2Layer("", meta["0"]["a2_gscale"])
hook(None, nameless)
check(
    math.isclose(float(1.0 / nameless.w2_input_scale.item()),
                 meta["0"]["a2_gscale"] / div["0"], rel_tol=1e-9),
    "a2 value fallback no longer applies to a nameless main layer",
)
check(
    any("L0 (value)" in m for m in LOGGER.messages),
    "a2 value fallback did not log its match",
)

# A draft layer whose stored scale exactly matches main layer 0 must not move.
draft = A2Layer("language_model.model.layers.40.ffn.experts", meta["0"]["a2_gscale"])
hook(None, draft)
check(
    math.isclose(float(1.0 / draft.w2_input_scale.item()), meta["0"]["a2_gscale"]),
    "a2 fix multiplied a draft layer (L40) by a main-layer scale",
)
check(
    any("draft layer excluded" in m and "layers.40" in m for m in LOGGER.messages),
    "a2 fix did not log the draft-layer exclusion",
)

for idx in ("41", "42"):
    draft = A2Layer(f"language_model.model.layers.{idx}.ffn.experts",
                    meta["3"]["a2_gscale"])
    hook(None, draft)
    check(
        math.isclose(float(1.0 / draft.w2_input_scale.item()),
                     meta["3"]["a2_gscale"]),
        f"a2 fix touched draft layer L{idx}",
    )

main = A2Layer("language_model.model.layers.39.ffn.experts", meta["39"]["a2_gscale"])
hook(None, main)
check(
    math.isclose(float(1.0 / main.w2_input_scale.item()),
                 meta["39"]["a2_gscale"] / div["39"], rel_tol=1e-9),
    "a2 fix did not apply the L39 main-layer divisor",
)


# A bad package (draft index in the table) must fail closed at install time.
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump({"div": {"40": 1.0}, "layers": {"40": {"a2_gscale": 1.0}}}, f)
    bad_path = f.name
try:
    os.environ["AFD_A2_SCALE_JSON"] = bad_path
    try:
        ov._install_activation_scale_fix()
        check(False, "a2 fix accepted a table key >= AFD_LAYERS")
    except RuntimeError as exc:
        check("non-main layer keys" in str(exc), f"unexpected refusal: {exc}")
finally:
    os.unlink(bad_path)
os.environ["AFD_A2_SCALE_JSON"] = A2_PATH

# --- seam: only 40 main layers with 384 experts are routed -------------------
os.environ["AFD_TRANSPORT"] = "1"
os.environ["AFD_SEAM"] = "1"
ov._install_moe_seam()


class ExpertLayer:
    def __init__(self, name, n_experts, monolithic=False):
        self.layer_name = name
        self.global_num_experts = n_experts
        self.quant_method = types.SimpleNamespace(
            is_monolithic=monolithic, moe_quant_config=None, activation_config=None
        )
        self.apply_router_weight_on_input = False


RECORDER.posts.clear()
for name, n in (
    ("language_model.model.layers.40.ffn.experts", 384),
    ("language_model.model.layers.41.ffn.experts", 384),
    ("language_model.model.layers.42.ffn.experts", 384),
    ("language_model.model.layers.39.ffn.experts", 128),
    ("language_model.model.layers.39.ffn.experts", 384),
):
    layer = ExpertLayer(name, n)
    out = RoutedExperts.forward_modular(layer, None, None, None)
    check(out[0] == "orig", f"seam hook skipped the original forward for {name}")
check(
    RECORDER.posts == [39],
    f"seam routed draft/wrong-width layers: posts={RECORDER.posts}",
)

RECORDER.collects.clear()
MoERunner._maybe_apply_routed_scale_to_output(
    ExpertLayer("language_model.model.layers.40.ffn.experts", 384), None, Tensor(1.0)
)
check(not RECORDER.collects, "seam collected the draft layer's fused output")
MoERunner._maybe_apply_routed_scale_to_output(
    ExpertLayer("language_model.model.layers.39.ffn.experts", 384), None, Tensor(1.0)
)
check(RECORDER.collects == [39], f"seam collect wrong: {RECORDER.collects}")

if failures:
    print("FAIL")
    for msg in failures:
        print(f"  - {msg}")
    sys.exit(1)
print("PASS: a2 main-only (40 keys, L40-42 excluded, fallback main-only); "
      "seam routed only over L0-39 with 384 experts")
