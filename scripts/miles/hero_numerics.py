"""Observe hero HF/SGLang layer boundaries and replay identical inputs in HF.

Baseline capture is observational. Optional ablations replace selected serving
outputs with HF arithmetic on the actual serving inputs; weights and gates stay
unchanged. Hooks use SGLang's public forward_hooks interface.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import sglang
import torch
from olmo_sglang import register
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from open_instruct.miles import fla_compat

_ROUTER_OBSERVATIONS = {}
_ACTIVE_MLP = None
_ROUTER_HOOKED = False
_REFERENCE_HF = None
_LAST_HIDDEN = None


BOUNDARIES = (
    "self_attn",
    "mlp",
    "pre_attention_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)
DETAILS = (
    "mlp.latent_down_proj",
    "mlp.latent_up_proj",
    "mlp.shared_expert",
    "mlp.experts",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.qkv_proj",
    "self_attn.f_proj_1",
    "self_attn.f_proj_2",
    "self_attn.beta_proj",
    "self_attn.g_proj_1",
    "self_attn.g_proj_2",
    "self_attn.g_proj",
    "self_attn.o_norm",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
)


def names(layers):
    return ["model.embed_norm", "model.norm"] + [
        f"model.layers.{i}.{suffix}" for i in range(layers) for suffix in BOUNDARIES + DETAILS
    ]


def tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return tensor(value[0])
    return None


def input_tensor(args, kwargs, name):
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    if name.endswith(".self_attn") and len(args) > 1 and isinstance(args[0], torch.Tensor) and args[0].ndim == 1:
        return args[1]
    return tensor(args)


def metric(actual, reference):
    actual, reference = actual.float().reshape(-1), reference.float().reshape(-1)
    if actual.shape != reference.shape:
        return {"shape_mismatch": [list(actual.shape), list(reference.shape)]}
    delta = actual - reference
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
        "equal_fraction": (actual == reference).float().mean().item(),
        "reference_rms": reference.square().mean().sqrt().item(),
    }


def reference_substitution(name, saved, output, control, model_path):
    """Diagnostic intervention: use checkpoint HF arithmetic on actual SGLang inputs."""
    global _REFERENCE_HF, _LAST_HIDDEN
    mode = control.get("override", "")
    if not mode:
        return output
    if name == "model.norm":
        _LAST_HIDDEN = tensor(output).detach()
    layer_suffix = name.split("layers.")[-1].split(".", 1)[-1]
    is_norm = name in ["model.norm", "model.embed_norm"] or layer_suffix.endswith("_layernorm")
    selected = (
        mode == "experts"
        and name.endswith(".mlp.experts")
        or mode == "mlp"
        and layer_suffix == "mlp"
        or mode == "norms"
        and is_norm
        or mode == "attention"
        and layer_suffix == "self_attn"
        or mode in ["all", "all+head"]
        and (is_norm or layer_suffix in ["mlp", "self_attn"])
        or mode == "all+head"
        and name == "logits_processor"
    )
    if not selected:
        return output
    if _REFERENCE_HF is None:
        fla_compat.install_kda_triton_compat()
        _REFERENCE_HF = load_hf(model_path)
    if name == "logits_processor":
        logits = _REFERENCE_HF.lm_head(_LAST_HIDDEN.unsqueeze(0))[0, -1:].float()
        output.next_token_logits = logits
        return output
    module = _REFERENCE_HF.get_submodule(name)
    value = saved["input"].cuda()
    if name.endswith(".mlp.experts"):
        replacement = module(
            value, topk_ids=saved["topk_ids"].cuda().long(), topk_weights=saved["topk_weights"].cuda()
        )
    elif layer_suffix == "self_attn":
        value = value.unsqueeze(0)
        length = value.shape[1]
        mask = torch.full((length, length), float("-inf"), device=value.device, dtype=value.dtype).triu(1)
        replacement = module(
            hidden_states=value,
            past_key_values=None,
            position_embeddings=None,
            attention_mask=mask[None, None],
            position_ids=torch.arange(length, device=value.device)[None],
        )[0]
    elif layer_suffix == "mlp":
        replacement = module(value.unsqueeze(0))
    else:
        replacement = module(value)
    replacement = replacement.reshape_as(tensor(output))
    if name == "model.norm":
        _LAST_HIDDEN = replacement.detach()
    return replacement


def trace_factory(config):
    """SGLang hook factory; a discarded priming request installs pre-hooks."""
    global _ROUTER_HOOKED
    model_module = sys.modules.get("olmo_sglang.models.olmo3_moe")
    if not _ROUTER_HOOKED and model_module is not None:
        original_router = model_module.fp32_router_logits

        def observe_router(hidden, weight):
            result = original_router(hidden, weight)
            if _ACTIVE_MLP is not None:
                _ROUTER_OBSERVATIONS[_ACTIVE_MLP] = {
                    "router_weight": weight.detach().cpu().clone(),
                    "router_logits": result.detach().cpu().clone(),
                    "router_compute_dtype": str(result.dtype),
                }
            return result

        model_module.fp32_router_logits = observe_router
        _ROUTER_HOOKED = True
    root = Path(config["root"])
    name = config["module"]
    installed = False
    saved = {}

    def before(module, args):
        global _ACTIVE_MLP
        del module
        if name.endswith(".mlp"):
            _ACTIVE_MLP = name
        control = json.loads((root / "control.json").read_text())
        if control["case"] == "prime":
            saved.clear()
            return
        value = input_tensor(args, {}, name)
        if value is not None:
            saved["input"] = value.detach().cpu().clone()
        saved["case"] = control["case"]
        if name.endswith(".experts") and len(args) > 1:
            topk = args[1]
            saved["topk_ids"] = topk.topk_ids.detach().cpu().clone()
            saved["topk_weights"] = topk.topk_weights.detach().cpu().clone()

    def after(module, args, output):
        nonlocal installed
        if not installed:
            module.register_forward_pre_hook(before)
            installed = True
        if "case" not in saved:
            return
        control = json.loads((root / "control.json").read_text())
        output = reference_substitution(name, saved, output, control, config.get("model"))
        value = tensor(output)
        if name == "logits_processor":
            value = output.next_token_logits
        if value is not None:
            saved["output"] = value.detach().cpu().clone()
        if name in _ROUTER_OBSERVATIONS:
            saved.update(_ROUTER_OBSERVATIONS.pop(name))
        target = root / "sglang" / saved["case"] / (("logits" if name == "logits_processor" else name) + ".pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        if control.get("record", True) or name == "logits_processor":
            torch.save(saved.copy(), target)
        saved.clear()
        return output

    return after


def hf_trace(model, ids, output, serving=None):
    collected = {}
    handles = []
    modules = dict(model.named_modules())
    selected = [name for name in names(model.config.num_hidden_layers) if name in modules]
    injected = []

    def before(name, module, args, kwargs):
        del module
        value = input_tensor(args, kwargs, name)
        top_level = (
            name in ["model.norm", "model.embed_norm"] or name.split("layers.")[-1].split(".", 1)[-1] in BOUNDARIES
        )
        if serving is not None and top_level and name in serving:
            replacement = serving[name]["input"].to(device=value.device, dtype=value.dtype).reshape_as(value)
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs, hidden_states=replacement)
            else:
                args = (replacement, *args[1:])
            value = replacement
            injected.append(name)
        collected[name] = {"input": value.detach().cpu().clone()}
        if name.endswith(".experts"):
            for key in ["topk_ids", "topk_weights"]:
                if key in kwargs:
                    collected[name][key] = kwargs[key].detach().cpu().clone()
        return args, kwargs

    def after(name, module, args, kwargs, result):
        del module, args, kwargs
        value = tensor(result)
        if value is not None:
            collected[name]["output"] = value.detach().cpu().clone()

    for name in selected:
        handles.append(
            modules[name].register_forward_pre_hook(lambda m, a, k, name=name: before(name, m, a, k), with_kwargs=True)
        )
        handles.append(
            modules[name].register_forward_hook(
                lambda m, a, k, o, name=name: after(name, m, a, k, o), with_kwargs=True
            )
        )
    try:
        with torch.no_grad():
            result = model(torch.tensor([ids], device="cuda"), use_cache=False)
        collected["logits"] = {"output": result.logits[0, -1].detach().cpu()}
    finally:
        for handle in handles:
            handle.remove()
    torch.save(collected, output)
    return collected, injected


def load_hf(source):
    model, info = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    return model.cuda().eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--ablations", default="", help="Comma-separated diagnostic HF substitutions")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    fla_compat.install_kda_triton_compat()
    cases = {"16": [5 + i % 30 for i in range(16)], "81": [5 + i % 30 for i in range(81)]}
    cases["83"] = cases["81"] + [26, 27]
    (root / "cases.json").write_text(json.dumps(cases))
    model = load_hf(args.model)
    layers = model.config.num_hidden_layers
    for case, ids in cases.items():
        hf_trace(model, ids, root / f"hf-{case}.pt")
        print("HF_TRACE", case, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    register()
    (root / "control.json").write_text(json.dumps({"case": "prime"}))
    engine = sglang.Engine(
        model_path=args.model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tp_size=1,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        attention_backend="triton",
        sampling_backend="pytorch",
        context_length=256,
        max_total_tokens=512,
        max_running_requests=1,
        max_mamba_cache_size=8,
        chunked_prefill_size=128,
        mem_fraction_static=args.memory_fraction,
        skip_server_warmup=True,
        forward_hooks=[
            {
                "name": name or "logits",
                "target_modules": [name],
                "hook_factory": "hero_numerics:trace_factory",
                "config": {"root": str(root), "module": name, "model": args.model},
            }
            for name in [*names(layers), "logits_processor"]
        ],
    )
    try:
        sampling = {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True}
        engine.generate(input_ids=cases["16"], sampling_params=sampling)
        for case, ids in cases.items():
            (root / "control.json").write_text(json.dumps({"case": case}))
            result = engine.generate(input_ids=ids, sampling_params=sampling, return_logprob=True, top_logprobs_num=20)
            (root / f"sglang-{case}.json").write_text(json.dumps(result))
            print("SGLANG_TRACE", case, flush=True)
        ablations = {}
        for mode in filter(None, args.ablations.split(",")):
            assert mode in ["experts", "mlp", "norms", "attention", "all", "all+head"]
            ablations[mode] = {}
            for case, ids in cases.items():
                label = f"{mode}-{case}"
                (root / "control.json").write_text(json.dumps({"case": label, "override": mode, "record": False}))
                result = engine.generate(
                    input_ids=ids, sampling_params=sampling, return_logprob=True, top_logprobs_num=20
                )
                observed = (
                    torch.load(root / "sglang" / label / "logits.pt", weights_only=True)["output"].float().reshape(-1)
                )
                reference = (
                    torch.load(root / f"hf-{case}.pt", weights_only=True)["logits"]["output"].float().reshape(-1)
                )
                indices = observed.topk(20).indices
                ablations[mode][case] = {
                    "logprobs": metric(observed.log_softmax(-1), reference.log_softmax(-1)),
                    "top20_logprobs": metric(observed.log_softmax(-1)[indices], reference.log_softmax(-1)[indices]),
                    "generated": result,
                }
                (root / "ablations.json").write_text(json.dumps(ablations, indent=2))
                print("ABLATION", mode, case, ablations[mode][case]["top20_logprobs"], flush=True)
    finally:
        engine.shutdown()
    model = load_hf(args.model)
    report = {"model": args.model, "gpu": torch.cuda.get_device_name(), "cases": {}}
    for case, ids in cases.items():
        serving = {p.stem: torch.load(p, weights_only=True) for p in (root / "sglang" / case).glob("*.pt")}
        baseline = torch.load(root / f"hf-{case}.pt", weights_only=True)
        replay, injected = hf_trace(model, ids, root / f"hf-replay-{case}.pt", serving)
        comparisons = {}
        for name in baseline.keys() & serving.keys():
            comparisons[name] = {
                kind: metric(serving[name][kind], baseline[name][kind])
                for kind in ["input", "output"]
                if kind in serving[name] and kind in baseline[name]
            }
            if name in injected:
                comparisons[name]["same_input_output"] = metric(serving[name]["output"], replay[name]["output"])
        router_checks = {}
        hf_modules = dict(model.named_modules())
        with torch.no_grad():
            for name, values in serving.items():
                if "router_weight" not in values:
                    continue
                router = hf_modules[name].router
                hidden = values["input"].cuda().reshape(1, -1, model.config.hidden_size)
                reference_logits = F.linear(hidden.float(), router.gate.weight.float()).cpu()
                expected_weights, expected_ids = router(hidden)
                actual = serving[name + ".experts"]
                original_ids = baseline[name + ".experts"]["topk_ids"].reshape_as(actual["topk_ids"])
                router_checks[name] = {
                    "sglang_weight_dtype": str(values["router_weight"].dtype),
                    "hf_weight_dtype": str(router.gate.weight.dtype),
                    "weights_exact": torch.equal(values["router_weight"], router.gate.weight.detach().cpu()),
                    "sglang_compute_dtype": values["router_compute_dtype"],
                    "same_input_logits": metric(values["router_logits"], reference_logits),
                    "same_input_topk_ids_exact": torch.equal(
                        actual["topk_ids"].long().reshape(-1), expected_ids.cpu().long().reshape(-1)
                    ),
                    "same_input_topk_weights": metric(actual["topk_weights"], expected_weights.cpu()),
                    "free_running_topk_slot_agreement": (actual["topk_ids"] == original_ids).float().mean().item(),
                }
        hf_logits = baseline["logits"]["output"].float().reshape(-1)
        sg_logits = serving["logits"]["output"].float().reshape(-1)
        indices = sg_logits.topk(20).indices
        report["cases"][case] = {
            "boundaries": comparisons,
            "router_checks": router_checks,
            "injected": injected,
            "logprobs": metric(sg_logits.log_softmax(-1), hf_logits.log_softmax(-1)),
            "top20_logprobs": metric(sg_logits.log_softmax(-1)[indices], hf_logits.log_softmax(-1)[indices]),
        }
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print("COMPARISON", case, report["cases"][case]["top20_logprobs"], flush=True)
    print("DIAGNOSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
