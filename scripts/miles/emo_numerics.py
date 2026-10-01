"""Compare a tiny EMO checkpoint across cached serving, full prefill and Core.

Diagnostic only: retain every score and report differences without relaxing the
RL acceptance threshold. Run in the pinned MILES image with one CUDA device.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from miles.backends.core_utils import moe_models
from olmo_core.nn.moe.v2 import olmo3
from olmo_core.nn.moe.v2.hf import configuration_olmo3moe, modeling_olmo3moe
from olmo_sglang import register
from safetensors.torch import load_file
from sglang import Engine
from sglang.srt.models import registry

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.execution import emo


def difference(left, right):
    a, b = torch.as_tensor(left), torch.as_tensor(right)
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Expected aligned finite scores")
    error = (a - b).abs()
    return {"mean_abs": error.mean().item(), "max_abs": error.max().item(), "signed_mean": (a - b).mean().item()}


def selected_scores(logits, ids, prompt_length):
    logits = logits[0, prompt_length - 1 : -1].float()
    return logits.log_softmax(-1).gather(-1, ids[0, prompt_length:, None]).flatten().cpu().tolist()


def serving(path, trace_path=None, token_record=None):
    register()
    if trace_path is not None:
        os.environ["EMO_NUMERICS_TRACE"] = str(trace_path)
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "scripts.miles.emo_trace_models"
        registry.ModelRegistry.register("scripts.miles.emo_trace_models", overwrite=True, strict=True)
    engine = Engine(
        model_path=str(path),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tp_size=1,
        disable_radix_cache=True,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        attention_backend="torch_native",
        sampling_backend="pytorch",
        context_length=512,
        max_total_tokens=4096,
        mem_fraction_static=0.15,
        skip_server_warmup=True,
    )
    prompt = [1] + [3] * 14
    try:
        if token_record is None:
            generated = engine.generate(
                input_ids=prompt,
                sampling_params={"temperature": 1.0, "max_new_tokens": 32, "ignore_eos": True},
                return_logprob=True,
            )
            entries = generated["meta_info"]["output_token_logprobs"]
            tokens = prompt + [entry[1] for entry in entries]
        else:
            tokens = token_record["tokens"]
            prompt = tokens[: token_record["prompt_length"]]
        rescored = engine.generate(
            input_ids=tokens,
            sampling_params={"temperature": 1.0, "max_new_tokens": 1},
            return_logprob=True,
            logprob_start_len=0,
        )
        input_entries = rescored["meta_info"]["input_token_logprobs"]
        if [entry[1] for entry in input_entries] != tokens:
            raise ValueError("Serving prefill scores do not align with supplied token IDs")
        record = dict(
            tokens=tokens,
            prompt_length=len(prompt),
            serving_prefill=[entry[0] for entry in input_entries[len(prompt) :]],
        )
        if token_record is None:
            record["serving_decode"] = [entry[0] for entry in entries]
        return record
    finally:
        engine.shutdown()


def route_difference(actual, weights, ids, logits):
    """Compare mixing by expert identity, independent of top-k slot ordering."""
    count = logits.shape[-1]
    expected_ids = ids.detach().cpu().reshape_as(actual["ids"]).long()
    expected_weights = weights.detach().cpu().reshape_as(actual["weights"])
    dense_actual = torch.zeros(actual["ids"].shape[0], count).scatter_(1, actual["ids"].long(), actual["weights"])
    dense_expected = torch.zeros_like(dense_actual).scatter_(1, expected_ids, expected_weights)
    return {
        "logits": difference(actual["logits"].float(), logits.detach().cpu().reshape_as(actual["logits"]).float()),
        "mixing_by_expert": difference(dense_actual, dense_expected),
        "expert_set_agreement": float(
            (actual["ids"].sort(-1).values == expected_ids.sort(-1).values).all(-1).float().mean()
        ),
        "serving_logits_dtype": str(actual["logits"].dtype),
        "serving_weights_dtype": str(actual["weights"].dtype),
        "reference_logits_dtype": str(logits.dtype),
    }


def trace_comparisons(reference, tokens, trace_path, native=None):
    """Compare actual common layer outputs and every loaded parameter on CPU."""
    serving_trace = torch.load(trace_path, map_location="cpu", weights_only=True)
    if not torch.equal(serving_trace["tokens"].flatten(), tokens.cpu().flatten()):
        raise ValueError("Layer trace token IDs do not match the numerical probe")
    outputs, handles = {}, []

    def capture(name):
        def hook(module, args, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach().cpu().clone()

        return hook

    for name, module in reference.named_modules():
        if name:
            handles.append(module.register_forward_hook(capture(name)))
    try:
        with torch.no_grad():
            reference(tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    # SGLang combines Q/K/V and gate/up GEMMs. Reconstruct their HF outputs.
    for name in list(outputs):
        for source, target, parts in (
            ("q_proj", "qkv_proj", ("q_proj", "k_proj", "v_proj")),
            ("gate_proj", "gate_up_proj", ("gate_proj", "up_proj")),
        ):
            if name.endswith("." + source):
                prefix = name[: -len(source)]
                outputs[prefix + target] = torch.cat([outputs[prefix + part] for part in parts], dim=-1)
    comparisons = {}
    for name, actual in serving_trace["outputs"].items():
        expected = outputs.get(name)
        if expected is not None and expected.numel() == actual.numel():
            comparisons[name] = difference(actual.float(), expected.reshape_as(actual).float())
    parameters = reference.state_dict()
    weight_comparisons = {}
    for name, actual in serving_trace["parameters"].items():
        if name.endswith("experts.w13_weight"):
            prefix = name.removesuffix("w13_weight")
            expected = torch.stack(
                [
                    torch.cat([parameters[f"{prefix}{i}.{part}.weight"] for part in ("gate_proj", "up_proj")])
                    for i in range(actual.shape[0])
                ]
            )
        elif name.endswith("experts.w2_weight"):
            prefix = name.removesuffix("w2_weight")
            expected = torch.stack([parameters[f"{prefix}{i}.down_proj.weight"] for i in range(actual.shape[0])])
        elif name.endswith("qkv_proj.weight"):
            prefix = name.removesuffix("qkv_proj.weight")
            expected = torch.cat([parameters[prefix + part + ".weight"] for part in ("q_proj", "k_proj", "v_proj")])
        elif name.endswith("gate_up_proj.weight"):
            prefix = name.removesuffix("gate_up_proj.weight")
            expected = torch.cat([parameters[prefix + part + ".weight"] for part in ("gate_proj", "up_proj")])
        else:
            expected = parameters[name]
        weight_comparisons[name] = difference(actual.float(), expected.detach().cpu().float())
    routers = {}
    for name, actual in serving_trace.get("routers", {}).items():
        # First compare routing on identical *serving* hidden states, isolating
        # the router from errors already present in upstream attention/norms.
        hidden = serving_trace["inputs"][name].to(tokens.device).unsqueeze(0)
        layer = int(name.split(".")[2])
        router = reference.model.layers[layer].mlp.router
        with torch.no_grad():
            weights, ids = router(hidden)
            logits = torch.nn.functional.linear(hidden.float(), router.gate.weight.float())
            routers[name] = {"hf_on_serving_input": route_difference(actual, weights, ids, logits)}
            if native is not None:
                core_router = native.blocks[str(layer)].routed_experts_router
                weights, ids, _, aux = core_router(hidden, scores_only=False)
                routers[name]["core_emo_on_serving_input"] = route_difference(actual, weights, ids, aux[1])
            mlp = reference.model.layers[layer].mlp
            if mlp.latent_down_proj is not None:
                raise ValueError("Isolated expert comparison currently requires the full-width tiny fixture")
            routed = mlp.experts(
                hidden.flatten(0, 1),
                topk_ids=actual["ids"].to(tokens.device).long(),
                topk_weights=actual["weights"].to(tokens.device),
            )
            routers[name]["experts_on_serving_input_and_routes"] = difference(
                serving_trace["outputs"][name + ".experts"].float(), routed.cpu().float()
            )
            if mlp.shared_expert is not None:
                shared = mlp.shared_expert(hidden).flatten(0, 1)
                routers[name]["shared_expert_on_serving_input"] = difference(
                    serving_trace["outputs"][name + ".shared_expert"].float(), shared.cpu().float()
                )
        routers[name]["serving"] = {key: value.tolist() for key, value in actual.items()}
    return {"layers": comparisons, "parameters": weight_comparisons, "routers": routers}


def models(path, record, trace_path=None):
    hf = configuration_olmo3moe.Olmo3MoeConfig.from_pretrained(path)
    options = CoreConfig(router_aux_loss_weight=0, router_z_loss_weight=0, activation_checkpointing=False)
    config = moe_models.model_config_from_hf(hf, options)
    moe_models.prepare_model_config(config, hf, options)
    model = config.build(init_device="cuda").eval()
    state = load_file(str(path / "model.safetensors"))
    olmo3.load_olmo3_moe_hf_state(model, hf, state)
    reference = modeling_olmo3moe.Olmo3MoeForCausalLM(hf).to(device="cuda", dtype=torch.bfloat16).eval()
    reference.load_state_dict(state)
    tokens = torch.tensor([record["tokens"]], device="cuda")
    with torch.no_grad():
        native_logits = model(tokens)
        hf_logits = reference(tokens, use_cache=False).logits
        record["core"] = selected_scores(native_logits, tokens, record["prompt_length"])
        record["hf_reference"] = selected_scores(hf_logits, tokens, record["prompt_length"])
        record["core_hf_logits"] = difference(native_logits.float().cpu(), hf_logits.float().cpu())
    record["comparisons"] = {
        f"{left}_vs_{right}": difference(record[left], record[right])
        for left, right in (
            ("serving_decode", "serving_prefill"),
            ("core", "serving_prefill"),
            ("hf_reference", "serving_prefill"),
            ("core", "hf_reference"),
        )
        if left in record and right in record
    }
    if trace_path is not None:
        record["trace"] = trace_comparisons(reference, tokens, trace_path, model)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens-from", type=Path, help="Rescore the exact tokens in a previous diagnostic JSON")
    parser.add_argument(
        "--non-emo-control", action="store_true", help="Repeat in a fresh process with EMO metadata removed"
    )
    parser.add_argument(
        "--ordinary", action="store_true", help="Require an ordinary non-EMO checkpoint for the control"
    )
    parser.add_argument(
        "--trace", action="store_true", help="Capture actual serving layers and compare to HF (tiny only)"
    )
    args = parser.parse_args()
    metadata = json.loads((args.checkpoint / "config.json").read_text())
    if args.ordinary and any(value is not None for key, value in metadata.items() if key.startswith("emo_")):
        raise ValueError("Ordinary control must have no active EMO metadata")
    if not args.ordinary and emo.resolve_hf(metadata, "full_pool") != metadata:
        raise ValueError("Use the explicitly prepared full-pool checkpoint")
    trace_path = args.output.with_suffix(".pt") if args.trace else None
    token_record = json.loads(args.tokens_from.read_text()) if args.tokens_from else None
    record = serving(args.checkpoint, trace_path, token_record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    record = models(args.checkpoint, record, trace_path)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record["comparisons"], indent=2))
    if args.non_emo_control:
        control = args.output.parent / "non-emo-checkpoint"
        control.mkdir()
        for source in args.checkpoint.iterdir():
            if source.name != "config.json":
                (control / source.name).symlink_to(source.resolve())
        (control / "config.json").write_text(
            json.dumps({k: v for k, v in metadata.items() if not k.startswith("emo_")})
        )
        control_output = args.output.with_name("non-emo-control.json")
        env = dict(os.environ, SGLANG_EXTERNAL_MODEL_PACKAGE="olmo_sglang.models")
        command = [
            sys.executable,
            "-m",
            "scripts.miles.emo_numerics",
            str(control),
            str(control_output),
            "--ordinary",
            "--tokens-from",
            str(args.output),
        ]
        if args.trace:
            command.append("--trace")
        subprocess.run(command, env=env, check=True)
        ordinary = json.loads(control_output.read_text())
        record["emo_vs_non_emo"] = {
            name: difference(record[name], ordinary[name]) for name in ("core", "hf_reference", "serving_prefill")
        }
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(json.dumps(record["emo_vs_non_emo"], indent=2))


if __name__ == "__main__":
    main()
