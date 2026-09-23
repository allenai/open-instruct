"""Compare actual Core scoring/training forwards, exported HF, and SGLang.

Diagnostic only. Optional serving arithmetic prototype; no optimizer update.
"""

import argparse
import gc
import hashlib
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path

import sglang
import torch
from olmo_core import config as core_config
from olmo_core.nn import attention
from olmo_core.nn.moe.v2 import olmo3, routed_experts
from olmo_sglang import register
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct.miles import fla_compat

BOUNDARIES = {
    "attention": ("self_attn", "attention"),
    "attention_pre_norm": ("pre_attention_layernorm", "attention_input_norm"),
    "attention_post_norm": ("post_attention_layernorm", "attention_norm"),
    "ffn_pre_norm": ("pre_feedforward_layernorm", "feed_forward_input_norm"),
    "ffn_post_norm": ("post_feedforward_layernorm", "feed_forward_norm"),
}


def metric(actual, reference):
    actual, reference = actual.float().reshape(-1), reference.float().reshape(-1)
    if actual.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} != {reference.shape}")
    delta = actual - reference
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
        "equal_fraction": (actual == reference).float().mean().item(),
    }


def release():
    gc.collect()
    torch.cuda.empty_cache()


def cpu_tensor(value):
    if isinstance(value, (tuple, list)):
        value = value[0]
    return value.detach().cpu().clone()


@contextmanager
def capture_block(block, backend):
    captures, handles = {}, []

    def attach(module, name):
        def before(_module, args, kwargs):
            captures[name + ".input"] = cpu_tensor(args[0] if args else kwargs["hidden_states"])

        def after(_module, _args, output):
            captures[name + ".output"] = cpu_tensor(output)

        handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(module.register_forward_hook(after))

    attach(block, "block")
    for label, names in BOUNDARIES.items():
        attach(getattr(block, names[backend == "core"]), label)
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()


def capture_logits_factory(config):
    root = Path(config["root"])

    def after(_module, _args, output):
        label = (root / "case.txt").read_text()
        target = root / f"{label}.pt"
        if not target.exists() and output.next_token_logits is not None:
            torch.save(cpu_tensor(output.next_token_logits).reshape(-1), target)

    return after


def prompts(model):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    cases = {str(n): [5 + i % 30 for i in range(n)] for n in [16, 81]}
    cases["83"] = cases["81"] + [26, 27]
    for name, question in {
        "math": "A shop sold 18 notebooks on Monday and twice as many on Tuesday. Each costs $3. What was the total revenue? Explain your calculation.",
        "code": "Write a Python function that returns the first non-repeating character in a string, or None if every character repeats. Explain its time complexity.",
    }.items():
        cases[name] = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if not isinstance(cases[name], list) or not all(isinstance(token, int) for token in cases[name]):
            raise TypeError(f"Expected flat token IDs for {name}, got {type(cases[name])}")
    return cases


def serving(args, cases):
    register()
    observations = {}
    for backend in ["torch_native"] if args.compatibility_probe else ["triton", "torch_native"]:
        root = args.output / backend
        root.mkdir()
        (root / "case.txt").write_text("warmup")
        extra_hooks = []
        if args.compatibility_probe:
            modules = [f"model.layers.{i}.mlp.experts" for i in range(1, 16)]
            modules += ["model.embed_norm", "model.norm"] + [
                f"model.layers.{i}.{suffix}"
                for i in range(16)
                for suffix in [
                    "pre_attention_layernorm",
                    "post_attention_layernorm",
                    "pre_feedforward_layernorm",
                    "post_feedforward_layernorm",
                ]
            ]
            extra_hooks = [
                {
                    "name": name,
                    "target_modules": [name],
                    "hook_factory": "hero_serving_arithmetic:compatibility_factory",
                    "config": {"root": str(root), "module": name},
                }
                for name in modules
            ]
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
            attention_backend=backend,
            sampling_backend="pytorch",
            context_length=1024,
            max_total_tokens=2048,
            max_running_requests=1,
            max_mamba_cache_size=8,
            chunked_prefill_size=1024,
            mem_fraction_static=0.6,
            skip_server_warmup=True,
            forward_hooks=extra_hooks
            + [
                {
                    "name": "capture_logits",
                    "target_modules": ["logits_processor"],
                    "hook_factory": "hero_core_fidelity:capture_logits_factory",
                    "config": {"root": str(root)},
                }
            ],
        )
        try:
            modes = ["original", "rounded_experts", "rounded_experts_norms"] if args.compatibility_probe else [""]
            if args.compatibility_probe:
                engine.generate(input_ids=cases["16"], sampling_params={"temperature": 0, "max_new_tokens": 1})
            for mode in modes:
                label = f"{backend}/{mode}" if mode else backend
                target = args.output / label
                target.mkdir(exist_ok=True)
                observations[label] = {}
                for name, ids in list(cases.items()):
                    if name not in ["16", "81", "83", "math", "code"]:
                        continue
                    (root / "case.txt").write_text(f"{mode}/{name}" if mode else name)
                    result = engine.generate(
                        input_ids=ids,
                        sampling_params={"temperature": 0, "max_new_tokens": 4, "ignore_eos": True},
                        return_logprob=True,
                        top_logprobs_num=20,
                    )
                    observations[label][name] = result
                    cases[f"{label}-{name}-rollout"] = ids + result["output_ids"]
                    print("SERVING", label, name, result["output_ids"], flush=True)
                (target / "generations.json").write_text(json.dumps(observations[label], indent=2))
        finally:
            engine.shutdown()
        release()
    return observations


def summarize_logits(logits, ids):
    logits = logits[0].float()
    target = torch.tensor(ids[1:], device=logits.device)
    selected = logits[:-1].gather(-1, target[:, None]).squeeze(-1) - logits[:-1].logsumexp(-1)
    return {"last": logits[-1].detach().cpu(), "selected": selected.detach().cpu()}


def distribution_comparison(actual, reference):
    a, b = actual.float().log_softmax(-1), reference.float().log_softmax(-1)
    indices = b.topk(20).indices
    return {
        "full_vocab_logprobs": metric(a, b),
        "reference_top20_logprobs": metric(a[indices], b[indices]),
        "kl_reference_to_actual": (b.exp() * (b - a)).sum().item(),
        "greedy_equal": a.argmax().item() == b.argmax().item(),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fla_compat.install_kda_triton_compat()
    cases = prompts(args.model)
    serving_results = serving(args, cases)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2))
    records, captures = {}, {}
    model, info = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    model = model.cuda().eval()
    hf_config = model.config
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    for profile, attn, layout in [
        ("hf_eager", "eager", False),
        ("hf_sdpa", "sdpa", False),
        ("hf_core_layout_sdpa", "sdpa", True),
    ]:
        os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "1" if layout else "0"
        model.config._attn_implementation = attn
        records[profile], captures[profile] = {}, {}
        with torch.no_grad():
            for name, ids in cases.items():
                with ExitStack() as stack:
                    layer_values = (
                        [stack.enter_context(capture_block(b, "hf")) for b in model.model.layers]
                        if name in ["16", "81"]
                        else []
                    )
                    logits = model(torch.tensor([ids], device="cuda"), use_cache=False).logits
                    records[profile][name] = summarize_logits(logits, ids)
                    captures[profile][name] = layer_values
                    del logits
                print("HF_FORWARD", profile, name, flush=True)
        torch.save(records, args.output / "logits-and-scores.pt")
    del model
    release()
    os.environ["OLMO_HF_MOE_CORE_REFERENCE"] = "0"
    torch.distributed.init_process_group(
        backend="nccl", init_method=f"file://{args.output / 'rendezvous'}", rank=0, world_size=1
    )
    config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf_config,
        dtype=core_config.DType.bfloat16,
        attention_backend=attention.AttentionBackendName.torch,
        router_aux_loss_weight=0.0,
        router_z_loss_weight=0.0,
    )
    config.recompute_each_block = False
    model = config.build(init_device="meta")
    longest = max(map(len, cases.values()))
    model.init_weights(max_seq_len=longest, max_local_microbatch_size=longest, device=torch.device("cuda"))
    olmo3.load_olmo3_moe_hf_state(model, hf_config, state)
    for block in model.routed_blocks():
        block.routed_experts_router.set_load_balancing_process_group(torch.distributed.group.WORLD)
    del state
    report = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(),
        "diagnosis_only": True,
        "core_grouped_mm": routed_experts.use_torch_grouped_mm(),
        "comparisons": {},
        "layer_comparisons": {},
        "source_sha256": {
            m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [olmo3, routed_experts]
        },
    }
    for profile, grad in [("core_scoring", False), ("core_grad_forward", True)]:
        model.train(grad)
        records[profile], captures[profile] = {}, {}
        with torch.set_grad_enabled(grad):
            for name, ids in cases.items():
                with ExitStack() as stack:
                    layer_values = (
                        [stack.enter_context(capture_block(b, "core")) for b in model.blocks.values()]
                        if name in ["16", "81"]
                        else []
                    )
                    logits = model(torch.tensor([ids], device="cuda"))
                    records[profile][name] = summarize_logits(logits, ids)
                    captures[profile][name] = layer_values
                    del logits
                print("CORE_FORWARD", profile, name, flush=True)
        torch.save(records, args.output / "logits-and-scores.pt")
    with torch.no_grad():
        model.eval()
        for name in ["16", "81"]:
            report["layer_comparisons"][name] = {}
            for profile in ["hf_eager", "hf_core_layout_sdpa"]:
                report["layer_comparisons"][name][profile] = []
                for index, block in enumerate(model.blocks.values()):
                    reference = captures[profile][name][index]
                    with capture_block(block, "core") as isolated:
                        block(reference["block.input"].cuda())
                    report["layer_comparisons"][name][profile].append(
                        {
                            "layer": index,
                            "same_block_input": {k: metric(isolated[k], v) for k, v in reference.items()},
                            "accumulated": {
                                k: metric(captures["core_scoring"][name][index][k], v) for k, v in reference.items()
                            },
                        }
                    )
    for profile, values in records.items():
        if profile == "core_scoring":
            continue
        report["comparisons"][profile] = {
            name: {
                **distribution_comparison(value["last"], records["core_scoring"][name]["last"]),
                "selected_token_logprobs": metric(value["selected"], records["core_scoring"][name]["selected"]),
            }
            for name, value in values.items()
        }
    report["serving_comparisons"] = {}
    for backend, values in serving_results.items():
        report["serving_comparisons"][backend] = {}
        for name, result in values.items():
            sg = torch.load(args.output / backend / f"{name}.pt", weights_only=True)
            rollout = records["core_scoring"][f"{backend}-{name}-rollout"]["selected"]
            behavior = torch.tensor([v[0] for v in result["meta_info"]["output_token_logprobs"]])
            scored = rollout[len(cases[name]) - 1 :]
            report["serving_comparisons"][backend][name] = {
                **distribution_comparison(sg, records["core_scoring"][name]["last"]),
                "rollout_logprobs": behavior.tolist(),
                "core_logprobs": scored.tolist(),
                "rollout_vs_core": metric(behavior, scored),
                "importance_ratio_core_over_serving": (scored - behavior).exp().tolist(),
            }
    torch.save(records, args.output / "logits-and-scores.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    torch.distributed.destroy_process_group()
    print(
        "FIDELITY_COMPLETE",
        json.dumps(
            {
                k: max(v["reference_top20_logprobs"]["max_abs"] for v in cases.values())
                for k, cases in report["comparisons"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compatibility-probe", action="store_true")
    run(parser.parse_args())
