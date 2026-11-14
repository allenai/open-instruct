"""Qualify Olmo expert quantization kernels and inspect actual loaded precisions.

This GPU-only gate is not an accuracy benchmark. It records the serving layer
inventory and deterministic generations before any longer AIME/RL submission.
"""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

FP8_EXCLUSIONS = ("self_attn", "shared_expert", "latent_down_proj", "latent_up_proj", "model.layers.0.mlp", "lm_head")


def audited_scheduler(*args, **kwargs):
    """Instrument one model forward for a read-only inventory in each GPU worker."""
    model_module = importlib.import_module("olmo_sglang.models.olmo3_moe")
    model_type = model_module.Olmo3MoeForCausalLM
    original = model_type.forward

    def forward(model, *forward_args, **forward_kwargs):
        if not getattr(model, "_study_audited", False):
            modules = []
            for name, module in model.named_modules():
                method = getattr(module, "quant_method", None)
                if method is not None:
                    modules.append(
                        {
                            "name": name,
                            "class": type(module).__name__,
                            "quant_method": type(method).__name__,
                            "parameters": {
                                key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                                for key, value in module.named_parameters(recurse=False)
                            },
                        }
                    )
            destination = Path(os.environ["OI_QUANT_AUDIT_DIR"])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"layers-{os.getpid()}.json").write_text(json.dumps(modules, indent=2) + "\n")
            model._study_audited = True
        return original(model, *forward_args, **forward_kwargs)

    model_type.forward = forward
    scheduler = importlib.import_module("sglang.srt.managers.scheduler")
    return scheduler.run_scheduler_process(*args, **kwargs)


def verify_inventory(modules, precision):
    experts = [module for module in modules if module["name"].endswith(".experts")]
    if len(experts) != 15:
        raise ValueError(f"Expected 15 routed expert modules, found {len(experts)}")
    for module in modules:
        name, method = module["name"], module["quant_method"].lower()
        low_precision = "fp8" in method or "fp4" in method
        if not name.endswith(".experts") and low_precision:
            raise ValueError(f"Unexpected quantized non-expert module: {name}: {method}")
    for module in experts:
        method = module["quant_method"].lower()
        if precision == "fp8" and "fp8" not in method:
            raise ValueError(f"FP8 expert not quantized: {module['name']}")
        if precision == "nvfp4" and "fp4" not in method:
            raise ValueError(f"NVFP4 expert not quantized: {module['name']}")
        if precision.startswith("bf16") and ("fp8" in method or "fp4" in method):
            raise ValueError("BF16 control contains quantized experts")
    return {"routed_expert_modules": len(experts), "audited_modules": len(modules)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", required=True, choices=("bf16", "bf16-control", "fp8", "nvfp4"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["OI_QUANT_AUDIT_DIR"] = str(args.output.resolve())
    os.environ["OLMO_SGLANG_CORE_COMPAT"] = "auto" if args.precision == "bf16" else "off"
    if args.precision == "fp8":
        os.environ["SGLANG_FP8_IGNORED_LAYERS"] = ",".join(FP8_EXCLUSIONS)
    registration = importlib.import_module("olmo_sglang")
    registration.register()
    engine_type = importlib.import_module("sglang.srt.entrypoints.engine").Engine
    engine_type.run_scheduler_process_func = staticmethod(audited_scheduler)
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(args.model)
    options = dict(
        model_path=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tp_size=1,
        context_length=18432,
        max_running_requests=16,
        disable_radix_cache=True,
        max_total_tokens=393216,
        max_mamba_cache_size=128,
        mem_fraction_static=0.7,
        cuda_graph_backend_decode="full",
        cuda_graph_max_bs_decode=64,
        cuda_graph_backend_prefill="disabled",
        chunked_prefill_size=4096,
        attention_backend="triton",
        sampling_backend="pytorch",
        random_seed=17,
        enable_fp32_lm_head=True,
        kv_cache_dtype="bf16",
    )
    if args.precision == "fp8":
        options.update(quantization="fp8", moe_runner_backend="triton")
    elif args.precision == "nvfp4":
        options.update(quantization="nvfp4_online", moe_runner_backend="flashinfer_trtllm_routed")
    (args.output / "options.json").write_text(json.dumps(options, indent=2) + "\n")
    start = time.monotonic()
    engine = engine_type(**options)
    try:
        load_seconds = time.monotonic() - start
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Compute 6 times 7.\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = engine.generate(
            [prompt] * 16,
            sampling_params={"temperature": 0, "max_new_tokens": 64},
            return_logprob=True,
            top_logprobs_num=20,
        )
        (args.output / "outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
        inventories = list(args.output.glob("layers-*.json"))
        if len(inventories) != 1:
            raise ValueError(f"Expected one TP1 inventory, got {len(inventories)}")
        audit = verify_inventory(json.loads(inventories[0].read_text()), args.precision)
        if len(outputs) != 16 or any(not output.get("text") for output in outputs):
            raise ValueError("Incomplete generation smoke")
        (args.output / "qualified.json").write_text(
            json.dumps(
                {
                    "scope": "load, layer precision, 16 concurrent short generations; not full-length or refresh qualification",
                    "precision": args.precision,
                    "load_seconds": load_seconds,
                    **audit,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
