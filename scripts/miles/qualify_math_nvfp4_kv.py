"""Inference-only native NVFP4 expert/KV gate on the study's pinned runtime."""

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path


def tensor_info(value):
    if isinstance(value, (list, tuple)):
        return [tensor_info(item) for item in value]
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.numel() * value.element_size()}
    return None


def backend_info(backend, seen=None):
    seen = set() if seen is None else seen
    if id(backend) in seen:
        return None
    seen.add(id(backend))
    result = {"class": type(backend).__name__}
    for key in ("prefill_uses_native_fp4", "decode_uses_native_fp4", "uses_trtllm_gen_native_fp4"):
        if hasattr(backend, key):
            result[key] = bool(getattr(backend, key))
    for key, value in vars(backend).items():
        if "backend" in key and value is not None and hasattr(value, "forward_extend"):
            result[key] = backend_info(value, seen)
    return result


def audited_scheduler(*args, **kwargs):
    runner_type = importlib.import_module("sglang.srt.model_executor.model_runner").ModelRunner
    original = runner_type.__init__

    def init(runner, *init_args, **init_kwargs):
        original(runner, *init_args, **init_kwargs)
        pool = runner.token_to_kv_pool
        full = pool.full_kv_pool
        state = pool.mamba_pool.mamba_cache
        modules = []
        for name, module in runner.model.named_modules():
            method = getattr(module, "quant_method", None)
            if method is not None or name.endswith("lm_head"):
                modules.append(
                    {
                        "name": name,
                        "class": type(module).__name__,
                        "quant_method": type(method).__name__,
                        "parameters": {
                            key: tensor_info(value) for key, value in module.named_parameters(recurse=False)
                        },
                    }
                )
        audit = {
            "pool": type(pool).__name__,
            "full_pool": type(full).__name__,
            "kv_dtype": str(full.dtype),
            "quant_method": type(full.quant_method).__name__,
            "quant_name": full.quant_method.name,
            "full_layer_count": pool.full_layer_nums,
            "buffers": {
                key: tensor_info(getattr(full, key, None))
                for key in ("k_buffer", "v_buffer", "native_k_scale_buffer", "native_v_scale_buffer")
            },
            "recurrent_temporal": tensor_info(state.temporal),
            "recurrent_conv": tensor_info(state.conv),
            "backends": backend_info(runner.attn_backend),
            "modules": modules,
        }
        Path(os.environ["OI_KV_GATE_OUTPUT"], f"pool-audit-{os.getpid()}.json").write_text(
            json.dumps(audit, indent=2) + "\n"
        )

    runner_type.__init__ = init
    return importlib.import_module("sglang.srt.managers.scheduler").run_scheduler_process(*args, **kwargs)


def verify_audit(audit):
    if (
        audit["pool"] != "HybridLinearKVPool"
        or audit["full_layer_count"] != 2
        or audit["quant_name"] != "nvfp4"
        or audit["kv_dtype"] != "torch.float4_e2m1fn_x2"
    ):
        raise ValueError("Unexpected hybrid KV pool/recipe/layers")
    for key in ("k_buffer", "v_buffer", "native_k_scale_buffer", "native_v_scale_buffer"):
        if not audit["buffers"][key]:
            raise ValueError("Missing physical packed KV/scales")
    for key in ("k_buffer", "v_buffer"):
        if any(item["dtype"] != "torch.uint8" for item in audit["buffers"][key]):
            raise ValueError("Native packed KV storage must be uint8")
    if audit["recurrent_temporal"]["dtype"] != "torch.float32":
        raise ValueError("Recurrent temporal precision changed")
    if '"uses_trtllm_gen_native_fp4": true' not in json.dumps(audit["backends"]):
        raise ValueError("Native GenMHA FP4 backend not active")
    experts = [module for module in audit["modules"] if module["name"].endswith(".experts")]
    if len(experts) != 15 or any("fp4" not in module["quant_method"].lower() for module in experts):
        raise ValueError("Expected 15 NVFP4 routed expert modules")
    for module in audit["modules"]:
        if module in experts or module["class"] == "RadixAttention":
            continue
        if "fp4" in module["quant_method"].lower() or "fp8" in module["quant_method"].lower():
            raise ValueError(f"Quantized non-expert weight module: {module['name']}")
    head = [module for module in audit["modules"] if module["name"].endswith("lm_head")]
    if len(head) != 1 or head[0]["parameters"]["weight"]["dtype"] != "torch.bfloat16":
        raise ValueError("Head weights must remain BF16")


def verify_outputs(outputs, count):
    if len(outputs) != count:
        raise ValueError("Incomplete short generation")
    for output in outputs:
        scores = output["meta_info"].get("output_token_logprobs", [])
        if not scores or any(not math.isfinite(score[0]) for score in scores):
            raise ValueError("Missing/nonfinite output logprobs")
        candidates = output["meta_info"].get("output_top_logprobs", [])
        if not candidates or any(
            len(row) != 20 or any(not math.isfinite(item[0]) for item in row) for row in candidates
        ):
            raise ValueError("Missing/nonfinite top20 candidates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    options = json.loads(args.options.read_text())
    os.environ["OLMO_SGLANG_CORE_COMPAT"] = "off"
    os.environ["OI_KV_GATE_OUTPUT"] = str(args.output)
    importlib.import_module("olmo_sglang").register()
    server_args_type = importlib.import_module("sglang.srt.server_args").ServerArgs
    native = server_args_type(**options)
    native.resolve_once()
    resolved = native.resolved_dict()
    (args.output / "resolved-options.json").write_text(json.dumps(resolved, default=str, indent=2) + "\n")
    for key in ("prefill_attention_backend", "decode_attention_backend"):
        if resolved[key] != "trtllm_mha":
            raise ValueError(f"Unexpected resolved backend: {key}")
    assert resolved["enable_fp32_lm_head"] and resolved["page_size"] == 16
    engine_type = importlib.import_module("sglang.srt.entrypoints.engine").Engine
    engine_type.run_scheduler_process_func = staticmethod(audited_scheduler)
    engine = engine_type(**options)
    try:
        tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(options["model_path"])
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Compute 6 times 7. Reason step by step and put your answer within \\boxed{}.",
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
        (args.output / "short-outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
        verify_outputs(outputs, 16)
        audits = list(args.output.glob("pool-audit-*.json"))
        if len(audits) != 1:
            raise ValueError("Expected one TP1 pool audit")
        verify_audit(json.loads(audits[0].read_text()))
        engine.flush_cache()
        # Exact-length synthetic probe is isolated from accuracy samples.
        sentence = tokenizer.encode("Solve the following mathematics problem step by step. ", add_special_tokens=False)
        tokens = (sentence * (2048 // len(sentence) + 1))[:2048]
        outputs = engine.generate(
            input_ids=[tokens],
            sampling_params={"temperature": 1, "top_p": 0.95, "max_new_tokens": 16384, "ignore_eos": True},
            return_logprob=True,
        )
        (args.output / "length-output.json").write_text(json.dumps(outputs, indent=2) + "\n")
        if len(outputs) != 1 or outputs[0]["meta_info"]["completion_tokens"] != 16384:
            raise ValueError("Incomplete exact 16K generation")
        scores = outputs[0]["meta_info"]["output_token_logprobs"]
        if len(scores) != 16384 or any(not math.isfinite(score[0]) for score in scores):
            raise ValueError("Missing/nonfinite 16K logprobs")
        (args.output / "qualified.json").write_text(
            json.dumps(
                {
                    "evaluation": True,
                    "accuracy_validation": False,
                    "options": options,
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "short_requests": 16,
                    "exact_output_tokens": 16384,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
