"""
Generate sequences with vLLM and extract logprobs.

This script generates sequences using vLLM in either bf16 or fp32 LM head mode,
saving the results to JSON for later scoring with HuggingFace.

Run this script TWICE (once per mode) via run_logprobs_comparison.sh to avoid
vLLM memory issues when reloading models in the same process.

Usage:
    # Generate bf16 sequences
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run python scripts/analysis/get_vllm_logprobs.py \
        --mode bf16 --output ~/dev/logprobs_data/vllm_bf16.json

    # Generate fp32 sequences
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run python scripts/analysis/get_vllm_logprobs.py \
        --mode fp32 --output ~/dev/logprobs_data/vllm_fp32.json

    # Or use the wrapper script:
    ./scripts/analysis/run_logprobs_comparison.sh --model hamishivi/qwen3_openthoughts2
"""
import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import torch
import transformers
import vllm
from vllm.inputs import TokensPrompt

from open_instruct.vllm_utils import patch_vllm_for_fp32_logits

# Disable vLLM multiprocessing so patches apply in-process
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


@dataclass
class Config:
    """Configuration for vLLM logprobs generation."""
    model: str = "hamishivi/qwen3_openthoughts2"
    max_tokens: int = 512
    max_model_len: int = 8192
    seed: int = 42
    dtype: str = "bfloat16"
    # Can use high GPU utilization since each mode runs in separate process
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    add_special_tokens: bool = False
    disable_tf32: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_hash(self) -> str:
        """Get a hash of config for cache validation."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "What is the weather today?",
    "What is machine learning?",
    "What is 42 * 27?",
    "Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$."
    "What is Reinforcement Learning with Verifiable Rewards?"
    "Write me 2000 words about the goldfish Bob."
]


def setup_precision(config: Config):
    """Configure precision settings."""
    if config.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        print("TF32 disabled for precise fp32 comparisons")


def setup_fp32_lm_head_cache(llm: vllm.LLM) -> None:
    """Set up fp32 LM head weight cache on vLLM model."""
    try:
        model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
        model = model_runner.model
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None:
            weight = getattr(lm_head, "weight", None)
            if isinstance(weight, torch.Tensor):
                lm_head._open_instruct_fp32_weight = weight.float()
                print(f"  Set up fp32 LM head cache: {weight.shape}")
                return
    except AttributeError as e:
        print(f"  Warning: Could not access model for fp32 cache: {e}")


def get_vllm_logprobs(
    model_name: str,
    prompts: List[str],
    config: Config,
    use_fp32_lm_head: bool = False,
) -> tuple[List[Dict], Any]:
    """
    Generate sequences with vLLM and extract logprobs.

    Returns:
        tuple of (results_list, tokenizer)
        Each result contains: prompt, query, response, logprobs, n_tokens
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set fp32 mode
    if use_fp32_lm_head:
        os.environ["OPEN_INSTRUCT_FP32_LM_HEAD"] = "1"
        patch_vllm_for_fp32_logits(True)
        print("  [vLLM] FP32 LM head enabled")
    else:
        os.environ.pop("OPEN_INSTRUCT_FP32_LM_HEAD", None)

    llm = vllm.LLM(
        model=model_name,
        seed=config.seed,
        enforce_eager=True,
        max_model_len=config.max_model_len,
        dtype=config.dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
    )

    if use_fp32_lm_head:
        setup_fp32_lm_head_cache(llm)

    # Explicit sampling params
    sampling_params = vllm.SamplingParams(
        max_tokens=config.max_tokens,
        logprobs=1,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        seed=config.seed,
    )

    all_results = []
    mode_str = "fp32" if use_fp32_lm_head else "bf16"

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] '{prompt[:50]}...'")

        # Explicit tokenization control
        query = tokenizer(prompt, add_special_tokens=config.add_special_tokens)["input_ids"]

        outputs = llm.generate([TokensPrompt(prompt_token_ids=query)], sampling_params=sampling_params)
        output = outputs[0]
        gen = output.outputs[0]

        # Use token_ids alignment instead of dict key order
        response = list(gen.token_ids)

        logprobs = []
        for t, d in zip(response, gen.logprobs):
            if d is None:
                raise RuntimeError(f"Missing logprobs dict at position for token {t}")
            lp_info = d.get(t, None)
            if lp_info is None:
                raise RuntimeError(
                    f"Missing sampled token {t} in logprobs dict. "
                    f"Keys (first 5): {list(d.keys())[:5]}"
                )
            logprobs.append(lp_info.logprob)

        all_results.append({
            "prompt": prompt,
            "query": query,
            "response": response,
            "logprobs": logprobs,
            "n_tokens": len(response),
        })
        print(f"    Generated {len(response)} tokens")

    # Cleanup
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return all_results, tokenizer


def save_results(
    output_path: Path,
    config: Config,
    mode: str,
    results: List[Dict],
    metadata: Dict[str, Any],
):
    """Save vLLM results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "config": config.to_dict(),
        "mode": mode,
        "metadata": metadata,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {mode} results to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate vLLM logprobs for one mode (bf16 or fp32)")
    parser.add_argument("--mode", type=str, required=True, choices=["bf16", "fp32"],
                        help="LM head precision mode")
    parser.add_argument("--model", type=str, default="hamishivi/qwen3_openthoughts2")
    parser.add_argument("--prompts", type=str, nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to GENERATE per response")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max CONTEXT LENGTH vLLM reserves (prompt + output)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="vLLM GPU memory fraction (can be high since separate process)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (for larger models)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true", help="Allow TF32 (less precise)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file path")
    args = parser.parse_args()

    config = Config(
        model=args.model,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        seed=args.seed,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        add_special_tokens=args.add_special_tokens,
        disable_tf32=not args.allow_tf32,
    )

    use_fp32 = args.mode == "fp32"

    print("=" * 60)
    print(f"vLLM Logprobs Generation ({args.mode.upper()} mode)")
    print(f"Model: {config.model}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"GPU memory utilization: {config.gpu_memory_utilization}")
    print(f"Prompts: {len(args.prompts)}")
    print("=" * 60)

    setup_precision(config)

    metadata = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "vllm_version": vllm.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results, tokenizer = get_vllm_logprobs(
        config.model, args.prompts, config, use_fp32_lm_head=use_fp32
    )

    save_results(Path(args.output), config, args.mode, results, metadata)

    # Print summary
    total_tokens = sum(r["n_tokens"] for r in results)
    print(f"\nTotal tokens generated: {total_tokens}")


if __name__ == "__main__":
    main()
