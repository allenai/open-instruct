"""
Generate sequences with vLLM and extract logprobs.

This script generates sequences using vLLM in either bf16 or fp32 LM head mode,
saving the results to JSON for later scoring with HuggingFace.

Run this script TWICE (once per mode) via run_logprobs_comparison.sh to avoid
vLLM memory issues when reloading models in the same process.

Usage:
    # Generate bf16 sequences
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run python scripts/analysis/fp32-lm-head/get_vllm_logprobs.py \
        --mode bf16 --output ~/dev/logprobs_data/vllm_bf16.json

    # Generate fp32 sequences
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run python scripts/analysis/fp32-lm-head/get_vllm_logprobs.py \
        --mode fp32 --output ~/dev/logprobs_data/vllm_fp32.json

    # Or use the wrapper script:
    ./scripts/analysis/fp32-lm-head/run_logprobs_comparison.sh --model hamishivi/qwen3_openthoughts2
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import transformers
import vllm
from vllm.inputs import TokensPrompt

# Note: For TP>1, workers are separate processes. We patch LogitsProcessor
# inside each worker via apply_model (see fp32_worker_patch.py).
# Add this directory to PYTHONPATH so workers (child processes) can import the patch module.
_script_dir = str(Path(__file__).parent.resolve())
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
# Also set PYTHONPATH for child processes (vLLM workers)
_pythonpath = os.environ.get("PYTHONPATH", "")
if _script_dir not in _pythonpath:
    os.environ["PYTHONPATH"] = f"{_script_dir}:{_pythonpath}" if _pythonpath else _script_dir

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


@dataclass
class Config:
    """Configuration for vLLM logprobs generation."""
    model: str = "hamishivi/qwen3_openthoughts2"
    revision: Optional[str] = None
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


def setup_fp32_lm_head(llm: vllm.LLM) -> None:
    """
    Set up fp32 LM head on all vLLM workers (TP-safe).

    This patches LogitsProcessor AND sets the fp32 weight cache inside each worker,
    which is necessary for TP>1 where workers are separate processes.

    The worker function must be imported from a separate module (not __main__)
    to be picklable for multiprocessing.
    """
    # Import from separate module so it's picklable for multiprocessing
    from fp32_worker_patch import setup_fp32_in_worker

    try:
        # apply_model executes the function on each worker's local model instance
        results = llm.apply_model(setup_fp32_in_worker)
        for i, result in enumerate(results):
            print(f"    Worker {i}: {result}")

        # Verify all workers have the patch
        all_patched = all("patched" in r or "already patched" in r for r in results)
        all_cached = all("cache" in r for r in results)
        if all_patched and all_cached:
            print(f"  ✓ FP32 LM head enabled on {len(results)} worker(s)")
        else:
            print(f"  ⚠ FP32 setup incomplete on some workers")
    except Exception as e:
        print(f"  ✗ Could not set up fp32 via apply_model: {e}")
        import traceback
        traceback.print_exc()


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
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        revision=config.revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create vLLM engine
    llm = vllm.LLM(
        model=model_name,
        revision=config.revision,
        seed=config.seed,
        max_model_len=config.max_model_len,
        dtype=config.dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
    )

    # Set up fp32 LM head AFTER creating LLM (patches LogitsProcessor inside workers)
    if use_fp32_lm_head:
        print("  [vLLM] Setting up FP32 LM head...")
        setup_fp32_lm_head(llm)

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

    mode_str = "fp32" if use_fp32_lm_head else "bf16"

    # Tokenize all prompts upfront
    print(f"  Tokenizing {len(prompts)} prompts...")
    queries = [
        tokenizer(prompt, add_special_tokens=config.add_special_tokens)["input_ids"]
        for prompt in prompts
    ]

    # Batch generate all at once
    print(f"  Generating responses for all {len(prompts)} prompts...")
    token_prompts = [TokensPrompt(prompt_token_ids=q) for q in queries]
    outputs = llm.generate(token_prompts, sampling_params=sampling_params)

    # Process outputs
    all_results = []
    for i, (prompt, query, output) in enumerate(zip(prompts, queries, outputs)):
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
        print(f"  [{i+1}/{len(prompts)}] Generated {len(response)} tokens for '{prompt[:40]}...'")

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
    parser.add_argument("--revision", type=str, default=None,
                        help="Model revision/branch/tag (e.g., 'step_0100' for checkpoint)")
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
        revision=args.revision,
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
    if config.revision:
        print(f"Revision: {config.revision}")
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
