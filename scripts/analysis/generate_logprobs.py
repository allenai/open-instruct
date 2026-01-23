"""
Generate sequences and compute logprobs for vLLM vs HuggingFace comparison.

This script measures the numerical alignment between vLLM (used for generation in GRPO)
and HuggingFace (used for training in GRPO) to evaluate the FP32 LM head fix.

=== METHODOLOGY ===

The script runs 4 steps:
1. Generate sequences with vLLM using bf16 LM head → get logprobs
2. Generate sequences with vLLM using fp32 LM head → get logprobs
3. Score the bf16 sequences with HF using bf16 LM head → get logprobs
4. Score the fp32 sequences with HF using fp32 LM head → get logprobs

Comparisons are done with MATCHED precision on both sides:
- vLLM bf16 vs HF bf16 (baseline - measures raw vLLM/HF mismatch)
- vLLM fp32 vs HF fp32 (with fix - should show tighter alignment)

This isolates the precision effect: if FP32 reduces the mismatch, it means the
fix improves training signal quality in GRPO (where vLLM generates and HF trains).

=== KEY FIXES IMPLEMENTED ===

Based on ChatGPT analysis, this script includes correctness fixes:
1. vLLM token alignment: Uses `gen.token_ids` instead of dict key order
2. HF exact slicing: Slices to exactly `len(response)` tokens
3. TF32 disabled: Prevents TF32 from masking precision differences
4. Explicit sampling: Pins temperature, top_p, top_k, seed
5. Config hashing: Includes all params for cache validation

=== OUTPUT ===

Saves JSON with:
- vllm_bf16_results: Sequences and logprobs from vLLM bf16
- vllm_fp32_results: Sequences and logprobs from vLLM fp32
- comparisons: Dict with hf_bf16, hf_fp32 logprobs for scoring

Use plot_logprobs.py to visualize the results.

=== USAGE ===

    # Generate with defaults (hamishivi/qwen3_openthoughts2)
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run python scripts/analysis/generate_logprobs.py \
        --output-dir ~/dev/logprobs_data/

    # Quick test with smaller model
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run python scripts/analysis/generate_logprobs.py \
        --model Qwen/Qwen2.5-0.5B --max-tokens 100 --output-dir ~/dev/logprobs_data/

    # Qwen3-30B-A3B MoE with recommended sampling (temp=0.7, top_p=0.8, top_k=20, min_p=0)
    HF_HUB_ENABLE_HF_TRANSFER=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run python scripts/analysis/generate_logprobs.py \
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --max-tokens 2048 --max-model-len 8192 \
        --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
        --output-dir ~/dev/logprobs_data/

    # With longer generations
    VLLM_ALLOW_INSECURE_SERIALIZATION=1 uv run python scripts/analysis/generate_logprobs.py \
        --model hamishivi/qwen3_openthoughts2 --max-tokens 2048 --output-dir ~/dev/logprobs_data/
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

import numpy as np
import torch
import transformers
import vllm
from vllm.inputs import TokensPrompt

from open_instruct import model_utils
from open_instruct.vllm_utils import patch_vllm_for_fp32_logits

# Disable vLLM multiprocessing so patches apply in-process
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


@dataclass
class Config:
    """Configuration for logprobs generation."""
    model: str = "hamishivi/qwen3_openthoughts2"
    max_tokens: int = 512
    max_model_len: int = 8192
    seed: int = 42
    dtype: str = "bfloat16"
    # NOTE: Keep utilization low (~0.4) because vLLM V1 doesn't fully release
    # memory between loads in the same Python process. Higher values (0.8+) cause
    # OOM on the second vLLM load (bf16 → fp32). To use higher utilization, would
    # need subprocess-based loading.
    gpu_memory_utilization: float = 0.4
    attn_implementation: str = "sdpa"
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
]


def setup_precision(config: Config):
    """Configure precision settings."""
    if config.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        print("TF32 disabled for precise fp32 comparisons")


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
        Each result contains: query, response, logprobs, prompt
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
    )

    if use_fp32_lm_head:
        _setup_fp32_lm_head_cache(llm)

    # Explicit sampling params to avoid defaults changing behavior
    sampling_params = vllm.SamplingParams(
        max_tokens=config.max_tokens,
        logprobs=1,  # Request logprobs for sampled token
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

        # FIXED: Use token_ids alignment instead of dict key order
        response = list(gen.token_ids)  # Sampled tokens in order

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

    # Aggressive cleanup for vLLM V1 engine
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(3)  # Give vLLM time to fully cleanup

    return all_results, tokenizer


def _setup_fp32_lm_head_cache(llm: vllm.LLM) -> None:
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


def get_hf_logprobs(
    model_name: str,
    vllm_results: List[Dict],
    tokenizer,
    config: Config,
    use_fp32_lm_head: bool = True,
) -> List[List[float]]:
    """
    Compute HF logprobs for the same sequences generated by vLLM.

    Args:
        use_fp32_lm_head: If True, compute LM head in fp32. If False, use bf16.
    """
    head_mode = "fp32" if use_fp32_lm_head else "bf16"
    print(f"  Loading HF model ({head_mode} LM head)...")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, config.dtype),
        device_map="cuda",
        attn_implementation=config.attn_implementation,
        use_cache=False,
    )

    all_hf_logprobs = []

    for i, result in enumerate(vllm_results):
        print(f"  [{i+1}/{len(vllm_results)}] Computing HF logprobs...")

        query = result["query"]
        response = result["response"]

        hf_logprobs = _compute_hf_logprobs_single(
            model, query, response, tokenizer, use_fp32_lm_head
        )

        # Verify length match
        if len(hf_logprobs) != len(response):
            raise RuntimeError(
                f"Length mismatch: HF={len(hf_logprobs)}, vLLM={len(response)}"
            )

        all_hf_logprobs.append(hf_logprobs)

        # Show per-prompt stats
        vllm_lp = np.array(result["logprobs"])
        hf_lp = np.array(hf_logprobs)
        diff = np.abs(vllm_lp - hf_lp)
        print(f"    Tokens: {len(response)}, Mean |diff|: {diff.mean():.6f}, Max: {diff.max():.6f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return all_hf_logprobs


def _compute_hf_logprobs_single(
    model,
    query: List[int],
    response: List[int],
    tokenizer,
    use_fp32_lm_head: bool,
) -> List[float]:
    """Compute logprobs for a single sequence."""
    # Concatenate query and response
    full_sequence = query + response
    input_ids = torch.tensor(full_sequence, dtype=torch.long, device="cuda")[None, :]

    # Create attention mask (all 1s, no padding needed for single sequence)
    attention_mask = torch.ones_like(input_ids)

    # Create position ids
    position_ids = torch.arange(len(full_sequence), device="cuda")[None, :]

    with torch.inference_mode():
        # Get hidden states from base model
        base_output = model.model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        hidden_states = base_output.last_hidden_state

        # Compute LM head
        if use_fp32_lm_head:
            hidden_fp32 = hidden_states.float()
            w = model.lm_head.weight.float()
            b = model.lm_head.bias.float() if getattr(model.lm_head, "bias", None) is not None else None
            logits = torch.nn.functional.linear(hidden_fp32, w, b)
        else:
            # BF16 head (control comparison)
            w = model.lm_head.weight
            b = model.lm_head.bias if getattr(model.lm_head, "bias", None) is not None else None
            logits = torch.nn.functional.linear(hidden_states, w, b)

        # Always do log_softmax in fp32
        logits = logits.to(torch.float32)
        logprobs_all = model_utils.log_softmax_and_gather(logits, input_ids[:, 1:])

        # FIXED: Slice exactly len(response) tokens starting after query
        start = len(query) - 1  # -1 because we predict next token
        end = start + len(response)
        logprobs = logprobs_all[:, start:end]

    return logprobs[0].tolist()


def find_worst_offenders(
    vllm_logprobs: List[float],
    hf_logprobs: List[float],
    response: List[int],
    tokenizer,
    top_n: int = 10,
) -> List[Dict]:
    """Find tokens with largest logprob differences."""
    diffs = np.abs(np.array(vllm_logprobs) - np.array(hf_logprobs))
    worst_indices = np.argsort(diffs)[-top_n:][::-1]

    offenders = []
    for idx in worst_indices:
        token_id = response[idx]
        offenders.append({
            "position": int(idx),
            "token_id": int(token_id),
            "token_str": tokenizer.decode([token_id]),
            "vllm_logprob": float(vllm_logprobs[idx]),
            "hf_logprob": float(hf_logprobs[idx]),
            "diff": float(diffs[idx]),
        })
    return offenders


def save_results(
    output_dir: Path,
    config: Config,
    prompts: List[str],
    vllm_bf16_results: List[Dict],
    vllm_fp32_results: List[Dict],
    comparisons: Dict[str, List[List[float]]],
    metadata: Dict[str, Any],
):
    """Save all results to a single JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_safe = config.model.replace("/", "_")
    output_path = output_dir / f"logprobs_{model_safe}_{config.get_hash()}.json"

    data = {
        "config": config.to_dict(),
        "metadata": metadata,
        "prompts": prompts,
        "vllm_bf16_results": vllm_bf16_results,
        "vllm_fp32_results": vllm_fp32_results,
        "comparisons": comparisons,  # Keys: "vllm_bf16", "vllm_fp32", "hf_bf16", "hf_fp32"
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved results to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate logprobs for vLLM vs HF comparison")
    parser.add_argument("--model", type=str, default="hamishivi/qwen3_openthoughts2")
    parser.add_argument("--prompts", type=str, nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to GENERATE per response (output length)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max CONTEXT LENGTH vLLM reserves (prompt + output combined)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4,
                        help="vLLM GPU memory fraction. Keep low (~0.4) for sequential bf16/fp32 runs")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true", help="Allow TF32 (less precise)")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    config = Config(
        model=args.model,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        seed=args.seed,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attn_implementation=args.attn_implementation,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        add_special_tokens=args.add_special_tokens,
        disable_tf32=not args.allow_tf32,
    )

    print("=" * 60)
    print("Logprobs Generation")
    print(f"Model: {config.model}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"Prompts: {len(args.prompts)}")
    print(f"Config hash: {config.get_hash()}")
    print("=" * 60)

    setup_precision(config)

    # Collect metadata
    metadata = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "vllm_version": vllm.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    comparisons = {}

    # Step 1: Generate with vLLM bf16
    print("\n" + "=" * 60)
    print("Step 1: vLLM BF16 generation")
    print("=" * 60)
    vllm_results, tokenizer = get_vllm_logprobs(
        config.model, args.prompts, config, use_fp32_lm_head=False
    )
    comparisons["vllm_bf16"] = [r["logprobs"] for r in vllm_results]

    # Memory cleanup
    print("\nCleaning up GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(5)

    # Step 2: Score with vLLM fp32 (rescore same prompts, will generate same due to seed)
    print("\n" + "=" * 60)
    print("Step 2: vLLM FP32 generation")
    print("=" * 60)
    vllm_fp32_results, _ = get_vllm_logprobs(
        config.model, args.prompts, config, use_fp32_lm_head=True
    )
    comparisons["vllm_fp32"] = [r["logprobs"] for r in vllm_fp32_results]

    # Memory cleanup
    print("\nCleaning up GPU memory...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(5)

    # Step 3: Score bf16 sequences with HF bf16 head (matched precision comparison)
    print("\n" + "=" * 60)
    print("Step 3: HF BF16 head logprobs (on bf16 sequences)")
    print("=" * 60)
    hf_bf16_logprobs = get_hf_logprobs(
        config.model, vllm_results, tokenizer, config, use_fp32_lm_head=False
    )
    comparisons["hf_bf16"] = hf_bf16_logprobs

    # Step 4: Score fp32 sequences with HF fp32 head (matched precision comparison)
    print("\n" + "=" * 60)
    print("Step 4: HF FP32 head logprobs (on fp32 sequences)")
    print("=" * 60)
    hf_fp32_logprobs = get_hf_logprobs(
        config.model, vllm_fp32_results, tokenizer, config, use_fp32_lm_head=True
    )
    comparisons["hf_fp32"] = hf_fp32_logprobs

    # Save results
    output_path = save_results(
        Path(args.output_dir),
        config,
        args.prompts,
        vllm_results,
        vllm_fp32_results,
        comparisons,
        metadata,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_tokens_bf16 = sum(r["n_tokens"] for r in vllm_results)
    total_tokens_fp32 = sum(r["n_tokens"] for r in vllm_fp32_results)
    print(f"\nTotal tokens (bf16 sequences): {total_tokens_bf16}")
    print(f"Total tokens (fp32 sequences): {total_tokens_fp32}")

    # Compute diffs - matched precision comparisons
    vllm_bf16_lp = np.concatenate(comparisons["vllm_bf16"])
    vllm_fp32_lp = np.concatenate(comparisons["vllm_fp32"])
    hf_bf16_lp = np.concatenate(comparisons["hf_bf16"])
    hf_fp32_lp = np.concatenate(comparisons["hf_fp32"])

    diff_bf16 = np.abs(vllm_bf16_lp - hf_bf16_lp)
    diff_fp32 = np.abs(vllm_fp32_lp - hf_fp32_lp)

    print(f"\nvLLM BF16 vs HF BF16 (matched precision baseline):")
    print(f"  Mean |diff|: {diff_bf16.mean():.6f}")
    print(f"  Max |diff|:  {diff_bf16.max():.6f}")
    print(f"  n_tokens:    {len(vllm_bf16_lp)}")

    print(f"\nvLLM FP32 vs HF FP32 (matched precision with fix):")
    print(f"  Mean |diff|: {diff_fp32.mean():.6f}")
    print(f"  Max |diff|:  {diff_fp32.max():.6f}")
    print(f"  n_tokens:    {len(vllm_fp32_lp)}")

    if diff_bf16.mean() > 0:
        improvement = (diff_bf16.mean() - diff_fp32.mean()) / diff_bf16.mean() * 100
        print(f"\nImprovement: {improvement:.1f}% reduction in mean |diff| with FP32")

    # Show worst offenders for first prompt (bf16 comparison)
    print(f"\nWorst offenders (prompt 1, vLLM bf16 vs HF bf16):")
    offenders = find_worst_offenders(
        vllm_results[0]["logprobs"],
        hf_bf16_logprobs[0],
        vllm_results[0]["response"],
        tokenizer,
        top_n=5,
    )
    for off in offenders:
        print(f"  pos={off['position']:4d} token={off['token_str']!r:12s} "
              f"vllm={off['vllm_logprob']:8.4f} hf={off['hf_logprob']:8.4f} "
              f"diff={off['diff']:.4f}")


if __name__ == "__main__":
    main()
