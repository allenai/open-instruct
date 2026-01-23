#!/usr/bin/env python3
"""
Debug script to verify logprob alignment between vLLM and HuggingFace on IDENTICAL weights.

This script:
1. Loads a model into both vLLM and HuggingFace
2. Generates some sequences with vLLM
3. Computes logprobs for those EXACT sequences using both:
   - vLLM's compute_logprobs (not generation logprobs!)
   - HuggingFace forward pass
4. Compares them to isolate precision issues from weight sync timing issues

Usage:
    python scripts/analysis/fp32-lm-head/debug_logprob_alignment.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B \
        --num_sequences 10 \
        --fp32_lm_head  # optional
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def compute_hf_logprobs(model, tokenizer, sequences: list[list[int]], temperature: float = 1.0) -> list[list[float]]:
    """Compute logprobs using HuggingFace model."""
    all_logprobs = []

    for seq in sequences:
        input_ids = torch.tensor([seq], device=model.device)

        with torch.no_grad():
            outputs = model(input_ids[:, :-1], return_dict=True)
            logits = outputs.logits

            # Apply temperature
            logits = logits / (temperature + 1e-7)

            # Compute log softmax
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Gather logprobs for actual tokens
            target_ids = input_ids[:, 1:]
            token_logprobs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

            all_logprobs.append(token_logprobs[0].float().cpu().tolist())

    return all_logprobs


def compute_vllm_logprobs(llm: LLM, sequences: list[list[int]], temperature: float = 1.0) -> list[list[float]]:
    """Compute logprobs using vLLM for existing sequences (not generation)."""
    from vllm import TokensPrompt
    from vllm.sampling_params import SamplingParams

    all_logprobs = []

    # We'll use prompt_logprobs to get logprobs for the input tokens
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=1,  # We don't want to generate, just get prompt logprobs
        prompt_logprobs=1,  # Get logprobs for each prompt token
    )

    for seq in sequences:
        # Feed the sequence as a prompt
        prompt = TokensPrompt(prompt_token_ids=seq)
        outputs = llm.generate([prompt], sampling_params)

        # Extract prompt logprobs (skip first token which has no logprob)
        output = outputs[0]
        if output.prompt_logprobs is None:
            print(f"Warning: No prompt_logprobs for sequence")
            all_logprobs.append([0.0] * (len(seq) - 1))
            continue

        # prompt_logprobs[i] contains logprobs for token i given tokens 0..i-1
        # Skip index 0 (no previous context)
        logprobs = []
        for i in range(1, len(seq)):
            if output.prompt_logprobs[i] is None:
                logprobs.append(float('nan'))
            else:
                # Get the logprob of the actual token
                token_id = seq[i]
                if token_id in output.prompt_logprobs[i]:
                    logprobs.append(output.prompt_logprobs[i][token_id].logprob)
                else:
                    # Token not in top-k, use a very negative value
                    logprobs.append(-100.0)

        all_logprobs.append(logprobs)

    return all_logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num_sequences", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fp32_lm_head", action="store_true", help="Use FP32 for LM head")
    parser.add_argument("--output", type=str, default="/tmp/debug_logprob_alignment.png")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name_or_path}")
    print(f"FP32 LM head: {args.fp32_lm_head}")

    # Set environment for vLLM fp32 mode
    if args.fp32_lm_head:
        os.environ["OPEN_INSTRUCT_FP32_LM_HEAD"] = "1"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    hf_model.eval()

    # Apply FP32 LM head to HF model if requested
    if args.fp32_lm_head:
        from open_instruct.model_utils import enable_fp32_lm_head
        enable_fp32_lm_head(hf_model)
        print("Enabled FP32 LM head for HuggingFace model")

    # Load vLLM model
    print("Loading vLLM model...")

    # Import and apply vLLM patch if fp32 mode
    if args.fp32_lm_head:
        from open_instruct.vllm_utils import patch_vllm_for_fp32_logits
        patch_vllm_for_fp32_logits(enabled=True)
        print("Applied FP32 LM head patch to vLLM")

    llm = LLM(
        model=args.model_name_or_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )

    # Generate some sequences using vLLM
    print(f"\nGenerating {args.num_sequences} sequences...")
    prompts = [
        "The capital of France is",
        "In mathematics, 2 + 2 equals",
        "The color of the sky is usually",
        "Water freezes at",
        "The largest planet in our solar system is",
        "Python is a programming language that",
        "Machine learning models are trained using",
        "The speed of light is approximately",
        "DNA stands for",
        "The year World War II ended was",
    ][:args.num_sequences]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_length,
        logprobs=1,
    )

    outputs = llm.generate(prompts, sampling_params)

    # Get full sequences (prompt + generation)
    sequences = []
    for output in outputs:
        prompt_ids = list(output.prompt_token_ids)
        gen_ids = list(output.outputs[0].token_ids)
        full_seq = prompt_ids + gen_ids
        sequences.append(full_seq)
        print(f"Sequence length: {len(full_seq)}")

    # Compute logprobs using both methods
    print("\nComputing HuggingFace logprobs...")
    hf_logprobs = compute_hf_logprobs(hf_model, tokenizer, sequences, args.temperature)

    print("Computing vLLM logprobs...")
    vllm_logprobs = compute_vllm_logprobs(llm, sequences, args.temperature)

    # Flatten and compare
    all_hf = []
    all_vllm = []

    for hf_lp, vllm_lp in zip(hf_logprobs, vllm_logprobs):
        min_len = min(len(hf_lp), len(vllm_lp))
        all_hf.extend(hf_lp[:min_len])
        all_vllm.extend(vllm_lp[:min_len])

    all_hf = np.array(all_hf)
    all_vllm = np.array(all_vllm)

    # Filter out invalid values
    valid_mask = ~np.isnan(all_hf) & ~np.isnan(all_vllm) & (all_vllm > -50)
    all_hf = all_hf[valid_mask]
    all_vllm = all_vllm[valid_mask]

    print(f"\nTotal valid tokens: {len(all_hf)}")

    # Statistics
    diff = np.abs(all_hf - all_vllm)
    print(f"\nLogprob differences:")
    print(f"  Mean: {diff.mean():.6f}")
    print(f"  Max:  {diff.max():.6f}")
    print(f"  Std:  {diff.std():.6f}")

    # Check for extreme mismatches
    extreme = diff > 1.0
    print(f"  Tokens with diff > 1.0: {extreme.sum()} ({100*extreme.sum()/len(diff):.2f}%)")

    if extreme.sum() > 0:
        print("\nExtreme mismatch examples:")
        indices = np.where(extreme)[0][:10]
        for i in indices:
            print(f"  HF: {all_hf[i]:.4f}, vLLM: {all_vllm[i]:.4f}, diff: {diff[i]:.4f}")

    # Convert to probabilities and plot
    hf_prob = np.exp(all_hf)
    vllm_prob = np.exp(all_vllm)

    from scipy import stats
    corr, _ = stats.pearsonr(hf_prob, vllm_prob)
    mae = np.mean(np.abs(hf_prob - vllm_prob))

    print(f"\nProbability statistics:")
    print(f"  Pearson correlation: {corr:.4f}")
    print(f"  MAE: {mae:.6f}")

    # Plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(vllm_prob, hf_prob, alpha=0.3, s=5)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect alignment')
    ax.set_xlabel("vLLM Probability")
    ax.set_ylabel("HuggingFace Probability")
    ax.set_title(f"vLLM vs HF Logprob Alignment (Same Weights)\nFP32 LM head: {args.fp32_lm_head}\nr={corr:.4f}, MAE={mae:.6f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
