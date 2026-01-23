"""
Plot logprobs comparison from saved generation data.

This script visualizes the alignment between vLLM and HuggingFace logprobs
to evaluate the effectiveness of the FP32 LM head fix for GRPO training.

Comparisons are done with MATCHED precision on both sides:
- vLLM bf16 vs HF bf16 (baseline - shows mismatch without fix)
- vLLM fp32 vs HF fp32 (with fix - should show tighter alignment)

The key insight: if FP32 reduces the logprob mismatch between vLLM (generator)
and HF (trainer), it means the precision fix improves training signal quality.

Outputs:
- Scatter plots showing logprob alignment (ideal: tight diagonal line)
- Histograms showing distribution of absolute differences
- Summary statistics (mean, max, percentiles)
- Worst offending tokens for debugging

Usage:
    # Plot from saved data (probabilities 0-1, default)
    uv run python scripts/analysis/plot_logprobs.py \
        --input ~/dev/logprobs_data/logprobs_*.json \
        --output-dir ~/dev/plots/

    # Plot logprobs (raw scale, harder to interpret)
    uv run python scripts/analysis/plot_logprobs.py \
        --input ~/dev/logprobs_data/logprobs_*.json \
        --output-dir ~/dev/plots/ --use-logprobs

    # Show worst offenders for debugging
    uv run python scripts/analysis/plot_logprobs.py \
        --input ~/dev/logprobs_data/logprobs_*.json \
        --output-dir ~/dev/plots/ --show-offenders 20
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import transformers


def load_results(input_path: Path) -> Dict[str, Any]:
    """Load saved logprobs results."""
    with open(input_path) as f:
        return json.load(f)


def compute_statistics(
    logprobs_a: np.ndarray,
    logprobs_b: np.ndarray,
    use_probs: bool = False,
) -> Dict[str, float]:
    """Compute comparison statistics."""
    if use_probs:
        vals_a = np.exp(logprobs_a)
        vals_b = np.exp(logprobs_b)
    else:
        vals_a = logprobs_a
        vals_b = logprobs_b

    diff = np.abs(vals_a - vals_b)

    return {
        "mean_diff": float(diff.mean()),
        "max_diff": float(diff.max()),
        "std_diff": float(diff.std()),
        "median_diff": float(np.median(diff)),
        "p95_diff": float(np.percentile(diff, 95)),
        "p99_diff": float(np.percentile(diff, 99)),
    }


def find_worst_offenders(
    logprobs_a: List[float],
    logprobs_b: List[float],
    response: List[int],
    tokenizer,
    top_n: int = 20,
) -> List[Dict]:
    """Find tokens with largest differences."""
    diffs = np.abs(np.array(logprobs_a) - np.array(logprobs_b))
    worst_indices = np.argsort(diffs)[-top_n:][::-1]

    offenders = []
    for idx in worst_indices:
        token_id = response[idx]
        offenders.append({
            "position": int(idx),
            "token_id": int(token_id),
            "token_str": tokenizer.decode([token_id]),
            "logprob_a": float(logprobs_a[idx]),
            "logprob_b": float(logprobs_b[idx]),
            "diff": float(diffs[idx]),
        })
    return offenders


def create_scatter_plot(
    data: Dict[str, Any],
    output_dir: Path,
    use_probs: bool = False,
):
    """Create scatter plot comparing logprobs/probs with matched precision."""
    config = data["config"]
    comparisons = data["comparisons"]

    metric_name = "Probability" if use_probs else "Logprob"

    # Get data - matched precision comparisons
    vllm_bf16 = np.concatenate(comparisons["vllm_bf16"])
    hf_bf16 = np.concatenate(comparisons["hf_bf16"])
    vllm_fp32 = np.concatenate(comparisons["vllm_fp32"])
    hf_fp32 = np.concatenate(comparisons["hf_fp32"])

    # Convert to probabilities if requested
    if use_probs:
        vllm_bf16 = np.exp(vllm_bf16)
        hf_bf16 = np.exp(hf_bf16)
        vllm_fp32 = np.exp(vllm_fp32)
        hf_fp32 = np.exp(hf_fp32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Set plot range
    if use_probs:
        plot_min, plot_max = -0.02, 1.02
    else:
        all_vals = [vllm_bf16, hf_bf16, vllm_fp32, hf_fp32]
        plot_min = min(v.min() for v in all_vals) - 0.5
        plot_max = max(v.max() for v in all_vals) + 0.5

    # Plot 1: vLLM bf16 vs HF bf16 (matched precision baseline)
    ax = axes[0]
    ax.scatter(vllm_bf16, hf_bf16, c='coral', s=20, alpha=0.5, edgecolors='none')
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel(f'vLLM {metric_name} (bf16)', fontsize=11)
    ax.set_ylabel(f'HF {metric_name} (bf16)', fontsize=11)
    ax.set_title('BF16 Baseline\n(vLLM bf16 vs HF bf16)', fontsize=12, fontweight='bold')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    stats_bf16 = compute_statistics(
        np.concatenate(comparisons["vllm_bf16"]),
        np.concatenate(comparisons["hf_bf16"]),
        use_probs
    )
    stats_text = f'Mean |diff|: {stats_bf16["mean_diff"]:.4f}\nMax |diff|: {stats_bf16["max_diff"]:.4f}\nn={len(vllm_bf16)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 2: vLLM fp32 vs HF fp32 (matched precision with fix)
    ax = axes[1]
    ax.scatter(vllm_fp32, hf_fp32, c='mediumseagreen', s=20, alpha=0.5, edgecolors='none')
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel(f'vLLM {metric_name} (fp32)', fontsize=11)
    ax.set_ylabel(f'HF {metric_name} (fp32)', fontsize=11)
    ax.set_title('FP32 Fix\n(vLLM fp32 vs HF fp32)', fontsize=12, fontweight='bold')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    stats_fp32 = compute_statistics(
        np.concatenate(comparisons["vllm_fp32"]),
        np.concatenate(comparisons["hf_fp32"]),
        use_probs
    )
    stats_text = f'Mean |diff|: {stats_fp32["mean_diff"]:.4f}\nMax |diff|: {stats_fp32["max_diff"]:.4f}\nn={len(vllm_fp32)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Compute improvement
    if stats_bf16["mean_diff"] > 0:
        improvement = (stats_bf16["mean_diff"] - stats_fp32["mean_diff"]) / stats_bf16["mean_diff"] * 100
        improvement_text = f"FP32 reduces mean |diff| by {improvement:.1f}%"
    else:
        improvement_text = ""

    # Title
    model_name = config["model"].split("/")[-1]
    # Handle both old format (vllm_results) and new format (vllm_bf16_results)
    if "vllm_bf16_results" in data:
        total_tokens_bf16 = sum(r["n_tokens"] for r in data["vllm_bf16_results"])
        total_tokens_fp32 = sum(r["n_tokens"] for r in data["vllm_fp32_results"])
        token_info = f"{total_tokens_bf16} bf16 / {total_tokens_fp32} fp32 tokens"
    else:
        total_tokens = sum(r["n_tokens"] for r in data["vllm_results"])
        token_info = f"{total_tokens} tokens"
    plt.suptitle(
        f'{metric_name} Alignment: vLLM vs HF\n'
        f'Model: {model_name} | {len(data["prompts"])} prompts | {token_info}\n'
        f'{improvement_text}',
        fontsize=12, y=1.05
    )

    plt.tight_layout()

    # Save
    model_safe = config["model"].replace("/", "_")
    metric_suffix = "probs" if use_probs else "logprobs"
    output_path = output_dir / f"comparison_{metric_suffix}_{model_safe}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_path}")

    return output_path


def create_diff_histogram(
    data: Dict[str, Any],
    output_dir: Path,
    use_probs: bool = False,
):
    """Create histogram of differences with matched precision comparisons."""
    config = data["config"]
    comparisons = data["comparisons"]

    metric_name = "Probability" if use_probs else "Logprob"

    # Matched precision data
    vllm_bf16 = np.concatenate(comparisons["vllm_bf16"])
    hf_bf16 = np.concatenate(comparisons["hf_bf16"])
    vllm_fp32 = np.concatenate(comparisons["vllm_fp32"])
    hf_fp32 = np.concatenate(comparisons["hf_fp32"])

    if use_probs:
        vllm_bf16 = np.exp(vllm_bf16)
        hf_bf16 = np.exp(hf_bf16)
        vllm_fp32 = np.exp(vllm_fp32)
        hf_fp32 = np.exp(hf_fp32)

    # Matched precision diffs
    diff_bf16 = np.abs(vllm_bf16 - hf_bf16)
    diff_fp32 = np.abs(vllm_fp32 - hf_fp32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Compute shared axis limits for visual comparison
    x_max = max(diff_bf16.max(), diff_fp32.max()) * 1.05
    bins = np.linspace(0, x_max, 51)  # Shared bins

    # Histogram 1: bf16 vs bf16 diffs
    ax = axes[0]
    counts_bf16, _, _ = ax.hist(diff_bf16, bins=bins, color='coral', alpha=0.7, edgecolor='darkred')
    ax.axvline(diff_bf16.mean(), color='red', linestyle='--', label=f'Mean: {diff_bf16.mean():.4f}')
    ax.set_xlabel(f'|{metric_name} Diff|', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('BF16 Baseline\n(vLLM bf16 vs HF bf16)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram 2: fp32 vs fp32 diffs
    ax = axes[1]
    counts_fp32, _, _ = ax.hist(diff_fp32, bins=bins, color='mediumseagreen', alpha=0.7, edgecolor='darkgreen')
    ax.axvline(diff_fp32.mean(), color='green', linestyle='--', label=f'Mean: {diff_fp32.mean():.4f}')
    ax.set_xlabel(f'|{metric_name} Diff|', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('FP32 Fix\n(vLLM fp32 vs HF fp32)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lock axes to same scale for visual comparison
    y_max = max(counts_bf16.max(), counts_fp32.max()) * 1.1
    for ax in axes:
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

    model_name = config["model"].split("/")[-1]
    plt.suptitle(f'{metric_name} Difference Distribution - {model_name}', fontsize=13, y=1.02)
    plt.tight_layout()

    model_safe = config["model"].replace("/", "_")
    metric_suffix = "probs" if use_probs else "logprobs"
    output_path = output_dir / f"histogram_{metric_suffix}_{model_safe}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Histogram saved to {output_path}")

    return output_path


def print_summary(data: Dict[str, Any], use_probs: bool = False):
    """Print summary statistics."""
    config = data["config"]
    comparisons = data["comparisons"]
    metadata = data.get("metadata", {})

    metric_name = "Probability" if use_probs else "Logprob"

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nModel: {config['model']}")
    print(f"Max tokens: {config['max_tokens']}")
    print(f"Prompts: {len(data['prompts'])}")
    print(f"TF32 disabled: {config.get('disable_tf32', 'unknown')}")

    if metadata:
        print(f"\nVersions:")
        print(f"  torch: {metadata.get('torch_version', 'unknown')}")
        print(f"  transformers: {metadata.get('transformers_version', 'unknown')}")
        print(f"  vllm: {metadata.get('vllm_version', 'unknown')}")

    # Handle both old format (vllm_results) and new format (vllm_bf16_results)
    if "vllm_bf16_results" in data:
        total_tokens_bf16 = sum(r["n_tokens"] for r in data["vllm_bf16_results"])
        total_tokens_fp32 = sum(r["n_tokens"] for r in data["vllm_fp32_results"])
        print(f"\nTotal tokens (bf16 sequences): {total_tokens_bf16}")
        print(f"Total tokens (fp32 sequences): {total_tokens_fp32}")
    else:
        total_tokens = sum(r["n_tokens"] for r in data["vllm_results"])
        print(f"\nTotal tokens: {total_tokens}")

    # Stats for each comparison (matched precision)
    print(f"\n{metric_name} Statistics (matched precision):")
    print("-" * 50)

    vllm_bf16 = np.concatenate(comparisons["vllm_bf16"])
    hf_bf16 = np.concatenate(comparisons["hf_bf16"])
    vllm_fp32 = np.concatenate(comparisons["vllm_fp32"])
    hf_fp32 = np.concatenate(comparisons["hf_fp32"])

    # Matched precision: bf16 vs bf16
    stats_bf16 = compute_statistics(vllm_bf16, hf_bf16, use_probs)
    print(f"\nvLLM BF16 vs HF BF16 (baseline):")
    print(f"  Mean |diff|:   {stats_bf16['mean_diff']:.6f}")
    print(f"  Max |diff|:    {stats_bf16['max_diff']:.6f}")
    print(f"  Std |diff|:    {stats_bf16['std_diff']:.6f}")
    print(f"  Median |diff|: {stats_bf16['median_diff']:.6f}")
    print(f"  P95 |diff|:    {stats_bf16['p95_diff']:.6f}")
    print(f"  P99 |diff|:    {stats_bf16['p99_diff']:.6f}")

    # Matched precision: fp32 vs fp32
    stats_fp32 = compute_statistics(vllm_fp32, hf_fp32, use_probs)
    print(f"\nvLLM FP32 vs HF FP32 (with fix):")
    print(f"  Mean |diff|:   {stats_fp32['mean_diff']:.6f}")
    print(f"  Max |diff|:    {stats_fp32['max_diff']:.6f}")
    print(f"  Std |diff|:    {stats_fp32['std_diff']:.6f}")
    print(f"  Median |diff|: {stats_fp32['median_diff']:.6f}")
    print(f"  P95 |diff|:    {stats_fp32['p95_diff']:.6f}")
    print(f"  P99 |diff|:    {stats_fp32['p99_diff']:.6f}")

    if stats_bf16['mean_diff'] > 0:
        improvement = (stats_bf16['mean_diff'] - stats_fp32['mean_diff']) / stats_bf16['mean_diff'] * 100
        print(f"\nImprovement: {improvement:.1f}% reduction in mean |diff| with FP32")


def print_offenders(data: Dict[str, Any], top_n: int = 20):
    """Print worst offending tokens (bf16 comparison)."""
    config = data["config"]
    comparisons = data["comparisons"]

    # Handle both old format (vllm_results) and new format (vllm_bf16_results)
    vllm_results = data.get("vllm_bf16_results", data.get("vllm_results", []))

    print("\n" + "=" * 60)
    print(f"WORST OFFENDERS (top {top_n} per prompt, bf16 comparison)")
    print("=" * 60)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"])

    for i, result in enumerate(vllm_results):
        print(f"\nPrompt {i+1}: '{result['prompt'][:50]}...'")
        print("-" * 50)

        offenders = find_worst_offenders(
            comparisons["vllm_bf16"][i],
            comparisons["hf_bf16"][i],
            result["response"],
            tokenizer,
            top_n=min(top_n, 10),  # Limit per prompt
        )

        for off in offenders:
            print(f"  pos={off['position']:4d} token={off['token_str']!r:15s} "
                  f"vllm={off['logprob_a']:8.4f} hf={off['logprob_b']:8.4f} "
                  f"diff={off['diff']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot logprobs comparison")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file from generate_logprobs.py")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--use-logprobs", action="store_true", help="Plot logprobs instead of probabilities (default: probabilities 0-1)")
    parser.add_argument("--show-offenders", type=int, default=0, help="Show N worst offending tokens per prompt")
    parser.add_argument("--no-scatter", action="store_true", help="Skip scatter plot")
    parser.add_argument("--no-histogram", action="store_true", help="Skip histogram")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading data from {input_path}")
    data = load_results(input_path)

    # Default to probabilities (0-1), use --use-logprobs to switch
    use_probs = not args.use_logprobs

    print_summary(data, use_probs=use_probs)

    if not args.no_scatter:
        create_scatter_plot(data, output_dir, use_probs=use_probs)

    if not args.no_histogram:
        create_diff_histogram(data, output_dir, use_probs=use_probs)

    if args.show_offenders > 0:
        print_offenders(data, top_n=args.show_offenders)


if __name__ == "__main__":
    main()
