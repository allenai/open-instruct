#!/usr/bin/env python3
"""
Plot logprob/prob alignment between vLLM and trainer (reproduces ScaleRL Figure 3).

This script analyzes saved logprob data from GRPO training runs and creates
scatter plots showing the alignment between vLLM inference logprobs and
trainer logprobs, similar to Figure 3 in the ScaleRL paper.

Usage:
    # First, run training with --save_logprob_samples to save data
    python open_instruct/grpo_fast.py ... --save_logprob_samples

    # Then plot the results (logprobs)
    python scripts/analysis/plot_logprob_alignment.py \
        --data_dir /tmp/grpo_run/logprob_samples \
        --output plot.png

    # Or plot as probabilities (0-1 range)
    python scripts/analysis/plot_logprob_alignment.py \
        --data_dir /tmp/grpo_run/logprob_samples \
        --output plot.png --use_probs

    # Compare multiple runs
    python scripts/analysis/plot_logprob_alignment.py \
        --data_dirs /tmp/run1/logprob_samples /tmp/run2/logprob_samples \
        --labels "No FP32" "FP32 Cache" \
        --output comparison.png --use_probs
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_logprob_samples(data_dir: str, max_samples: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """Load saved logprob samples from training run."""
    vllm_logprobs = []
    trainer_logprobs = []

    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Load all JSON files in the directory
    for json_file in sorted(data_path.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
            vllm_logprobs.extend(data.get("vllm_logprobs", []))
            trainer_logprobs.extend(data.get("trainer_logprobs", []))

        if len(vllm_logprobs) >= max_samples:
            break

    # Convert to numpy and truncate
    vllm = np.array(vllm_logprobs[:max_samples])
    trainer = np.array(trainer_logprobs[:max_samples])

    return vllm, trainer


def create_scatter_plot(
    vllm_logprobs: np.ndarray,
    trainer_logprobs: np.ndarray,
    output_path: str,
    title: str = "Logprob Alignment: vLLM vs Trainer",
    use_probs: bool = False,
):
    """Create scatter plot showing logprob/prob alignment."""
    if use_probs:
        # Convert to probabilities (0-1 range)
        vllm_vals = np.exp(vllm_logprobs)
        trainer_vals = np.exp(trainer_logprobs)
        xlabel = "vLLM Inference Prob"
        ylabel = "Trainer Prob"
        title = title.replace("Logprob", "Prob")
    else:
        vllm_vals = vllm_logprobs
        trainer_vals = trainer_logprobs
        xlabel = "vLLM Inference Logprob"
        ylabel = "Trainer Logprob"

    # Calculate statistics
    correlation, p_value = stats.pearsonr(vllm_vals, trainer_vals)
    mae = np.mean(np.abs(vllm_vals - trainer_vals))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot with alpha for density visualization
    ax.scatter(vllm_vals, trainer_vals, alpha=0.1, s=1, c='blue')

    # Add diagonal line (perfect alignment)
    min_val = min(vllm_vals.min(), trainer_vals.min())
    max_val = max(vllm_vals.max(), trainer_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect alignment')

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{title}\nPearson r = {correlation:.4f}, MAE = {mae:.4f}', fontsize=14)

    # Add legend
    ax.legend(loc='upper left')

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Grid
    ax.grid(True, alpha=0.3)

    # Set axis limits for probs
    if use_probs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {output_path}")
    print(f"Statistics: Pearson r = {correlation:.4f}, MAE = {mae:.4f}, n = {len(vllm_vals)}")

    return correlation, mae


def create_comparison_plot(
    data_dirs: list[str],
    labels: list[str],
    output_path: str,
    max_samples: int = 10000,
    use_probs: bool = False,
):
    """Create side-by-side comparison plot for multiple runs."""
    n_plots = len(data_dirs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    for ax, data_dir, label in zip(axes, data_dirs, labels):
        vllm, trainer = load_logprob_samples(data_dir, max_samples)

        if use_probs:
            vllm_vals = np.exp(vllm)
            trainer_vals = np.exp(trainer)
            xlabel = "vLLM Prob"
            ylabel = "Trainer Prob"
        else:
            vllm_vals = vllm
            trainer_vals = trainer
            xlabel = "vLLM Logprob"
            ylabel = "Trainer Logprob"

        correlation, _ = stats.pearsonr(vllm_vals, trainer_vals)
        mae = np.mean(np.abs(vllm_vals - trainer_vals))

        ax.scatter(vllm_vals, trainer_vals, alpha=0.1, s=1, c='blue')

        min_val = min(vllm_vals.min(), trainer_vals.min())
        max_val = max(vllm_vals.max(), trainer_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{label}\nr = {correlation:.4f}, MAE = {mae:.4f}', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        if use_probs:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    plt.suptitle("vLLM vs Trainer Alignment (ScaleRL Figure 3 Style)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot logprob/prob alignment between vLLM and trainer")
    parser.add_argument("--data_dir", type=str, help="Directory containing logprob sample JSON files")
    parser.add_argument("--data_dirs", type=str, nargs="+", help="Multiple directories for comparison plot")
    parser.add_argument("--labels", type=str, nargs="+", help="Labels for comparison plot")
    parser.add_argument("--output", type=str, default="logprob_alignment.png", help="Output plot path")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples to plot")
    parser.add_argument("--use_probs", action="store_true", help="Plot probabilities (0-1) instead of logprobs")

    args = parser.parse_args()

    if args.data_dirs:
        # Comparison mode
        labels = args.labels or [f"Run {i+1}" for i in range(len(args.data_dirs))]
        create_comparison_plot(args.data_dirs, labels, args.output, args.max_samples, args.use_probs)
    elif args.data_dir:
        # Single plot mode
        vllm, trainer = load_logprob_samples(args.data_dir, args.max_samples)
        create_scatter_plot(vllm, trainer, args.output, use_probs=args.use_probs)
    else:
        parser.error("Either --data_dir or --data_dirs must be provided")


if __name__ == "__main__":
    main()
