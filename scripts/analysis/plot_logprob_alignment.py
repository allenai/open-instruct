#!/usr/bin/env python3
"""
Plot logprob alignment between vLLM and trainer (reproduces ScaleRL Figure 3).

This script analyzes saved logprob data from GRPO training runs and creates
scatter plots showing the alignment between vLLM inference logprobs and
trainer logprobs, similar to Figure 3 in the ScaleRL paper.

Usage:
    # First, run training with --save_logprob_samples to save data
    python open_instruct/grpo_fast.py ... --save_logprob_samples

    # Then plot the results
    python scripts/analysis/plot_logprob_alignment.py \
        --data_dir /tmp/grpo_run/logprob_samples \
        --output plot.png
"""

import argparse
import json
import os
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
):
    """Create scatter plot showing logprob alignment."""
    # Calculate statistics
    correlation, p_value = stats.pearsonr(vllm_logprobs, trainer_logprobs)
    mae = np.mean(np.abs(vllm_logprobs - trainer_logprobs))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot with alpha for density visualization
    ax.scatter(vllm_logprobs, trainer_logprobs, alpha=0.1, s=1, c='blue')

    # Add diagonal line (perfect alignment)
    min_val = min(vllm_logprobs.min(), trainer_logprobs.min())
    max_val = max(vllm_logprobs.max(), trainer_logprobs.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect alignment')

    # Labels and title
    ax.set_xlabel('vLLM Inference Logprobs', fontsize=12)
    ax.set_ylabel('Trainer Logprobs', fontsize=12)
    ax.set_title(f'{title}\nPearson r = {correlation:.4f}, MAE = {mae:.4f}', fontsize=14)

    # Add legend
    ax.legend(loc='upper left')

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Grid
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {output_path}")
    print(f"Statistics: Pearson r = {correlation:.4f}, MAE = {mae:.4f}, n = {len(vllm_logprobs)}")

    return correlation, mae


def create_comparison_plot(
    data_dirs: list[str],
    labels: list[str],
    output_path: str,
    max_samples: int = 10000,
):
    """Create side-by-side comparison plot for multiple runs."""
    n_plots = len(data_dirs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))

    if n_plots == 1:
        axes = [axes]

    for ax, data_dir, label in zip(axes, data_dirs, labels):
        vllm, trainer = load_logprob_samples(data_dir, max_samples)
        correlation, _ = stats.pearsonr(vllm, trainer)

        ax.scatter(vllm, trainer, alpha=0.1, s=1, c='blue')

        min_val = min(vllm.min(), trainer.min())
        max_val = max(vllm.max(), trainer.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        ax.set_xlabel('vLLM Logprobs', fontsize=11)
        ax.set_ylabel('Trainer Logprobs', fontsize=11)
        ax.set_title(f'{label}\nr = {correlation:.4f}', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot logprob alignment between vLLM and trainer")
    parser.add_argument("--data_dir", type=str, help="Directory containing logprob sample JSON files")
    parser.add_argument("--data_dirs", type=str, nargs="+", help="Multiple directories for comparison plot")
    parser.add_argument("--labels", type=str, nargs="+", help="Labels for comparison plot")
    parser.add_argument("--output", type=str, default="logprob_alignment.png", help="Output plot path")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples to plot")

    args = parser.parse_args()

    if args.data_dirs:
        # Comparison mode
        labels = args.labels or [f"Run {i+1}" for i in range(len(args.data_dirs))]
        create_comparison_plot(args.data_dirs, labels, args.output, args.max_samples)
    elif args.data_dir:
        # Single plot mode
        vllm, trainer = load_logprob_samples(args.data_dir, args.max_samples)
        create_scatter_plot(vllm, trainer, args.output)
    else:
        parser.error("Either --data_dir or --data_dirs must be provided")


if __name__ == "__main__":
    main()
