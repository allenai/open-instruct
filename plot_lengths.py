#!/usr/bin/env python3
"""
Plot histograms of token lengths from cached JSON files produced by tokenize_dataset.py.

Usage:
    python plot_lengths.py <input_paths...> [--output <output.png>] [--bins <N>] [--max-tokens <N>]

Examples:
    python plot_lengths.py lengths/ultrachat.json
    python plot_lengths.py lengths/*.json --output comparison.png
    python plot_lengths.py lengths/ds1.json lengths/ds2.json --bins 100 --max-tokens 8192
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot token length histograms.")
    parser.add_argument("inputs", nargs="+", type=str, help="Input JSON file(s) from tokenize_dataset.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: show interactively)")
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins (default: 80)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Clip x-axis to this value")
    parser.add_argument("--log-y", action="store_true", help="Use log scale on y-axis")
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 6],
                        help="Figure size in inches (default: 14 6)")
    return parser.parse_args()


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def make_label(data: dict, path: str) -> str:
    """Build a short label for the legend."""
    name = data.get("dataset", Path(path).stem)
    subset = data.get("subset")
    if subset:
        name += f" ({subset})"
    return name


def plot_single(ax, lengths, label, color, bins, max_tokens):
    """Plot one dataset's histogram with stat lines."""
    arr = np.array(lengths)
    if max_tokens:
        arr_plot = arr[arr <= max_tokens]
        clipped = len(arr) - len(arr_plot)
    else:
        arr_plot = arr
        clipped = 0

    mean = np.mean(arr)
    median = np.median(arr)
    p10 = np.percentile(arr, 10)
    p90 = np.percentile(arr, 90)

    ax.hist(arr_plot, bins=bins, alpha=0.55, color=color, edgecolor="none", label=label)

    # Stat lines
    ymin, ymax = ax.get_ylim()
    line_kw = dict(linewidth=1.5, alpha=0.85)
    ax.axvline(mean, color=color, linestyle="-", label=f"Mean: {mean:,.0f}", **line_kw)
    ax.axvline(median, color=color, linestyle="--", label=f"Median: {median:,.0f}", **line_kw)
    ax.axvline(p10, color=color, linestyle=":", label=f"P10: {p10:,.0f}", **line_kw)
    ax.axvline(p90, color=color, linestyle=":", label=f"P90: {p90:,.0f}", **line_kw)

    # Shade the 10-90 percentile region
    ax.axvspan(p10, p90, alpha=0.07, color=color)

    return {
        "n": len(arr),
        "mean": mean,
        "median": median,
        "std": np.std(arr),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p10": p10,
        "p90": p90,
        "clipped": clipped,
    }


COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be123c", "#4f46e5"]


def main():
    args = parse_args()

    datasets = []
    for path in args.inputs:
        data = load_data(path)
        datasets.append((path, data))

    multi = len(datasets) > 1

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    for i, (path, data) in enumerate(datasets):
        label = make_label(data, path)
        color = COLORS[i % len(COLORS)]
        lengths = data["lengths"]
        tokenizer = data.get("tokenizer", "unknown")

        stats = plot_single(ax, lengths, label, color, args.bins, args.max_tokens)

        print(f"--- {label} ---")
        print(f"  Tokenizer:  {tokenizer}")
        print(f"  Samples:    {stats['n']:,}")
        print(f"  Mean:       {stats['mean']:,.0f}")
        print(f"  Median:     {stats['median']:,.0f}")
        print(f"  Std:        {stats['std']:,.0f}")
        print(f"  Range:      [{stats['min']:,}, {stats['max']:,}]")
        print(f"  P10-P90:    [{stats['p10']:,.0f}, {stats['p90']:,.0f}]")
        if stats["clipped"]:
            print(f"  Clipped:    {stats['clipped']:,} samples beyond --max-tokens")

    ax.set_xlabel("Token Length", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    if args.log_y:
        ax.set_yscale("log")

    if args.max_tokens:
        ax.set_xlim(right=args.max_tokens)

    title = "Token Length Distribution"
    if not multi:
        title += f" — {make_label(datasets[0][1], datasets[0][0])}"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
