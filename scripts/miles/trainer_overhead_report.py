"""Summarize actual Core optimizer updates, excluding startup warmup.

Compare native PPO/GRPO with SCMC using logs from completed training jobs.
This measures the whole loss-path change, not an isolated probe operation.
"""

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

FIELDS = {
    "step_seconds": "train/step_seconds",
    "forward_backward_seconds": "train/profile/forward_backward_host_seconds",
    "optimizer_seconds": "train/profile/optimizer_host_seconds",
    "post_backward_seconds": "train/profile/post_backward_checks_seconds",
    "peak_allocated_bytes": "train/allocator/peak_allocated_bytes_max",
    "peak_reserved_bytes": "train/allocator/peak_reserved_bytes_max",
    "grad_norm": "train/grad_norm",
    "active_response_tokens": "train/active_response_tokens",
}


def load_steps(path):
    steps = {}
    for line in path.read_text().splitlines():
        match = re.search(r"Core optimizer step (\d+): (\{.*\})", line)
        if match:
            step = int(match[1])
            row = ast.literal_eval(match[2])
            if step in steps and steps[step] != row:
                raise ValueError(f"Conflicting optimizer step {step} in {path}; separate job attempts")
            steps[step] = row
    return [{"step": step, **row} for step, row in sorted(steps.items())]


def summarize(path, warmup, expected_steps):
    rows = load_steps(path)
    if [row["step"] for row in rows] != list(range(1, expected_steps + 1)):
        raise ValueError(f"Expected exactly {expected_steps} contiguous completed updates in {path}")
    selected = rows[warmup:]
    if not selected:
        raise ValueError("Warmup excludes every update")
    stats = {}
    for field, key in FIELDS.items():
        values = np.array([row[key] for row in selected], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {key} in {path}")
        stats[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "std": float(values.std()),
        }
    if stats["grad_norm"]["min"] <= 0:
        raise ValueError(f"An update had zero gradient norm in {path}")
    return {
        "source": str(path.resolve()),
        "completed_steps": len(rows),
        "measured_steps": len(selected),
        "stats": stats,
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scmc32", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--expected-steps", type=int, default=20)
    args = parser.parse_args()
    runs, rows = {}, []
    for name in ("baseline", "scmc32"):
        runs[name], steps = summarize(getattr(args, name), args.warmup, args.expected_steps)
        rows.extend({"mode": name, **row} for row in steps)
    overhead = {}
    for field in FIELDS:
        base = runs["baseline"]["stats"][field]["mean"]
        value = runs["scmc32"]["stats"][field]["mean"]
        overhead[field] = {
            "absolute_difference": value - base,
            "increase_pct": 100 * (value / base - 1) if base else None,
        }
    report = {
        "runs": runs,
        "warmup_steps": args.warmup,
        "overhead": overhead,
        "scope": "Native PPO/GRPO baseline versus MILES REINFORCE score centering with TIS and MC32 array probes; full loss-path comparison.",
        "limitations": "Separate jobs, synthetic balanced rewards, fixed response lengths, independently evolving policies. Native host wall timings, with no added CUDA synchronization; stage spans are not completed GPU execution times. Not a convergence test or paired same-trajectory benchmark.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trainer-report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "trainer-steps.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    labels = ["PPO/GRPO", "SCMC32 + TIS"]
    xs = np.arange(2)
    for offset, field, label in [
        (-0.25, "step_seconds", "Optimizer update"),
        (0, "forward_backward_seconds", "Forward/backward host span"),
        (0.25, "optimizer_seconds", "Optimizer host span"),
    ]:
        axes[0].bar(xs + offset, [run["stats"][field]["mean"] for run in runs.values()], width=0.25, label=label)
    axes[0].set_ylabel("Mean seconds per measured update")
    axes[0].set_title(f"Update overhead: {overhead['step_seconds']['increase_pct']:+.2f}%")
    axes[0].legend()
    for offset, field, label in [
        (-0.18, "peak_allocated_bytes", "Allocated"),
        (0.18, "peak_reserved_bytes", "Reserved"),
    ]:
        axes[1].bar(
            xs + offset, [run["stats"][field]["max"] / 2**30 for run in runs.values()], width=0.36, label=label
        )
    axes[1].set_ylabel("Peak memory across trainer ranks (GiB)")
    axes[1].set_title("Trainer GPU memory")
    axes[1].legend()
    for ax in axes:
        ax.set_xticks(xs, labels)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Olmo H066 · actual training · Jupiter · 4 trainer GPUs, EP4")
    fig.supxlabel(
        f"First {args.warmup} updates excluded. Full loss-path comparison; synthetic rewards, separate runs."
    )
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"trainer-overhead.{suffix}", dpi=170)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
