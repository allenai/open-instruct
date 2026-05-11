"""Plot per-task accuracy and length over training steps for one or more runs.

Walks `cse-579-experiments/results/<run_prefix>_step_<N>/<task>/` for every
matching run prefix, reads each task's primary metric and token-length mean
from disk, and emits a multi-panel PNG plus a tidy CSV of the underlying data.

Usage:
    uv run python cse-579-experiments/plot_trajectories.py \\
        --run "Baseline:baseline_think_run_4b_base_mixed_32k" \\
        --run "Linear α=1.0:lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant" \\
        --out cse-579-experiments/results/trajectories.png

`--run` takes either `<display>:<prefix>` or just `<prefix>` (in which case
the prefix is used as the display name).

Only reads files we already commit (metrics.json + length_stats.json), so the
plot can be regenerated without re-fetching from Beaker.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
STEP_RE = re.compile(r"_step_(\d+)$")

# Cleaner display names for the tasks our pipeline currently emits.
TASK_DISPLAY: dict[str, str] = {
    "aime": "AIME 2025 (pass@1)",
    "alpaca_eval": "AlpacaEval v3 (LC winrate)",
    "ifeval_mt_wildchat_unused_withRewrite": "IFEval (wildchat)",
    "ifeval_mt_ood_wildchat_unused_withRewrite": "IFEval (wildchat OOD)",
    "ifeval_ood": "IFEval (OOD)",
    "livecodebench_codegeneration": "LiveCodeBench (pass@1)",
    "minerva_math_500": "Minerva Math 500",
}


def discover_runs(run_prefix: str) -> dict[int, Path]:
    by_step: dict[int, Path] = {}
    for child in RESULTS_DIR.iterdir():
        if not child.is_dir() or not child.name.startswith(run_prefix + "_step_"):
            continue
        m = STEP_RE.search(child.name)
        if m:
            by_step[int(m.group(1))] = child
    return dict(sorted(by_step.items()))


def load_subtask_metrics(run_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for task_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for stm in sorted(task_dir.glob("task-*-metrics.json")):
            meta = json.loads(stm.read_text())
            task_name = meta.get("task_name") or stm.name
            primary_metric = (meta.get("task_config") or {}).get("primary_metric")
            primary_value = (meta.get("metrics") or {}).get(primary_metric) if primary_metric else None
            n_items = meta.get("num_instances")
            stem = stm.name.removesuffix("-metrics.json")
            length_path = stm.parent / f"{stem}-length_stats.json"
            tok_mean = tok_corr_mean = tok_inc_mean = None
            if length_path.exists():
                length = json.loads(length_path.read_text())
                tok_mean = (length.get("all_rollouts") or {}).get("tokens", {}).get("mean")
                tok_corr_mean = (length.get("correct_rollouts") or {}).get("tokens", {}).get("mean")
                tok_inc_mean = (length.get("incorrect_rollouts") or {}).get("tokens", {}).get("mean")
            records[task_name] = {
                "primary_metric": primary_metric,
                "primary_score": primary_value,
                "n_items": n_items,
                "tok_mean": tok_mean,
                "tok_corr_mean": tok_corr_mean,
                "tok_inc_mean": tok_inc_mean,
            }
    return records


def _parse_run_arg(arg: str) -> tuple[str, str]:
    """Returns (display_name, prefix). Accepts 'name:prefix' or bare 'prefix'."""
    if ":" in arg:
        name, prefix = arg.split(":", 1)
        return name.strip(), prefix.strip()
    return arg, arg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="'<display_name>:<run_prefix>' (or bare prefix). Repeatable.")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "trajectories.png")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    runs = [_parse_run_arg(r) for r in args.run]

    # display -> step -> task -> record
    data: dict[str, dict[int, dict[str, dict]]] = {}
    all_tasks: set[str] = set()
    for display, prefix in runs:
        per_step = {}
        for step, run_dir in discover_runs(prefix).items():
            recs = load_subtask_metrics(run_dir)
            per_step[step] = recs
            all_tasks.update(recs.keys())
        data[display] = per_step

    if not all_tasks:
        print("No data found.")
        return

    tasks_sorted = sorted(all_tasks)

    if args.csv is not None:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "step", "task", "primary_metric", "primary_score",
                        "n_items", "tok_mean", "tok_corr_mean", "tok_inc_mean"])
            for display, per_step in data.items():
                for step in sorted(per_step):
                    for task, rec in per_step[step].items():
                        w.writerow([display, step, task, rec["primary_metric"], rec["primary_score"],
                                    rec["n_items"], rec["tok_mean"], rec["tok_corr_mean"], rec["tok_inc_mean"]])
        print(f"wrote csv: {args.csv}")

    ncols = len(tasks_sorted)
    fig, axes = plt.subplots(2, ncols, figsize=(3.4 * ncols, 6.0), squeeze=False)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ti, task in enumerate(tasks_sorted):
        ax_score = axes[0][ti]
        ax_len = axes[1][ti]
        ax_score.set_title(TASK_DISPLAY.get(task, task), fontsize=10)
        for ri, (display, _) in enumerate(runs):
            color = color_cycle[ri % len(color_cycle)]
            steps = sorted(s for s, recs in data[display].items() if task in recs)
            if not steps:
                continue
            scores = [data[display][s][task]["primary_score"] for s in steps]
            tok_mean = [data[display][s][task]["tok_mean"] for s in steps]
            tok_corr = [data[display][s][task]["tok_corr_mean"] for s in steps]
            tok_inc = [data[display][s][task]["tok_inc_mean"] for s in steps]
            ax_score.plot(steps, scores, marker="o", color=color, linewidth=1.6)
            ax_len.plot(steps, tok_mean, color=color, marker="o", linestyle="-",
                        linewidth=1.6, label=f"{display} (all)")
            if any(v is not None for v in tok_corr):
                ax_len.plot(steps, tok_corr, color=color, marker=".", linestyle="--",
                            linewidth=1.1, alpha=0.85, label=f"{display} (correct)")
            if any(v is not None for v in tok_inc):
                ax_len.plot(steps, tok_inc, color=color, marker=".", linestyle=":",
                            linewidth=1.1, alpha=0.85, label=f"{display} (incorrect)")

        ax_score.set_xlabel("training step")
        if ti == 0:
            ax_score.set_ylabel("primary score")
            ax_len.set_ylabel("response tokens (mean)")
        ax_score.grid(True, alpha=0.3)
        ax_len.set_xlabel("training step")
        ax_len.set_yscale("symlog", linthresh=10)
        ax_len.grid(True, alpha=0.3)

    # Single figure-level legend at the bottom, one row.
    handles, labels = axes[1][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.0), fontsize=9, frameon=False)

    # Leave space at the bottom for the legend.
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote plot: {args.out}")


if __name__ == "__main__":
    main()
