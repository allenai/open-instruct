"""Plot per-task accuracy and length over training steps for one or more runs.

Walks `cse-579-experiments/results/<run>/step_<N>/<task>/` for every named run,
reads each task's primary metric and token-length mean from disk, and emits a
multi-panel PNG plus a tidy CSV of the underlying data.

Usage:
    uv run python cse-579-experiments/plot_trajectories.py \\
        --run "Baseline:baseline_think_run_4b_base_mixed_32k" \\
        --run "Linear α=1.0:lenshape_qwen_4b_base_mixed_linear_p1.0_wconstant" \\
        --out cse-579-experiments/results/trajectories.png

`--run` takes either `<display>:<run>` or just `<run>` (in which case
the run name is used as the display name).

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
STEP_RE = re.compile(r"^step_(\d+)$")

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


def discover_runs(run: str) -> dict[int, Path]:
    by_step: dict[int, Path] = {}
    run_root = RESULTS_DIR / run
    if not run_root.is_dir():
        return by_step
    for child in run_root.iterdir():
        if not child.is_dir():
            continue
        m = STEP_RE.match(child.name)
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


def _pareto_front(points: list[dict]) -> list[dict]:
    """Return the non-dominated checkpoints (maximize score, minimize tokens).

    A point is dominated if another has score >= and tokens <=. Sorting by tokens
    ascending (ties: higher score first) and keeping points that strictly beat the
    best score seen so far yields the upper-left staircase frontier, sorted by tokens.
    """
    pts = sorted(points, key=lambda p: (p["tokens"], -p["score"]))
    front: list[dict] = []
    best = float("-inf")
    for p in pts:
        if p["score"] > best:
            front.append(p)
            best = p["score"]
    return front


def _parse_run_arg(arg: str) -> tuple[str, str, dict | None]:
    """Returns (display_name, run, style).

    Accepts 'name:run' or bare 'run', with an optional trailing style block
    separated by ';' as 'color,marker,linestyle' (e.g.
    'Linear α=1.0:my_run;tab:blue,o,-'). Style fields may be left empty.
    """
    style: dict | None = None
    if ";" in arg:
        arg, stylestr = arg.split(";", 1)
        parts = (stylestr.split(",") + ["", "", ""])[:3]
        style = {
            "color": parts[0] or None,
            "marker": parts[1] or "o",
            "linestyle": parts[2] or "-",
        }
    if ":" in arg:
        name, run = arg.split(":", 1)
        return name.strip(), run.strip(), style
    return arg, arg, style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="'<display_name>:<run>' (or bare run). Repeatable.")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "trajectories.png")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    runs = [_parse_run_arg(r) for r in args.run]

    # display -> step -> task -> record
    data: dict[str, dict[int, dict[str, dict]]] = {}
    all_tasks: set[str] = set()
    for display, run, _style in runs:
        per_step = {}
        for step, run_dir in discover_runs(run).items():
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
    fig, axes = plt.subplots(3, ncols, figsize=(3.4 * ncols, 9.0), squeeze=False)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # The default cycle only has 10 colors; with more runs, switch to a 20-color
    # map so every run stays visually distinct.
    if len(runs) > len(color_cycle):
        color_cycle = list(plt.get_cmap("tab20").colors)

    for ti, task in enumerate(tasks_sorted):
        ax_score = axes[0][ti]
        ax_len = axes[1][ti]
        ax_pareto = axes[2][ti]
        ax_score.set_title(TASK_DISPLAY.get(task, task), fontsize=10)
        pareto_points: list[dict] = []  # every checkpoint's (tokens, score) for this task
        for ri, (display, _run, style) in enumerate(runs):
            if style and style.get("color"):
                color = style["color"]
                marker = style.get("marker", "o")
                linestyle = style.get("linestyle", "-")
            else:
                color = color_cycle[ri % len(color_cycle)]
                marker, linestyle = "o", "-"
            steps = sorted(s for s, recs in data[display].items() if task in recs)
            if not steps:
                continue
            scores = [data[display][s][task]["primary_score"] for s in steps]
            tok_mean = [data[display][s][task]["tok_mean"] for s in steps]
            ax_score.plot(steps, scores, marker=marker, color=color, linewidth=1.5,
                          linestyle=linestyle, markersize=4)
            ax_len.plot(steps, tok_mean, color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.5, markersize=4, label=display)
            for t, sc in zip(tok_mean, scores):
                if t is not None and sc is not None:
                    pareto_points.append({"tokens": t, "score": sc, "color": color, "marker": marker})

        # Row 3 (Pareto): all checkpoints as faint dots; the global non-dominated
        # frontier (high score, few tokens) as a connecting line with the frontier
        # checkpoints highlighted as large run-colored markers.
        for p in pareto_points:
            ax_pareto.scatter(p["tokens"], p["score"], s=10, color=p["color"],
                              alpha=0.3, linewidths=0, zorder=1)
        front = _pareto_front(pareto_points)
        if front:
            ax_pareto.plot([p["tokens"] for p in front], [p["score"] for p in front],
                           color="0.4", linewidth=1.2, zorder=2)
            for p in front:
                ax_pareto.scatter(p["tokens"], p["score"], s=70, color=p["color"],
                                  marker=p["marker"], edgecolors="black", linewidths=0.6, zorder=3)

        ax_score.set_xlabel("training step")
        if ti == 0:
            ax_score.set_ylabel("primary score")
            ax_len.set_ylabel("response tokens (mean)")
            ax_pareto.set_ylabel("primary score")
        ax_score.grid(True, alpha=0.3)
        ax_len.set_xlabel("training step")
        ax_len.set_yscale("symlog", linthresh=10)
        ax_len.grid(True, alpha=0.3)
        ax_pareto.set_xlabel("response tokens (mean, symlog)")
        ax_pareto.set_xscale("symlog", linthresh=10)
        ax_pareto.grid(True, alpha=0.3)

    # Single figure-level legend at the bottom, one row.
    handles, labels = axes[1][0].get_legend_handles_labels()
    legend_rows = 1
    if handles:
        ncol = min(len(labels), 6)
        legend_rows = (len(labels) + ncol - 1) // ncol
        fig.legend(handles, labels, loc="lower center", ncol=ncol,
                   bbox_to_anchor=(0.5, 0.0), fontsize=9, frameon=False)

    # Leave space at the bottom for the legend (more rows -> more room).
    fig.tight_layout(rect=(0, 0.04 + 0.03 * legend_rows, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote plot: {args.out}")


if __name__ == "__main__":
    main()
