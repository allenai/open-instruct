"""Summarize cse-579-experiments/results/<run>/<task>/{metrics,length_stats}.json.

Usage:
    uv run python cse-579-experiments/summarize.py                  # all runs
    uv run python cse-579-experiments/summarize.py <run_dir> ...    # specific runs

Output is a markdown table per run that the experiment .md files can include.
Pulls numbers directly from metrics.json / length_stats.json — no hand-typed
values, so the table stays in sync with whatever fetch_eval_results.sh put
on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _primary_scores(metrics: dict) -> list[str]:
    """Return the all_primary_scores strings from a metrics.json blob."""
    return metrics.get("all_primary_scores", []) or []


def _summarize_run(run_dir: Path) -> str:
    lines: list[str] = []
    lines.append(f"### `{run_dir.name}`")
    lines.append("")
    lines.append("| Task | Primary score | n | Resp len mean | median | p90 |")
    lines.append("|------|---------------|---|---------------|--------|-----|")

    for task_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        metrics = _load(task_dir / "metrics.json")
        length = _load(task_dir / "length_stats.json")
        if metrics is None:
            lines.append(f"| {task_dir.name} | _missing metrics.json_ | – | – | – | – |")
            continue
        scores = _primary_scores(metrics)
        if not scores:
            score_str = "_no all_primary_scores key_"
        else:
            score_str = "<br>".join(scores)
        n = "–"
        tasks = metrics.get("tasks") or []
        if tasks:
            n = str(sum(t.get("num_instances", 0) for t in tasks))
        if length is None:
            lmean = lmed = lp90 = "–"
        else:
            lmean = f"{length['mean']:.0f}"
            lmed = f"{length['median']:.0f}"
            lp90 = f"{length['p90']:.0f}"
        lines.append(f"| {task_dir.name} | {score_str} | {n} | {lmean} | {lmed} | {lp90} |")
    lines.append("")
    lines.append("_Response lengths are character counts of model continuations. "
                 "Computed by `fetch_eval_results.sh` and saved per-task in `length_stats.json`._")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", help="Run directories under results/ (default: all).")
    args = parser.parse_args()

    if args.runs:
        run_dirs = [RESULTS_DIR / r for r in args.runs]
    else:
        run_dirs = sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()) if RESULTS_DIR.exists() else []

    missing = [d for d in run_dirs if not d.exists()]
    if missing:
        for m in missing:
            print(f"# NOT FOUND: {m}")
        return

    if not run_dirs:
        print("(no result directories under cse-579-experiments/results/)")
        return

    for d in run_dirs:
        print(_summarize_run(d))
        print()


if __name__ == "__main__":
    main()
