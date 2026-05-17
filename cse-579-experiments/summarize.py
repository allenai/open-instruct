"""Summarize cse-579-experiments/results/<run>/step_<N>/<task>/{metrics,length_stats}.json.

Usage:
    uv run python cse-579-experiments/summarize.py                       # all runs, all steps
    uv run python cse-579-experiments/summarize.py <run> ...             # all steps of named runs
    uv run python cse-579-experiments/summarize.py <run>/step_<N> ...    # specific step(s)

Output is a markdown table per (run, step) that the experiment .md files can include.
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


def _fmt(x: float | int | None, digits: int = 0) -> str:
    if x is None:
        return "–"
    if digits == 0:
        return f"{x:.0f}"
    return f"{x:.{digits}f}"


def _row_for_block(prefix: str, block: dict | None) -> str:
    if block is None:
        return " | ".join([prefix] + ["–"] * 5)
    tok = block.get("tokens") or {}
    if not tok or tok.get("n", 0) == 0:
        return " | ".join([prefix, "0", "–", "–", "–", "–"])
    return " | ".join([
        prefix,
        str(tok.get("n")),
        _fmt(tok.get("mean"), 1),
        _fmt(tok.get("std"), 1),
        _fmt(tok.get("p50")),
        _fmt(tok.get("p90")),
    ])


def _summarize_subtask(subtask_metrics: Path) -> list[str]:
    """One Markdown row per subtask (per `task-NNN-{name}-metrics.json`).

    Emits THREE rows for each subtask: all / correct / incorrect rollouts.
    """
    stem = subtask_metrics.name.removesuffix("-metrics.json")
    length_path = subtask_metrics.parent / f"{stem}-length_stats.json"
    meta = _load(subtask_metrics) or {}

    task_name = meta.get("task_name") or stem
    primary_metric = (meta.get("task_config") or {}).get("primary_metric") or "?"
    primary_value = (meta.get("metrics") or {}).get(primary_metric)
    primary_str = "?" if primary_value is None else f"{primary_value:.4g}"
    n_items = meta.get("num_instances", "?")

    length = _load(length_path)
    rows: list[str] = []
    if length is None:
        # No predictions retrieved (e.g. fetched before fix). Emit one empty row.
        rows.append(
            f"| `{task_name}` | {primary_metric}={primary_str} | n={n_items} | _no length data_ | – | – | – | – | – |"
        )
        return rows

    n_corr = length.get("n_items_correct")
    n_inc = length.get("n_items_incorrect")
    n_unk = length.get("n_items_unknown")
    header_cell = (
        f"`{task_name}` | {primary_metric}={primary_str} | "
        f"n={n_items} (✓ {n_corr}, ✗ {n_inc}, ? {n_unk})"
    )

    rows.append("| " + header_cell + " | " + _row_for_block("**all**", length.get("all_rollouts")) + " |")
    rows.append("| ↳ correct | | | " + _row_for_block("**✓**", length.get("correct_rollouts")) + " |")
    rows.append("| ↳ incorrect | | | " + _row_for_block("**✗**", length.get("incorrect_rollouts")) + " |")
    return rows


def _summarize_run(run_dir: Path) -> str:
    # run_dir is .../results/<run>/step_<N>; show "<run>/step_<N>" as the header.
    rel = f"{run_dir.parent.name}/{run_dir.name}" if run_dir.parent != RESULTS_DIR else run_dir.name
    lines: list[str] = []
    lines.append(f"### `{rel}`")
    lines.append("")
    lines.append(
        "| Task | Primary | Items (✓/✗/?) | Subset | gens | Tok mean | Tok std | Tok p50 | Tok p90 |"
    )
    lines.append(
        "|------|---------|----------------|--------|------|----------|---------|---------|---------|"
    )

    for task_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        subtask_metrics_files = sorted(task_dir.glob("task-*-metrics.json"))
        if not subtask_metrics_files:
            lines.append(f"| _{task_dir.name}: no task-*-metrics.json_ | – | – | – | – | – | – | – | – |")
            continue
        for stm in subtask_metrics_files:
            lines.extend(_summarize_subtask(stm))
    lines.append("")
    lines.append(
        "_Tok columns are TOKEN counts across all model_output samples (pass@k "
        "contributes k samples per item). Stratified into all / correct items / "
        "incorrect items based on the per-item primary metric; ✓ rows are samples "
        "from items where the primary metric > 0, ✗ rows are samples from items "
        "where it equals 0. '? items' are items whose per-item metrics don't "
        "expose the primary metric (e.g. alpaca's length_controlled_winrate is "
        "aggregate-only). Full distributions in per-subtask `*-length_stats.json`._"
    )
    lines.append("")
    lines.append(
        "_**Limitation for pass@k tasks (e.g. AIME with k=32):** oe-eval "
        "predictions only record item-level correctness, not per-sample. We "
        "inherit the item-level label to all k samples, which over-counts "
        "correct rollouts whenever some-but-not-all of the k samples passed. "
        "Revisit by re-verifying per-sample (extract `\\boxed{...}` per "
        "continuation) once a run actually has AIME items with pass_at_k > 0._"
    )
    return "\n".join(lines)


def _all_step_dirs() -> list[Path]:
    """Walk results/<run>/step_<N>/ — all runs, all steps, sorted."""
    if not RESULTS_DIR.exists():
        return []
    out: list[Path] = []
    for run_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        out.extend(sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("step_")))
    return out


def _expand_arg(arg: str) -> list[Path]:
    """A positional arg is either '<run>' (expand to all its step dirs) or
    '<run>/step_<N>' (single step dir)."""
    p = RESULTS_DIR / arg
    if not p.exists():
        return [p]  # caller surfaces as NOT FOUND
    if p.is_dir() and any(child.name.startswith("step_") for child in p.iterdir() if child.is_dir()):
        return sorted(child for child in p.iterdir() if child.is_dir() and child.name.startswith("step_"))
    return [p]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*",
                        help="Either '<run>' (all steps) or '<run>/step_<N>' (one step). Default: all.")
    args = parser.parse_args()

    if args.runs:
        run_dirs: list[Path] = []
        for r in args.runs:
            run_dirs.extend(_expand_arg(r))
    else:
        run_dirs = _all_step_dirs()

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
