"""Compute length statistics for one (sub)task's predictions.jsonl.

Reads:
  - predictions.jsonl: one row per eval item; row.model_output is a list of
    one (or k, for pass@k tasks) sample dicts containing num_tokens + continuation.
  - the task's metrics.json: provides task_config.primary_metric, which we use
    to flag each ROW as correct/incorrect (per-item correctness).

Writes a length_stats.json with three nested blocks of length stats:
  - all_rollouts:       stats over every sample
  - correct_rollouts:   stats over samples belonging to items deemed correct
  - incorrect_rollouts: stats over samples from items deemed incorrect

Why split by ITEM correctness, not sample correctness:
  Most eval predictions only carry per-item metrics. For pass@k tasks (AIME)
  there are k samples per item but only an item-level correctness number, so
  we can't distinguish which of the k was the correct one. Stratifying by item
  correctness is the most fine-grained correct/incorrect split available
  without re-running the verifier per-sample.

Why both token and character lengths:
  Reward shaping operated on TOKEN counts (`len(response)` on tokenized
  responses in data_loader.py). Char counts are kept as a sanity-check
  unit that's easy to eyeball alongside raw outputs.

Usage:
    python compute_length_stats.py <predictions.jsonl> <task_metrics.json> <out.json>
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

# Bin edges shared across tasks so histograms are directly comparable across
# runs. Covers tiny single-token answers up through near-max-context responses.
TOKEN_BIN_EDGES = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
CHAR_BIN_EDGES = [0, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 200000]


def _percentiles(xs: list[int], pct: list[int]) -> dict[str, float]:
    if not xs:
        return {f"p{p}": 0 for p in pct}
    s = sorted(xs)
    n = len(s)
    out = {}
    for p in pct:
        idx = min(n - 1, max(0, int(math.floor(p / 100 * n))))
        out[f"p{p}"] = s[idx]
    return out


def _summary(xs: list[int]) -> dict:
    if not xs:
        return {"n": 0}
    mean = sum(xs) / len(xs)
    if len(xs) > 1:
        var = statistics.pvariance(xs)
        std = math.sqrt(var)
    else:
        var = 0.0
        std = 0.0
    out = {
        "n": len(xs),
        "min": min(xs),
        "max": max(xs),
        "mean": mean,
        "std": std,
        "var": var,
    }
    out.update(_percentiles(xs, [10, 25, 50, 75, 90, 95, 99]))
    return out


def _histogram(xs: list[int], edges: list[int]) -> dict:
    counts = [0] * (len(edges) - 1)
    for x in xs:
        for i in range(len(edges) - 1):
            if x < edges[i + 1] or i == len(counts) - 1:
                counts[i] += 1
                break
    return {
        "edges": edges,
        "counts": counts,
        "labels": [f"[{edges[i]},{edges[i + 1]})" for i in range(len(counts))],
    }


def _stat_block(xs_tok: list[int], xs_char: list[int]) -> dict:
    block = {
        "tokens": _summary(xs_tok),
        "tokens_histogram": _histogram(xs_tok, TOKEN_BIN_EDGES) if xs_tok else None,
        "tokens_raw": xs_tok,
        "chars": _summary(xs_char),
        "chars_histogram": _histogram(xs_char, CHAR_BIN_EDGES) if xs_char else None,
    }
    return block


def _per_item_correct(row_metrics: dict | None, primary_metric: str | None) -> bool | None:
    """Return True/False if we can determine per-item correctness, else None."""
    if not isinstance(row_metrics, dict) or not primary_metric:
        return None
    v = row_metrics.get(primary_metric)
    if isinstance(v, (int, float)):
        return v > 0
    return None


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: compute_length_stats.py <predictions.jsonl> <task_metrics.json> <out.json>",
              file=sys.stderr)
        sys.exit(2)

    pred_path = Path(sys.argv[1])
    task_metrics_path = Path(sys.argv[2])
    dst = Path(sys.argv[3])

    primary_metric = None
    if task_metrics_path.exists():
        meta = json.loads(task_metrics_path.read_text())
        primary_metric = (meta.get("task_config") or {}).get("primary_metric")

    all_tok: list[int] = []
    all_char: list[int] = []
    correct_tok: list[int] = []
    correct_char: list[int] = []
    incorrect_tok: list[int] = []
    incorrect_char: list[int] = []

    n_items = 0
    n_items_correct = 0
    n_items_incorrect = 0
    n_items_unknown = 0
    samples_per_item: set[int] = set()

    with pred_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_items += 1
            outputs = obj.get("model_output") or []
            samples_per_item.add(len(outputs))
            correct = _per_item_correct(obj.get("metrics"), primary_metric)
            if correct is True:
                n_items_correct += 1
            elif correct is False:
                n_items_incorrect += 1
            else:
                n_items_unknown += 1

            for out in outputs:
                ntok = out.get("num_tokens")
                cont = out.get("continuation")
                if isinstance(ntok, int):
                    all_tok.append(ntok)
                    if correct is True:
                        correct_tok.append(ntok)
                    elif correct is False:
                        incorrect_tok.append(ntok)
                if isinstance(cont, str):
                    all_char.append(len(cont))
                    if correct is True:
                        correct_char.append(len(cont))
                    elif correct is False:
                        incorrect_char.append(len(cont))

    result = {
        "primary_metric": primary_metric,
        "n_items": n_items,
        "n_items_correct": n_items_correct,
        "n_items_incorrect": n_items_incorrect,
        "n_items_unknown": n_items_unknown,
        "n_generations": len(all_tok) if all_tok else len(all_char),
        "samples_per_item": sorted(samples_per_item) if samples_per_item else [],
        "all_rollouts": _stat_block(all_tok, all_char),
        "correct_rollouts": _stat_block(correct_tok, correct_char),
        "incorrect_rollouts": _stat_block(incorrect_tok, incorrect_char),
    }

    dst.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
