"""Metrics computation for evolving rubrics.

Provides functions to:
- Collect per-rubric-key statistics (mean, std) from scored rubrics.
- Filter/prune the rubric buffer by deactivating low-signal rubrics.
- Produce a metrics dict suitable for logging to wandb / console.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


# ------------------------------------------------------------------
# Generation-phase metrics (from update_ground_truths_with_evolving_rubrics)
# ------------------------------------------------------------------


def compute_generation_metrics(
    valid_evolving_rubric_rate: float,
    avg_num_ground_truths: float,
    avg_num_evolving_rubrics: float,
    avg_num_active_buffer_rubrics: float,
    skipped_count: int,
) -> dict[str, float]:
    """Build a metrics dict from the values returned by ``update_ground_truths_with_evolving_rubrics``.

    Args:
        valid_evolving_rubric_rate: Fraction of prompts that received valid evolving rubrics.
        avg_num_ground_truths: Mean number of ground-truth (persistent) rubrics per prompt.
        avg_num_evolving_rubrics: Mean number of newly generated evolving rubrics per prompt.
        avg_num_active_buffer_rubrics: Mean number of active rubrics in the buffer per query.
        skipped_count: Number of rubric generation attempts that returned None.

    Returns:
        Dict of ``"objective/..."`` metric keys → float values.
    """
    return {
        "objective/valid_evolving_rubric_rate": valid_evolving_rubric_rate,
        "objective/avg_num_ground_truths": avg_num_ground_truths,
        "objective/avg_num_evolving_rubrics": avg_num_evolving_rubrics,
        "objective/avg_num_active_buffer_rubrics": avg_num_active_buffer_rubrics,
        "objective/skipped_evolving_rubrics": float(skipped_count),
    }


# ------------------------------------------------------------------
# Rubric-key-level statistics (computed from per-response rubric scores)
# ------------------------------------------------------------------


def compute_rubric_key_stats(rubric_scores_by_key: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Compute mean and standard deviation for each rubric key.

    Args:
        rubric_scores_by_key: Mapping from rubric key (e.g. ``query::title``)
            to the list of scores that rubric received across responses.

    Returns:
        Mapping from rubric key to ``{"mean": …, "std": …}``.
    """
    stats: dict[str, dict[str, float]] = {}
    for key, scores_list in rubric_scores_by_key.items():
        arr = np.array(scores_list, dtype=np.float64)
        stats[key] = {"mean": float(arr.mean()), "std": float(arr.std())}
    return stats


def aggregate_rubric_key_metrics(rubric_key_stats: dict[str, dict[str, float]]) -> dict[str, float]:
    """Aggregate per-rubric-key stats into loggable summary metrics.

    Metrics produced:
    - ``rubric_keys/avg_mean``: Average of per-key means.
    - ``rubric_keys/avg_std``: Average of per-key standard deviations.
    - ``rubric_keys/num_all_zero_rubrics_ratio``: Fraction of keys with mean=0 and std=0.
    - ``rubric_keys/num_all_same_value_non_zero_rubrics_ratio``: Fraction with mean>0 and std=0.

    Args:
        rubric_key_stats: Output of :func:`compute_rubric_key_stats`.

    Returns:
        Dict of ``"rubric_keys/..."`` metric keys → float values.
    """
    if not rubric_key_stats:
        return {}

    means = [s["mean"] for s in rubric_key_stats.values()]
    stds = [s["std"] for s in rubric_key_stats.values()]
    n = len(rubric_key_stats)

    num_all_zero = sum(1 for s in rubric_key_stats.values() if s["mean"] == 0 and s["std"] == 0)
    num_same_nonzero = sum(1 for s in rubric_key_stats.values() if s["mean"] > 0 and s["std"] == 0)

    return {
        "rubric_keys/avg_mean": float(np.mean(means)),
        "rubric_keys/avg_std": float(np.mean(stds)),
        "rubric_keys/num_all_zero_rubrics_ratio": num_all_zero / n,
        "rubric_keys/num_all_same_value_non_zero_rubrics_ratio": num_same_nonzero / n,
    }


# ------------------------------------------------------------------
# Rubric buffer filtering
# ------------------------------------------------------------------


def filter_rubric_buffer(
    rubric_buffer: dict[str, Any],
    rubric_key_stats: dict[str, dict[str, float]],
    max_active_rubrics: int,
    create_rubric_key_fn: Any | None = None,
) -> dict[str, float]:
    """Prune the rubric buffer by deactivating low-signal evolving rubrics.

    The pruning strategy (from the original implementation):
    1. **Deactivate zero-std rubrics**: rubrics whose scores had zero variance
       across responses are uninformative and are moved to inactive.
    2. **Cap active rubrics**: if a query still exceeds ``max_active_rubrics``,
       keep only the top rubrics ranked by score variance (std).

    Args:
        rubric_buffer: Mutable mapping ``query → {active_rubrics, inactive_rubrics, persistent_rubrics}``.
        rubric_key_stats: Per-rubric-key statistics from :func:`compute_rubric_key_stats`.
        max_active_rubrics: Maximum number of active evolving rubrics per query.
        create_rubric_key_fn: Optional callable ``(query, rubric) → str`` that produces
            the key used in ``rubric_key_stats``.  Defaults to ``f"{query}::{rubric['title']}"``.

    Returns:
        Dict of buffer-filtering metrics:
        - ``rubric_buffer/deactivated_zero_std``: number moved due to zero std.
        - ``rubric_buffer/deactivated_over_cap``: number moved due to cap.
        - ``rubric_buffer/avg_active_rubrics``: mean active rubrics per query after filtering.
    """
    if create_rubric_key_fn is None:

        def create_rubric_key_fn(query: str, rubric: dict) -> str:
            return f"{query}::{rubric.get('title', '')}"

    # Phase 1: deactivate zero-std rubrics
    moved_zero_std = 0
    rubrics_by_query_std: dict[str, list[tuple[dict, float]]] = {}

    for rubric_key, stats in rubric_key_stats.items():
        for query, buffer_data in rubric_buffer.items():
            active_rubrics = buffer_data.get("active_rubrics", [])
            for rubric in active_rubrics:
                if create_rubric_key_fn(query, rubric) == rubric_key:
                    if stats["std"] == 0:
                        buffer_data["active_rubrics"].remove(rubric)
                        buffer_data["inactive_rubrics"].append(rubric)
                        moved_zero_std += 1
                    else:
                        rubrics_by_query_std.setdefault(query, []).append((rubric, stats["std"]))
                    break
            else:
                continue
            break

    # Phase 2: cap per-query active rubrics
    moved_cap = 0
    for query, rubric_std_pairs in rubrics_by_query_std.items():
        buffer_data = rubric_buffer[query]
        active_rubrics = buffer_data["active_rubrics"]

        if len(active_rubrics) > max_active_rubrics:
            rubric_std_pairs.sort(key=lambda x: x[1], reverse=True)
            keys_to_keep = {create_rubric_key_fn(query, rubric) for rubric, _ in rubric_std_pairs[:max_active_rubrics]}

            new_active: list[dict] = []
            for rubric in active_rubrics:
                rk = create_rubric_key_fn(query, rubric)
                if rk in keys_to_keep or rk not in rubric_key_stats:
                    new_active.append(rubric)
                else:
                    buffer_data["inactive_rubrics"].append(rubric)
                    moved_cap += 1
            buffer_data["active_rubrics"] = new_active

    if moved_zero_std > 0 or moved_cap > 0:
        logger.info(
            f"[Rubric buffer filtering] Deactivated {moved_zero_std} zero-std rubrics, {moved_cap} over-cap rubrics"
        )

    # Compute average active rubrics
    active_counts = [len(b["active_rubrics"]) for b in rubric_buffer.values()]
    avg_active = float(np.mean(active_counts)) if active_counts else 0.0

    return {
        "rubric_buffer/deactivated_zero_std": float(moved_zero_std),
        "rubric_buffer/deactivated_over_cap": float(moved_cap),
        "rubric_buffer/avg_active_rubrics": avg_active,
    }
