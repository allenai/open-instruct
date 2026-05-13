"""Global metrics for evolving rubrics.

Provides functions to compute and log:
- Average reward from evolving rubrics vs persistent (static) rubrics.
- Number of newly generated evolving rubrics per step.
- Number of active rubrics in the buffer.
- Rubric buffer filtering (deactivating uninformative rubrics).
"""

from typing import Any

import numpy as np

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def compute_rubric_reward_metrics(
    per_rubric_scores: list[list[tuple[float, float]]], per_rubric_types: list[list[str]]
) -> dict[str, float]:
    """Compute average reward split by rubric type (evolving vs persistent).

    Args:
        per_rubric_scores: For each response, a list of ``(score, weight)`` tuples
            — one per rubric that was scored.
        per_rubric_types: For each response, a list of rubric type strings
            (``"persistent"`` or ``"evolving"``) aligned with ``per_rubric_scores``.

    Returns:
        Dict with:
        - ``evolving_rubrics/avg_evolving_reward``: mean weighted score from evolving rubrics.
        - ``evolving_rubrics/std_evolving_reward``: std of weighted scores from evolving rubrics.
        - ``evolving_rubrics/avg_persistent_reward``: mean weighted score from persistent rubrics.
        - ``evolving_rubrics/std_persistent_reward``: std of weighted scores from persistent rubrics.
    """
    evolving_weighted_scores: list[float] = []
    persistent_weighted_scores: list[float] = []

    for scores, types in zip(per_rubric_scores, per_rubric_types):
        for (score, weight), rtype in zip(scores, types):
            weighted = score * weight
            if rtype == "evolving":
                evolving_weighted_scores.append(weighted)
            else:
                persistent_weighted_scores.append(weighted)

    ev_arr = np.array(evolving_weighted_scores) if evolving_weighted_scores else np.array([])
    ps_arr = np.array(persistent_weighted_scores) if persistent_weighted_scores else np.array([])

    return {
        "evolving_rubrics/avg_evolving_reward": float(ev_arr.mean()) if ev_arr.size > 0 else 0.0,
        "evolving_rubrics/std_evolving_reward": float(ev_arr.std()) if ev_arr.size > 0 else 0.0,
        "evolving_rubrics/avg_persistent_reward": float(ps_arr.mean()) if ps_arr.size > 0 else 0.0,
        "evolving_rubrics/std_persistent_reward": float(ps_arr.std()) if ps_arr.size > 0 else 0.0,
    }


def compute_rubric_count_metrics(
    avg_num_evolving_rubrics: float, avg_num_active_buffer_rubrics: float
) -> dict[str, float]:
    """Compute rubric count metrics for logging.

    Args:
        avg_num_evolving_rubrics: Mean number of newly generated evolving rubrics per prompt.
        avg_num_active_buffer_rubrics: Mean number of active rubrics in the buffer per query.

    Returns:
        Dict with:
        - ``evolving_rubrics/num_new_rubrics``: newly generated evolving rubrics per prompt.
        - ``evolving_rubrics/num_active_rubrics``: active rubrics in the buffer per query.
    """
    return {
        "evolving_rubrics/num_new_rubrics": avg_num_evolving_rubrics,
        "evolving_rubrics/num_active_rubrics": avg_num_active_buffer_rubrics,
    }


def filter_rubric_buffer(
    rubric_buffer: dict[str, Any],
    rubric_key_stats: dict[str, dict[str, float]],
    max_active_rubrics: int,
    create_rubric_key_fn: Any | None = None,
) -> None:
    """Prune the rubric buffer by deactivating low-signal evolving rubrics.

    The pruning strategy:
    1. **Deactivate zero-std rubrics**: rubrics whose scores had zero variance
       across responses are uninformative and are moved to inactive.
    2. **Cap active rubrics**: if a query still exceeds ``max_active_rubrics``,
       keep only the top rubrics ranked by score variance (std).

    Args:
        rubric_buffer: Mutable mapping ``query → {active_rubrics, inactive_rubrics, persistent_rubrics}``.
        rubric_key_stats: Per-rubric-key statistics — mapping from key to ``{"mean": …, "std": …}``.
        max_active_rubrics: Maximum number of active evolving rubrics per query.
        create_rubric_key_fn: Optional callable ``(query, rubric) → str`` that produces
            the key used in ``rubric_key_stats``.  Defaults to ``f"{query}::{rubric['title']}"``.
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
