"""High-level orchestration for evolving rubrics during GRPO training.

``RubricManager`` is the only public class.  ``DataPreparationActor``
imports it from ``open_instruct.rubrics`` and delegates all rubric
state management to it.
"""

import asyncio
from typing import Any

import ray

from open_instruct import logger_utils
from open_instruct.rubrics.metrics import compute_rubric_count_metrics
from open_instruct.rubrics.rubric_utils import (
    _generate_instance_wise_evolving_rubrics,
    initialize_rubric_buffer,
    save_evolving_rubric_cache_safe,
    update_ground_truths_with_evolving_rubrics,
)

logger = logger_utils.setup_logger(__name__)


class RubricManager:
    """Manages evolving rubric state across training steps.

    Encapsulates the rubric buffer, config, and vLLM engine handles so
    that callers only need a single import and two calls
    (``__init__`` and ``run_step``).
    """

    def __init__(self, streaming_config, ground_truths: list) -> None:
        self._num_samples = streaming_config.num_samples_per_prompt_rollout
        self._max_active = streaming_config.max_active_rubrics
        self._cache_dir: str | None = streaming_config.cache_evolving_rubric_data_dir
        self._buffer = initialize_rubric_buffer(ground_truths)
        self._vllm_engines: list = []
        logger.info(f"Initialized rubric buffer with {len(self._buffer)} unique queries")

    def set_vllm_engines(self, engines: list) -> None:
        """Register vLLM engine handles (called after engines are created)."""
        self._vllm_engines = engines

    def run_step(
        self, *, decoded_responses: list[str], ground_truths: list, indices: list[int] | None, step: int
    ) -> dict[str, Any]:
        """Run one evolving-rubric cycle and return metrics."""
        metrics: dict[str, Any] = {}
        try:
            metrics, self._buffer = _run_evolving_rubric_step(
                decoded_responses=decoded_responses,
                ground_truths=ground_truths,
                indices=indices,
                num_samples=self._num_samples,
                max_active=self._max_active,
                cache_dir=self._cache_dir,
                rubric_buffer=self._buffer,
                vllm_engines=self._vllm_engines,
                step=step,
            )
        except Exception:
            logger.exception("Error in evolving rubric step")
        return metrics


# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------


def _run_evolving_rubric_step(
    *,
    decoded_responses: list[str],
    ground_truths: list,
    indices: list[int] | None,
    num_samples: int,
    max_active: int,
    cache_dir: str | None,
    rubric_buffer: dict[str, Any] | None,
    vllm_engines: list,
    step: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    metrics: dict[str, Any] = {}

    all_evolving_rubrics, num_subsampled = asyncio.run(
        _generate_instance_wise_evolving_rubrics(
            responses=decoded_responses,
            ground_truths=ground_truths,
            num_samples_per_prompt_rollout=num_samples,
            rubric_buffer=rubric_buffer,
        )
    )

    (
        updated_ground_truths,
        valid_rate,
        avg_gt_rubrics,
        avg_evolving_rubrics,
        avg_active_buffer,
        rubric_buffer,
        skipped,
    ) = update_ground_truths_with_evolving_rubrics(
        ground_truths=ground_truths,
        all_evolving_rubrics=all_evolving_rubrics,
        num_samples_per_prompt_rollout=num_samples,
        rubric_buffer=rubric_buffer,
    )

    if rubric_buffer is not None:
        _cap_active_rubrics(rubric_buffer, max_active)

    _push_ground_truth_overrides(updated_ground_truths, indices, vllm_engines)

    metrics.update(compute_rubric_count_metrics(avg_evolving_rubrics, avg_active_buffer))
    metrics["evolving_rubrics/valid_rate"] = valid_rate
    metrics["evolving_rubrics/avg_gt_rubrics"] = avg_gt_rubrics
    metrics["evolving_rubrics/skipped"] = skipped

    if cache_dir:
        save_evolving_rubric_cache_safe(
            cache_dir=cache_dir,
            training_step=step,
            decoded_responses=decoded_responses,
            ground_truths=ground_truths,
            all_evolving_rubrics=all_evolving_rubrics,
            num_subsampled_answers_list=num_subsampled,
            num_samples_per_prompt_rollout=num_samples,
            use_full_responses=True,
            answer_length_limit_in_words=None,
        )

    logger.info(
        f"Step {step}: evolving rubrics generated. "
        f"valid_rate={valid_rate:.2f}, avg_new={avg_evolving_rubrics:.1f}, "
        f"avg_active_buffer={avg_active_buffer:.1f}, skipped={skipped}"
    )

    return metrics, rubric_buffer


def _cap_active_rubrics(rubric_buffer: dict[str, Any], max_active: int) -> None:
    """Enforce max_active_rubrics per query using FIFO eviction.

    When per-rubric variance stats are not yet available (early training),
    ``filter_rubric_buffer`` from metrics.py is a no-op.  This function
    provides a simple fallback: keep the *most recently added* rubrics and
    move the oldest excess ones to inactive.
    """
    for query, buf in rubric_buffer.items():
        active = buf.get("active_rubrics", [])
        if len(active) > max_active:
            evicted = active[:-max_active]
            buf["active_rubrics"] = active[-max_active:]
            buf.setdefault("inactive_rubrics", []).extend(evicted)
            logger.debug(f"Capped {len(evicted)} rubrics for query '{query[:60]}...'")


def _push_ground_truth_overrides(updated_ground_truths: list, indices: list[int] | None, vllm_engines: list) -> None:
    """Send updated ground truths to vLLM engines for future reward computation."""
    if not vllm_engines or not indices:
        return

    overrides: dict[int, Any] = {}
    seen: set[int] = set()
    for gt, idx in zip(updated_ground_truths, indices):
        if idx not in seen:
            overrides[idx] = gt
            seen.add(idx)

    if overrides:
        ray.get([engine.update_ground_truths.remote(overrides) for engine in vllm_engines])
        logger.info(f"Pushed {len(overrides)} ground truth overrides to {len(vllm_engines)} vLLM engines")
