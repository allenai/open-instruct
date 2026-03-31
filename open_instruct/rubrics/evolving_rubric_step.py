"""High-level orchestration for a single evolving-rubric step.

This module is the only entry point that ``DataPreparationActor`` needs to
import.  It bundles rubric generation, ground-truth updates, buffer
filtering, cache saving, and metric collection so that the caller in
``data_loader.py`` stays thin.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import ray

from open_instruct import logger_utils
from open_instruct.rubrics.metrics import compute_rubric_count_metrics, filter_rubric_buffer
from open_instruct.rubrics.rubric_utils import (
    _generate_instance_wise_evolving_rubrics,
    initialize_rubric_buffer,
    save_evolving_rubric_cache_safe,
    update_ground_truths_with_evolving_rubrics,
)

logger = logger_utils.setup_logger(__name__)


@dataclass
class EvolvingRubricConfig:
    """Subset of StreamingDataLoaderConfig relevant to evolving rubrics."""

    apply_evolving_rubric_reward: bool = False
    num_samples_per_prompt_rollout: int = 4
    max_active_rubrics: int = 5
    cache_evolving_rubric_data_dir: str | None = None

    @classmethod
    def from_streaming_config(cls, cfg) -> "EvolvingRubricConfig":
        return cls(
            apply_evolving_rubric_reward=cfg.apply_evolving_rubric_reward,
            num_samples_per_prompt_rollout=cfg.num_samples_per_prompt_rollout,
            max_active_rubrics=cfg.max_active_rubrics,
            cache_evolving_rubric_data_dir=cfg.cache_evolving_rubric_data_dir,
        )


def init_rubric_buffer(ground_truths: list) -> dict[str, Any]:
    """Thin wrapper so callers need only one import."""
    return initialize_rubric_buffer(ground_truths)


def run_evolving_rubric_step(
    *,
    decoded_responses: list[str],
    ground_truths: list,
    indices: list[int] | None,
    config: EvolvingRubricConfig,
    rubric_buffer: dict[str, Any] | None,
    vllm_engines: list,
    step: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one evolving-rubric cycle: generate, update buffer, push overrides.

    Returns ``(metrics_dict, updated_rubric_buffer)``.
    """
    metrics: dict[str, Any] = {}

    try:
        loop = asyncio.new_event_loop()
        all_evolving_rubrics, num_subsampled = loop.run_until_complete(
            _generate_instance_wise_evolving_rubrics(
                responses=decoded_responses,
                ground_truths=ground_truths,
                num_samples_per_prompt_rollout=config.num_samples_per_prompt_rollout,
                rubric_buffer=rubric_buffer,
            )
        )
        loop.close()

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
            num_samples_per_prompt_rollout=config.num_samples_per_prompt_rollout,
            rubric_buffer=rubric_buffer,
        )

        if rubric_buffer is not None:
            filter_rubric_buffer(rubric_buffer, {}, config.max_active_rubrics)

        _push_ground_truth_overrides(
            updated_ground_truths, indices, config.num_samples_per_prompt_rollout, vllm_engines
        )

        metrics.update(compute_rubric_count_metrics(avg_evolving_rubrics, avg_active_buffer))
        metrics["evolving_rubrics/valid_rate"] = valid_rate
        metrics["evolving_rubrics/avg_gt_rubrics"] = avg_gt_rubrics
        metrics["evolving_rubrics/skipped"] = skipped

        if config.cache_evolving_rubric_data_dir:
            save_evolving_rubric_cache_safe(
                cache_dir=config.cache_evolving_rubric_data_dir,
                training_step=step,
                decoded_responses=decoded_responses,
                ground_truths=ground_truths,
                all_evolving_rubrics=all_evolving_rubrics,
                num_subsampled_answers_list=num_subsampled,
                num_samples_per_prompt_rollout=config.num_samples_per_prompt_rollout,
                use_full_responses=True,
                answer_length_limit_in_words=None,
            )

        logger.info(
            f"Step {step}: evolving rubrics generated. "
            f"valid_rate={valid_rate:.2f}, avg_new={avg_evolving_rubrics:.1f}, "
            f"avg_active_buffer={avg_active_buffer:.1f}, skipped={skipped}"
        )
    except Exception:
        logger.exception("Error in evolving rubric step")

    return metrics, rubric_buffer


def _push_ground_truth_overrides(
    updated_ground_truths: list,
    indices: list[int] | None,
    num_samples_per_prompt_rollout: int,
    vllm_engines: list,
) -> None:
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
        logger.info(
            f"Pushed {len(overrides)} ground truth overrides to {len(vllm_engines)} vLLM engines"
        )
