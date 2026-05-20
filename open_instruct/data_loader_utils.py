# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grouped advantage helpers for GRPO-style reward normalization."""

import contextlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from open_instruct import data_types
from open_instruct.utils import combine_reward_metrics


@dataclass
class NeverGiveUpAccumulationState:
    """State for carrying pending never_give_up retries across accumulation calls."""

    pending_results: dict[str, list[data_types.GenerationResult]] = field(default_factory=dict)
    pending_metrics: dict[str, list[dict[str, Any] | None]] = field(default_factory=dict)
    pending_best_reward: dict[str, float] = field(default_factory=dict)
    pending_response_counts: dict[str, int] = field(default_factory=dict)
    pending_reward_sums: dict[str, float] = field(default_factory=dict)
    pending_attempt_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class PendingNeverGiveUpState:
    results: list[data_types.GenerationResult]
    metrics: list[dict[str, Any] | None]
    best_reward: float | None
    response_count: int
    reward_sum: float
    attempt_count: int


def get_never_give_up_retry_suffix(prompt_id: str, epoch_number: int, index: int) -> str:
    """Return a numeric retry suffix for a never_give_up requeue."""
    base_prompt_id = f"{epoch_number}_{index}"
    if prompt_id == base_prompt_id:
        return "_1"
    retry_count = int(prompt_id.removeprefix(f"{base_prompt_id}_"))
    return f"_{retry_count + 1}"


def get_never_give_up_chain_id(prompt_id: str) -> str:
    """Return the base epoch/index prompt id shared by a never_give_up retry chain."""
    prompt_id_parts = prompt_id.split("_")
    if len(prompt_id_parts) == 2:
        return prompt_id
    if len(prompt_id_parts) == 3:
        return "_".join(prompt_id_parts[:2])
    raise ValueError(f"Unexpected prompt_id format for never_give_up retry tracking: {prompt_id}")


def should_requeue_never_give_up(never_give_up: float, never_give_up_int: int, resample_number: int) -> bool:
    if never_give_up_int > 0:
        return resample_number < never_give_up_int
    return np.random.random() < never_give_up


def pop_pending_never_give_up_state(
    never_give_up_state: NeverGiveUpAccumulationState,
    chain_id: str,
    current_model_step: int | None,
    maintain_pending_ngu_age: int,
    never_give_up_state_lock: Any = None,
) -> PendingNeverGiveUpState:
    lock = never_give_up_state_lock or contextlib.nullcontext()
    with lock:
        pending_results = never_give_up_state.pending_results.pop(chain_id, [])
        pending_metrics = never_give_up_state.pending_metrics.pop(chain_id, [])
        pending_best_reward = never_give_up_state.pending_best_reward.pop(chain_id, None)
        pending_response_count = never_give_up_state.pending_response_counts.pop(chain_id, 0)
        pending_reward_sum = never_give_up_state.pending_reward_sums.pop(chain_id, 0.0)
        pending_attempt_count = never_give_up_state.pending_attempt_counts.pop(chain_id, 0)

    if current_model_step is None:
        return PendingNeverGiveUpState(
            pending_results,
            pending_metrics,
            pending_best_reward,
            pending_response_count,
            pending_reward_sum,
            pending_attempt_count,
        )

    filtered_pending = []
    for pending_result, pending_metric in zip(pending_results, pending_metrics, strict=False):
        pending_model_step = pending_result.model_step
        if pending_model_step is None or current_model_step - pending_model_step <= maintain_pending_ngu_age:
            filtered_pending.append((pending_result, pending_metric))

    if not filtered_pending:
        return PendingNeverGiveUpState(
            [], [], pending_best_reward, pending_response_count, pending_reward_sum, pending_attempt_count
        )
    return PendingNeverGiveUpState(
        [pending_result for pending_result, _ in filtered_pending],
        [pending_metric for _, pending_metric in filtered_pending],
        pending_best_reward,
        pending_response_count,
        pending_reward_sum,
        pending_attempt_count,
    )


def store_pending_never_give_up_state(
    never_give_up_state: NeverGiveUpAccumulationState,
    chain_id: str,
    pending_state: PendingNeverGiveUpState,
    best_reward: float,
    attempt_count: int,
    never_give_up_state_lock: Any = None,
) -> None:
    lock = never_give_up_state_lock or contextlib.nullcontext()
    with lock:
        if pending_state.results:
            never_give_up_state.pending_results[chain_id] = pending_state.results
            never_give_up_state.pending_metrics[chain_id] = pending_state.metrics
        else:
            never_give_up_state.pending_results.pop(chain_id, None)
            never_give_up_state.pending_metrics.pop(chain_id, None)
        never_give_up_state.pending_best_reward[chain_id] = best_reward
        never_give_up_state.pending_response_counts[chain_id] = pending_state.response_count
        never_give_up_state.pending_reward_sums[chain_id] = pending_state.reward_sum
        never_give_up_state.pending_attempt_counts[chain_id] = attempt_count


def should_accept_never_give_up_batch(
    reward_scores: np.ndarray,
    pending_best_reward: float | None,
    filter_zero_std_samples: bool,
    never_give_up_accept_on: Literal["better", "different"],
) -> bool:
    if not filter_zero_std_samples:
        return True

    non_zero_std_reward = np.std(reward_scores) != 0
    if pending_best_reward is None:
        return bool(non_zero_std_reward)
    current_reward = float(reward_scores.max())
    if never_give_up_accept_on == "different":
        return bool(non_zero_std_reward or current_reward != pending_best_reward)
    if never_give_up_accept_on == "better":
        return current_reward > pending_best_reward
    raise ValueError(f"Invalid never_give_up_accept_on: {never_give_up_accept_on!r}")


def merge_generation_results(
    results: list[data_types.GenerationResult], reward_metrics: list[dict[str, Any] | None]
) -> tuple[data_types.GenerationResult, dict[str, Any]]:
    """Merge buffered never_give_up attempts into a single logical prompt group."""
    if len(results) == 0:
        raise ValueError("Cannot merge an empty list of GenerationResults.")

    combined_responses = []
    combined_finish_reasons = []
    combined_masks = []
    combined_logprobs = []
    combined_num_calls = []
    combined_timeouts = []
    combined_tool_errors = []
    combined_tool_outputs = []
    combined_tool_runtimes = []
    combined_tool_calleds = []
    combined_tool_call_stats = []
    combined_rollout_states = []
    total_prompt_tokens = 0
    total_response_tokens = 0
    max_generation_time = 0.0
    earliest_start_time = float("inf")

    for result in results:
        combined_responses.extend(result.responses)
        combined_finish_reasons.extend(result.finish_reasons)
        combined_masks.extend(result.masks)
        if result.logprobs is not None:
            combined_logprobs.extend(result.logprobs)
        combined_num_calls.extend(result.request_info.num_calls)
        combined_timeouts.extend(result.request_info.timeouts)
        combined_tool_errors.extend(result.request_info.tool_errors)
        combined_tool_outputs.extend(result.request_info.tool_outputs)
        combined_tool_runtimes.extend(result.request_info.tool_runtimes)
        combined_tool_calleds.extend(result.request_info.tool_calleds)
        combined_tool_call_stats.extend(result.request_info.tool_call_stats)
        combined_rollout_states.extend(result.request_info.rollout_states)
        if result.token_statistics is not None:
            total_prompt_tokens += result.token_statistics.num_prompt_tokens
            total_response_tokens += result.token_statistics.num_response_tokens
            max_generation_time = max(max_generation_time, result.token_statistics.generation_time)
            if result.start_time is not None:
                earliest_start_time = min(earliest_start_time, result.start_time)

    combined_request_info = data_types.RequestInfo(
        num_calls=combined_num_calls,
        timeouts=combined_timeouts,
        tool_errors=combined_tool_errors,
        tool_outputs=combined_tool_outputs,
        tool_runtimes=combined_tool_runtimes,
        tool_calleds=combined_tool_calleds,
        tool_call_stats=combined_tool_call_stats,
        rollout_states=combined_rollout_states,
    )
    token_statistics = None
    if any(result.token_statistics is not None for result in results):
        token_statistics = data_types.TokenStatistics(
            num_prompt_tokens=total_prompt_tokens,
            num_response_tokens=total_response_tokens,
            generation_time=max_generation_time,
            earliest_start_time=None if earliest_start_time == float("inf") else earliest_start_time,
        )

    merged_result = data_types.GenerationResult(
        responses=combined_responses,
        finish_reasons=combined_finish_reasons,
        masks=combined_masks,
        request_info=combined_request_info,
        index=results[-1].index,
        prompt_id=results[-1].prompt_id,
        token_statistics=token_statistics,
        start_time=0.0 if earliest_start_time == float("inf") else earliest_start_time,
        logprobs=combined_logprobs,
        reward_scores=[score for result in results for score in (result.reward_scores or [])],
        reward_metrics=combine_reward_metrics([metrics if metrics is not None else {} for metrics in reward_metrics]),
        model_step=results[-1].model_step,
    )
    return merged_result, merged_result.reward_metrics or {}


def _expand_per_group_values(values: list[float], prompt_sample_counts: list[int], dtype) -> np.ndarray:
    if len(values) != len(prompt_sample_counts):
        raise ValueError(
            f"Mismatch between values and prompt_sample_counts: {len(values)} vs {len(prompt_sample_counts)}."
        )
    return np.concatenate(
        [np.full(sample_count, value, dtype=dtype) for value, sample_count in zip(values, prompt_sample_counts)]
    )


def expand_grouped_scores(scores: np.ndarray, prompt_sample_counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-group means/stds and expand them back to per-sample arrays."""
    if sum(prompt_sample_counts) != len(scores):
        raise ValueError(
            "Mismatch between prompt_sample_counts and scores: "
            f"{sum(prompt_sample_counts)} grouped samples vs {len(scores)} scores."
        )

    group_means = []
    group_stds = []
    start = 0
    for sample_count in prompt_sample_counts:
        group_scores = scores[start : start + sample_count]
        group_means.append(float(group_scores.mean()))
        group_stds.append(float(group_scores.std()))
        start += sample_count

    return (
        _expand_per_group_values(group_means, prompt_sample_counts, dtype=scores.dtype),
        _expand_per_group_values(group_stds, prompt_sample_counts, dtype=scores.dtype),
    )


def _apply_anchor_pos_advantage_rescaling(
    advantages: np.ndarray, reward_scores: np.ndarray, prompt_sample_counts: list[int]
) -> np.ndarray:
    """Keep max-reward sample advantages fixed and scale the rest so each group sums to zero."""
    if sum(prompt_sample_counts) != len(advantages):
        raise ValueError(
            "Mismatch between prompt_sample_counts and advantages: "
            f"{sum(prompt_sample_counts)} grouped samples vs {len(advantages)} advantages."
        )
    if len(reward_scores) != len(advantages):
        raise ValueError(f"reward_scores length {len(reward_scores)} != advantages length {len(advantages)}.")

    out = advantages.astype(np.float64, copy=True)
    reward_scores = reward_scores.astype(np.float64, copy=False)
    start = 0
    for sample_count in prompt_sample_counts:
        group_adv = out[start : start + sample_count]
        group_scores = reward_scores[start : start + sample_count]
        is_positive = np.isclose(group_scores, float(np.max(group_scores)))
        if not np.any(~is_positive):
            start += sample_count
            continue

        sum_pos = float(group_adv[is_positive].sum())
        neg_vals = group_adv[~is_positive]
        sum_neg = float(neg_vals.sum())
        if sum_neg != 0:
            group_adv[~is_positive] = neg_vals * (-sum_pos / sum_neg)
        start += sample_count
    return out.astype(advantages.dtype, copy=False)


def expand_grouped_advantage_scales(
    prompt_sample_counts: list[int], prompt_baseline_sample_counts: list[int] | None = None
) -> np.ndarray:
    """Expand per-group sample_count / baseline_sample_count scales."""
    if prompt_baseline_sample_counts is None:
        prompt_baseline_sample_counts = prompt_sample_counts
    if len(prompt_sample_counts) != len(prompt_baseline_sample_counts):
        raise ValueError(
            "Mismatch between prompt_sample_counts and prompt_baseline_sample_counts: "
            f"{len(prompt_sample_counts)} vs {len(prompt_baseline_sample_counts)}."
        )

    advantage_scales = []
    for sample_count, baseline_sample_count in zip(prompt_sample_counts, prompt_baseline_sample_counts, strict=True):
        if baseline_sample_count < sample_count:
            raise ValueError(
                "Each prompt_baseline_sample_count must be >= its prompt_sample_count, got "
                f"{baseline_sample_count} < {sample_count}."
            )
        advantage_scales.append(np.full(sample_count, sample_count / baseline_sample_count, dtype=np.float32))
    return np.concatenate(advantage_scales)


def compute_grouped_advantages(
    scores: np.ndarray,
    prompt_sample_counts: list[int],
    prompt_baseline_sample_counts: list[int] | None = None,
    prompt_baseline_reward_sums: list[float] | None = None,
    advantage_normalization_type: str = "centered",
    ngu_count_rescale: Literal["anchor_pos", "ratio", "count_ratio"] | None = None,
    ngu_count_baseline: bool = True,
) -> np.ndarray:
    """Compute per-sample advantages from raw scores grouped by prompt."""
    mean_grouped_rewards, std_grouped_rewards = expand_grouped_scores(scores, prompt_sample_counts)

    have_ngu_baseline = prompt_baseline_sample_counts is not None and prompt_baseline_reward_sums is not None
    if have_ngu_baseline:
        ngu_group_means = [s / c for s, c in zip(prompt_baseline_reward_sums, prompt_baseline_sample_counts)]
        ngu_mean_baseline = _expand_per_group_values(ngu_group_means, prompt_sample_counts, dtype=scores.dtype)
        if ngu_count_baseline:
            mean_grouped_rewards = ngu_mean_baseline

    if advantage_normalization_type == "standard":
        advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
    elif advantage_normalization_type == "centered":
        advantages = scores - mean_grouped_rewards
    elif advantage_normalization_type == "max_rl":
        advantages = (scores - mean_grouped_rewards) / (mean_grouped_rewards + 1e-8)
    else:
        raise ValueError(f"Invalid advantage normalization type: {advantage_normalization_type}")

    if ngu_count_rescale == "anchor_pos":
        advantages = _apply_anchor_pos_advantage_rescaling(advantages, scores, prompt_sample_counts)
    elif ngu_count_rescale == "ratio":
        advantages = advantages * expand_grouped_advantage_scales(prompt_sample_counts, prompt_baseline_sample_counts)
    elif ngu_count_rescale == "count_ratio":
        advantages = advantages / (ngu_mean_baseline + 1e-8)

    return advantages
