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

from typing import Literal

import numpy as np


def _expand_per_group_values(values: list[float], prompt_sample_counts: list[int], dtype) -> np.ndarray:
    """Expand per-group scalars into a per-sample array by repeating each value ``sample_count`` times."""
    if len(values) != len(prompt_sample_counts):
        raise ValueError(
            f"Mismatch between values and prompt_sample_counts: {len(values)} vs {len(prompt_sample_counts)}."
        )
    return np.concatenate(
        [np.full(sample_count, value, dtype=dtype) for value, sample_count in zip(values, prompt_sample_counts)]
    )


def expand_grouped_scores(scores: np.ndarray, prompt_sample_counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-group mean/std from ``scores`` grouped by ``prompt_sample_counts`` and expand back to per-sample
    arrays for advantage normalization."""
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
    """Per prompt group: keep advantages at max-reward samples fixed; scale other advantages so the group sum is zero.

    Positive samples are identified from ``reward_scores`` (group max reward; ties count). Each negative advantage is
    multiplied by ``-(sum of positive advantages) / (sum of negative advantages)``. Groups with no negatives, or with
    zero sum on negatives, are left unchanged.
    """
    if sum(prompt_sample_counts) != len(advantages):
        raise ValueError(
            "Mismatch between prompt_sample_counts and advantages: "
            f"{sum(prompt_sample_counts)} grouped samples vs {len(advantages)} advantages."
        )
    if len(reward_scores) != len(advantages):
        raise ValueError(f"reward_scores length {len(reward_scores)} != advantages length {len(advantages)}.")
    out = advantages.astype(np.float64, copy=True)
    r = reward_scores.astype(np.float64, copy=False)
    start = 0
    for sample_count in prompt_sample_counts:
        group_adv = out[start : start + sample_count]
        group_r = r[start : start + sample_count]
        max_r = float(np.max(group_r))
        is_positive = np.isclose(group_r, max_r)
        if not np.any(~is_positive):
            start += sample_count
            continue
        sum_pos = float(group_adv[is_positive].sum())
        neg_vals = group_adv[~is_positive]
        sum_neg = float(neg_vals.sum())
        if sum_neg == 0:
            start += sample_count
            continue
        scale = -sum_pos / sum_neg
        group_adv[~is_positive] = neg_vals * scale
        start += sample_count
    return out.astype(advantages.dtype, copy=False)


def expand_grouped_advantage_scales(
    prompt_sample_counts: list[int], prompt_baseline_sample_counts: list[int] | None = None
) -> np.ndarray:
    """Expand per-group advantage scales for max-RL retry-chain normalization."""
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
    """Compute per-sample advantages from raw scores grouped by prompt.

    ``ngu_count_baseline`` controls whether the NGU pending sums/counts are used for the GRPO mean baseline:

    * ``True`` (default): use ``prompt_baseline_reward_sums`` / ``prompt_baseline_sample_counts`` (which include
      discarded NGU attempts) for the baseline mean. Std is always computed from the in-batch scores only since NGU
      tracks aggregate sums but not individual reward values.
    * ``False``: ignore pending NGU sums/counts and compute the mean from the in-batch scores only (standard GRPO
      mean over the training batch).

    ``ngu_count_rescale`` controls never-give-up baseline handling (applied after mean/std normalization):

    * ``None``: no extra step after normalization.
    * ``anchor_pos``: after normalization, per prompt group keep advantages at max-reward samples (by raw ``scores``)
      fixed and scale other advantages so the group sum is zero.
    * ``ratio``: multiply each advantage by ``sample_count / baseline_sample_count``.
    * ``count_ratio``: divide each advantage by the per-group NGU-count mean baseline (computed from
      ``prompt_baseline_reward_sums`` / ``prompt_baseline_sample_counts``), regardless of ``ngu_count_baseline``.
    """
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
