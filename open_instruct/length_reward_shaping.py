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
"""Dynamic length-aware reward shaping for GRPO/RLVR.

Among the correct responses in a prompt group, this rewards the shortest
correct response most and decays the reward for longer correct responses.
Incorrect responses are unaffected. See the project proposal for motivation.
"""

import numpy as np

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# Allowed values for length_reward_shaping_method.
SHAPING_METHODS = ("none", "linear", "exponential", "rank", "binary_shortest", "soft_threshold")

# Allowed values for length_reward_warmup_type.
WARMUP_TYPES = ("constant", "linear", "solve_rate")


def _correct_mask(scores: np.ndarray, correctness_threshold: float) -> np.ndarray:
    """Boolean mask of correct responses (score > threshold)."""
    return scores > correctness_threshold


def _shape_group(
    scores: np.ndarray, lengths: np.ndarray, method: str, decay_param: float, correctness_threshold: float
) -> np.ndarray:
    """Shape rewards within a single prompt group.

    Args:
        scores: 1D array of raw rewards for the N responses in the group.
        lengths: 1D array of response token lengths for the N responses.
        method: shaping method to apply.
        decay_param: alpha (linear) or lambda (exponential) decay coefficient,
            or threshold fraction for soft_threshold (default 1.0 = median).
        correctness_threshold: a response is "correct" if its score exceeds this.

    Returns:
        1D array of shaped rewards, same shape as `scores`.
    """
    correct = _correct_mask(scores, correctness_threshold)
    # Need at least one correct response with a positive length to compute L_min.
    if not correct.any():
        return scores.copy()
    correct_lengths = lengths[correct]
    if correct_lengths.size == 0:
        return scores.copy()

    shaped = scores.astype(np.float64).copy()
    L_min = float(correct_lengths.min())
    # Avoid division by zero on degenerate (empty) responses.
    L_min_safe = max(L_min, 1.0)

    if method == "linear":
        # r_adj = r * max(0, 1 - alpha * (L - L_min) / L_min)
        rel = (lengths.astype(np.float64) - L_min) / L_min_safe
        factor = np.maximum(0.0, 1.0 - decay_param * rel)
        shaped[correct] = scores[correct] * factor[correct]
    elif method == "exponential":
        # r_adj = r * exp(-lambda * (L - L_min) / L_min)
        rel = (lengths.astype(np.float64) - L_min) / L_min_safe
        factor = np.exp(-decay_param * rel)
        shaped[correct] = scores[correct] * factor[correct]
    elif method == "rank":
        # Sort correct responses by length ascending; reward scales as 1 / (1 + rank).
        # Ties in length share the same rank (dense ranking).
        idxs_correct = np.where(correct)[0]
        cl = lengths[idxs_correct].astype(np.float64)
        # dense rank of each correct response by length ascending
        order = np.argsort(cl, kind="stable")
        ranks = np.empty_like(order)
        prev_len = None
        cur_rank = -1
        for idx in order:
            length_val = cl[idx]
            if prev_len is None or length_val != prev_len:
                cur_rank += 1
                prev_len = length_val
            ranks[idx] = cur_rank
        factor = 1.0 / (1.0 + ranks.astype(np.float64))
        shaped[idxs_correct] = scores[idxs_correct] * factor
    elif method == "binary_shortest":
        # Only the shortest-length correct response(s) keep their reward.
        is_shortest = correct & (lengths == L_min)
        shaped[correct & ~is_shortest] = 0.0
    elif method == "soft_threshold":
        # Full reward up to (decay_param * median) of correct length, then linear decay.
        # decay_param defaults to 1.0 (median); >1 gives a wider plateau.
        median_len = float(np.median(correct_lengths))
        threshold = decay_param * median_len
        # Linear decay slope normalized so that L=2*threshold halves the reward.
        factor = np.where(
            lengths.astype(np.float64) <= threshold,
            1.0,
            np.maximum(0.0, 1.0 - (lengths.astype(np.float64) - threshold) / max(threshold, 1.0)),
        )
        shaped[correct] = scores[correct] * factor[correct]
    else:
        raise ValueError(f"Unknown length_reward_shaping_method: {method}")

    return shaped.astype(scores.dtype, copy=False)


def apply_length_reward_shaping(
    scores_per_prompt: np.ndarray,
    lengths_per_prompt: np.ndarray,
    method: str,
    decay_param: float,
    warmup_weight: float = 1.0,
    correctness_threshold: float = 0.0,
) -> np.ndarray:
    """Apply length-aware reward shaping to a batch of prompt groups.

    Shaping is per group (per prompt). Within each group, only correct responses
    are reweighted; incorrect responses keep their original (typically zero) score.
    The shaped scores are blended with the original scores by `warmup_weight`:
    0.0 = no shaping, 1.0 = full shaping.

    Args:
        scores_per_prompt: shape (num_prompts, num_samples), raw rewards.
        lengths_per_prompt: shape (num_prompts, num_samples), token lengths.
        method: shaping method (see SHAPING_METHODS).
        decay_param: alpha for linear, lambda for exponential, fraction for soft_threshold.
        warmup_weight: 0..1 multiplier for blending shaped vs unshaped scores.
        correctness_threshold: a response is correct if score > threshold.

    Returns:
        Shaped scores with the same shape as scores_per_prompt.
    """
    if method == "none" or warmup_weight <= 0.0:
        return scores_per_prompt.copy()
    if scores_per_prompt.shape != lengths_per_prompt.shape:
        raise ValueError(f"shape mismatch: scores {scores_per_prompt.shape} vs lengths {lengths_per_prompt.shape}")

    shaped = np.empty_like(scores_per_prompt, dtype=np.float64)
    for i in range(scores_per_prompt.shape[0]):
        shaped[i] = _shape_group(
            scores_per_prompt[i], lengths_per_prompt[i], method, decay_param, correctness_threshold
        )

    if warmup_weight >= 1.0:
        return shaped.astype(scores_per_prompt.dtype, copy=False)
    blended = warmup_weight * shaped + (1.0 - warmup_weight) * scores_per_prompt.astype(np.float64)
    return blended.astype(scores_per_prompt.dtype, copy=False)


def compute_warmup_weight(
    step: int,
    num_training_steps: int,
    warmup_type: str,
    warmup_fraction: float,
    solve_rate_threshold: float,
    group_solve_rate: float | None,
) -> float:
    """Compute the warm-up multiplier for the length penalty at this training step.

    Args:
        step: current training step (0-indexed).
        num_training_steps: total training steps planned.
        warmup_type: "constant" (always 1.0), "linear" (ramp 0→1 over warmup_fraction
            of training), or "solve_rate" (0 until group solve rate exceeds threshold,
            then 1).
        warmup_fraction: fraction of training over which to linearly ramp up.
        solve_rate_threshold: solve-rate cutoff for the "solve_rate" type.
        group_solve_rate: current batch's mean solve rate (fraction of correct
            responses); required when warmup_type == "solve_rate".

    Returns:
        weight in [0, 1].
    """
    if warmup_type == "constant":
        return 1.0
    if warmup_type == "linear":
        if warmup_fraction <= 0.0 or num_training_steps <= 0:
            return 1.0
        end_step = max(1, int(warmup_fraction * num_training_steps))
        return float(min(1.0, max(0.0, step / end_step)))
    if warmup_type == "solve_rate":
        if group_solve_rate is None:
            return 0.0
        return 1.0 if group_solve_rate >= solve_rate_threshold else 0.0
    raise ValueError(f"Unknown length_reward_warmup_type: {warmup_type}")
