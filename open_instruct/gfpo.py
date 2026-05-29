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
"""Group Filtered Policy Optimization (GFPO).

GFPO (Shrivastava et al., 2025, arXiv:2508.09726) curbs response-length
inflation in RLVR by *oversampling* a larger group of G responses per prompt,
then training only on the top-k responses ranked by a target metric — response
length (shortest-first) or token efficiency (reward/length, highest-first).
Advantages are normalized over the retained subset only; the (G - k)
non-retained responses receive zero advantage and contribute no policy gradient.

Unlike length-aware reward shaping (open_instruct/length_reward_shaping.py),
GFPO leaves the reward untouched and filters purely on the metric — correctness
stays entirely in the reward, so it is not conditioned on in the filter. The two
methods are mutually exclusive.
"""

import numpy as np

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# Allowed values for gfpo_filter_metric.
GFPO_METRICS = ("none", "shortest", "token_efficiency")


def apply_gfpo_filter(
    scores_per_prompt: np.ndarray, lengths_per_prompt: np.ndarray, retain_k: int, metric: str
) -> np.ndarray:
    """Select the top-k responses per prompt group by the GFPO metric.

    Args:
        scores_per_prompt: shape (num_prompts, G), raw rewards.
        lengths_per_prompt: shape (num_prompts, G), response token lengths.
        retain_k: number of responses to keep per group (k). Clamped to G.
        metric: "shortest" (keep k smallest lengths) or "token_efficiency"
            (keep k largest reward/length).

    Returns:
        Boolean keep-mask of shape (num_prompts, G); True for retained responses.
        Each row has exactly min(retain_k, G) True entries.
    """
    if metric not in ("shortest", "token_efficiency"):
        raise ValueError(f"gfpo_filter_metric must be 'shortest' or 'token_efficiency', got {metric!r}")
    if scores_per_prompt.shape != lengths_per_prompt.shape:
        raise ValueError(f"shape mismatch: scores {scores_per_prompt.shape} vs lengths {lengths_per_prompt.shape}")

    num_prompts, group_size = scores_per_prompt.shape
    k = min(retain_k, group_size)

    if metric == "shortest":
        # Rank by length ascending; keep the k shortest.
        rank_metric = lengths_per_prompt.astype(np.float64)
        keep_smallest = True
    else:  # token_efficiency
        # reward / length; keep the k highest. Guard div-by-zero on empty responses.
        lengths_safe = np.maximum(lengths_per_prompt.astype(np.float64), 1.0)
        rank_metric = scores_per_prompt.astype(np.float64) / lengths_safe
        keep_smallest = False

    order = np.argsort(rank_metric, axis=1, kind="stable")  # ascending per row
    keep_idx = order[:, :k] if keep_smallest else order[:, group_size - k :]

    mask = np.zeros_like(scores_per_prompt, dtype=bool)
    np.put_along_axis(mask, keep_idx, True, axis=1)
    return mask


def compute_gfpo_advantages(
    scores_per_prompt: np.ndarray, keep_mask: np.ndarray, normalization_type: str = "standard"
) -> np.ndarray:
    """Compute GFPO advantages: normalize over the retained subset, zero the rest.

    The per-group baseline (mean, and std for "standard") is computed over the
    retained responses only (mu_S, sigma_S in the paper). Non-retained responses
    get advantage 0 so they contribute no policy gradient, while still occupying
    the batch (and the token-level loss denominator) per the GFPO objective.

    Args:
        scores_per_prompt: shape (num_prompts, G), raw rewards.
        keep_mask: shape (num_prompts, G), boolean from apply_gfpo_filter.
        normalization_type: "standard" (subtract mu_S, divide by sigma_S) or
            "centered" (subtract mu_S only).

    Returns:
        Advantages of shape (num_prompts, G); 0 for non-retained responses.
    """
    if scores_per_prompt.shape != keep_mask.shape:
        raise ValueError(f"shape mismatch: scores {scores_per_prompt.shape} vs mask {keep_mask.shape}")

    scores = scores_per_prompt.astype(np.float64)
    # Restrict the group statistics to the retained subset via NaN-masking.
    subset = np.where(keep_mask, scores, np.nan)
    mu = np.nanmean(subset, axis=1, keepdims=True)
    adv = scores - mu
    if normalization_type == "standard":
        sigma = np.nanstd(subset, axis=1, keepdims=True)
        adv = adv / (sigma + 1e-8)
    elif normalization_type != "centered":
        raise ValueError(f"Invalid advantage normalization type: {normalization_type}")
    adv = adv * keep_mask
    return adv
