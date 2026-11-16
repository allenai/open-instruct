"""Experimental binary-reference standard deviation for bounded GRPO rewards.

Use with the H235 pinned-runtime installer. The native rollout grouping, mean
centering, epsilon, and raw reward recording remain unchanged.
"""

import torch


def reference_std(rewards: torch.Tensor) -> torch.Tensor:
    """Return sqrt(n/(n-1) * mean * (1-mean)) for one reward per rollout.

    Rewards must have meaningful endpoints 0 and 1. Constant groups have no
    advantage, including at either endpoint. Do not silently clip invalid grades.
    """
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("binary-reference normalization requires a nonempty vector")
    if not torch.isfinite(rewards).all() or ((rewards < 0) | (rewards > 1)).any():
        raise ValueError("binary-reference normalization requires finite rewards in [0, 1]")
    if rewards.numel() == 1:
        return rewards.new_zeros(())
    mean = rewards.mean()
    return (mean * (1 - mean) * (rewards.numel() / (rewards.numel() - 1))).sqrt()
