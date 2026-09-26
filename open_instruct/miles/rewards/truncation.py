"""Exclude truncated responses from the policy objective (overlong filtering).

MILES scores a response that hit the length limit like any other: the verifier
reads the unfinished text, and an answer mentioned mid-reasoning can earn full
reward. Configure this function as ``miles.custom_reward_post_process_path`` to
leave truncated responses out of training instead:

- they get zero advantage and ``remove_sample``, which zeroes their loss mask;
- each group's baseline (mean, and standard deviation when enabled) uses only the
  responses that finished.

Raw rewards are returned unchanged so reward metrics stay comparable across runs.
Excluded responses still count in response-averaged loss denominators.
"""

from collections import defaultdict

import torch
from miles.utils.types import Sample


def exclude_truncated(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Return raw rewards and group-normalized rewards computed over finished responses only."""
    raw = [sample.get_reward_value(args) for sample in samples]
    normalized = [0.0] * len(samples)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("exclude_truncated requires every sample to carry its prompt group index")
        if sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True
        else:
            groups[sample.group_index].append(index)
    for indices in groups.values():
        rollouts = [samples[i].rollout_id if samples[i].rollout_id is not None else samples[i].index for i in indices]
        if len(set(rollouts)) != len(rollouts):
            raise NotImplementedError("exclude_truncated supports single-segment rollouts only")
        if not args.rewards_normalization:
            for i in indices:
                normalized[i] = raw[i]
            continue
        values = torch.tensor([raw[i] for i in indices], dtype=torch.float)
        centered = values - values.mean()
        if args.advantage_estimator in ("grpo", "gspo") and args.grpo_std_normalization and len(indices) > 1:
            std = values.std()
            if std > 0:
                centered = centered / (std + 1e-6)
        for i, value in zip(indices, centered.tolist(), strict=True):
            normalized[i] = value
    return raw, normalized
