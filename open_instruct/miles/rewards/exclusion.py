"""Exclude ungraded responses, and optionally truncated ones, from the policy objective.

A response whose verifier or judge failed for infrastructure reasons has no grade.
Scoring it zero would place a correct answer below its siblings' baseline and push
the policy away from it. Dropping its whole prompt group instead wastes every
graded sibling. These post-processing hooks keep the group and leave only the
ungraded response out:

- it gets zero advantage and ``remove_sample``, which zeroes its loss mask;
- each group's baseline (mean, and standard deviation when enabled) uses only the
  responses that were graded, and finished when truncation exclusion is selected;
- the matching dynamic filter checks reward variation over the same responses and
  drops a group with fewer than two of them.

The reward adapter marks an ungraded response with ``metadata["reward_omission"]``
and returns reward 0 when the configured post-process declares ``masks_ungraded``;
without such a post-process it returns no reward and MILES drops the group.

``exclude_ungraded`` is the default post-process. ``exclude_truncated`` additionally
excludes length-truncated responses (overlong filtering) and requires
``core.reward_zero_truncated=false``. Raw rewards are returned unchanged so reward
metrics stay comparable across runs; an excluded response contributes its stored
reward (0 for ungraded responses) to raw metrics and still counts in
response-averaged loss denominators.
"""

from collections import defaultdict

import torch
from miles.rollout.filter_hub import base_types, common_filters
from miles.utils.types import Sample

UNGRADED_KEY = "reward_omission"


def ungraded(sample) -> bool:
    """Whether the reward adapter recorded a failed grade for this response."""
    metadata = getattr(sample, "metadata", None)
    return isinstance(metadata, dict) and UNGRADED_KEY in metadata


def _excluded(sample, *, truncated: bool) -> bool:
    return ungraded(sample) or (truncated and sample.status == Sample.Status.TRUNCATED)


def _filter_reward_variance(args, samples, *, truncated: bool, **kwargs):
    included = [sample for sample in base_types.iter_samples(samples) if not _excluded(sample, truncated=truncated)]
    if len(included) < 2:
        reason = "insufficient_finished_responses" if truncated else "insufficient_graded_responses"
        return base_types.FilterOutput(keep=False, reason=reason)
    return common_filters.apply_reward_nonzero_std_filter(args, included, **kwargs)


def _exclude(args, samples: list[Sample], *, truncated: bool) -> tuple[list[float], list[float]]:
    raw = [sample.get_reward_value(args) for sample in samples]
    normalized = [0.0] * len(samples)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.group_index is None:
            raise ValueError("reward exclusion requires every sample to carry its prompt group index")
        if _excluded(sample, truncated=truncated):
            sample.remove_sample = True
        else:
            groups[sample.group_index].append(index)
    for indices in groups.values():
        rollouts = [samples[i].rollout_id if samples[i].rollout_id is not None else samples[i].index for i in indices]
        if len(set(rollouts)) != len(rollouts):
            raise NotImplementedError("reward exclusion supports single-segment rollouts only")
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


def exclude_ungraded(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Return raw rewards and group-normalized rewards computed over graded responses only."""
    return _exclude(args, samples, truncated=False)


def exclude_truncated(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Return raw and group-normalized rewards computed over graded, finished responses only."""
    return _exclude(args, samples, truncated=True)


def filter_graded_reward_variance(args, samples, **kwargs):
    """Drop groups with no reward variation among graded responses."""
    return _filter_reward_variance(args, samples, truncated=False, **kwargs)


def filter_finished_reward_variance(args, samples, **kwargs):
    """Drop groups with no reward variation among graded, finished responses."""
    return _filter_reward_variance(args, samples, truncated=True, **kwargs)


# The reward adapter reads this marker to decide between returning reward 0 for an
# ungraded response (this post-process masks it) and returning no reward (MILES
# drops the group).
exclude_ungraded.masks_ungraded = True
exclude_truncated.masks_ungraded = True
