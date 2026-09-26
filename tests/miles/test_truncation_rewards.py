"""Truncated responses are left out of advantages and training loss masks."""

from types import SimpleNamespace

import pytest
from miles.utils.types import Sample

from open_instruct.miles.rewards.truncation import exclude_truncated


def args(std=False, normalize=True):
    return SimpleNamespace(
        reward_key=None, rewards_normalization=normalize, advantage_estimator="grpo", grpo_std_normalization=std
    )


def sample(group, reward, truncated=False, index=None):
    status = Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED
    return Sample(group_index=group, index=index, reward=reward, status=status)


def test_truncated_responses_leave_the_baseline_and_the_loss():
    samples = [
        sample(0, 1.0, index=0),
        sample(0, 0.0, index=1),
        sample(0, 1.0, truncated=True, index=2),  # a leaked reward on unfinished text
        sample(1, 1.0, truncated=True, index=3),
        sample(1, 0.0, index=4),
    ]
    raw, normalized = exclude_truncated(args(), samples)
    assert raw == [1.0, 0.0, 1.0, 1.0, 0.0]
    # Group 0's baseline is the finished mean 0.5; the leaked 1.0 does not raise it.
    assert normalized == pytest.approx([0.5, -0.5, 0.0, 0.0, 0.0])
    assert [s.remove_sample for s in samples] == [False, False, True, True, False]


def test_standard_deviation_scaling_uses_finished_responses():
    samples = [sample(0, 1.0, index=0), sample(0, 0.0, index=1), sample(0, 0.0, truncated=True, index=2)]
    _, normalized = exclude_truncated(args(std=True), samples)
    assert normalized[0] == pytest.approx(0.5 / 0.7071068, rel=1e-4)
    assert normalized[2] == 0.0


def test_group_index_is_required():
    with pytest.raises(ValueError, match="group index"):
        exclude_truncated(args(), [sample(None, 1.0, index=0)])
