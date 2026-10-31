"""The GRPO audit recomputes advantages the way the reward post-process hook assigned them."""

import pytest

from open_instruct.miles.megatron_grpo import grpo_audit
from open_instruct.miles.rewards import rewards


def row(index, reward, status="completed", ungraded=False):
    metadata = {rewards.EXCLUDED_KEY: "code:service_error"} if ungraded else {}
    return {"sample_index": index, "reward": reward, "status": status, "metadata": metadata}


def test_full_groups_center_on_the_group_mean():
    expected, mixed = grpo_audit.expected_advantages([row(0, 1.0), row(1, 0.0), row(2, 0.0), row(3, 0.0)], 4, False)
    assert expected == pytest.approx({0: 0.75, 1: -0.25, 2: -0.25, 3: -0.25})
    assert mixed == 1


def test_ungraded_samples_leave_the_baseline():
    rows = [row(0, 1.0), row(1, 0.0, ungraded=True), row(2, 0.0), row(3, 1.0)]
    expected, mixed = grpo_audit.expected_advantages(rows, 4, False)
    assert expected == pytest.approx({0: 1 / 3, 1: 0.0, 2: -2 / 3, 3: 1 / 3})
    assert mixed == 1


def test_truncated_samples_leave_the_baseline_only_when_excluded():
    rows = [row(0, 1.0), row(1, 1.0, status="truncated"), row(2, 0.0), row(3, 0.0)]
    kept, _ = grpo_audit.expected_advantages(rows, 4, False)
    assert kept == pytest.approx({0: 0.5, 1: 0.5, 2: -0.5, 3: -0.5})
    excluded, _ = grpo_audit.expected_advantages(rows, 4, False, exclude_truncated=True)
    assert excluded == pytest.approx({0: 2 / 3, 1: 0.0, 2: -1 / 3, 3: -1 / 3})


def test_fully_ungraded_group_has_zero_advantages_and_no_signal():
    rows = [row(0, 0.0, ungraded=True), row(1, 0.0, ungraded=True)]
    expected, mixed = grpo_audit.expected_advantages(rows, 2, True)
    assert expected == {0: 0.0, 1: 0.0}
    assert mixed == 0
