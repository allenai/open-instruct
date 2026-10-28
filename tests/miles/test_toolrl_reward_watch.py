"""The experiment guard must not mistake format-only or unfinished rewards for signal."""

import json
from types import SimpleNamespace

import pytest
from miles.utils.types import Sample
from scripts.miles import toolrl_reward_watch


def args(tmp_path):
    return SimpleNamespace(
        save=str(tmp_path / "checkpoints"),
        reward_key=None,
        rewards_normalization=True,
        advantage_estimator="grpo",
        grpo_std_normalization=False,
    )


def sample(index, reward, *, calls=True, truncated=False):
    return Sample(
        group_index=0,
        index=index,
        reward=reward,
        response="recorded response",
        status=Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED,
        metadata={"verifiers": [{"name": "toolrl_think_prefix", "target": {"calls": [{}] if calls else []}}]},
    )


def test_positive_call_rewards_preserve_advantages_and_audit(tmp_path):
    result = toolrl_reward_watch.post_process(args(tmp_path), [sample(0, 4), sample(1, -3)])
    assert result == ([4, -3], [3.5, -3.5])
    summary = json.loads((tmp_path / "toolrl-reward-watch.jsonl").read_text())
    assert summary["positive_call_responses"] == 1
    rows = [json.loads(x) for x in (tmp_path / "toolrl-reward-samples" / "batch_00000.jsonl").read_text().splitlines()]
    assert rows[0]["response"] == "recorded response"


@pytest.mark.parametrize("calls,truncated,reward", [(True, False, -3), (False, False, 1), (True, True, 4)])
def test_four_batches_are_allowed_before_aborting(tmp_path, calls, truncated, reward):
    config = args(tmp_path)
    for _ in range(4):
        toolrl_reward_watch.post_process(config, [sample(0, reward, calls=calls, truncated=truncated)])
    with pytest.raises(RuntimeError, match="before update five"):
        toolrl_reward_watch.post_process(config, [sample(0, 4)])
    rows = [json.loads(x) for x in (tmp_path / "toolrl-reward-watch.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and rows[-1]["guard_passed"] is False


@pytest.mark.parametrize("positive_batch", range(4))
def test_any_positive_in_first_four_passes_without_later_streak_rule(tmp_path, positive_batch):
    config = args(tmp_path)
    for batch in range(8):
        toolrl_reward_watch.post_process(config, [sample(0, 4 if batch == positive_batch else -3)])
    rows = [json.loads(x) for x in (tmp_path / "toolrl-reward-watch.jsonl").read_text().splitlines()]
    assert rows[-1]["guard_passed"] is True
    assert rows[-1]["first_four_positive_call_responses"] == 1
