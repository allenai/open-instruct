"""Exercise exclusion through the real MILES runtime when its CPU dependencies are installed.

Use the MILES revision in runtime/miles/runtime.lock.json on PYTHONPATH. Ordinary
CPU CI skips these tests when MILES is absent; stub-based unit tests run there.
"""

import asyncio
import copy
import importlib
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import EXCLUDE_UNGRADED, CoreConfig
from open_instruct.miles.rewards import rewards

ESTIMATORS = ["grpo", "gspo", "reinforce_plus_plus_baseline", "reinforce_plus_plus", "ppo"]


@pytest.fixture
def runtime():
    conversion = pytest.importorskip("miles.ray.rollout.train_data_conversion")
    exclusion = importlib.import_module("open_instruct.miles.rewards.exclusion")
    return SimpleNamespace(conversion=conversion, exclusion=exclusion, Sample=exclusion.Sample)


def arguments(estimator, normalize, std):
    return SimpleNamespace(
        reward_key=None,
        advantage_estimator=estimator,
        rewards_normalization=normalize,
        grpo_std_normalization=std,
        rollout_top_logprobs_num=0,
        use_dynamic_global_batch_size=False,
    )


def sample(runtime, group, index, reward):
    value = runtime.Sample(
        prompt="Question", response="answer", tokens=[11, 12, 13], response_length=2, group_index=group, metadata={}
    )
    value.index = index
    value.reward = reward
    value.status = runtime.Sample.Status.COMPLETED
    return value


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("std", [False, True])
def test_no_failures_match_native_rewards_exactly(runtime, estimator, normalize, std):
    args = arguments(estimator, normalize, std)
    # Unequal group sizes, interleaved groups, a singleton and constant rewards.
    rows = [(0, 0.1), (1, 0.75), (0, 0.9), (2, 1.0), (0, 0.0), (2, 1.0)]
    samples = [sample(runtime, group, i, reward) for i, (group, reward) in enumerate(rows)]
    expected = runtime.conversion._post_process_rewards(args, samples, None)
    for hook in (runtime.exclusion.exclude_ungraded, runtime.exclusion.exclude_truncated):
        actual_samples = copy.deepcopy(samples)
        assert hook(args, actual_samples) == expected
        assert not any(value.remove_sample for value in actual_samples)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("normalize", [False, True])
def test_failed_grades_reach_native_conversion_with_zero_loss_masks(runtime, monkeypatch, estimator, normalize):
    async def verify(tokens, response, target, **kwargs):
        if response == "unavailable":
            raise RuntimeError("injected verifier outage")
        return SimpleNamespace(score=float(response), cost=0.0)

    monkeypatch.setattr(rewards, "_registry", lambda path: {"test": SimpleNamespace(async_call=verify)})
    args = arguments(estimator, normalize, True)
    args.olmo_core = CoreConfig(reward_config="unused")
    args.custom_reward_post_process_path = EXCLUDE_UNGRADED
    # A mixed graded group, a single surviving grade, and an all-failed group.
    rows = [(0, "1"), (0, "unavailable"), (0, "0"), (1, "0.75"), (1, "unavailable"), (2, "unavailable")]
    samples = [sample(runtime, group, i, None) for i, (group, _) in enumerate(rows)]
    for value, (_, response) in zip(samples, rows, strict=True):
        value.response = response
        value.metadata = {"verifiers": [{"name": "test", "target": "unused"}]}
        value.loss_mask = [1, 0]  # Existing observation masks must survive.
    for value, reward in zip(samples, asyncio.run(rewards.rollout_reward(args, samples)), strict=True):
        value.reward = reward

    graded = [value for value in samples if value.response != "unavailable"]
    _, native_rewards = runtime.conversion._post_process_rewards(args, graded, None)
    expected = dict(zip((value.index for value in graded), native_rewards, strict=True))
    converted = runtime.conversion.convert_samples_to_train_data(
        args, samples, {}, None, runtime.exclusion.exclude_ungraded
    )
    assert converted["rewards"] == [expected.get(value.index, 0.0) for value in samples]
    assert converted["loss_masks"] == [[1, 0], [0, 0], [1, 0], [1, 0], [0, 0], [0, 0]]
    assert converted["rollout_mask_sums"] == [1, 0, 1, 1, 0, 0]
    assert converted["raw_reward"] == [1.0, 0.0, 0.0, 0.75, 0.0, 0.0]
