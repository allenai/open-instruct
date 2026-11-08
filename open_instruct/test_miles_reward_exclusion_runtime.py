"""Exercise exclusion through the real MILES runtime when its CPU dependencies are installed.

Use the MILES revision in runtime/miles/runtime.lock.json on PYTHONPATH. Ordinary
CPU CI skips these tests when MILES is absent; stub-based unit tests run there.
"""

import asyncio
import copy
import importlib
import math
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from open_instruct.miles.configuration.config import EXCLUDE_UNGRADED, CoreConfig
from open_instruct.miles.rewards import rewards

ESTIMATORS = ["grpo", "gspo", "reinforce_plus_plus_baseline", "reinforce_plus_plus", "ppo"]


@pytest.fixture
def runtime():
    conversion = pytest.importorskip("miles.ray.rollout.train_data_conversion")
    exclusion = importlib.import_module("open_instruct.miles.rewards.exclusion")
    return SimpleNamespace(conversion=conversion, exclusion=exclusion, Sample=exclusion.Sample)


@pytest.fixture
def training_runtime(runtime, tmp_path):
    parallel = importlib.import_module("miles.backends.training_utils.parallel")
    objective = importlib.import_module("miles.backends.training_utils.loss.objective")
    previous = parallel._parallel_state
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1)
    group = parallel.GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    parallel.set_parallel_state(
        parallel.ParallelState(
            **{key: group for key in ("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp")}
        )
    )
    runtime.objective = objective
    runtime.filters = importlib.import_module("miles.rollout.filter_hub.common_filters")
    try:
        yield runtime
    finally:
        parallel.set_parallel_state(previous)
        dist.destroy_process_group()


def training_arguments(args, count):
    # Exercise native advantage calculation, whitening, and the full policy loss.
    vars(args).update(
        qkv_format="thd",
        gamma=1.0,
        lambd=0.95,
        kl_coef=0.0,
        kl_loss_type="k1",
        use_rollout_logprobs=False,
        skip_actor_forward_only=False,
        use_opd=False,
        normalize_advantages=args.advantage_estimator.startswith("reinforce_plus_plus"),
        rollout_temperature=1.0,
        allgather_cp=False,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        debug_unified_grad_fused_logprob=False,
        use_sampling_support_replay=False,
        loss_type="policy_loss",
        custom_loss_function_path=None,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=0.0,
        observe_training_entropy=False,
        use_kl_loss=False,
        use_unbiased_kl=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_pg_loss_reducer_function_path=None,
        use_opsm=False,
        calculate_per_token_loss=False,
        recompute_loss_function=False,
        global_batch_size=count,
    )
    return args


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
@pytest.mark.parametrize("variance_filter", [False, True])
@pytest.mark.parametrize("all_zero", [False, True])
def test_failed_grades_through_filter_returns_and_backward(
    training_runtime, monkeypatch, estimator, normalize, variance_filter, all_zero
):
    runtime = training_runtime

    async def verify(tokens, response, target, **kwargs):
        if response == "unavailable":
            raise RuntimeError("injected verifier outage")
        return SimpleNamespace(score=float(response), cost=0.0)

    monkeypatch.setattr(rewards, "_registry", lambda path: {"test": SimpleNamespace(async_call=verify)})
    args = arguments(estimator, normalize, True)
    args.olmo_core = CoreConfig(reward_config="unused", filter_zero_std_groups=variance_filter)
    args.custom_reward_post_process_path = EXCLUDE_UNGRADED
    # A mixed graded group, a single surviving grade, and an all-failed group.
    rows = [(0, "1"), (0, "unavailable"), (0, "0"), (1, "0.75"), (1, "unavailable"), (2, "unavailable")]
    if all_zero:
        rows = [(group, "0" if response != "unavailable" else response) for group, response in rows]
    samples = [sample(runtime, group, i, None) for i, (group, _) in enumerate(rows)]
    for value, (_, response) in zip(samples, rows, strict=True):
        value.response = response
        value.metadata = {"verifiers": [{"name": "test", "target": "unused"}]}
        value.loss_mask = [1, 0]  # Existing observation masks must survive.
    for value, reward in zip(samples, asyncio.run(rewards.rollout_reward(args, samples)), strict=True):
        value.reward = reward

    group_filter = (
        runtime.exclusion.filter_graded_reward_variance
        if variance_filter
        else runtime.exclusion.filter_graded_responses
    )
    groups = [[value for value in samples if value.group_index == group] for group in range(3)]
    results = [runtime.filters.apply_preput_filters(args, group_filter, group) for group in groups]
    assert results[2].reason == "no_trainable_responses"
    assert [bool(result.keep) for result in results] == [
        not (variance_filter and all_zero),
        not variance_filter,
        False,
    ]
    samples = [value for group, result in zip(groups, results, strict=True) if result.keep for value in group]
    if not samples:
        # No optimizer batch exists: collector must request replacement groups.
        assert variance_filter and all_zero
        return
    graded = [value for value in samples if value.response != "unavailable"]
    _, native_rewards = runtime.conversion._post_process_rewards(args, graded, None)
    expected = dict(zip((value.index for value in graded), native_rewards, strict=True))
    converted = runtime.conversion.convert_samples_to_train_data(
        args, samples, {}, None, runtime.exclusion.exclude_ungraded
    )
    assert converted["rewards"] == [expected.get(value.index, 0.0) for value in samples]
    assert converted["loss_masks"] == (
        [[1, 0], [0, 0], [1, 0]] if variance_filter else [[1, 0], [0, 0], [1, 0], [1, 0], [0, 0]]
    )
    count = len(samples)
    training_arguments(args, count)
    converted.update(
        unconcat_tokens=[torch.tensor(value.tokens) for value in samples],
        total_lengths=[3] * count,
        rollout_mask_sums=[torch.tensor(total) for total in converted["rollout_mask_sums"]],
        loss_masks=[torch.tensor(mask) for mask in converted["loss_masks"]],
        log_probs=[torch.full((2,), -math.log(16)) for _ in samples],
        values=[torch.zeros(2) for _ in samples],
    )
    runtime.objective.compute_advantages_and_returns(args, converted)
    assert all(torch.isfinite(value).all() for key in ("advantages", "returns") for value in converted[key])
    logits = torch.zeros((1, 3 * count, 16), requires_grad=True)
    loss, _, _ = runtime.objective.loss_function(args, converted, num_microbatches=1, logits=logits)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    for index, value in enumerate(samples):
        if value.response == "unavailable":
            assert torch.count_nonzero(logits.grad[:, index * 3 : (index + 1) * 3]) == 0
    assert bool(logits.grad.abs().sum() > 0) is (not all_zero)
