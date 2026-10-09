"""CPU contracts for failure-triggered continuation search and configuration."""

import asyncio
import copy
import dataclasses
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import BREAD_ROLLOUT, CoreConfig, RunConfig
from open_instruct.miles.rollout import bread


def samples(scores=None, *, status="completed"):
    return [
        SimpleNamespace(
            prompt="question|assistant:",
            tokens=[] if scores is None else [1, 2],
            response="" if scores is None else "wrong",
            response_length=0 if scores is None else 1,
            reward=None if scores is None else {"score": score},
            status="pending" if scores is None else status,
            metadata={"bread": {"sections": ["first", "second", "third", "final answer"]}},
        )
        for score in ([None, None] if scores is None else scores)
    ]


def test_search_preserves_inputs_and_retries_from_pristine_context():
    pristine, initial = samples(), samples([0, 0])
    before = copy.deepcopy(pristine)
    calls = []

    async def generate(group):
        calls.append(copy.deepcopy(group))
        scores = ([0, 0], [0, 1], [1, 0])[len(calls) - 1]
        for sample, score in zip(group, scores):
            sample.reward = {"score": score}
            sample.status = "completed"
            sample.response = "student continuation"
            sample.response_length = 3
            sample.tokens = [1, 2, 3, 4]
        return group

    result = asyncio.run(bread.rescue(pristine, initial, generate))
    assert pristine == before
    assert len(calls) == 2
    assert calls[0][0].prompt.endswith("first\n\n")
    assert calls[1][0].prompt.endswith("first\n\nsecond\n\n")
    assert all(not sample.tokens and sample.response_length == 0 for group in calls for sample in group)
    assert result.event["outcome"] == "rescued"
    assert result.event["selected_sections"] == 2
    assert result.group[0].response == "student continuation"
    assert initial[0].response == "wrong"


@pytest.mark.parametrize(
    "scores,status",
    [
        ([0, 1], "completed"),
        ([1, 1], "completed"),
        ([0, 0], "aborted"),
        ([0, 0], "failed"),
        ([0, 0], "pending"),
        ([0, 0.5], "completed"),
        ([0, float("nan")], "completed"),
    ],
)
def test_only_terminal_all_zero_binary_groups_trigger_search(scores, status):
    async def forbidden(group):
        pytest.fail("unexpected guidance")

    original = samples(scores, status=status)
    result = asyncio.run(bread.rescue(samples(), original, forbidden))
    assert result.group is original


def test_truncated_zero_rewards_search_continue_and_supply_probe_negatives():
    calls = []

    async def generate(group):
        calls.append(group)
        if len(calls) == 1:
            return samples([0, 0], status="truncated")
        result = samples([1, 0])
        result[1].status = "truncated"
        return result

    result = asyncio.run(bread.rescue(samples(), samples([0, 0], status="truncated"), generate))
    assert len(calls) == 2
    assert result.event["initial_rewards"] == [0, 0]
    assert result.event["attempts"][0]["rewards"] == [0, 0]
    assert result.event["selected_sections"] == 2
    assert result.event["outcome"] == "rescued"
    assert bread.rewards(result.group, "score") == [1, 0]


def test_exhausted_search_never_exposes_complete_reference():
    calls = []

    async def generate(group):
        calls.append(group)
        return samples([0, 0])

    initial = samples([0, 0])
    result = asyncio.run(bread.rescue(samples(), initial, generate, max_prefixes=2))
    assert len(calls) == 2
    assert all("final answer" not in group[0].prompt for group in calls)
    assert result.group is initial
    assert result.event["outcome"] == "exhausted"


def test_context_budget_stops_before_generation_and_boxed_answer_stops_prefixes():
    async def forbidden(group):
        pytest.fail("context overflow")

    result = asyncio.run(bread.rescue(samples(), samples([0, 0]), forbidden, fits=lambda _: False))
    assert result.event["outcome"] == "context_limit"
    assert list(bread.prefixes(["setup", r"answer \boxed{7}", "end"], 9)) == [(1, "setup\n\n")]


def test_all_correct_probe_stops_without_resampling():
    calls = []

    async def generate(group):
        calls.append(group)
        return samples([1, 1])

    result = asyncio.run(bread.rescue(samples(), samples([0, 0]), generate))
    assert len(calls) == 1
    assert result.event["outcome"] == "probe_all_correct"
    assert bread.rewards(result.group, "score") == [1, 1]


def test_cancellation_propagates():
    async def cancel(group):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(bread.rescue(samples(), samples([0, 0]), cancel))


def config(**options):
    return RunConfig(
        CoreConfig(reward_zero_truncated=True),
        {
            "hf_checkpoint": "/fixture/hf",
            "rollout_function_path": BREAD_ROLLOUT,
            "rollout_batch_size": 2,
            "global_batch_size": 4,
            "n_samples_per_prompt": 2,
            **options,
        },
    )


def test_bread_is_opt_in_and_rejects_async_and_other_generation_hooks():
    config().validate()
    assert (
        "rollout_function_path"
        not in RunConfig(CoreConfig(reward_zero_truncated=True), {"hf_checkpoint": "/fixture"}).resolved_miles()
    )
    for options in (
        {"fully_async": True},
        {"custom_generate_function_path": "external.generate"},
        {"partial_rollout": True},
        {"sglang_enable_deterministic_inference": True},
    ):
        with pytest.raises(ValueError, match="BREAD"):
            config(**options).validate()
    with pytest.raises(ValueError, match="bread_max_groups_per_batch"):
        dataclasses.replace(
            config(), core=CoreConfig(bread_max_groups_per_batch=1, reward_zero_truncated=True)
        ).validate()


def test_bread_requires_truncation_zero_and_long_eval_serving_capacity():
    with pytest.raises(ValueError, match="reward_zero_truncated"):
        dataclasses.replace(config(), core=CoreConfig()).validate()
    candidate = dataclasses.replace(
        config(sglang_context_length=65536, rollout_max_context_len=32768),
        core=CoreConfig(
            reward_zero_truncated=True,
            max_sequence_length=32768,
            bread_eval_panel_size=128,
            bread_eval_long_response_len=32768,
        ),
    )
    candidate.validate()
    with pytest.raises(ValueError, match="context space"):
        dataclasses.replace(candidate, miles={**candidate.miles, "sglang_context_length": 32768}).validate()
