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
    assert len(calls) == 3
    assert calls[0][0].prompt.endswith("first\n\n")
    assert calls[1][0].prompt.endswith("first\n\nsecond\n\n")
    assert calls[1] == calls[2]  # fresh winning-prefix rollout, no probe response reuse
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
        ([0, 0], "truncated"),
        ([0, 0], "aborted"),
        ([0, 0.5], "completed"),
        ([0, float("nan")], "completed"),
    ],
)
def test_only_finished_all_zero_binary_groups_trigger_search(scores, status):
    async def forbidden(group):
        pytest.fail("unexpected guidance")

    original = samples(scores, status=status)
    result = asyncio.run(bread.rescue(samples(), original, forbidden))
    assert result.group is original


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


def test_all_correct_probe_gets_fresh_group_but_not_a_fabricated_negative():
    calls = []

    async def generate(group):
        calls.append(group)
        return samples([1, 1])

    result = asyncio.run(bread.rescue(samples(), samples([0, 0]), generate))
    assert len(calls) == 2
    assert result.event["outcome"] == "fresh_group_not_mixed"
    assert bread.rewards(result.group, "score") == [1, 1]


def test_cancellation_propagates():
    async def cancel(group):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(bread.rescue(samples(), samples([0, 0]), cancel))


def config(**options):
    return RunConfig(
        CoreConfig(),
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
    assert "rollout_function_path" not in RunConfig(CoreConfig(), {"hf_checkpoint": "/fixture"}).resolved_miles()
    for options in (
        {"fully_async": True},
        {"custom_generate_function_path": "external.generate"},
        {"partial_rollout": True},
        {"sglang_enable_deterministic_inference": True},
    ):
        with pytest.raises(ValueError, match="BREAD"):
            config(**options).validate()
    with pytest.raises(ValueError, match="bread_max_groups_per_batch"):
        dataclasses.replace(config(), core=CoreConfig(bread_max_groups_per_batch=1)).validate()
