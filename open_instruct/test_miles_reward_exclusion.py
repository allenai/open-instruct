"""Ungraded responses are masked inside their prompt group instead of dropping the group."""

import asyncio
import dataclasses
import enum
import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from open_instruct.miles.configuration.config import (
    EXCLUDE_TRUNCATED,
    EXCLUDE_UNGRADED,
    FINISHED_STD_FILTER,
    GRADED_STD_FILTER,
    CoreConfig,
)
from open_instruct.miles.rewards import rewards


def masking_hook(args, samples):
    raise AssertionError("not called")


def plain_hook(args, samples):
    raise AssertionError("not called")


masking_hook.masks_ungraded = True


def _install_miles_stubs():
    """Prefer the installed MILES package; otherwise stub the three symbols the hooks import."""
    try:
        importlib.import_module("miles.rollout.filter_hub.common_filters")
        importlib.import_module("miles.utils.types")
        return
    except ImportError:
        pass

    @dataclasses.dataclass
    class FilterOutput:
        keep: bool
        reason: str | None = None

    def iter_samples(group):
        for item in group:
            if isinstance(item, list):
                yield from item
            else:
                yield item

    def apply_reward_nonzero_std_filter(args, samples, **kwargs):
        values = [sample.get_reward_value(args) for sample in iter_samples(samples)]
        keep = bool(torch.tensor(values, dtype=torch.float64).std() > 1e-8)
        return FilterOutput(keep=keep, reason=None if keep else f"zero_std_{round(values[0], 1)}")

    class Status(enum.Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    class Sample:
        def __init__(self, **fields):
            self.status = Status.PENDING
            self.reward = None
            self.index = None
            self.rollout_id = None
            self.remove_sample = False
            self.__dict__.update(fields)

        def get_reward_value(self, args):
            return self.reward

    Sample.Status = Status
    modules = {
        "miles": types.ModuleType("miles"),
        "miles.rollout": types.ModuleType("miles.rollout"),
        "miles.rollout.filter_hub": types.ModuleType("miles.rollout.filter_hub"),
        "miles.rollout.filter_hub.base_types": types.ModuleType("miles.rollout.filter_hub.base_types"),
        "miles.rollout.filter_hub.common_filters": types.ModuleType("miles.rollout.filter_hub.common_filters"),
        "miles.utils": types.ModuleType("miles.utils"),
        "miles.utils.types": types.ModuleType("miles.utils.types"),
    }
    modules["miles.rollout.filter_hub.base_types"].FilterOutput = FilterOutput
    modules["miles.rollout.filter_hub.base_types"].iter_samples = iter_samples
    modules[
        "miles.rollout.filter_hub.common_filters"
    ].apply_reward_nonzero_std_filter = apply_reward_nonzero_std_filter
    modules["miles.utils.types"].Sample = Sample
    sys.modules.update(modules)


_install_miles_stubs()
exclusion = importlib.import_module("open_instruct.miles.rewards.exclusion")
truncation = importlib.import_module("open_instruct.miles.rewards.truncation")
Sample = importlib.import_module("miles.utils.types").Sample


def make_sample(reward, *, group_index, index, truncated=False, ungraded=False, rollout_id=None):
    sample = Sample(
        prompt="Question", response="answer", tokens=[], response_length=0, group_index=group_index, metadata={}
    )
    sample.index = index
    sample.rollout_id = rollout_id
    sample.reward = reward
    sample.status = Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED
    if ungraded:
        sample.metadata["reward_omission"] = {"verifier": "test", "error": "RuntimeError: unavailable"}
    return sample


def args(**overrides):
    values = dict(
        reward_key=None, rewards_normalization=True, advantage_estimator="grpo", grpo_std_normalization=False
    )
    return SimpleNamespace(**(values | overrides))


def batch():
    # Group 0 has one ungraded response; group 1 has one truncated response.
    return [
        make_sample(1.0, group_index=0, index=0),
        make_sample(0.0, group_index=0, index=1),
        make_sample(1.0, group_index=0, index=2),
        make_sample(0.0, group_index=0, index=3, ungraded=True),
        make_sample(1.0, group_index=1, index=4),
        make_sample(0.0, group_index=1, index=5, truncated=True),
        make_sample(0.0, group_index=1, index=6),
    ]


def test_exclude_ungraded_masks_only_the_ungraded_response():
    samples = batch()
    raw, normalized = exclusion.exclude_ungraded(args(), samples)
    assert raw == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    # Group 0's baseline uses the three graded responses (mean 2/3); group 1 keeps
    # its truncated response, which the zero-truncation default scores as a grade.
    assert normalized == pytest.approx([1 / 3, -2 / 3, 1 / 3, 0.0, 2 / 3, -1 / 3, -1 / 3])
    assert [sample.remove_sample for sample in samples] == [False, False, False, True, False, False, False]


def test_exclude_truncated_masks_ungraded_and_truncated_responses():
    samples = batch()
    _, normalized = exclusion.exclude_truncated(args(), samples)
    assert normalized == pytest.approx([1 / 3, -2 / 3, 1 / 3, 0.0, 0.5, 0.0, -0.5])
    assert [sample.remove_sample for sample in samples] == [False, False, False, True, False, True, False]


def test_std_normalization_and_raw_passthrough_use_graded_responses_only():
    samples = batch()[:4]
    _, normalized = exclusion.exclude_ungraded(args(grpo_std_normalization=True), samples)
    values = torch.tensor([1.0, 0.0, 1.0])
    expected = ((values - values.mean()) / (values.std() + 1e-6)).tolist() + [0.0]
    assert normalized == pytest.approx(expected)
    _, unnormalized = exclusion.exclude_ungraded(args(rewards_normalization=False), batch()[:4])
    assert unnormalized == [1.0, 0.0, 1.0, 0.0]


def test_exclusion_requires_group_index_and_single_segment_rollouts():
    with pytest.raises(ValueError, match="group index"):
        exclusion.exclude_ungraded(args(), [make_sample(1.0, group_index=None, index=0)])
    segments = [make_sample(1.0, group_index=0, index=i, rollout_id=7) for i in range(2)]
    with pytest.raises(NotImplementedError, match="single-segment"):
        exclusion.exclude_ungraded(args(), segments)


def test_graded_variance_filter_ignores_ungraded_placeholders():
    graded_constant = [make_sample(1.0, group_index=0, index=i) for i in range(2)]
    graded_constant.append(make_sample(0.0, group_index=0, index=2, ungraded=True))
    # The placeholder zero must not keep an all-correct group in the batch.
    assert bool(exclusion.filter_graded_reward_variance(args(), graded_constant).keep) is False
    mixed = batch()[:4]
    assert bool(exclusion.filter_graded_reward_variance(args(), mixed).keep) is True
    too_few = [make_sample(1.0, group_index=0, index=0), make_sample(0.0, group_index=0, index=1, ungraded=True)]
    result = exclusion.filter_graded_reward_variance(args(), too_few)
    assert (result.keep, result.reason) == (False, "insufficient_graded_responses")
    # Truncated responses count as graded under the zero-truncation default.
    assert bool(exclusion.filter_graded_reward_variance(args(), batch()[4:]).keep) is True


def test_finished_variance_filter_also_ignores_truncated_responses():
    group = batch()[4:]
    group.pop()  # one finished graded response and one truncated response remain
    result = exclusion.filter_finished_reward_variance(args(), group)
    assert (result.keep, result.reason) == (False, "insufficient_finished_responses")


def test_configured_paths_resolve_to_masking_hooks_and_keep_compatibility():
    for path in (EXCLUDE_UNGRADED, EXCLUDE_TRUNCATED):
        assert rewards.masks_ungraded(SimpleNamespace(custom_reward_post_process_path=path)) is True
    assert truncation.exclude_truncated is exclusion.exclude_truncated
    assert truncation.filter_finished_reward_variance is exclusion.filter_finished_reward_variance
    module, _, symbol = GRADED_STD_FILTER.rpartition(".")
    assert getattr(importlib.import_module(module), symbol) is exclusion.filter_graded_reward_variance
    module, _, symbol = FINISHED_STD_FILTER.rpartition(".")
    assert getattr(importlib.import_module(module), symbol) is exclusion.filter_finished_reward_variance


@pytest.fixture(autouse=True)
def clear_hook_cache():
    rewards._masks_ungraded_path.cache_clear()
    yield
    rewards._masks_ungraded_path.cache_clear()


def test_masks_ungraded_reads_the_hook_marker():
    assert rewards.masks_ungraded(SimpleNamespace()) is False
    assert rewards.masks_ungraded(SimpleNamespace(custom_reward_post_process_path=None)) is False
    assert rewards.masks_ungraded(SimpleNamespace(custom_reward_post_process_path=f"{__name__}.plain_hook")) is False
    assert rewards.masks_ungraded(SimpleNamespace(custom_reward_post_process_path=f"{__name__}.masking_hook")) is True


def scored_sample(response="ok", group_index=0):
    return SimpleNamespace(
        group_index=group_index,
        tokens=[],
        response_length=0,
        response=response,
        prompt="Question",
        metadata={"verifiers": [{"name": "test", "target": "answer"}]},
    )


@pytest.fixture
def scoring(monkeypatch, tmp_path):
    async def verify(tokens, response, target, **kwargs):
        if response == "failure":
            raise RuntimeError("service unavailable")
        return SimpleNamespace(
            score=float("nan") if response == "nan" else float(response == "ok"),
            cost=0.0,
            diagnostics={"status": response} if response in {"timeout", "service_error"} else None,
        )

    monkeypatch.setattr(rewards, "_registry", lambda path: {"test": SimpleNamespace(async_call=verify)})
    return SimpleNamespace(
        olmo_core=CoreConfig(reward_config="unused"),
        save=str(tmp_path),
        custom_reward_post_process_path=f"{__name__}.masking_hook",
    )


def test_training_failures_become_marked_zeros_when_the_hook_masks_them(scoring):
    samples = [scored_sample(value) for value in ("ok", "failure", "wrong", "timeout", "service_error", "nan")]
    assert asyncio.run(rewards.rollout_reward(scoring, samples)) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    marked = [index for index, sample in enumerate(samples) if "reward_omission" in sample.metadata]
    assert marked == [1, 4, 5]
    assert samples[1].metadata["reward_omission"] == {"verifier": "test", "error": "RuntimeError: service unavailable"}
    assert all("reward_components" not in samples[index].metadata for index in marked)
    # A legitimate zero and a math timeout keep their grades and components.
    assert samples[2].metadata["reward_components"][0]["score"] == 0.0
    assert samples[3].metadata["verifier_diagnostics"]["test"]["status"] == "timeout"
    assert rewards.ungraded_count(samples) == 3


def test_recovery_clears_the_marker_and_evaluation_stays_strict(scoring):
    sample = scored_sample("failure")
    assert asyncio.run(rewards.rollout_reward(scoring, sample)) == 0.0
    sample.response = "ok"
    assert asyncio.run(rewards.rollout_reward(scoring, sample)) == 1.0
    assert "reward_omission" not in sample.metadata
    with pytest.raises(RuntimeError, match="service unavailable"):
        asyncio.run(rewards.rollout_reward(scoring, scored_sample("failure", group_index=None)))


def test_without_a_masking_hook_training_failures_still_omit_the_group(scoring):
    scoring.custom_reward_post_process_path = f"{__name__}.plain_hook"
    assert asyncio.run(rewards.rollout_reward(scoring, scored_sample("failure"))) is None
