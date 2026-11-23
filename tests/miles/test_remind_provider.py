"""Review selection, feedback through the real filter, and pending-work resume."""

import asyncio
import copy
import dataclasses
import json

import pytest
import test_prompt_provider
import torch
from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

from open_instruct.miles.configuration import run_spec
from open_instruct.miles.configuration.config import ZERO_STD_FILTER, CoreConfig, RunConfig
from open_instruct.miles.datasets import recording
from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource
from open_instruct.miles.rollout.prompt_provider import SELECTION_ID
from open_instruct.miles.rollout.provider_source import ProviderRolloutDataSource
from open_instruct.miles.rollout.remind_provider import ReMindPromptProvider

args = test_prompt_provider.args


def provider(args, **kwargs):
    dataset = ProviderRolloutDataSource(args).dataset
    return ReMindPromptProvider(lambda: dataset, shuffle=True, seed=17, **kwargs)


def feedback(sample, rewards=(1, 1), *, valid=True, **extra):
    return {
        "kind": "group",
        SELECTION_ID: sample.metadata[SELECTION_ID],
        "group_size": len(rewards),
        "filter_decision": "filtered",
        "responses": [{"reward": r, "status": "completed", "validity": {"valid": valid}} for r in rewards],
        **extra,
    }


def test_fifo_review_success_exits_regression_returns_and_no_duplicates(args):
    p = provider(args, review_fraction=0.5, enqueue_probability=1)
    a = p.next_samples(1)[0]
    p.feedback([feedback(a), feedback(a)])
    assert len(p.queue) == 1
    review = p.next_samples(1)[0]
    assert review.prompt == a.prompt
    assert len(p.queue) == 0 and len(p.tracked) == 1
    p.feedback([feedback(review, (1, 0))])
    b = p.next_samples(1)[0]
    p.feedback([feedback(b)])
    review = p.next_samples(1)[0]
    assert review.prompt == a.prompt  # Older regression precedes newly mastered b.
    p.feedback([feedback(review)])
    assert len(p.queue) == len(p.tracked) == 1
    p.next_samples(1)
    assert p.next_samples(1)[0].prompt == b.prompt


def test_review_budget_chunk_invariance_and_empty_queue_fallback(args):
    large, small = provider(args), provider(args)
    assert [s.prompt for s in large.next_samples(100)] == [small.next_samples(1)[0].prompt for _ in range(100)]
    assert large.state_dict() == small.state_dict()
    # Populate enough mastered rows; two of the next 100 admissions are reviews.
    p = provider(args, enqueue_probability=1)
    for s in p.next_samples(11):
        p.feedback([feedback(s)])
    baseline_offset = p.base.sample_offset
    p.next_samples(100)
    assert sum(review for _, review in p.pending.values()) == 2
    assert p.base.epoch_id * 11 + p.base.sample_offset == baseline_offset + 98


@pytest.mark.parametrize("rewards,valid", [((0, 0), True), ((0, 1), True), ((1, 1), False), ((0.5, 1), True)])
def test_only_verified_binary_mastery_enters_queue(args, rewards, valid):
    p = provider(args, enqueue_probability=1)
    p.feedback([feedback(p.next_samples(1)[0], rewards, valid=valid)])
    assert not p.queue


def test_aborted_retry_and_unknown_review_do_not_lose_prompt(args):
    p = provider(args, review_fraction=1, enqueue_probability=1)
    a = p.next_samples(1)[0]
    p.feedback([feedback(a, filter_decision="aborted")])
    assert p.pending
    p.feedback([feedback(a)])
    review = p.next_samples(1)[0]
    p.feedback([feedback(review, valid=False)])
    assert len(p.queue) == 1
    review = p.next_samples(1)[0]
    p.feedback([feedback(review, (0, 0))])
    assert len(p.queue) == 1  # All-zero regression still returns for review.


def test_truncation_zero_and_named_reward(args):
    p = provider(args, review_fraction=1, enqueue_probability=1)
    a = p.next_samples(1)[0]
    p.feedback([feedback(a, ({"score": 1}, {"score": 1}), reward_key="score")])
    assert len(p.queue) == 1
    review = p.next_samples(1)[0]
    record = feedback(review, (1, 0), reward_zero_truncated=True)
    record["responses"][1].update(status="truncated", truncated=True, validity={"valid": None})
    p.feedback([record])
    assert len(p.queue) == 1


def test_state_restores_rng_queue_pending_and_fractional_schedule(args):
    p = provider(args, review_fraction=0.3, enqueue_probability=0.7)
    for _ in range(35):
        p.feedback([feedback(p.next_samples(1)[0])])
    pending = p.next_samples(10)
    state = copy.deepcopy(p.state_dict())
    restored = provider(args, review_fraction=0.3, enqueue_probability=0.7)
    restored.load_state_dict(state)
    for item in reversed(pending):
        record = feedback(item, (1, 0))
        p.feedback([record])
        restored.feedback([record])
    assert [s.to_dict() for s in p.next_samples(50)] == [s.to_dict() for s in restored.next_samples(50)]
    assert p.state_dict() == restored.state_dict()
    with pytest.raises(ValueError, match="configuration"):
        provider(args).load_state_dict(state)


@pytest.mark.parametrize("failure", ["selection", "expansion"])
def test_failed_delivery_restores_review_queue_pending_cursor_and_ids(args, monkeypatch, failure):
    args.n_samples_per_prompt = 2
    p = provider(args, review_fraction=0.5, enqueue_probability=1)
    source = ProviderRolloutDataSource(args, provider=p)
    first = source.get_samples(1)[0]
    p.feedback([feedback(first[0])])
    source.add_samples([first])
    saved = copy.deepcopy(p.state_dict())
    counters = source.sample_group_index, source.sample_index
    select, deepcopy = p.next_samples, copy.deepcopy
    copies = 0

    def fail_selection(count):
        select(count)
        raise RuntimeError("selection failed after advancing")

    def fail_expansion(value, *a, **kw):
        nonlocal copies
        if isinstance(value, Sample):
            copies += 1
            if copies == 2:
                raise RuntimeError("expansion failed after assigning an ID")
        return deepcopy(value, *a, **kw)

    with monkeypatch.context() as patch:
        if failure == "selection":
            patch.setattr(p, "next_samples", fail_selection)
        else:
            patch.setattr(copy, "deepcopy", fail_expansion)
        with pytest.raises(RuntimeError, match="failed after"):
            source.get_samples(4)
    assert p.state_dict() == saved
    assert source.buffer == [first]
    assert (source.sample_group_index, source.sample_index) == counters
    retry, review = source.get_samples(2)
    assert retry is first
    assert review[0].prompt == first[0].prompt
    assert review[0].metadata["prompt_provider_review"] is True
    assert (review[0].group_index, review[0].index) == counters


def setup_remind(args, tmp_path):
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    (checkpoint / "workflow-model.json").write_text(
        json.dumps({"identity": {"source": {"path": "/model", "files": []}}})
    )
    args.hf_checkpoint = str(checkpoint)
    args.olmo_core = CoreConfig(
        prompt_provider="remind",
        records_root=str(tmp_path / "records"),
        remind_review_fraction=1,
        remind_enqueue_probability=1,
    )
    args.n_samples_per_prompt = 2
    args.rollout_batch_size = args.global_batch_size = 2
    args.max_weight_staleness = args.async_data_buffer_capacity_factor = 2
    args.reward_key = None
    args.dynamic_sampling_filter_path = ZERO_STD_FILTER
    args.core_records_factory = "open_instruct.miles.datasets.recording.create_recorder"


def scored(group, rewards):
    for sample, reward in zip(group, rewards):
        sample.reward = reward
        sample.status = Sample.Status.COMPLETED
        sample.metadata.update(verifiers=[{"name": "math"}], verifier_diagnostics={"math": {"status": "ok"}})
        sample.weight_versions = [WeightVersionsPerCall([WeightVersionSpan("1", 1, 2)])]
    return DataBufferInput(prompt_group=group, group=group)


def test_legacy_cursor_tracks_fresh_draws_and_repairs_old_checkpoint(args, tmp_path):
    setup_remind(args, tmp_path)
    source = DashboardDrainingRolloutDataSource(args)
    try:
        fresh = source.get_samples(12)  # Cross the 11-row dataset boundary.
        source.feedback([feedback(fresh[0][0]) | {"group_index": fresh[0][0].group_index}])
        source.acknowledge_groups(fresh)
        delegate = source._delegate
        assert (delegate.epoch_id, delegate.sample_offset) == (1, 1)
        review = source.get_samples(1)
        assert review[0][0].prompt == fresh[0][0].prompt
        assert (delegate.epoch_id, delegate.sample_offset) == (1, 1)
        source.add_samples(review)
        assert test_prompt_provider.payload(source.get_samples(1)) == test_prompt_provider.payload(review)
        assert (delegate.epoch_id, delegate.sample_offset) == (1, 1)
        source.save(0)
        path = tmp_path / "rollout" / "global_dataset_state_dict_0.pt"
        state = torch.load(path, weights_only=True)
        assert (state["epoch_id"], state["sample_offset"]) == (1, 1)
        base = state["prompt_provider"]["base"]
        assert (state["epoch_id"], state["sample_offset"]) == (base["epoch_id"], base["sample_offset"])
        state.update(epoch_id=0, sample_offset=0)  # Older ReMind checkpoint mirrors.
        torch.save(state, path)
    finally:
        source.close()

    args.load, args.start_rollout_id = args.save, 1
    restored = DashboardDrainingRolloutDataSource(args)
    try:
        restored.load(0)
        assert restored._delegate.provider.state_dict() == state["prompt_provider"]
        assert (restored._delegate.epoch_id, restored._delegate.sample_offset) == (1, 1)
        restored.save(1)  # Correct even before any new draws.
        saved = torch.load(tmp_path / "rollout" / "global_dataset_state_dict_1.pt", weights_only=True)
        assert (saved["epoch_id"], saved["sample_offset"]) == (1, 1)
        assert test_prompt_provider.payload(restored.get_samples(1)) == test_prompt_provider.payload(review)
    finally:
        restored.close()


def test_replacing_remind_dataset_cannot_silently_reuse_review_history(args, tmp_path):
    setup_remind(args, tmp_path)
    source = ProviderRolloutDataSource(args)
    fresh = source.get_samples(1)
    source.provider.feedback([feedback(fresh[0][0])])
    source.add_samples(fresh)
    state = copy.deepcopy(source.provider.state_dict())
    original = source.dataset
    source.dataset = copy.deepcopy(original)
    for action in (
        lambda: source.get_samples(2),  # Pop a buffered group, then try a review.
        lambda: source.provider.feedback([]),
        source.provider.state_dict,
        lambda: source.provider.load_state_dict(state),
    ):
        with pytest.raises(ValueError, match="ReMind dataset cannot be replaced"):
            action()
    assert source.buffer == fresh and source.sample_group_index == 1
    source.dataset = original
    assert source.provider.state_dict() == state
    assert source.get_samples(2)[1][0].prompt == fresh[0][0].prompt


def test_real_filtered_feedback_and_pending_resume(args, tmp_path):
    setup_remind(args, tmp_path)
    source = DashboardDrainingRolloutDataSource(args)
    buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, source.add_samples))
    recorder = buffer._delegate._records
    try:
        first = source.get_samples(1)[0]
        assert asyncio.run(buffer.put(scored(first, [1, 1]))) is False
        source.acknowledge_groups([first])
        review = source.get_samples(1)[0]
        assert review[0].prompt == first[0].prompt
        source.save(0)  # Review is in flight, not in the queue.
        recorder.close()
        source.close()
        args.load, args.start_rollout_id = args.save, 1
        restored = DashboardDrainingRolloutDataSource(args)
        try:
            restored.load(0)
            replay = restored.get_samples(1)[0]
            assert [s.to_dict() for s in replay] == [s.to_dict() for s in review]
            buffer2 = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, restored.add_samples))
            try:

                async def finish():
                    assert await buffer2.put(scored(replay, [1, 0])) is True
                    assert (await asyncio.wait_for(buffer2.get(current_version=1), 1)).group == replay

                asyncio.run(finish())
                restored.acknowledge_groups([replay])
                assert restored.get_samples(1)[0][0].prompt == first[0].prompt
                assert buffer2.get_metrics()["rollout/records/failed_total"] == 0
            finally:
                buffer2._delegate._records.close()
        finally:
            restored.close()
    finally:
        recorder.close()
        source.close()


@pytest.mark.parametrize(
    "changes", [{"remind_review_fraction": 0}, {"remind_enqueue_probability": -1}, {"prompt_provider": "unknown"}]
)
def test_config_rejects_invalid_options(changes):
    with pytest.raises(ValueError):
        CoreConfig(**changes)


def test_config_requires_feedback_and_rejects_exclusions():
    with pytest.raises(ValueError, match="ReMind requires"):
        RunConfig(CoreConfig(prompt_provider="remind"), {}).validate()
    core = CoreConfig(prompt_provider="remind", records_root="/tmp/records", max_policy_lag=2)
    RunConfig(core, {"fully_async": True, "hf_checkpoint": "/model", "global_batch_size": 2}).validate()
    with pytest.raises(ValueError, match="exclusion"):
        RunConfig(
            dataclasses.replace(core, selection_table="/tmp/table", selection_sha256="0" * 64), {"fully_async": True}
        ).validate()


def test_missing_feedback_binding_fails_closed(args, tmp_path):
    setup_remind(args, tmp_path)
    with pytest.raises(RuntimeError, match="no bound"):
        recording.create_recorder(args)


def test_run_spec_selects_remind_without_changing_preparation(tmp_path):
    spec = run_spec.RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "remind-example",
            "model": {"source": "model", "format": "hf"},
            "output": {"root": str(tmp_path / "run")},
            "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 8}]},
            "core": {"prompt_provider": "remind", "max_policy_lag": 2},
            "miles": {"fully_async": True},
            "records": {"enabled": True, "root": str(tmp_path / "records")},
        },
        config_path=tmp_path / "run.toml",
    )
    compiled = spec.compile()
    assert compiled.core.prompt_provider == "remind"
    assert compiled.core.remind_review_fraction == 0.02
    assert compiled.miles["fully_async"] is True


def test_unscored_discard_releases_pending_selection_and_requeues_review(args, tmp_path):
    setup_remind(args, tmp_path)
    source = DashboardDrainingRolloutDataSource(args)
    try:
        fresh = source.get_samples(1)[0]
        record = feedback(fresh[0]) | {"group_index": fresh[0].group_index}
        source.feedback([record])
        source.acknowledge_groups([fresh])
        review = source.get_samples(1)[0]
        source.acknowledge_groups([review])  # Discarded before any outcome exists.
        assert not source._delegate.provider.pending
        assert source.get_samples(1)[0][0].prompt == fresh[0].prompt
    finally:
        source.close()
