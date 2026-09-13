"""The refresh contract preserves behavior provenance and owns live requests."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from miles.backends.training_utils.loss_hub.corrections import vanilla_tis_function
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample

from open_instruct.miles import policy_refresh
from open_instruct.miles.async_buffer import RefreshPolicyDataBuffer
from open_instruct.miles.refreshing_rollout import RefreshingRolloutFn


def sample(group=0, versions=(0, 1)):
    value = Sample(
        group_index=group,
        tokens=[9, 10, 11, 12],
        response_length=3,
        rollout_log_probs=[-0.8, -1.2, -0.6],
        status=Sample.Status.COMPLETED,
    )
    policy_refresh.record_response(
        value,
        dict(
            weight_version=str(versions[-1]),
            weight_versions=[
                dict(version=str(versions[0]), start=0, end=2),
                dict(version=str(versions[-1]), start=2, end=3),
            ],
            output_token_logprobs=[[-0.8, 10], [-1.2, 11], [-0.6, 12]],
        ),
    )
    return value


def args():
    return SimpleNamespace(
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        global_batch_size=2,
        max_weight_staleness=1,
        async_data_buffer_capacity_factor=1,
        dynamic_sampling_filter_path=None,
        olmo_core=SimpleNamespace(engine_drain_timeout=1),
    )


def entry(index=0, versions=(0, 1)):
    values = [sample(index, versions) for _ in range(2)]
    return DataBufferInput(prompt_group=values, group=values)


def test_behavior_scores_and_routes_are_unchanged_and_provenance_serializes():
    value = sample()
    assert value.rollout_log_probs == [-0.8, -1.2, -0.6]
    assert value.weight_versions == ["0", "1"]
    assert value.oldest_weight_version == 0
    assert value.train_metadata == value.metadata
    restored = Sample.from_dict(value.to_dict())
    assert restored.train_metadata == value.train_metadata
    assert restored.weight_versions == value.weight_versions
    batch = dict(metadata=[restored.train_metadata], response_lengths=[3], weight_versions=[restored.weight_versions])
    assert policy_refresh.validate_batch(batch)[0][0] == dict(version=0, start=0, end=2)
    batch["weight_versions"] = [["1"]]
    with pytest.raises(ValueError, match="staleness ledger"):
        policy_refresh.validate_batch(batch)


@pytest.mark.parametrize("change", ["gap", "overlap", "future", "missing", "boolean"])
def test_bad_spans_fail_closed(change):
    spans = [dict(version=0, start=0, end=2), dict(version=1, start=2, end=3)]
    if change == "gap":
        spans[0]["end"] = 1
    elif change == "overlap":
        spans[1]["start"] = 1
    elif change == "future":
        spans[0]["version"] = 2
    elif change == "missing":
        spans.pop()
    else:
        spans[0]["version"] = True
    with pytest.raises(ValueError):
        policy_refresh.validate_spans(spans, 3, replay_version=1)


def test_repeated_refresh_and_empty_span_boundaries():
    spans = [
        dict(version=0, start=0, end=1),
        dict(version=1, start=1, end=1),
        dict(version=2, start=1, end=2),
        dict(version=2, start=2, end=3),
    ]
    assert policy_refresh.validate_spans(spans, 3, replay_version=2) == [
        dict(version=0, start=0, end=1),
        dict(version=2, start=1, end=3),
    ]


def test_mixed_group_is_accepted_but_old_prefix_expires():
    async def scenario():
        unused = []
        buffer = RefreshPolicyDataBuffer(DataBufferConstructorInput(args(), unused.append))
        mixed = entry()
        # Different siblings may have different publication boundaries.
        mixed.group[1] = sample(0, (1, 1))
        await buffer.put(mixed)
        assert (await buffer.get(current_version=1)) is mixed
        stale = entry(1)
        await buffer.put(stale)
        pending = asyncio.create_task(buffer.get(current_version=2))
        await asyncio.sleep(0)
        assert unused == [stale.prompt_group]
        assert not pending.done()
        fresh = entry(2, (1, 2))
        await buffer.put(fresh)
        assert await asyncio.wait_for(pending, 1) is fresh

    asyncio.run(scenario())


def producer():
    p = RefreshingRolloutFn.__new__(RefreshingRolloutFn)
    p.args = args()
    p._refreshing = p._publication_paused = p._stopping = p._shutdown_complete = False
    p._refresh_started = None
    p._boundary_capacity = None
    p._output = RefreshPolicyDataBuffer(DataBufferConstructorInput(args(), lambda group: None))
    p._producer_resumed = asyncio.Event()
    p._producer_resumed.set()
    p._producer_idle = asyncio.Event()
    p._stop_requested = asyncio.Event()
    p._producing_groups = {1: entry(1).group}
    p._worker = None
    return p


def test_refresh_does_not_cancel_or_wait_for_live_requests():
    async def scenario():
        p = producer()
        blocked = asyncio.Event()
        p._worker = asyncio.create_task(blocked.wait())
        await asyncio.wait_for(p.begin_refresh(), 0.1)
        assert not p._producer_resumed.is_set() and not p._worker.done()
        assert 1 in p._producing_groups
        with pytest.raises(RuntimeError, match="Cannot start"):
            await p.begin_refresh()
        await p.end_refresh()
        assert p._producer_resumed.is_set() and not p._worker.done()
        blocked.set()
        await p._worker

    asyncio.run(scenario())


def test_lifecycle_drains_saturated_queue_without_new_admission():
    async def scenario():
        p = producer()
        await p._output.put(entry(0))

        async def complete():
            await p._output.put(entry(1))
            p._producing_groups.clear()
            p._producer_idle.set()

        p._worker = asyncio.create_task(complete())
        await asyncio.wait_for(p.prepare_publication(), 1)
        assert p._worker.done() and not p._worker.cancelled()
        assert len(p._output._delegate._buffer) == 2
        await p.finish_publication()
        assert not p._producer_resumed.is_set()
        await p._output.get(current_version=1)
        p._resume_if_buffer_allows()
        assert p._producer_resumed.is_set()

    asyncio.run(scenario())


def test_failed_publication_never_reopens_and_shutdown_cancels_owned_work():
    async def scenario():
        p = producer()
        p._worker = asyncio.create_task(asyncio.Event().wait())
        await p.begin_refresh()
        with pytest.raises(RuntimeError, match="did not finish"):
            await p.shutdown()
        assert p._worker.cancelled()
        assert not p._producer_resumed.is_set()

    asyncio.run(scenario())


def test_existing_tis_uses_original_prefix_denominator_without_masking():
    behavior = torch.tensor([-0.8, -1.2, -0.6])
    scored = behavior + torch.tensor([0.1, -0.2, 0.0])
    gradient_forward = scored + 0.02
    ratio = (gradient_forward - scored).exp()
    masks = [torch.ones(3)]
    corrected, returned_masks, metrics = vanilla_tis_function(
        SimpleNamespace(tis_clip_low=0.5, tis_clip=2.0),
        pg_loss=-ratio,
        train_log_probs=[scored],
        rollout_log_probs=[behavior],
        loss_masks=masks,
    )
    torch.testing.assert_close(corrected, -(gradient_forward - behavior).exp())
    assert returned_masks is masks
    assert metrics["tis"][0] != 1 and metrics["tis"][1] != 1 and metrics["tis"][2] == 1
    assert not metrics["tis_clipfrac"].any()


def test_sample_reset_discards_refresh_training_provenance():
    value = sample()
    value.reset_for_retry()
    assert value.train_metadata is None and value.weight_versions == []
    assert value.rollout_log_probs is None


def test_refresh_score_metrics_separate_prefix_from_fresh_suffix():
    value = sample()
    batch = dict(
        metadata=[value.train_metadata],
        response_lengths=[3],
        weight_versions=[value.weight_versions],
        rollout_log_probs=[torch.tensor(value.rollout_log_probs)],
        log_probs=[torch.tensor(value.rollout_log_probs) + torch.tensor([0.1, -1.0, 0.0])],
        loss_masks=[torch.ones(3)],
    )
    result = policy_refresh.score_metrics(batch, current_version=1, clip_low=0.5, clip_high=2)
    assert result["mixed_responses"] == 1
    assert result["current_version_token_fraction"] == pytest.approx(1 / 3)
    assert result["historical_prefix"]["tis_clip_fraction"] == 0.5
    assert result["latest_forward"]["mean_abs_logratio"] == 0
