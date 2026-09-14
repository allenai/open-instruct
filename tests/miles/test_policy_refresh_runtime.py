"""The refresh contract preserves behavior provenance and owns live requests."""

import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from miles.backends.training_utils.loss_hub.corrections import vanilla_tis_function
from miles.ray.rollout import train_data_conversion
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample

from open_instruct.miles import policy_refresh, refreshing_rollout
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
    assert value.train_metadata["policy_refresh"] == value.metadata["policy_refresh"]
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


def test_metadata_survives_dp_reordering_with_numpy_lengths():
    values = [sample(0, (0, 1)), sample(1, (1, 2))]
    data = dict(
        metadata=[v.train_metadata for v in values],
        weight_versions=[v.weight_versions for v in values],
        response_lengths=np.array([3, 3]),
        total_lengths=[4, 4],
    )
    shards = train_data_conversion._package_shards(None, data, [[1], [0]])
    assert policy_refresh.validate_batch(shards[0])[0][-1]["version"] == 2
    assert policy_refresh.validate_batch(shards[1])[0][-1]["version"] == 1


def test_queue_discard_metrics_capture_generated_lengths_before_retry_reset():
    async def scenario():
        retried = []

        def retry(group):
            retried.append(group)
            for value in group:
                value.response_length = 0
                value.weight_versions = []

        buffer = RefreshPolicyDataBuffer(DataBufferConstructorInput(args(), retry))
        stale = entry(0, (0, 1))
        current = entry(1, (2, 2))
        await buffer.put(stale)
        pending = asyncio.create_task(buffer.get(current_version=2))
        await buffer.put(current)
        assert (await pending) is current
        assert retried == [stale.prompt_group]
        metrics = buffer.get_metrics()
        prefix = "rollout/fully_async/completed_queue/"
        assert metrics[prefix + "dropped_samples"] == 2
        assert metrics[prefix + "dropped_response_tokens"] == 6
        assert metrics[prefix + "dropped_samples_by_age/2"] == 2
        assert metrics[prefix + "delivered_samples"] == 2
        assert metrics[prefix + "dropped_samples_fraction"] == 0.5
        assert buffer.get_metrics()[prefix + "dropped_samples"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("deadline,expires", [(0.2, False), (0.001, True)])
def test_request_deadline_is_independent_of_drain_and_cancels_timeout(monkeypatch, deadline, expires):
    async def scenario():
        p = producer()
        p.args.olmo_core.engine_drain_timeout = 0.0001
        p.args.olmo_core.refresh_request_timeout = deadline
        p.args.sglang_router_ip, p.args.sglang_router_port = "localhost", 1234
        finalized = []

        async def post(*args, **kwargs):
            try:
                await asyncio.sleep(0.02)
                return {"meta_info": {"finish_reason": {"type": "stop"}}}
            finally:
                finalized.append(True)

        async def update(*args):
            pass

        monkeypatch.setattr(refreshing_rollout, "post", post)
        monkeypatch.setattr(refreshing_rollout, "compute_prompt_ids_from_sample", lambda *args: [9])
        monkeypatch.setattr(refreshing_rollout, "compute_request_payload", lambda *args, **kwargs: ({}, None))
        monkeypatch.setattr(refreshing_rollout, "update_sample_from_response", update)
        monkeypatch.setattr(policy_refresh, "record_response", lambda *args: {"spans": [], "replay_version": 0})
        request = SimpleNamespace(args=p.args, state=None, sample=sample(), sampling_params={})
        if expires:
            with pytest.raises(TimeoutError, match="core.refresh_request_timeout"):
                await p._generate_response(request)
        else:
            result = await p._generate_response(request)
            assert result.samples is request.sample
        assert finalized == [True]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_type", ["read", "status"])
def test_transport_failure_logs_request_identity_without_resampling(monkeypatch, caplog, failure_type):
    async def scenario():
        p = producer()
        p.args.olmo_core.refresh_request_timeout = 1
        p.args.sglang_router_ip, p.args.sglang_router_port = "localhost", 1234
        value = sample()
        value.index = 17
        calls = []
        failure = httpx.ReadError("connection reset")
        if failure_type == "status":
            response = httpx.Response(503, request=httpx.Request("POST", "http://localhost:1234/generate"))
            failure = httpx.HTTPStatusError("unavailable", request=response.request, response=response)

        async def post(url, payload, **kwargs):
            calls.append((url, dict(payload), kwargs))
            raise failure

        monkeypatch.setattr(refreshing_rollout, "post", post)
        monkeypatch.setattr(refreshing_rollout, "compute_prompt_ids_from_sample", lambda *args: [9])
        monkeypatch.setattr(
            refreshing_rollout,
            "compute_request_payload",
            lambda *args, **kwargs: ({"input_ids": [9], "private": "secret prompt"}, None),
        )
        request = SimpleNamespace(args=p.args, state=None, sample=value, sampling_params={})
        original_scores = list(value.rollout_log_probs)
        with pytest.raises(httpx.HTTPError) as caught:
            await p._generate_response(request)
        assert caught.value is failure
        assert len(calls) == 1
        _, payload, kwargs = calls[0]
        assert kwargs == {"max_retries": 1, "headers": {"x-miles-request-id": payload["rid"]}}
        assert value.rollout_log_probs == original_scores
        assert f"request={payload['rid']} group=0 sample=17" in caplog.text
        assert "delivery=unknown" in caplog.text
        assert f"error={type(failure).__name__}" in caplog.text
        assert "secret prompt" not in caplog.text
        if failure_type == "status":
            assert "status=503" in caplog.text

    asyncio.run(scenario())
