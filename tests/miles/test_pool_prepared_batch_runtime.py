"""Exercise finite pools through real cross-loop prefetch, conversion and recovery."""

import asyncio
from copy import copy
from dataclasses import replace

import pytest
import test_pool_provider
import test_prepared_batch_runtime as prepared
import test_prompt_provider
from miles.backends.core_utils.rollout.async_buffer import RefreshPolicyDataBuffer
from miles.backends.core_utils.rollout.errors import PromptPoolExhausted
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput

from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource

args = test_prompt_provider.args
harness = prepared.harness


@pytest.fixture(params=[("uniform", False), ("uniform", True), ("exposure", True)])
def pooled_harness(harness, args, tmp_path, request):
    policy, priority = request.param
    test_pool_provider.setup(args, tmp_path, policy, priority=priority)
    core = args.olmo_core
    vars(args).update(vars(prepared.config()))
    args.olmo_core = replace(core, publication_mode="refresh")
    source = DashboardDrainingRolloutDataSource(args)
    h = harness
    h.source = h.executor.data_source = h.producer.data_source = source
    h.executor.args = h.producer.args = args
    h.producer._pool_exposure = True
    h.producer._active_tasks = set()
    h.producer._ready_completion_counts = {}
    h.producer._output = RefreshPolicyDataBuffer(
        DataBufferConstructorInput(args, lambda group: source.discard_groups([group]))
    )
    try:
        yield h
    finally:
        # Join the producer before unbinding its feedback recorder.
        prepared.async_utils.run(h.producer.release_reservation(consume=False))
        h.producer._output._delegate._records.close()
        source.close()


def restore(h):
    args = copy(h.source.args)
    args.olmo_core = replace(args.olmo_core)
    args.load, args.start_rollout_id = args.save, 1
    source = DashboardDrainingRolloutDataSource(args)
    source.load(0)
    return source


@pytest.mark.parametrize("cancel", [False, True])
def test_pool_prefetch_checkpoint_and_cancel_preserve_uncredited_prompts(pooled_harness, cancel):
    h = pooled_harness

    async def scenario():
        await prepared.on_producer(prepared.seed(h))
        await h.executor.get(0)
        assert sum(h.source.provider.trained.values()) == 0
        # The stale group must release its pool reservation, while the two
        # accepted groups remain outstanding until the next optimizer commit.
        await prepared.on_producer(prepared.seed(h, versions=(0, 1)))
        await h.executor.core_prefetch(1)
        await prepared.on_producer(prepared.seed(h, versions=(1,)))
        await h.executor._core_prepared_batch.task
        assert set(h.source._exposure_batches) == {0, 1}
        assert 2 not in h.source.provider.pending
        expected = test_prompt_provider.payload(list(h.source._pending_groups.values()))
        assert len(expected) == 2
        await h.executor.core_commit_exposure(0)
        h.source.save(0)
        assert sum(h.source.provider.trained.values()) == 2
        if cancel:
            await h.executor.core_cancel_prepared()
            await h.executor.core_cancel_prepared()
            assert test_prompt_provider.payload(h.source.get_samples(2)) == expected
            assert sum(h.source.provider.trained.values()) == 2
            # Cancellation after staging must still permit a recovery save.
            h.source.save(0)
            assert len(h.removed) == 1
        else:
            h.executor.set_weight_version(3)
            await h.executor.core_get_prepared(1)
            assert not h.source._pending_groups
            await h.executor.core_commit_exposure(1)
            await h.executor.core_commit_exposure(1)
            assert sum(h.source.provider.trained.values()) == 4
            assert not h.source.provider.pending
        assert h.producer._reserved_groups is None
        assert h.producer._output._delegate._capacity == 2
        restored = restore(h)
        try:
            assert sum(restored.provider.trained.values()) == 2
            assert test_prompt_provider.payload(restored.get_samples(2)) == expected
            assert not restored._exposure_batches
        finally:
            restored.close()

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("remaining", [0, 1])
def test_pool_exhaustion_during_prefetch_preserves_last_completed_update(pooled_harness, remaining):
    h = pooled_harness

    async def scenario():
        await prepared.on_producer(prepared.seed(h))
        await h.executor.get(0)
        await h.executor.core_commit_exposure(0)
        # The fixture has 18 finite prompts. Model terminal drops of all but
        # an incomplete final batch, then exercise the actual exhaustion path.
        h.source.discard_groups(h.source.get_samples(16 - remaining))
        await prepared.on_producer(prepared.seed(h, versions=(1,) * remaining))
        await h.executor.core_prefetch(1)
        h.executor.set_weight_version(3)
        with pytest.raises(PromptPoolExhausted):
            await h.executor.core_get_prepared(1)
        assert not h.source._exposure_batches
        expected = test_prompt_provider.payload(list(h.source._pending_groups.values()))
        assert len(expected) == remaining
        # The driver saves before cancelling the exhausted prefetch.
        h.source.save(0)
        await h.executor.core_cancel_prepared()
        assert h.producer._reserved_groups is None
        assert h.producer._output._delegate._capacity == 2
        assert sum(h.source.provider.trained.values()) == 2
        assert not h.removed
        restored = restore(h)
        try:
            assert sum(restored.provider.trained.values()) == 2
            assert test_prompt_provider.payload(restored.get_samples(remaining)) == expected
        finally:
            restored.close()

    asyncio.run(asyncio.wait_for(scenario(), 10))


@pytest.mark.parametrize("cancel_at", ["prepared", "conversion_failure", "handoff"])
def test_cancelled_pool_batch_can_retry_same_update(pooled_harness, monkeypatch, cancel_at):
    h = pooled_harness

    async def scenario():
        await prepared.on_producer(prepared.seed(h))
        await h.executor.get(0)
        current = list(h.source._exposure_batches[0])
        convert = prepared.rollout_executor.convert_samples_to_train_data
        if cancel_at == "conversion_failure":

            def fail_conversion(*args, **kwargs):
                raise ValueError("injected conversion failure")

            monkeypatch.setattr(prepared.rollout_executor, "convert_samples_to_train_data", fail_conversion)
        await prepared.on_producer(prepared.seed(h))
        await h.executor.core_prefetch(1)
        if cancel_at == "conversion_failure":
            with pytest.raises(ValueError, match="injected conversion failure"):
                await h.executor._core_prepared_batch.task
            monkeypatch.setattr(prepared.rollout_executor, "convert_samples_to_train_data", convert)
        else:
            await h.executor._core_prepared_batch.task
        following = list(h.source._exposure_batches[1])
        pending_before = h.source.provider.state_dict()
        if cancel_at == "handoff":
            h.executor.set_weight_version(3)
            release = h.producer.release_reservation
            loop = asyncio.get_running_loop()

            async def cancel_after_transfer(*, consume):
                groups = await release(consume=consume)
                if consume:
                    loop.call_soon_threadsafe(consumer.cancel)
                    await asyncio.sleep(0.01)
                return groups

            monkeypatch.setattr(h.producer, "release_reservation", cancel_after_transfer)
            consumer = asyncio.create_task(h.executor.core_get_prepared(1))
            with pytest.raises(asyncio.CancelledError):
                await consumer
        await h.executor.core_cancel_prepared()
        await h.executor.core_cancel_prepared()
        # Releasing the unused next batch must not erase the update in training,
        # credit exposure, or change selection reservations and sampling history.
        assert h.source._exposure_batches == {0: current}
        assert h.source.provider.state_dict() == pending_before
        assert h.producer._output._delegate._capacity == 2
        if cancel_at == "handoff":
            monkeypatch.setattr(h.producer, "release_reservation", release)
            # Retrying after publication takes the ordinary sequential path.
            await h.executor.core_commit_exposure(0)
            await prepared.on_producer(prepared.seed(h))
            await h.executor.get(1)
        else:
            # Before publication, retry prefetch while update 0 still trains.
            await prepared.on_producer(prepared.seed(h))
            await h.executor.core_prefetch(1)
            await h.executor._core_prepared_batch.task
            assert h.source._exposure_batches == {0: current, 1: following}
            await h.executor.core_commit_exposure(0)
            h.executor.set_weight_version(3)
            await h.executor.core_get_prepared(1)
        await h.executor.core_commit_exposure(1)
        assert sum(h.source.provider.trained.values()) == 4
        assert not h.source.provider.pending
        assert not h.source._exposure_batches

    asyncio.run(asyncio.wait_for(scenario(), 10))


def test_partial_requeue_does_not_release_staged_pool_batch(pooled_harness):
    h = pooled_harness

    async def scenario():
        await prepared.on_producer(prepared.seed(h))
        await h.executor.core_prefetch(1)
        await h.executor._core_prepared_batch.task
        staged = dict(h.source._exposure_batches)
        pending = h.source.provider.state_dict()
        identities = list(h.source._pending_groups)
        with pytest.raises(RuntimeError, match="part of a staged exposure batch"):
            h.source.requeue_pending_groups(identities[:1])
        assert h.source._exposure_batches == staged
        assert h.source.provider.state_dict() == pending
        assert not h.source._delegate.buffer
        assert list(h.source._pending_groups) == identities
        await h.executor.core_cancel_prepared()
        assert not h.source._exposure_batches

    asyncio.run(asyncio.wait_for(scenario(), 10))
