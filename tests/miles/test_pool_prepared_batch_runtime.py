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


@pytest.fixture
def pooled_harness(harness, args, tmp_path):
    test_pool_provider.setup(args, tmp_path, "uniform")
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
