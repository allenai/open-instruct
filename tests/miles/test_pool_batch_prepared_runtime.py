"""Verify pool cohorts through actual cross-loop prefetch and cursor recovery."""

import asyncio
from types import SimpleNamespace

import pytest
import test_prepared_batch_runtime as prepared
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput

from open_instruct.miles.rollout.pool_batch_buffer import PoolBatchDataBuffer

harness = prepared.harness


@pytest.mark.parametrize("cancel", [False, True])
def test_pool_cohort_prefetch_save_and_cancel(harness, cancel):
    h = harness
    args = h.executor.args
    args.async_data_buffer_capacity_factor = 4
    args.rollout_seed = 17
    args.olmo_core = SimpleNamespace(
        prepared_batch_overlap=True,
        publication_mode="refresh",
        pool_batch_fraction=1,
        pool_batch_values=["if", "math"],
        pool_batch_field="pool",
    )
    for i, sample in enumerate(h.source.dataset.samples):
        sample.metadata = {"pool": "if" if i % 4 in (0, 3) else "math"}
    h.producer._output = PoolBatchDataBuffer(
        DataBufferConstructorInput(args, lambda group: h.source.acknowledge_groups([group]))
    )

    async def scenario():
        await prepared.on_producer(prepared.seed(h, versions=(1,) * 4))
        first = await h.executor.get(0)
        assert first.sample_indices == [0, 1, 6, 7]  # oldest ready pool: IF
        assert len(h.source._pending_groups) == 2
        await h.executor.core_prefetch(1)
        await h.executor._core_prepared_batch.task
        assert h.producer._output._delegate._capacity == 6
        h.source.save(0)
        restored = prepared.make_source(h.root, load=h.root)
        restored.load(0)
        regenerated = restored.get_samples(2)
        assert [group[0].group_index for group in regenerated] == [1, 2]
        assert all(sample.metadata["pool"] == "math" for group in regenerated for sample in group)
        assert all(not sample.tokens for group in regenerated for sample in group)
        if cancel:
            await h.executor.core_cancel_prepared()
            assert len(h.source._pending_groups) == 2
            recovered = h.source.get_samples(2)
            assert [group[0].group_index for group in recovered] == [1, 2]
        else:
            h.executor.set_weight_version(3)
            following = await h.executor.core_get_prepared(1)
            assert following.sample_indices == [2, 3, 4, 5]
            assert not h.source._pending_groups
        assert h.producer._output._delegate._capacity == 8
        assert not h.producer._output.queue.cohort

    asyncio.run(asyncio.wait_for(scenario(), 10))
