"""Run homogeneous packing through native stale filtering and capacity accounting."""

import asyncio
import contextlib
import io
import sys
from types import SimpleNamespace

import pytest
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils import arguments as native_arguments
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

from open_instruct.miles.configuration import arguments as core_arguments
from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.rollout.pool_batch_buffer import PoolBatchDataBuffer


def args(**overrides):
    return SimpleNamespace(
        **(
            {
                "rollout_batch_size": 2,
                "n_samples_per_prompt": 2,
                "global_batch_size": 4,
                "max_weight_staleness": 2,
                "async_data_buffer_capacity_factor": 4,
                "dynamic_sampling_filter_path": None,
                "reward_key": None,
                "fully_async": True,
                "rollout_seed": 17,
                "olmo_core": CoreConfig(
                    pool_batch_fraction=1, pool_batch_field="pool", pool_batch_values=["if", "math"]
                ),
            }
            | overrides
        )
    )


def entry(pool, identity, version=1):
    samples = [
        Sample(
            metadata={"pool": pool},
            group_index=identity,
            weight_versions=[WeightVersionsPerCall([WeightVersionSpan(str(version), 1, 2)])],
            reward=float(i),
            status=Sample.Status.COMPLETED,
        )
        for i in range(2)
    ]
    return DataBufferInput(prompt_group=samples, group=samples)


def test_native_stale_filter_and_ready_batch():
    async def scenario():
        rejected = []
        buffer = PoolBatchDataBuffer(DataBufferConstructorInput(args(), rejected.append))
        stale = entry("if", 0, version=0)
        i1, m1, m2, i2 = [entry(pool, i + 1) for i, pool in enumerate(["if", "math", "math", "if"])]
        for item in [stale, i1, m1]:
            await buffer.put(item)
        assert await buffer.get_ready(current_version=3) is None
        assert rejected == [stale.prompt_group]
        await buffer.put(m2)
        assert await buffer.get_ready(current_version=3) is m1
        buffer.retain_prepared_capacity()
        assert await buffer.get_ready(current_version=3) is m2
        buffer.retain_prepared_capacity()
        await buffer.release_prepared_capacity(2)
        await buffer.put(i2)
        assert await buffer.get(current_version=3) is i1
        assert await buffer.get(current_version=3) is i2
        assert buffer._delegate._capacity == 8
        metrics = buffer.get_metrics()
        assert metrics["rollout/fully_async/stale_groups_filtered"] == 1
        assert metrics["rollout/pool_batches/homogeneous"] == 2

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_waits_for_matching_pool_without_admitting_mixed_batch():
    async def scenario():
        buffer = PoolBatchDataBuffer(DataBufferConstructorInput(args(), lambda group: None))
        i1, m1, m2 = entry("if", 1), entry("math", 2), entry("math", 3)
        await buffer.put(i1)
        await buffer.put(m1)
        waiting = asyncio.create_task(buffer.get(current_version=1))
        await asyncio.sleep(0)
        assert not waiting.done()
        await buffer.put(m2)
        assert await waiting is m1
        assert await buffer.get(current_version=1) is m2
        assert len(buffer.queue) == 1

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_rejects_deadlock_prone_capacity():
    with pytest.raises(ValueError, match="capacity"):
        PoolBatchDataBuffer(DataBufferConstructorInput(args(async_data_buffer_capacity_factor=2), lambda group: None))


def test_partial_prepared_cancellation_restarts_a_complete_cohort():
    async def scenario():
        buffer = PoolBatchDataBuffer(DataBufferConstructorInput(args(), lambda group: None))
        i1, i2, m1, m2 = entry("if", 1), entry("if", 2), entry("math", 3), entry("math", 4)
        for item in [i1, i2, m1, m2]:
            await buffer.put(item)
        assert await buffer.get(current_version=1) is i1
        buffer.retain_prepared_capacity()
        # The producer returns i1's pristine prompt to the source on cancellation.
        await buffer.release_prepared_capacity(1)
        assert not buffer.queue.cohort
        assert await buffer.get(current_version=1) is m1
        assert await buffer.get(current_version=1) is m2
        regenerated = entry("if", 1)
        await buffer.put(regenerated)
        assert await buffer.get(current_version=1) is i2
        assert await buffer.get(current_version=1) is regenerated
        assert len(buffer.queue) == 0

    asyncio.run(asyncio.wait_for(scenario(), 5))


@pytest.mark.parametrize("capacity", [8, 16])
def test_native_parser_preserves_capacity_for_pool_validation(monkeypatch, capacity):
    config = RunConfig(
        CoreConfig(pool_batch_fraction=1, pool_batch_values=[str(i) for i in range(7)], max_policy_lag=12),
        {
            "hf_checkpoint": "model",
            "rollout_batch_size": 64,
            "n_samples_per_prompt": 8,
            "global_batch_size": 512,
            "fully_async": True,
            "async_data_buffer_capacity_factor": capacity,
        },
    )
    monkeypatch.setattr(sys, "argv", ["test", *config.arguments()])
    with contextlib.redirect_stderr(io.StringIO()):
        parsed = core_arguments.load_core_args(native_arguments.get_miles_extra_args_provider())
    assert parsed.async_data_buffer_capacity_factor == capacity
    assert parsed.custom_async_data_buffer_path == "open_instruct.miles.rollout.pool_batch_buffer.PoolBatchDataBuffer"
