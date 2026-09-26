"""Exercise the actual pinned MILES buffer on CPU when the backend is available."""

import asyncio
from importlib import import_module
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import ZERO_STD_FILTER
from open_instruct.miles.publication import policy_refresh

pytest.importorskip("miles")
buffers = import_module("open_instruct.miles.rollout.async_buffer")
native = import_module("miles.rollout.fully_async_data_buffer")
Sample = import_module("miles.utils.types").Sample


def configured(kind, unused, enabled=True):
    args = SimpleNamespace(
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        global_batch_size=2,
        max_weight_staleness=1,
        async_data_buffer_capacity_factor=1,
        dynamic_sampling_filter_path=ZERO_STD_FILTER if enabled else None,
        reward_key="score",
    )
    cls = buffers.RefreshPolicyDataBuffer if kind == "refresh" else buffers.HomogeneousPolicyDataBuffer
    return cls(native.DataBufferConstructorInput(args, unused.append))


def entry(rewards, index=0, *, mixed=False):
    samples = []
    for reward in rewards:
        sample = Sample(
            group_index=index,
            tokens=[9, 10, 11],
            response_length=2,
            rollout_log_probs=[-0.8, -1.2],
            reward={"score": reward, "unrelated": 123},
            status=Sample.Status.COMPLETED,
        )
        policy_refresh.record_response(
            sample,
            {
                "weight_version": "1",
                "weight_versions": [
                    {"version": "0" if mixed else "1", "start": 0, "end": 1},
                    {"version": "1", "start": 1, "end": 2},
                ],
                "output_token_logprobs": [[-0.8, 10], [-1.2, 11]],
            },
        )
        samples.append(sample)
    return native.DataBufferInput(prompt_group=samples, group=samples)


@pytest.mark.parametrize("kind", ["homogeneous", "refresh"])
@pytest.mark.parametrize("reward", [0.0, 1.0, 0.5])
def test_constant_groups_drop_without_retry_and_mixed_groups_refill(kind, reward):
    async def scenario():
        unused = []
        buffer = configured(kind, unused)
        waiting = asyncio.create_task(buffer.get(current_version=1))
        dropped = entry([reward, reward], mixed=kind == "refresh")
        assert await buffer.put(dropped) is False
        await asyncio.sleep(0)
        assert not waiting.done()
        assert not buffer._delegate._buffer
        assert unused == []  # dynamic drops must not use the stale/aborted retry handler
        accepted = entry([0.0, 1.0], index=1, mixed=kind == "refresh")
        assert await buffer.put(accepted) is True
        assert await asyncio.wait_for(waiting, 1) is accepted
        assert [s.reward["score"] for s in accepted.group] == [0.0, 1.0]
        metrics = buffer.get_metrics()
        assert metrics[f"rollout/dynamic_filter/drop_zero_std_{reward}"] == 1
        assert f"rollout/dynamic_filter/drop_zero_std_{reward}" not in buffer.get_metrics()

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["homogeneous", "refresh"])
def test_opt_out_retains_constant_groups(kind):
    async def scenario():
        buffer = configured(kind, [], enabled=False)
        item = entry([0, 0], mixed=kind == "refresh")
        assert await buffer.put(item) is True
        assert await asyncio.wait_for(buffer.get(current_version=1), 1) is item

    asyncio.run(scenario())


def test_all_rejected_collection_remains_cancellable():
    async def scenario():
        buffer = configured("refresh", [])
        waiting = asyncio.create_task(buffer.get(current_version=1))
        for index in range(8):
            assert await buffer.put(entry([0, 0], index=index, mixed=True)) is False
        await asyncio.sleep(0)
        assert not waiting.done()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert not buffer._delegate._buffer

    asyncio.run(scenario())


def test_aborted_groups_still_retry_and_invalid_provenance_still_fails():
    async def scenario():
        unused = []
        buffer = configured("refresh", unused)
        aborted = entry([0, 0], mixed=True)
        aborted.group[0].status = Sample.Status.ABORTED
        assert await buffer.put(aborted) is True
        assert unused == [aborted.prompt_group]
        assert not buffer._delegate._buffer
        invalid = entry([0, 0], mixed=True)
        invalid.group[0].train_metadata = {}
        with pytest.raises(ValueError, match="provenance"):
            await buffer.put(invalid)
        assert not buffer.get_metrics().get("rollout/dynamic_filter/drop_zero_std_0.0")

    asyncio.run(scenario())
