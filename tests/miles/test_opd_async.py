"""Exercise the async OPD boundary against the pinned native runtime."""

import asyncio
from types import SimpleNamespace

import pytest
from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample

from open_instruct.miles import async_buffer, async_rollout, opd_async


@pytest.mark.parametrize("fails", [False, True])
def test_async_eval_does_not_mutate_inflight_teacher_args(monkeypatch, fails):
    async def exercise():
        producer = opd_async.OPDAsyncRollout.__new__(opd_async.OPDAsyncRollout)
        args = SimpleNamespace(custom_rm_path="teacher", custom_reward_post_process_path="teacher-post", num_rollout=4)
        producer.args = args
        producer.state = SimpleNamespace(args=args, semaphore=asyncio.Semaphore(1))
        producer._producer_resumed = asyncio.Event()
        producer._producer_resumed.set()
        producer._eval_prompt_dataset_cache = {}

        async def evaluate(state, cache):
            assert not producer._producer_resumed.is_set()
            assert state.semaphore is producer.state.semaphore
            assert state.args.custom_rm_path.endswith("eval_reward")
            assert state.args.custom_reward_post_process_path is None
            await asyncio.sleep(0)
            assert args.custom_rm_path == "teacher"
            assert args.custom_reward_post_process_path == "teacher-post"
            if fails:
                raise RuntimeError("eval failure")
            return {"math": {"rewards": [1]}}

        monkeypatch.setattr(opd_async, "run_eval_datasets", evaluate)
        if fails:
            with pytest.raises(RuntimeError, match="eval failure"):
                await producer._call_eval(RolloutFnEvalInput(rollout_id=0))
        else:
            result = await producer._call_eval(RolloutFnEvalInput(rollout_id=0))
            assert result.data["math"]["rewards"] == [1]
        assert producer._producer_resumed.is_set()

    asyncio.run(exercise())


def test_async_prefetch_age_uses_consuming_update_and_resets_on_resume(monkeypatch):
    async def exercise():
        producer = opd_async.OPDAsyncRollout.__new__(opd_async.OPDAsyncRollout)
        producer.args = SimpleNamespace(max_weight_staleness=3)
        producer.data_source = SimpleNamespace()
        versions = []

        async def generate(self, input):
            versions.append(input.weight_version)
            return RolloutFnTrainOutput(samples=[[Sample(weight_versions=[1])]], metrics={})

        monkeypatch.setattr(async_rollout.ManagedFullyAsyncRolloutFn, "__call__", generate)
        first = await producer(RolloutFnTrainInput(rollout_id=70, weight_version=1))
        second = await producer(RolloutFnTrainInput(rollout_id=71, weight_version=1))
        assert versions == [1, 2]
        assert first.metrics["rollout/fully_async/consumed_age_max"] == 0
        assert second.metrics["rollout/fully_async/consumed_age_max"] == 1

    asyncio.run(exercise())


def test_measured_queue_counts_long_stale_work_and_waits():
    async def exercise():
        args = SimpleNamespace(
            rollout_batch_size=2,
            async_data_buffer_capacity_factor=2,
            dynamic_sampling_filter_path=None,
            max_weight_staleness=3,
        )
        dropped = []
        buffer = async_buffer.MeasuredDataBuffer(DataBufferConstructorInput(args, dropped.append))

        def entry(version, length):
            group = [Sample(weight_versions=[version], response_length=length, status=Sample.Status.COMPLETED)]
            return DataBufferInput(prompt_group=group, group=group)

        stale, fresh = entry(1, 16384), entry(5, 64)
        await buffer.put(stale)
        waiter = asyncio.create_task(buffer.get(current_version=5))
        await asyncio.sleep(0.01)
        await buffer.put(fresh)
        assert await waiter is fresh
        metrics = buffer.get_metrics()
        prefix = "rollout/fully_async/completed_queue/"
        assert metrics[prefix + "consumer_wait_seconds"] >= 0.01
        assert metrics[prefix + "dropped_response_tokens"] == 16384
        assert metrics[prefix + "delivered_samples"] == 1
        assert metrics[prefix + "dropped_samples_by_age/4"] == 1
        assert dropped == [stale.prompt_group]

    asyncio.run(exercise())


def test_prefetched_prompts_survive_checkpoint_before_their_update(tmp_path):
    args = SimpleNamespace(
        rollout_global_dataset=False,
        use_miles_dashboard=False,
        fully_async=True,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        save=str(tmp_path),
        load=None,
        rollout_shuffle=False,
        start_rollout_id=0,
    )
    source = opd_async.OPDAsyncDataSource(args)
    args.rollout_global_dataset = True

    class Dataset:
        samples = [Sample(prompt=f"prompt-{i}") for i in range(8)]

        def __len__(self):
            return len(self.samples)

    source._delegate.dataset = Dataset()
    current, prefetched = source.get_samples(2)
    source.consumer_rollout_id = 0
    source.acknowledge_groups([current])
    source.consumer_rollout_id = 1
    source.acknowledge_groups([prefetched])
    source.save(0)
    restored_args = SimpleNamespace(**vars(args))
    restored_args.rollout_global_dataset = False
    restored_args.load = str(tmp_path)
    restored_args.start_rollout_id = 1
    restored = opd_async.OPDAsyncDataSource(restored_args)
    restored_args.rollout_global_dataset = True
    restored._delegate.dataset = Dataset()
    restored.load(0)
    assert restored.get_samples(1)[0][0].prompt == prefetched[0].prompt
    source.save(1)
    assert not source._pending_groups
