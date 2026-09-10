"""Use actual MILES samples, buffers, and data cursors at the async boundary."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample

from open_instruct.miles.async_buffer import HomogeneousPolicyDataBuffer
from open_instruct.miles.data_source import DashboardDrainingRolloutDataSource


def make_source(root, *, load=None):
    args = SimpleNamespace(
        rollout_global_dataset=False,
        use_miles_dashboard=False,
        fully_async=True,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        save=str(root),
        load=str(load) if load else None,
        rollout_shuffle=False,
        start_rollout_id=1 if load else 0,
    )
    source = DashboardDrainingRolloutDataSource(args)
    args.rollout_global_dataset = True

    class Dataset:
        samples = [Sample(prompt=f"prompt-{i}") for i in range(32)]

        def __len__(self):
            return len(self.samples)

    source._delegate.dataset = Dataset()
    return source


def test_pending_prompt_cursor_regenerates_without_skipping_or_reusing_output(tmp_path):
    source = make_source(tmp_path)
    consumed, active, buffered = source.get_samples(3)
    source.acknowledge_groups([consumed])
    active[0].tokens = [10, 20]
    active[0].response_length = 1
    active[0].weight_versions = [7]
    source.add_samples([buffered])
    source.save(0)
    restored = make_source(tmp_path, load=tmp_path)
    restored.load(0)
    groups = restored.get_samples(3)
    assert [group[0].prompt for group in groups] == ["prompt-1", "prompt-2", "prompt-3"]
    assert [sample.index for group in groups for sample in group] == list(range(2, 8))
    assert all(not sample.tokens and not sample.weight_versions for group in groups for sample in group)
    restored.acknowledge_groups(groups)
    restored.save(1)
    assert (
        torch.load(tmp_path / "rollout/global_dataset_state_dict_1.pt", weights_only=True)["olmo_async_pending"][
            "groups"
        ]
        == []
    )


def test_interrupted_cursor_write_preserves_committed_file(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    source.get_samples(1)
    source.save(0)
    path = tmp_path / "rollout/global_dataset_state_dict_0.pt"
    original = path.read_bytes()

    def fail_save(state, stream):
        stream.write(b"partial")
        raise OSError("injected interruption")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interruption"):
        source.save(0)
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".cursor-*"))


def test_async_buffer_homogeneity_and_optimizer_step_lag_budget():
    async def exercise():
        rejected = []
        args = SimpleNamespace(
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            global_batch_size=2,
            max_weight_staleness=2,
            async_data_buffer_capacity_factor=2,
            dynamic_sampling_filter_path=None,
        )
        buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, rejected.append))
        assert args.max_weight_staleness == 2
        assert buffer._delegate._args.max_weight_staleness == 1

        def entry(versions):
            group = [Sample(weight_versions=[version], status=Sample.Status.COMPLETED) for version in versions]
            return DataBufferInput(prompt_group=group, group=group)

        mixed = entry([1, 2])
        stale = entry([0, 0])
        admitted = entry([1, 1])
        await buffer.put(mixed)
        await buffer.put(stale)
        await buffer.put(admitted)
        assert await asyncio.wait_for(buffer.get(current_version=2), 1) is admitted
        assert rejected == [mixed.prompt_group, stale.prompt_group]
        assert buffer.get_metrics()["rollout/fully_async/rejected_policy_groups"] == 1
        assert buffer.get_metrics()["rollout/fully_async/rejected_policy_groups"] == 0

    asyncio.run(exercise())


def test_async_buffer_accepts_homogeneous_multisegment_trajectories():
    async def exercise():
        args = SimpleNamespace(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            global_batch_size=2,
            max_weight_staleness=1,
            async_data_buffer_capacity_factor=1,
            dynamic_sampling_filter_path=None,
        )
        buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, lambda _: pytest.fail("rejected")))
        trajectories = [[Sample(weight_versions=[3], status=Sample.Status.COMPLETED)] for _ in range(2)]
        item = DataBufferInput(prompt_group=[trajectory[0] for trajectory in trajectories], group=trajectories)
        await buffer.put(item)
        assert await asyncio.wait_for(buffer.get(current_version=3), 1) is item

    asyncio.run(exercise())
