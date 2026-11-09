"""Prompt replay through the real buffered filter, recorder, and resume ledger."""

import asyncio
import copy

import pytest
import test_mixture_provider
import test_prompt_provider
import test_remind_provider
from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource
from open_instruct.miles.rollout.provider_source import ProviderRolloutDataSource

args = test_prompt_provider.args


def setup(args, tmp_path):
    test_remind_provider.setup_remind(args, tmp_path)
    args.olmo_core = CoreConfig(
        records_root=str(tmp_path / "records"),
        prompt_replay_enabled=True,
        prompt_replay_fraction=1,
        prompt_replay_max_successes=0,
        prompt_replay_truncation="none",
        prompt_replay_window=1,
        prompt_replay_length_drop=0.2,
        prompt_replay_answer_probe=True,
    )


def scored(group, rewards, length):
    result = test_remind_provider.scored(group, rewards)
    for sample in group:
        sample.response_length = length
        sample.tokens = [1] * (length + 1)
        sample.response = "The answer is 42."
        sample.metadata["verifiers"][0]["target"] = "42"
    return result


def test_real_zero_filter_replay_and_pending_resume(args, tmp_path):
    setup(args, tmp_path)
    source = DashboardDrainingRolloutDataSource(args)
    buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, source.add_samples))
    recorder = buffer._delegate._records
    try:
        failed = source.get_samples(1)[0]
        assert asyncio.run(buffer.put(scored(failed, [0, 0], 100))) is False
        source.acknowledge_groups([failed])
        shorter = source.get_samples(1)[0]
        assert asyncio.run(buffer.put(scored(shorter, [1, 1], 70))) is False
        source.acknowledge_groups([shorter])
        replay = source.get_samples(1)[0]
        assert replay[0].prompt == failed[0].prompt
        assert replay[0].metadata["prompt_replay_review"] is True
        assert source.provider_metrics()["rollout/prompt_replay/zero_literal_reference_found"] == 2
        source.save(0)
        recorder.close()
        source.close()
        args.load, args.start_rollout_id = args.save, 1
        restored = DashboardDrainingRolloutDataSource(args)
        try:
            restored.load(0)
            pending = restored.get_samples(1)[0]
            assert test_prompt_provider.payload([pending]) == test_prompt_provider.payload([replay])
            buffer2 = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, restored.add_samples))
            try:
                assert asyncio.run(buffer2.put(scored(pending, [1, 0], 70))) is True
                restored.acknowledge_groups([pending])
                metrics = buffer2.get_metrics()
                assert metrics["rollout/records/failed_total"] == 0
                assert metrics["rollout/prompt_replay/replay_mixed"] == 1
                assert metrics["rollout/prompt_replay/candidates"] == 0
            finally:
                buffer2._delegate._records.close()
        finally:
            restored.close()
    finally:
        recorder.close()
        source.close()


def test_delivery_rollback_does_not_copy_history(args, tmp_path, monkeypatch):
    setup(args, tmp_path)
    source = ProviderRolloutDataSource(args)
    first = source.get_samples(1)
    source.add_samples(first)
    before = source.provider.state_dict()
    original = source.provider.next_samples

    def fail(n):
        original(n)
        raise RuntimeError("failed after selection")

    monkeypatch.setattr(source.provider, "next_samples", fail)
    with pytest.raises(RuntimeError, match="failed after selection"):
        source.get_samples(3)
    assert source.provider.state_dict() == before
    assert source.buffer == first


def test_dataset_replacement_rejected(args, tmp_path):
    setup(args, tmp_path)
    source = ProviderRolloutDataSource(args)
    source.dataset = copy.deepcopy(source.dataset)
    with pytest.raises(RuntimeError, match="cannot be replaced"):
        source.get_samples(1)


def test_mixture_catalog_and_record_identity_match(args, tmp_path):
    test_mixture_provider.setup(args)
    args.olmo_core = CoreConfig(
        prompt_provider="mixture",
        prompt_mixture=args.olmo_core.prompt_mixture,
        prompt_replay_enabled=True,
        prompt_replay_fraction=0,
        records_root=str(tmp_path / "records"),
    )
    source = ProviderRolloutDataSource(args)
    groups = source.get_samples(100)
    assert {g[0].metadata["mixture_source"] for g in groups} == {"a", "b"}
    for group in groups:
        sample = group[0]
        key = sample.metadata["prompt_replay_prompt_key"]
        assert source.provider.samples[key].prompt == sample.prompt
        assert source.provider.samples[key].metadata["verifiers"] == sample.metadata["verifiers"]
        assert key.startswith(sample.metadata["mixture_source"] + ":")
