"""Exercise named pools through the actual loader, filter, recorder and checkpoint ledger."""

import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import test_mixture_provider
import test_prompt_provider
import test_remind_provider
from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput

from open_instruct import test_miles_run_data
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.datasets import run_data
from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource

args = test_prompt_provider.args


@pytest.mark.parametrize("policy", ["shuffle", "uniform", "weighted", "exposure"])
def test_source_pool_preparation_reaches_runtime_with_replay(args, tmp_path, monkeypatch, policy):
    setup(args, tmp_path, policy)
    old_path = Path(args.olmo_core.prompt_mixture)
    old_manifest = json.loads(old_path.read_text())
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"gsm8k": {"factory": "open_instruct.miles.rewards.verifiers.GSM8KVerifier"}}))
    sources = []
    for source in old_manifest["sources"]:
        path = old_path.parent / source["path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.pop("pool_id")
            row["metadata"].pop("pool_id")
            row["metadata"].pop("sampling_pool")
            row["metadata"]["verifiers"] = [{"name": "gsm8k", "target": row["label"]}]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        sources.append(
            {"name": source["name"], "pool": source["name"], "prompt_data": str(path), "reward_config": str(registry)}
        )
    hf = tmp_path / "tokenizer"
    hf.mkdir()
    monkeypatch.setattr(run_data, "_tokenizer", lambda _: test_miles_run_data.Tokenizer())
    sampling = {"policy": policy}
    if policy == "weighted":
        sampling["weights"] = {source["pool"]: 1 for source in sources}
    result = run_data.prepare_data(
        {"sources": sources, "sampling": sampling, "shuffle": args.rollout_shuffle, "seed": args.rollout_seed},
        hf,
        tmp_path / "assigned",
        max_prompt_length=2000,
        seed=args.rollout_seed,
    )
    args.prompt_data = result["prompt_data"]
    args.olmo_core = dataclasses.replace(
        args.olmo_core, prompt_mixture=result["prompt_mixture"], prompt_replay_enabled=True
    )
    loader = DashboardDrainingRolloutDataSource(args)
    try:
        groups = loader.get_samples(sum(source["rows"] for source in old_manifest["sources"]))
        assert {group[0].metadata["sampling_pool"] for group in groups} == {source["name"] for source in sources}
        assert all(group[0].metadata["sampling_pool"] == group[0].metadata["mixture_source"] for group in groups)
        assert all(group[0].metadata["prompt_replay_review"] is False for group in groups)
        assert not loader.provider.available()
    finally:
        loader.close()


def setup(args, tmp_path, policy="shuffle"):
    test_remind_provider.setup_remind(args, tmp_path)
    test_mixture_provider.setup(args)
    path = Path(args.olmo_core.prompt_mixture)
    manifest = json.loads(path.read_text())
    manifest["pools"] = {"math": {"label": "math"}, "if": {"label": "if"}, "empty": {"label": "empty"}}
    manifest["sampling"] = {"policy": policy}
    if policy == "weighted":
        manifest["sampling"]["weights"] = {"math": 3, "if": 1, "empty": 1}
    for source, pool in zip(manifest["sources"], ["math", "if"], strict=True):
        source.pop("fraction")
        data = path.parent / source["path"]
        rows = [json.loads(line) for line in data.read_text().splitlines()]
        for row in rows:
            row["pool_id"] = pool
            row["metadata"].update(pool_id=pool, sampling_pool=pool)
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        data.write_bytes(raw)
        source["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(manifest))
    args.olmo_core = CoreConfig(
        prompt_provider="pools", prompt_mixture=str(path), records_root=str(tmp_path / "records")
    )
    args.rollout_batch_size = 1
    args.n_samples_per_prompt = 2


@pytest.mark.parametrize("policy", ["shuffle", "uniform", "weighted", "exposure"])
def test_actual_filter_does_not_credit_until_optimizer_commit_and_restore(args, tmp_path, policy):
    setup(args, tmp_path, policy)
    source = DashboardDrainingRolloutDataSource(args)
    buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, source.add_samples))
    recorder = buffer._delegate._records
    try:
        failed = source.get_samples(1)[0]
        assert asyncio.run(buffer.put(test_remind_provider.scored(failed, [0, 0]))) is False
        source.acknowledge_groups([failed])
        assert not source.provider.pending
        kept = source.get_samples(1)[0]
        assert asyncio.run(buffer.put(test_remind_provider.scored(kept, [0, 1]))) is True
        source.acknowledge_groups([kept])
        assert sum(source.provider.trained.values()) == 0
        source.stage_training_batch(0, kept)
        source.commit_training_batch(0)
        source.commit_training_batch(0)  # RPC retry must not double-credit or append another receipt.
        assert sum(source.provider.trained.values()) == 1
        receipts = list((tmp_path / "records").glob("pool-exposure-*.jsonl"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text())
        assert receipt["update"] == 1 and receipt["selection_ids"] == [kept[0].metadata["pool_selection_id"]]
        outstanding = source.get_samples(1)
        source.save(0)
        recorder.close()
        source.close()
        args.load, args.start_rollout_id = args.save, 1
        restored = DashboardDrainingRolloutDataSource(args)
        try:
            restored.load(0)
            assert restored.provider.trained == source.provider.trained
            assert test_prompt_provider.payload(restored.get_samples(1)) == test_prompt_provider.payload(outstanding)
        finally:
            restored.close()
    finally:
        recorder.close()
        source.close()


@pytest.mark.parametrize("policy", ["shuffle", "uniform", "weighted", "exposure"])
def test_expired_queue_entry_releases_exposure_reservation(args, tmp_path, policy):
    setup(args, tmp_path, policy)
    source = DashboardDrainingRolloutDataSource(args)
    buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, source.add_samples))
    recorder = buffer._delegate._records
    try:
        kept = source.get_samples(1)[0]
        assert asyncio.run(buffer.put(test_remind_provider.scored(kept, [0, 1])))
        recorder.record_disposition(kept, accepted=False, staleness=99)
        source.acknowledge_groups([kept])
        assert not source.provider.pending
        assert sum(source.provider.trained.values()) == 0
        assert sum(source.provider.terminal.values()) == 1
    finally:
        recorder.close()
        source.close()
