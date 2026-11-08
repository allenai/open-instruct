"""Exercise named pools through the actual loader, filter, recorder and checkpoint ledger."""

import asyncio
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_mixture_provider
import test_prompt_provider
import test_remind_provider
from miles.backends.core_utils.rollout import async_rollout
from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

from open_instruct import test_miles_run_data
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.datasets import run_data
from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource

args = test_prompt_provider.args


@pytest.mark.parametrize("policy", ["shuffle", "uniform", "weighted", "exposure"])
def test_failed_pool_delivery_restores_cursors_and_preserves_prior_reservations(args, tmp_path, monkeypatch, policy):
    setup(args, tmp_path, policy)
    source = DashboardDrainingRolloutDataSource(args)
    try:
        source.get_samples(1)
        before = copy.deepcopy(source.provider.state_dict())
        existing = dict(source.provider.pending)
        select = source.provider.next_samples

        def fail_after_selection(n):
            select(n)
            raise RuntimeError("delivery failed after selection")

        monkeypatch.setattr(source.provider, "next_samples", fail_after_selection)
        with pytest.raises(RuntimeError, match="delivery failed"):
            source.get_samples(2)
        assert source.provider.state_dict() == before
        assert all(source.provider.pending[key] is entry for key, entry in existing.items())
        monkeypatch.setattr(source.provider, "next_samples", select)
        expected = [group[0].prompt for group in source.get_samples(2)]
        source.provider.load_state_dict(before)
        # Compare prompt selection; response-group counters are owned by the bridge.
        assert [s.prompt for s in source.provider.next_samples(2)] == expected
    finally:
        source.close()


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
        metrics = source.commit_training_batch(0)
        source.commit_training_batch(0)  # RPC retry must not double-credit or append another receipt.
        assert sum(source.provider.trained.values()) == 1
        assert metrics == source.provider.metrics()
        for key in ("proposed", "attempted", "filtered", "filtered_all_zero", "trained"):
            expected = 2 if key in ("proposed", "attempted") else 1
            assert sum(metrics[f"rollout/pool/{p}/{key}"] for p in source.provider.providers) == expected
        assert all(recorder.metrics()[k] == v for k, v in metrics.items())
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
        pool = kept[0].metadata["sampling_pool"]
        assert recorder.metrics()[f"rollout/pool/{pool}/discarded"] == 1
        assert recorder.metrics()[f"rollout/pool/{pool}/discarded_unobserved"] == 0
    finally:
        recorder.close()
        source.close()


@pytest.mark.parametrize(
    "case", ["incomplete", "mixed", "aborted", "verifier_failure", "stale", "trained", "shutdown"]
)
def test_actual_buffer_reservations_reconcile_through_resume(args, tmp_path, monkeypatch, case):
    setup(args, tmp_path, "exposure")
    args.async_unused_samples_handler = "drop"
    source = DashboardDrainingRolloutDataSource(args)

    # Avoid engine construction; exercise the managed producer's actual callback wiring.
    def base_init(self, input):
        self.args, self.data_source = input.args, input.data_source

    monkeypatch.setattr(async_rollout.FullyAsyncRolloutFn, "__init__", base_init)
    monkeypatch.setattr(async_rollout.async_capacity, "report", lambda *a: {"warnings": []})
    producer = async_rollout.ManagedFullyAsyncRolloutFn(SimpleNamespace(args=args, data_source=source))
    buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, producer._handle_unused))
    recorder = buffer._delegate._records
    try:
        group = source.get_samples(1)[0]
        item = test_remind_provider.scored(group, [0, 1])
        if case == "incomplete":
            item = DataBufferInput(prompt_group=group, group=group[:1])
        elif case == "mixed":
            group[1].weight_versions = [WeightVersionsPerCall([WeightVersionSpan("2", 1, 2)])]
        elif case == "aborted":
            group[0].status = Sample.Status.ABORTED
        elif case == "verifier_failure":
            group[0].reward = None
            group[0].metadata["verifier_diagnostics"] = {"math": {"status": "error"}}

        async def exercise():
            kept = await buffer.put(item)
            if kept is False:
                source.acknowledge_groups([group])
            if case == "stale":
                assert await buffer.get_ready(current_version=99) is None
            if case == "trained":
                assert await buffer.get_ready(current_version=1) is not None
                source.acknowledge_groups([group])
                assert len(source.provider.pending) == 1  # Acknowledged is not discarded.
                source.stage_training_batch(0, group)
                source.commit_training_batch(0)

        asyncio.run(exercise())
        if case == "shutdown":
            # Joined unconsumed work stays explicitly outstanding and regenerates on resume.
            source.requeue_pending_groups([group[0].group_index])
        assert sum(source.provider.terminal.values()) + len(source.provider.pending) == source.provider.draws == 1
        assert len(source.provider.pending) == int(case == "shutdown")
        assert sum(source.provider.trained.values()) == int(case == "trained")
        source.save(0)
        saved = source.provider.state_dict()
        recorder.close()
        source.close()
        args.load, args.start_rollout_id = args.save, 1
        restored = DashboardDrainingRolloutDataSource(args)
        try:
            restored.load(0)
            assert restored.provider.state_dict() == saved
            if case == "shutdown":
                recovered = restored.get_samples(1)
                assert recovered[0][0].group_index == group[0].group_index
                restored.discard_groups(recovered)
                assert not restored.provider.pending
        finally:
            restored.close()
    finally:
        recorder.close()
        source.close()


def test_collection_file_preserves_origins_and_release_in_receipts(args, tmp_path):
    setup(args, tmp_path)
    path = Path(args.olmo_core.prompt_mixture)
    manifest = json.loads(path.read_text())
    first = manifest["sources"][0]
    data = path.parent / first["path"]
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    for i, row in enumerate(rows):
        row["metadata"].update(
            origin_id=f"original-{i % 2}", record_id=f"row-{i}", release_id="release-v1", labeling_version="labels-v2"
        )
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    data.write_bytes(raw)
    first["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(manifest))
    args.rollout_shuffle = False
    manifest["shuffle"] = False
    path.write_text(json.dumps(manifest))
    args.olmo_core = dataclasses.replace(args.olmo_core, prompt_replay_enabled=True)
    source = DashboardDrainingRolloutDataSource(args)
    buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, lambda g: source.discard_groups([g])))
    recorder = buffer._delegate._records
    try:
        templates = source.dataset.datasets[0].origin_samples
        assert source.provider.providers["math"].base.samples[0] is templates[0]
        for update in range(2):
            group = source.get_samples(1)[0]
            group[0].metadata["generated_only"] = True
            assert "generated_only" not in templates[update].metadata
            assert asyncio.run(buffer.put(test_remind_provider.scored(group, [0, 1])))
            source.acknowledge_groups([group])
            source.stage_training_batch(update, group)
            source.commit_training_batch(update)
        replay = source.provider.providers["math"]
        assert set(replay.windows) == {"original-0", "original-1"}
        receipts = list((tmp_path / "records").glob("pool-exposure-*.jsonl"))
        records = [json.loads(line) for line in receipts[0].read_text().splitlines()]
        assert records[0]["population_identity"] == hashlib.sha256(path.read_bytes()).hexdigest()
        for i, receipt in enumerate(records):
            identity = receipt["selections"][0]
            assert identity["origin_id"] == f"original-{i}"
            assert identity["mixture_source"] == first["name"]
            assert identity["record_id"] == f"row-{i}"
            assert identity["release_id"] == "release-v1"
            assert identity["labeling_version"] == "labels-v2"
    finally:
        recorder.close()
        source.close()
