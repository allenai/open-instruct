"""Use the pinned MILES loader and buffered adapter to exercise separate source cursors."""

import asyncio
import copy
import dataclasses
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import test_prompt_provider
import test_remind_provider
import torch
from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput
from miles.utils import inference_records

from open_instruct.miles.datasets import recording
from open_instruct.miles.rollout.data_source import DashboardDrainingRolloutDataSource
from open_instruct.miles.rollout.provider_source import ProviderRolloutDataSource

args = test_prompt_provider.args


def setup(args, fractions=(0.8, 0.2)):
    root = Path(args.prompt_data).parent
    sources = []
    for name, size, fraction in zip(("a", "b"), (11, 7), fractions):
        path = root / f"{name}.jsonl"
        raw = "".join(
            json.dumps(
                {
                    "input": f"{name}-{i}",
                    "label": str(i),
                    "metadata": {
                        "run_prompt_tokens": 2,
                        "verifiers": [{"name": "math", "target": str(i)}],
                        "nested": [i],
                    },
                }
            )
            + "\n"
            for i in range(size)
        ).encode()
        path.write_bytes(raw)
        sources.append(
            {
                "name": name,
                "fraction": fraction,
                "path": path.name,
                "rows": size,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    path = root / "mixture.json"
    path.write_text(json.dumps({"schema_version": 1, "seed": 17, "shuffle": args.rollout_shuffle, "sources": sources}))
    args.prompt_data = str(root / sources[0]["path"])
    args.olmo_core = SimpleNamespace(prompt_provider="mixture", prompt_mixture=str(path), records_root=None)


def selected(source, n):
    return [group[0].to_dict() for group in source.get_samples(n)]


def test_proportions_chunking_and_per_source_without_replacement(args):
    setup(args)
    args.n_samples_per_prompt = 1
    large, small = ProviderRolloutDataSource(args), ProviderRolloutDataSource(args)
    actual = selected(large, 10000)
    assert actual == sum([selected(small, n) for n in (3, 0, 70, 5927, 4000)], [])
    counts = Counter(row["metadata"]["mixture_source"] for row in actual)
    assert abs(counts["a"] / len(actual) - 0.8) < 0.02
    epochs = defaultdict(list)
    for row in actual:
        epochs[row["metadata"]["mixture_source"], row["metadata"]["mixture_source_epoch"]].append(row["prompt"])
    for (name, epoch), prompts in epochs.items():
        assert len(prompts) == len(set(prompts))
        if epoch < max(e for n, e in epochs if n == name):
            assert len(prompts) == {"a": 11, "b": 7}[name]
    assert len(large.dataset) == 18


@pytest.mark.parametrize("shuffle", [True, False])
def test_single_source_is_identical_to_original_provider(args, shuffle):
    args.rollout_shuffle = shuffle
    setup(args, (1.0,))
    mixture = ProviderRolloutDataSource(args)
    args.olmo_core = SimpleNamespace(prompt_provider="shuffle")
    original = ProviderRolloutDataSource(args)
    for n in (1, 5, 7, 20):
        actual = selected(mixture, n)
        for row in actual:
            row["metadata"].pop("mixture_source")
            row["metadata"].pop("mixture_source_epoch")
        assert actual == selected(original, n)


def test_checkpoint_restores_cursors_rng_and_pending_groups(args):
    setup(args)
    source = DashboardDrainingRolloutDataSource(args)
    pending = source.get_samples(9)
    source.save(3)
    expected = test_prompt_provider.payload(source.get_samples(21))
    args.load = args.save
    resumed = DashboardDrainingRolloutDataSource(args)
    resumed.load(3)
    assert test_prompt_provider.payload(resumed.get_samples(9)) == test_prompt_provider.payload(pending)
    assert test_prompt_provider.payload(resumed.get_samples(21)) == expected
    # The real checkpoint uses weights_only=True, including Python RNG state.
    state = torch.load(Path(args.save) / "rollout/global_dataset_state_dict_3.pt", weights_only=True)
    assert state["prompt_provider"]["kind"] == "mixture"


def test_failed_fresh_draw_restores_buffer_all_cursors_and_rng(args, monkeypatch):
    setup(args)
    source = ProviderRolloutDataSource(args)
    retry = source.get_samples(2)
    source.add_samples(retry)
    before = copy.deepcopy(source.provider.state_dict())
    original = source.provider.next_samples

    def broken(n):
        original(n)
        raise RuntimeError("failure after advancement")

    monkeypatch.setattr(source.provider, "next_samples", broken)
    with pytest.raises(RuntimeError, match="advancement"):
        source.get_samples(10)
    assert source.provider.state_dict() == before
    assert test_prompt_provider.payload(source.buffer) == test_prompt_provider.payload(retry)


def test_mismatch_and_bad_cursor_fail_without_partial_restore(args):
    setup(args)
    source = ProviderRolloutDataSource(args)
    selected(source, 10)
    before = source.provider.state_dict()
    broken = copy.deepcopy(before)
    broken["cursors"][0]["epoch_id"] = 99
    broken["cursors"][1]["sample_offset"] = 100
    with pytest.raises(ValueError, match="cursor"):
        source.provider.load_state_dict(broken)
    assert source.provider.state_dict() == before
    for key, value in (("manifest_sha256", "changed"), ("seed", 2), ("sources", [])):
        with pytest.raises(RuntimeError, match=key):
            source.provider.load_state_dict(before | {key: value})


def test_changed_source_and_length_overflow_are_not_silent_drops(args):
    setup(args)
    source_file = Path(args.prompt_data).parent / "b.jsonl"
    source_file.write_text(source_file.read_text() + "\n")
    with pytest.raises(ValueError, match="hash changed"):
        ProviderRolloutDataSource(args)
    setup(args)
    # First source still loads normally; fail if a later source exceeds the runtime limit.
    args.rollout_max_prompt_len = 1
    # The fixture supplies no tokenizer; bypass upstream first-source filtering only.
    with (
        mock.patch("miles.utils.data.filter_long_prompt", side_effect=lambda rows, *a: rows),
        pytest.raises(ValueError, match="prompt length"),
    ):
        ProviderRolloutDataSource(args)


def test_feedback_records_source_attribution_from_pending_ledger(args):
    setup(args)
    source = DashboardDrainingRolloutDataSource(args)
    groups = source.get_samples(5)
    records = [{"kind": "group", "group_index": group[0].group_index} for group in groups]
    source.feedback(records)
    assert [row["mixture_source"] for row in records] == [group[0].metadata["mixture_source"] for group in groups]


def test_mixture_runtime_rejects_global_exclusions_before_loading_files(args):
    args.olmo_core = SimpleNamespace(
        prompt_provider="mixture", prompt_mixture="/missing-mixture.json", selection_table="/missing-exclusions.json"
    )
    with pytest.raises(ValueError, match="Mixtures cannot yet be combined with a frozen prompt-exclusion table"):
        DashboardDrainingRolloutDataSource(args)


def test_real_async_recorder_persists_source_on_kept_and_filtered_groups(args, tmp_path):
    test_remind_provider.setup_remind(args, tmp_path)
    core = args.olmo_core
    setup(args, (0.5, 0.5))
    args.olmo_core = dataclasses.replace(core, prompt_provider="mixture", prompt_mixture=args.olmo_core.prompt_mixture)
    source = DashboardDrainingRolloutDataSource(args)
    buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, source.add_samples))
    recorder = buffer._delegate._records
    try:
        groups = source.get_samples(6)
        for index, group in enumerate(groups):
            mixed = index % 2 == 0
            assert asyncio.run(buffer.put(test_remind_provider.scored(group, [1, 0] if mixed else [1, 1]))) is mixed
        recorder.close()
        rows = [
            json.loads(line)
            for path in (tmp_path / "records").rglob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        records = [row for row in rows if row.get("kind") == "group"]
        assert len(records) == 6
        assert {row["filter_decision"] for row in records} == {"passed", "filtered"}
        assert {row["mixture_source"] for row in records} == {"a", "b"}
        assert [row["mixture_source"] for row in records] == [group[0].metadata["mixture_source"] for group in groups]
    finally:
        recorder.close()
        source.close()


@pytest.fixture(params=["mixture", "remind"])
def recorded_source(args, tmp_path, request):
    test_remind_provider.setup_remind(args, tmp_path)
    if request.param == "mixture":
        core = args.olmo_core
        setup(args)
        args.olmo_core = dataclasses.replace(
            core, prompt_provider="mixture", prompt_mixture=args.olmo_core.prompt_mixture
        )
    source = DashboardDrainingRolloutDataSource(args)
    try:
        yield args, source
    finally:
        source.close()


def test_recording_unavailable_at_start_is_fatal_only_when_feedback_is_required(recorded_source, monkeypatch):
    args, source = recorded_source

    def unavailable(_):
        raise OSError("injected unavailable lineage")

    monkeypatch.setattr(recording, "lineage_identity", unavailable)
    if source.provider.requires_feedback:
        with pytest.raises(RuntimeError, match="requires an available"):
            recording.create_recorder(args)
    else:
        recorder = recording.create_recorder(args)
        try:
            assert recorder.unavailable
            recorder.record_group(source.get_samples(1)[0], decision="passed")
            assert recorder.counts["dropped"] == 1
            assert len(source.get_samples(1)) == 1
        finally:
            recorder.close()


@pytest.mark.parametrize("failure", ["unavailable", "build", "attribution"])
def test_recording_failures_are_nonfatal_for_fixed_mixtures_only(recorded_source, monkeypatch, failure):
    args, source = recorded_source
    recorder = recording.create_recorder(args)
    try:
        group = source.get_samples(1)[0]
        test_remind_provider.scored(group, [1, 0])
        state = copy.deepcopy(source.provider.state_dict())

        def broken(*a, **kw):
            raise RuntimeError("injected recording failure")

        with monkeypatch.context() as patch:
            if failure == "unavailable":
                patch.setattr(recorder, "unavailable", True)
            elif failure == "build":
                patch.setattr(inference_records, "response_record", broken)
            else:
                patch.setattr(source, "feedback", broken)
            if source.provider.requires_feedback:
                with pytest.raises(RuntimeError):
                    recorder.record_group(group, decision="passed")
            else:
                recorder.record_group(group, decision="passed")
                assert recorder.counts["dropped" if failure == "unavailable" else "failed"] == 1
        assert source.provider.state_dict() == state
        # Once recording recovers, attribution/feedback still work and draws continue.
        recorder.record_group(group, decision="passed")
        assert recorder.counts["queued"] == 1
        assert len(source.get_samples(1)) == 1
    finally:
        recorder.close()


def test_mixture_continues_when_record_store_is_unwritable(args, tmp_path):
    test_remind_provider.setup_remind(args, tmp_path)
    core = args.olmo_core
    setup(args)
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("Cannot create record directories below this file")
    args.olmo_core = dataclasses.replace(
        core, prompt_provider="mixture", prompt_mixture=args.olmo_core.prompt_mixture, records_root=str(blocked)
    )
    source = DashboardDrainingRolloutDataSource(args)
    recorder = recording.create_recorder(args)
    try:
        group = source.get_samples(1)[0]
        test_remind_provider.scored(group, [1, 0])
        recorder.record_group(group, decision="passed")
        recorder.close()  # Join the real writer so its filesystem failure is deterministic.
        assert recorder.unavailable
        assert recorder.counts["failed"] == 1
        dropped = recorder.counts["dropped"]
        recorder.record_group(source.get_samples(1)[0], decision="passed")
        assert recorder.counts["dropped"] == dropped + 1
    finally:
        recorder.close()
        source.close()
