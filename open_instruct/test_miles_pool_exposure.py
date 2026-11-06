"""Exposure feedback, finite pools, replay scope and preparation compatibility."""

import copy
import gzip
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct import test_miles_prompt_replay, test_miles_run_data, test_miles_run_spec
from open_instruct.miles.datasets import pools, run_data
from open_instruct.miles.rollout import pool_provider

environment = test_miles_run_data.environment


def provider(size=10000, replay=None):
    samples = [
        SimpleNamespace(
            prompt=f"{p} {i}",
            metadata={
                "sampling_pool": p,
                "pool_id": p,
                "mixture_source": p,
                "verifiers": [{"name": "math", "target": "42"}],
            },
        )
        for p in ["math", "if"]
        for i in range(size)
    ]
    return pool_provider.PoolPromptProvider(
        samples,
        pools={p: {"label": p} for p in ["math", "if", "empty"]},
        seed=17,
        shuffle=True,
        identity="fixture",
        replay=replay,
        sampling={"policy": "exposure"},
    )


def discard(p, sample):
    p.feedback([{"kind": "disposition", "disposition": "discarded", **sample.metadata}])


def test_unequal_delivery_converges_to_equal_trained_exposure():
    p = provider(30000)
    rng = random.Random(3)
    for update in range(150):
        accepted = []
        while len(accepted) < 32:
            sample = p.next_samples(1)[0]
            if rng.random() < {"math": 0.2, "if": 0.8}[sample.metadata["sampling_pool"]]:
                accepted.append(sample.metadata["pool_selection_id"])
            else:
                discard(p, sample)
        p.commit(update, accepted)
    assert abs(p.trained["math"] / sum(p.trained.values()) - 0.5) < 0.025
    assert p.proposed["math"] > p.proposed["if"] * 3
    assert p.trained["empty"] == 0


def test_only_completed_training_counts_and_duplicates_are_safe():
    p = provider(3)
    sample = p.next_samples(1)[0]
    record = test_miles_prompt_replay.record(sample, rewards=(0, 1), filter_decision="passed")
    p.feedback([record, {"kind": "disposition", "disposition": "acknowledged", **sample.metadata}])
    assert sum(p.trained.values()) == 0
    ids = [sample.metadata["pool_selection_id"]]
    p.commit(0, ids)
    p.commit(0, ids)
    assert sum(p.trained.values()) == 1
    with pytest.raises(ValueError):
        p.commit(1, ids)


def test_finite_exhaustion_no_implicit_epochs_and_target_skips_empty():
    p = provider(2)
    seen = set()
    for i in range(4):
        sample = p.next_samples(1)[0]
        assert sample.prompt not in seen
        seen.add(sample.prompt)
        p.commit(i, [sample.metadata["pool_selection_id"]])
    assert not p.available()
    with pytest.raises(StopIteration):
        p.next_samples(1)
    assert p.targets["empty"] == 0


def test_resume_pending_and_selection_rollback():
    p = provider(20)
    pending = p.next_samples(5)
    p.commit(0, [pending[0].metadata["pool_selection_id"]])
    restored = provider(20)
    restored.load_state_dict(p.state_dict())
    assert [s.prompt for s in p.next_samples(8)] == [s.prompt for s in restored.next_samples(8)]
    snapshot = p.selection_state_dict()
    expected = p.next_samples(7)
    p.load_selection_state_dict(snapshot)
    assert expected == p.next_samples(7)
    assert restored.trained == p.trained


def test_pool_replay_respects_pool_identity_and_original_verifier():
    p = provider(8, replay=dict(fraction=1, window=1, max_successes=0, truncation="any", length_drop=0.1))
    first = p.next_samples(1)[0]
    pool = first.metadata["sampling_pool"]
    p.feedback([test_miles_prompt_replay.record(first, caps=4, tokens=100)])
    # Shorter fresh observations unlock only this source/pool's replay.
    while True:
        sample = p.next_samples(1)[0]
        if sample.metadata["sampling_pool"] == pool:
            p.feedback([test_miles_prompt_replay.record(sample, rewards=(1, 1, 1, 1), tokens=50)])
            break
        discard(p, sample)
    for _ in range(20):
        sample = p.next_samples(1)[0]
        if sample.metadata.get("prompt_replay_review"):
            assert sample.prompt == first.prompt
            assert sample.metadata["sampling_pool"] == pool
            assert sample.metadata["verifiers"] == first.metadata["verifiers"]
            return
        discard(p, sample)
    pytest.fail("Replay never selected")


def test_pool_labels_are_unambiguous_and_canonical():
    config = {"math": {"label": "math"}}
    pools.validate(config)
    row = {"pool_id": "math", "metadata": {"verifiers": ["original"]}}
    pools.annotate(row, config)
    assert row["metadata"]["sampling_pool"] == "math"
    assert row["metadata"]["verifiers"] == ["original"]
    with pytest.raises(ValueError, match="unique"):
        pools.validate(config | {"other": {"label": "math"}})
    with pytest.raises(ValueError, match="top-level"):
        pools.annotate({"metadata": {"pool_id": "math"}}, config)


def test_pool_mode_validation_and_no_fraction_ambiguity(tmp_path):
    data = {
        "sources": [{"name": "one", "prompt_data": "/prompts.jsonl", "reward_config": "/rewards.json"}],
        "pools": {"math": {"label": "math"}, "empty": {"label": "empty"}},
    }
    run_data.validate_data(data)
    spec = test_miles_run_spec.spec(
        tmp_path,
        data=data,
        records={"enabled": True, "root": "/records"},
        core={"prepared_batch_overlap": False},
        **{"async": {"fully_async": True}},
    )
    # Use a single optimizer batch per collection; the default fixture may differ.
    spec.compile()
    bad = copy.deepcopy(data)
    bad["sources"][0]["fraction"] = 1
    with pytest.raises(ValueError, match="omit source fraction"):
        run_data.validate_data(bad)


def test_prepare_labeled_sources_preserves_payload_and_emits_pool_manifest(environment, tmp_path):
    row = {
        "input": "What is 6 times 7?",
        "label": "42",
        "pool_id": "mathematics",
        "metadata": {"prepared_sample_id": "row-1", "verifiers": [{"name": "gsm8k", "target": "42"}]},
    }
    data_path = tmp_path / "input.jsonl.gz"
    with gzip.open(data_path, "wt") as stream:
        stream.write(json.dumps(row) + "\n")
    rewards = tmp_path / "rewards.json"
    rewards.write_text(json.dumps({"gsm8k": {"factory": "open_instruct.miles.rewards.verifiers.GSM8KVerifier"}}))
    data = {
        "sources": [{"name": "original", "prompt_data": str(data_path), "reward_config": str(rewards)}],
        "pools": {"math": {"label": "mathematics"}, "empty": {"label": "empty"}},
    }
    result = test_miles_run_data.prepare(environment, data)
    manifest = json.loads(Path(result["prompt_mixture"]).read_text())
    assert manifest["pools"] == data["pools"]
    assert manifest["sampling"] == {"policy": "shuffle"}
    assert "fraction" not in manifest["sources"][0]
    actual = json.loads(Path(result["prompt_data"]).read_text())
    assert actual["pool_id"] == row["pool_id"]
    assert actual["metadata"]["sampling_pool"] == "math"
    assert actual["metadata"]["verifiers"] == row["metadata"]["verifiers"]
    assert actual["label"] == row["label"] and actual["input"] == row["input"]
    assert test_miles_run_data.prepare(environment, data) == result


def test_corrupt_restore_fails_without_moving_cursors():
    p = provider(20)
    p.next_samples(3)
    good = p.state_dict()
    broken = copy.deepcopy(good)
    broken["providers"]["if"]["offset"] = 999
    with pytest.raises(ValueError, match="cursor"):
        p.load_state_dict(broken)
    assert p.state_dict() == good
    broken = copy.deepcopy(good)
    broken["pending"].pop(0)
    with pytest.raises(ValueError, match="reconcile"):
        p.load_state_dict(broken)
    assert p.state_dict() == good
