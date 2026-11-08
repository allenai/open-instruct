"""CPU tests for replay evidence, scheduling and resume without the GPU runtime."""

import copy
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration import config, run_spec
from open_instruct.miles.datasets import replay_evidence, replay_summary
from open_instruct.miles.rollout import replay_provider


class Base:
    requires_feedback = False

    def __init__(self, samples):
        self.samples, self.offset = samples, 0

    def next_samples(self, n):
        result = [self.samples[(self.offset + i) % len(self.samples)] for i in range(n)]
        self.offset += n
        return result

    def state_dict(self):
        return {"offset": self.offset}

    def load_state_dict(self, state):
        self.offset = state["offset"]


def provider(**kwargs):
    samples = [
        SimpleNamespace(
            prompt=f"question {i}",
            metadata={"mixture_source": "math", "verifiers": [{"name": "math", "target": "42"}]},
        )
        for i in range(8)
    ]
    return replay_provider.ReplayPromptProvider(
        Base(samples),
        samples,
        fraction=kwargs.pop("fraction", 1),
        window=kwargs.pop("window", 2),
        max_successes=kwargs.pop("max_successes", 0),
        truncation=kwargs.pop("truncation", "none"),
        **kwargs,
    )


def record(sample=None, rewards=(0, 0, 0, 0), caps=0, tokens=100, valid=True, **kw):
    return {
        "kind": "group",
        "filter_decision": "filtered",
        "group_size": len(rewards),
        "expected_group_size": len(rewards),
        "reward_zero_truncated": True,
        "observation_id": "observation",
        "recorded_unix": 10,
        "task_key": "task",
        **(sample.metadata if sample else {}),
        "responses": [
            {
                "reward": reward,
                "truncated": i < caps,
                "status": "truncated" if i < caps else "completed",
                "validity": {"valid": valid},
                "response_tokens": tokens,
                "policy_versions": ["10"],
            }
            for i, reward in enumerate(rewards)
        ],
        **kw,
    }


@pytest.mark.parametrize(
    "caps,category", [(0, "zero_completed"), (1, "zero_some_truncated"), (4, "zero_all_truncated")]
)
def test_zero_split_and_group_sizes(caps, category):
    out = replay_evidence.classify(record(caps=caps))
    assert out["category"] == category and out["truncated"] == caps and out["group_size"] == 4
    assert replay_evidence.classify(record(rewards=(0,) * 8, caps=8))["category"] == "zero_all_truncated"


def test_verifier_failures_and_unknowns_are_not_difficulty():
    assert replay_evidence.classify(record(valid=False))["category"] == "invalid"
    assert replay_evidence.classify(record(valid=None))["category"] == "invalid"
    assert replay_evidence.classify(record(valid=None, caps=4))["category"] == "zero_all_truncated"
    assert replay_evidence.classify(record(valid=None, caps=3))["category"] == "invalid"
    assert replay_evidence.classify(record(expected_group_size=8))["category"] == "invalid"
    assert replay_evidence.classify(record(rewards=(float("nan"), 0)))["category"] == "invalid"
    assert replay_evidence.classify(record(filter_decision="aborted"))["category"] == "aborted"
    assert (
        replay_evidence.classify(record(rewards=({"score": 0},) * 4, reward_key="score"))["category"]
        == "zero_completed"
    )


@pytest.mark.parametrize(
    "text,target,expected",
    [
        ("Reasoning: 42. Still thinking", "42", True),
        ("142 or 42.5", "42", False),
        ("not 42", "42", True),
        ("x = \\boxed{42}", "42", True),
        ("answer", {"tests": []}, None),
        (None, 42, None),
    ],
)
def test_literal_probe_is_bounded_and_not_a_correctness_claim(text, target, expected):
    assert replay_evidence.literal_answer_probe(text, target) is expected


def test_waits_for_drop_and_full_fresh_window_then_replays_original_prompt():
    p = provider(length_drop=0.2)
    first = p.next_samples(1)[0]
    p.feedback([record(first, caps=4)])
    for _ in range(2):
        sample = p.next_samples(1)[0]
        assert not sample.metadata[replay_evidence.REPLAY]
        p.feedback([record(sample, rewards=(0, 1, 0, 1), tokens=90)])
    assert not p._eligible()  # 10%, not the requested20% drop.
    sample = p.next_samples(1)[0]
    p.feedback([record(sample, rewards=(0, 1, 0, 1), tokens=70)])
    replay = p.next_samples(1)[0]
    assert replay.prompt == first.prompt and replay.metadata[replay_evidence.REPLAY]
    before = copy.deepcopy(p.windows)
    p.feedback([record(replay, rewards=(0, 1, 0, 0), tokens=120)])
    assert p.windows == before  # Replay cannot inflate its own length baseline.
    assert p.counts["replay_mixed"] == 1
    assert not p.history[replay_provider.identity(first)[0]]["candidate"]


def test_source_local_window_does_not_respond_to_shorter_other_source():
    p = provider(length_drop=0.2)
    extra = SimpleNamespace(prompt="chat", metadata={"mixture_source": "chat"})
    p = replay_provider.ReplayPromptProvider(
        Base([*p.samples.values(), extra]),
        [*p.samples.values(), extra],
        fraction=1,
        length_drop=0.2,
        window=1,
        max_successes=0,
        truncation="none",
    )
    first = p.next_samples(1)[0]
    p.feedback([record(first, caps=4)])
    for sample in p.next_samples(8):
        p.feedback([record(sample, rewards=(0, 1, 0, 1), tokens=1 if sample.prompt == "chat" else 100)])
    assert not p._eligible()


@pytest.mark.parametrize("field", ["weight", "prepared_sample_id", "record_id"])
def test_replay_preserves_reward_and_row_identity_for_otherwise_identical_rows_on_resume(field):
    samples = [
        SimpleNamespace(
            prompt=prompt,
            metadata={"origin_id": "math", "verifiers": [{"name": "math", "target": "42", "weight": weight}]},
        )
        for prompt, weight in [("same", 1), ("same", 1), ("shorter", 1)]
    ]
    if field == "weight":
        samples[1].metadata["verifiers"][0][field] = 0.5
    else:
        samples[0].metadata[field], samples[1].metadata[field] = "first", "second"

    def make(rows):
        return replay_provider.ReplayPromptProvider(
            Base(rows), rows, fraction=1, window=1, length_drop=0.2, max_successes=0, truncation="none"
        )

    p = make(samples)
    assert len(p.samples) == 3
    # Native task identity stays comparable to inference records, while replay
    # identity additionally binds the full verifier configuration.
    assert replay_provider.identity(samples[0])[2] == replay_provider.identity(samples[1])[2]
    p.feedback([record(p.next_samples(1)[0], rewards=(0, 1))])
    failed = p.next_samples(1)[0]
    p.feedback([record(failed)])
    p.feedback([record(p.next_samples(1)[0], rewards=(0, 1), tokens=70)])
    state = p.state_dict()
    restored = make(samples)
    restored.load_state_dict(state)
    for current in (p, restored):
        replay = current.next_samples(1)[0]
        assert replay.metadata[replay_evidence.REPLAY]
        assert replay.metadata[replay_evidence.KEY] == failed.metadata[replay_evidence.KEY]
        assert replay.metadata["verifiers"] == failed.metadata["verifiers"]
        if field == "weight":
            assert replay.metadata["verifiers"][0]["weight"] == 0.5
        else:
            assert replay.metadata[field] == "second"
    changed = copy.deepcopy(samples)
    changed[1].metadata["verifiers"][0]["weight"] = 0.75
    with pytest.raises(RuntimeError, match="checkpoint mismatch for catalog"):
        make(changed).load_state_dict(state)


def test_duplicate_abort_acknowledgement_and_failed_replay_cooldown():
    p = provider(length_drop=0, window=1)
    first = p.next_samples(1)[0]
    p.feedback([record(first, filter_decision="aborted")])
    assert len(p.pending) == 1
    p.feedback([record(first), record(first)])
    assert p.counts["zero_completed"] == 1
    p.feedback([record(p.next_samples(1)[0], rewards=(0, 1, 0, 1))])
    retry = p.next_samples(1)[0]
    assert retry.metadata[replay_evidence.REPLAY]
    p.feedback(
        [
            {
                "kind": "disposition",
                "disposition": "acknowledged",
                replay_evidence.SELECTION_ID: retry.metadata[replay_evidence.SELECTION_ID],
            }
        ]
    )
    assert p.next_samples(1)[0].metadata[replay_evidence.REPLAY]
    retry_id = max(p.pending)
    retry.metadata[replay_evidence.SELECTION_ID] = retry_id
    p.feedback([record(retry)])
    assert not p.next_samples(1)[0].metadata[replay_evidence.REPLAY]


@pytest.mark.parametrize(
    "truncation,caps,expected",
    [("none", 0, True), ("any", 0, False), ("any", 1, True), ("all", 1, False), ("all", 4, True)],
)
def test_pool_modes(truncation, caps, expected):
    p = provider(truncation=truncation)
    p.feedback([record(p.next_samples(1)[0], caps=caps)])
    assert next(iter(p.history.values()))["candidate"] is expected


def test_checkpoint_and_cheap_selection_rollback_preserve_history():
    p = provider(fraction=0.5, length_drop=0, window=1)
    for sample in p.next_samples(4):
        p.feedback([record(sample)])
    p.next_samples(3)
    checkpoint = copy.deepcopy(p.state_dict())
    restored = provider(fraction=0.5, length_drop=0, window=1)
    restored.load_state_dict(checkpoint)
    assert [s.__dict__ for s in p.next_samples(7)] == [restored.next_samples(1)[0].__dict__ for _ in range(7)]
    assert p.state_dict() == restored.state_dict()
    before = copy.deepcopy(p.state_dict())
    cursor = copy.deepcopy(p.selection_state_dict())
    assert "history" not in cursor
    p.next_samples(11)
    p.load_selection_state_dict(cursor)
    assert p.state_dict() == before
    with pytest.raises(RuntimeError, match="config"):
        provider().load_state_dict(checkpoint)


def test_fraction_zero_is_tracking_only_and_source_order_unchanged():
    p = provider(fraction=0)
    observed = []
    for _ in range(30):
        sample = p.next_samples(1)[0]
        observed.append(sample.prompt)
        p.feedback([record(sample)])
    assert observed == [f"question {i % 8}" for i in range(30)]
    assert p.counts["zero_completed"] == 30 and p.counts["replayed"] == 0


def test_export_keeps_unique_prompts_and_later_success_and_bounds(tmp_path):
    manifest = tmp_path / "manifest-a.json"
    manifest.write_text(json.dumps({"run": {"attempt": "a", "name": "run"}, "protocol_sha256": "protocol"}))
    records = [record(), record(rewards=(0, 1, 0, 1), observation_id="later", recorded_unix=11)]
    records.append(record(observation_id="outside", responses=[{"policy_versions": ["401"]}]))
    (tmp_path / "records-a.jsonl").write_text("\n".join(json.dumps(r) for r in [records[0], *records]))
    out = replay_summary.summarize([manifest], tmp_path / "out")
    assert out["unique_ever_zero"] == 1 and out["ever_zero_latest_nonzero"] == 1
    assert out["counts"]["duplicate_observations"] == 1 and out["counts"]["outside_policy_range"] == 1
    assert len((tmp_path / "out/zero-prompts.jsonl").read_text().splitlines()) == 1


def test_configuration_requires_records_and_rejects_incompatible_strategies(tmp_path):
    with pytest.raises(ValueError, match="requires"):
        config.RunConfig(config.CoreConfig(prompt_replay_enabled=True), {}).validate()
    with pytest.raises(ValueError, match="below 1"):
        config.CoreConfig(prompt_replay_length_drop=1)
    spec = run_spec.RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "prompt-replay",
            "model": {"source": "model", "format": "hf"},
            "output": {"root": str(tmp_path / "run")},
            "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 8}]},
            "core": {"prompt_replay_enabled": True, "max_policy_lag": 2},
            "miles": {"fully_async": True},
            "records": {"enabled": True, "root": str(tmp_path / "records")},
        },
        config_path=tmp_path / "run.toml",
    )
    compiled = spec.compile()
    assert compiled.core.prompt_replay_enabled and compiled.core.prompt_provider == "shuffle"


def test_legacy_if_scored_status_is_reconciled_without_accepting_service_failures():
    row = record(valid=False)
    for response in row["responses"]:
        response["validity"]["components"] = {"ifeval": "scored"}
    assert replay_evidence.classify(row)["category"] == "zero_completed"
    assert replay_evidence.classify(row)["legacy_if_validity_reconciled"] == 4
    row["responses"][0]["validity"]["components"]["code"] = "service_error"
    assert replay_evidence.classify(row)["category"] == "invalid"


def test_checkpoint_rejects_inconsistent_pending_reviews():
    p = provider(window=1, length_drop=0)
    p.feedback([record(p.next_samples(1)[0])])
    p.feedback([record(p.next_samples(1)[0], rewards=(1, 1, 1, 1))])
    p.next_samples(1)
    state = p.state_dict()
    assert state["inflight"]
    state["inflight"] = []
    with pytest.raises(RuntimeError, match="inconsistent in-flight"):
        provider(window=1, length_drop=0).load_state_dict(state)


@pytest.mark.parametrize("k", [4, 8])
def test_one_success_with_truncations_enters_pool_but_partial_credit_does_not(k):
    p = provider(max_successes=1, truncation="half", window=1)
    sample = p.next_samples(1)[0]
    row = record(sample, rewards=(0,) * (k - 1) + (1,), caps=k - 1)
    p.feedback([row])
    assert next(iter(p.history.values()))["candidate"]
    assert row["prompt_replay"]["binary_successes"] == 1
    assert replay_evidence.eligible(replay_evidence.classify(row), truncation="half")
    row["responses"][-1]["reward"] = 0.5
    assert not replay_evidence.eligible(replay_evidence.classify(row), truncation="half")
    row["responses"][-1]["reward"] = 1
    row["responses"][0].update(truncated=False, status="completed", reward=1)
    assert not replay_evidence.eligible(replay_evidence.classify(row), truncation="any")


def test_truncation_thresholds_distinguish_any_and_half():
    outcome = replay_evidence.classify(record(rewards=(0, 0, 0, 1), caps=1))
    assert replay_evidence.eligible(outcome, truncation="any")
    assert not replay_evidence.eligible(outcome, truncation="half")


def test_historical_export_preserves_one_success_and_both_truncation_thresholds(tmp_path):
    manifest = tmp_path / "manifest-a.json"
    manifest.write_text(json.dumps({"run": {"attempt": "a"}}))
    rows = [
        record(rewards=(0, 0, 0, 1), caps=1, task_key="one-cap", observation_id="a"),
        record(rewards=(0, 0, 0, 1), caps=3, task_key="three-cap", observation_id="b"),
        record(rewards=(0, 0, 0, 0), caps=0, task_key="no-cap", observation_id="c"),
    ]
    (tmp_path / "records-a.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    any_cap = replay_summary.summarize([manifest], tmp_path / "any")
    half = replay_summary.summarize([manifest], tmp_path / "half", truncation="half")
    assert any_cap["unique_ever_eligible"] == 2
    assert half["unique_ever_eligible"] == 1
    assert any_cap["counts"]["successes_1/4_truncated_3/4"] == 1
    assert len((tmp_path / "any/low-success-prompts.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("rewards", [(0, 0.5), (0, 2), (0.5, 0.5)])
def test_nonbinary_observations_are_valid_but_ineligible(rewards):
    p = provider()
    sample = p.next_samples(1)[0]
    evidence = record(sample, rewards=rewards)
    p.feedback([evidence])
    outcome = evidence["prompt_replay"]
    assert outcome["category"] in {"mixed", "constant_nonzero"}
    assert outcome["binary_successes"] is None
    assert not outcome["candidate"]
    assert outcome["positive_responses"] == sum(r > 0 for r in rewards)


@pytest.mark.parametrize("timestamp", [None, "unknown"])
def test_historical_nondefault_threshold_and_missing_timestamp(tmp_path, timestamp):
    manifest = tmp_path / "manifest-a.json"
    manifest.write_text(json.dumps({"run": {"attempt": "a"}}))
    rows = [record(rewards=(0, 1, 1, 0), caps=1, recorded_unix=timestamp)]
    (tmp_path / "records-a.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    result = replay_summary.summarize([manifest], tmp_path / "out", max_successes=2)
    assert result["unique_ever_eligible"] == 1
    assert result["counts"]["successes_2/4_truncated_1/4"] == 1
