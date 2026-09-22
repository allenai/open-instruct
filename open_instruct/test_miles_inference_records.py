"""CPU checks of inference-record identity, validity, storage, isolation and configuration."""

import asyncio
import enum
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles import inference_records, rewards
from open_instruct.miles.config import CoreConfig
from open_instruct.miles.errors import InputError
from open_instruct.miles.run_spec import RunSpec


class Status(enum.Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"


def sample(
    index, reward, *, prompt="<system>Be brief.</system><user>What is 6 * 7?</user>", versions=("0",), **kwargs
):
    return SimpleNamespace(
        group_index=11,
        index=index,
        rollout_id=3,
        prompt=prompt,
        response=f"answer {index}",
        response_length=100 + index,
        reward=reward,
        status=kwargs.get("status", Status.COMPLETED),
        weight_versions=list(versions),
        metadata={
            "query": "What is 6 * 7?",
            "verifiers": [{"name": "math", "target": kwargs.get("target", "42"), "weight": 1.0}],
            "prepared_sample_id": "math:train:7",
            "source_dataset": "gsm8k",
            "source_row": 7,
            "run_prompt_token_ids_sha256": "tokens",
            "reward_components": [{"name": "math", "score": reward, "weight": 1.0, "cost": 0.0}],
            "verifier_diagnostics": {"math": {"status": kwargs.get("verifier", "ok")}},
        },
    )


def checkpoint(tmp_path, name="policy"):
    path = tmp_path / name
    path.mkdir(exist_ok=True)
    source = {"path": "/weka/source", "files": [{"path": "config.json", "size": 3, "mtime_ns": 1, "sha256": "x"}]}
    (path / "workflow-model.json").write_text(json.dumps({"identity": {"source": source}, "prepared_files": []}))
    return path


def make_args(tmp_path, *, responses="off", rate=None, run="run-a", policy=None, start=0, temperature=1.0):
    return SimpleNamespace(
        olmo_core=CoreConfig(
            records_root=str(tmp_path / "records"), records_responses=responses, records_response_sample_rate=rate
        ),
        hf_checkpoint=str(policy or checkpoint(tmp_path)),
        wandb_run_name=run,
        rollout_temperature=temperature,
        n_samples_per_prompt=4,
        start_rollout_id=start,
        rollout_seed=17,
    )


def drain(records):
    records.close()
    return records


def rows(directory, kind="group"):
    return [
        row
        for path in sorted(directory.glob("records-*.jsonl"))
        for row in map(json.loads, path.read_text().splitlines())
        if row["kind"] == kind
    ]


def test_task_key_includes_full_context_and_targets():
    first = inference_records.task_identity(sample(0, 1.0).metadata, "<system>Be brief.</system><user>Q</user>")
    other_system = inference_records.task_identity(
        sample(0, 1.0).metadata, "<system>Show work.</system><user>Q</user>"
    )
    other_target = inference_records.task_identity(
        sample(0, 1.0, target="41").metadata, "<system>Be brief.</system><user>Q</user>"
    )
    assert first["task_key"] != other_system["task_key"]
    assert first["task_key"] != other_target["task_key"]
    # The final-message hash is only a grouping hint and deliberately matches.
    assert first["query_sha256"] == other_system["query_sha256"]


def test_input_key_separates_tasks_that_share_prompt_tokens(tmp_path):
    records = inference_records.Recorder(make_args(tmp_path))
    records.record_group([sample(0, 1.0, target="42")], decision="passed")
    records.record_group([sample(0, 1.0, target="41")], decision="passed")
    drain(records)
    first, second = rows(next((tmp_path / "records").glob("*/*")))
    assert first["prompt_token_sha256"] == second["prompt_token_sha256"]
    assert first["input_key"] != second["input_key"]


@pytest.mark.parametrize(
    ("diagnostics", "verifiers", "components", "valid"),
    [
        (
            {"math": {"status": "ok"}, "code": {"status": "completed"}},
            ["math", "code"],
            {"math": "ok", "code": "completed"},
            True,
        ),
        ({"math": {"status": "ok"}}, ["math", "code"], {"math": "ok", "code": "unknown"}, None),
        ({}, ["math"], {"math": "unknown"}, None),
        ({"code": {"status": "service_error"}}, ["code"], {"code": "service_error"}, False),
        ({}, [], {}, None),
    ],
)
def test_every_expected_verifier_has_a_validity_state(diagnostics, verifiers, components, valid):
    metadata = {"verifier_diagnostics": diagnostics, "verifiers": [{"name": name} for name in verifiers]}
    assert inference_records.validity(metadata) == {"components": components, "valid": valid}


def test_plain_verifiers_report_adapter_completion(tmp_path):
    path = tmp_path / "rewards.json"
    path.write_text(json.dumps({"math_answer": {"factory": "open_instruct.ground_truth_utils.GSM8KVerifier"}}))
    scored = SimpleNamespace(
        tokens=[1, 2, 3],
        response_length=1,
        response="The answer is 42",
        prompt="6 times 7?",
        metadata={"verifiers": [{"name": "math_answer", "target": "42"}]},
    )
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config=str(path)))
    assert asyncio.run(rewards.registered_reward(args, [scored])) == [1.0]
    assert scored.metadata["verifier_diagnostics"]["math_answer"] == {"kind": "adapter", "status": "completed"}
    assert inference_records.validity(scored.metadata)["valid"] is True


@pytest.mark.parametrize(
    ("versions", "fresh", "scope"),
    [
        (["0"], True, "start_checkpoint"),
        (["0"], False, "run_version"),
        (["9"], True, "run_version"),
        (["9", "10"], True, "mixed"),
        ([], True, "unknown"),
    ],
)
def test_policy_scope_separates_exact_start_samples(versions, fresh, scope):
    assert inference_records.policy_scope(versions, fresh) == scope


def test_response_sampling_is_deterministic_by_group():
    identities = [f"observation-{index}" for index in range(20000)]
    chosen = [inference_records.include_responses("sample", 0.1, identity) for identity in identities]
    assert chosen == [inference_records.include_responses("sample", 0.1, identity) for identity in identities]
    assert 0.09 < sum(chosen) / len(chosen) < 0.11
    assert not inference_records.include_responses("off", None, "observation-0")
    assert inference_records.include_responses("all", None, "observation-0")


def test_recorder_writes_manifest_groups_and_dispositions(tmp_path, monkeypatch):
    monkeypatch.setenv("BEAKER_WORKLOAD_ID", "workload-1")
    records = inference_records.Recorder(make_args(tmp_path))
    filtered = [sample(i, 0.0) for i in range(4)]
    passed = [sample(i, float(i % 2), status=Status.TRUNCATED, versions=("0", "1")) for i in range(4)]
    records.record_group(filtered, decision="filtered", reason="zero_std_0.0")
    records.record_group(passed, decision="passed")
    records.record_disposition(passed, accepted=True, staleness=1)
    drain(records)
    directory = next((tmp_path / "records").glob("*/run-a-workload-1"))
    manifest = json.loads(next(directory.glob("manifest-*.json")).read_text())
    assert manifest["source"] == "train"
    assert manifest["lineage"]["basis"] == "source_inventory" and manifest["lineage"]["weights_hashed"] is False
    assert manifest["run"]["fresh"] is True and manifest["run"]["id"] == "workload-1"
    assert manifest["protocol"]["rollout_temperature"] == 1.0
    groups = rows(directory)
    assert [(g["filter_decision"], g["filter_reason"]) for g in groups] == [
        ("filtered", "zero_std_0.0"),
        ("passed", None),
    ]
    assert groups[0]["task_key"] == groups[1]["task_key"] and groups[0]["input_key"] == groups[1]["input_key"]
    assert groups[0]["observation_id"] != groups[1]["observation_id"]
    assert all(s.metadata["inference_record_id"] == groups[1]["observation_id"] for s in passed)
    response = groups[1]["responses"][1]
    assert (response["reward"], response["truncated"], response["policy_scope"]) == (1.0, True, "mixed")
    assert response["validity"] == {"components": {"math": "ok"}, "valid": True}
    assert groups[0]["responses"][0]["policy_scope"] == "start_checkpoint"
    assert "response" not in response
    [disposition] = rows(directory, "disposition")
    assert disposition == {
        **disposition,
        "observation_id": groups[1]["observation_id"],
        "disposition": "consumed",
        "staleness": 1,
    }
    assert records.metrics() == {
        "rollout/records/queued_total": 3,
        "rollout/records/written_total": 3,
        "rollout/records/dropped_total": 0,
        "rollout/records/failed_total": 0,
        "rollout/records/pending": 0,
    }


def test_same_lineage_different_protocol_shares_task_but_not_input_key(tmp_path):
    policy = checkpoint(tmp_path)
    first = inference_records.Recorder(make_args(tmp_path, run="first", policy=policy))
    second = inference_records.Recorder(
        make_args(tmp_path, run="second", policy=policy, responses="all", temperature=0.6)
    )
    first.record_group([sample(0, 1.0)], decision="passed")
    second.record_group([sample(0, 1.0)], decision="passed")
    drain(first), drain(second)
    [lineage] = list((tmp_path / "records").iterdir())
    [a] = rows(next(lineage.glob("first-*")))
    [b] = rows(next(lineage.glob("second-*")))
    assert a["task_key"] == b["task_key"] and a["input_key"] != b["input_key"]
    assert b["responses"][0]["response"] == "answer 0"


def test_blocked_store_never_blocks_the_caller(tmp_path, monkeypatch):
    release = threading.Event()
    original = Path.mkdir

    def slow_mkdir(self, *args, **kwargs):
        release.wait(10)
        return original(self, *args, **kwargs)

    args = make_args(tmp_path)
    monkeypatch.setattr(inference_records, "QUEUE_LIMIT", 2)
    monkeypatch.setattr(Path, "mkdir", slow_mkdir)
    records = inference_records.Recorder(args)
    started = time.monotonic()
    for _ in range(10):
        records.record_group([sample(0, 1.0)], decision="passed")
    assert time.monotonic() - started < 1.0
    counts = records.metrics()
    assert counts["rollout/records/dropped_total"] >= 7
    release.set()
    drain(records)
    counts = records.metrics()
    assert counts["rollout/records/written_total"] + counts["rollout/records/dropped_total"] == 10


def test_unavailable_store_is_reported_once_and_never_raises(tmp_path, caplog):
    records = inference_records.Recorder(make_args(tmp_path))
    Path(records.root).write_text("not a directory")
    for _ in range(3):
        records.record_group([sample(0, 1.0)], decision="passed")
    drain(records)
    counts = records.metrics()
    assert counts["rollout/records/failed_total"] == 1
    assert counts["rollout/records/written_total"] == 0
    assert caplog.text.count("Inference records disabled") == 1


def test_missing_checkpoint_disables_recording_without_raising(tmp_path, caplog):
    records = inference_records.Recorder(make_args(tmp_path, policy=tmp_path / "absent"))
    records.record_group([sample(0, 1.0)], decision="passed")
    assert records.unavailable and records.metrics()["rollout/records/dropped_total"] == 1
    assert "Inference records disabled" in caplog.text


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"records_root": "relative/store"}, "absolute"),
        ({"records_responses": "some"}, "records_responses"),
        ({"records_responses": "sample", "records_response_sample_rate": 1.5}, "records_response_sample_rate"),
    ],
)
def test_core_rejects_invalid_record_settings(fields, message):
    with pytest.raises(InputError, match=message):
        CoreConfig(**fields)


def document(tmp_path, **sections):
    return {
        "schema_version": 1,
        "name": "records-trial",
        "model": {"source": "model", "format": "hf"},
        "output": {"root": str(tmp_path / "run")},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 16}]},
        **sections,
    }


@pytest.mark.parametrize(
    "records",
    [{"responses": "sample"}, {"responses": "off", "response_sample_rate": 0.5}, {"response_sample_rate": 0.5}],
)
def test_sample_rate_is_required_only_for_sampled_responses(tmp_path, records):
    section = {"enabled": True, "root": "/weka/store", **records}
    with pytest.raises(InputError, match="required with, and only with"):
        RunSpec.from_dict(document(tmp_path, records=section, **ASYNC), config_path=tmp_path / "run.toml")


ASYNC = {"async": {"fully_async": True}}


def test_recording_requires_fully_async(tmp_path):
    with pytest.raises(InputError, match="requires fully_async=true"):
        RunSpec.from_dict(
            document(tmp_path, records={"enabled": True, "root": "/weka/store"}), config_path=tmp_path / "run.toml"
        )


def test_records_section_compiles_to_core_and_round_trips(tmp_path):
    section = {"enabled": True, "root": "store", "responses": "sample", "response_sample_rate": 0.25}
    spec = RunSpec.from_dict(document(tmp_path, records=section, **ASYNC), config_path=tmp_path / "run.toml")
    core = spec.compile().core
    assert core.records_root == str(tmp_path / "store")
    assert (core.records_responses, core.records_response_sample_rate) == ("sample", 0.25)
    assert spec.plan()["records"]["root"] == str(tmp_path / "store")
    again = RunSpec.from_dict(spec.to_dict(), config_path=tmp_path / "elsewhere" / "run.toml")
    assert again.compile().core.records_root == core.records_root


def test_records_are_off_unless_enabled(tmp_path):
    assert RunSpec.from_dict(document(tmp_path), config_path=tmp_path / "run.toml").compile().core.records_root is None
    disabled = document(tmp_path, records={"enabled": False, "root": "/weka/store"})
    assert RunSpec.from_dict(disabled, config_path=tmp_path / "run.toml").compile().core.records_root is None
    with pytest.raises(InputError, match="records.root is required"):
        RunSpec.from_dict(document(tmp_path, records={"enabled": True}), config_path=tmp_path / "run.toml")
    with pytest.raises(InputError, match="records"):
        RunSpec.from_dict(
            document(tmp_path, records={"enabled": True, "path": "/x"}), config_path=tmp_path / "run.toml"
        )
