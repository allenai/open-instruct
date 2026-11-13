"""Selection provenance survives the pinned recorder's metadata whitelist."""

import copy
import importlib
import json
from types import SimpleNamespace

import pytest

from open_instruct import test_miles_grading as shared
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import grading


@pytest.fixture
def native():
    pytest.importorskip("miles.utils.inference_records")
    return importlib.import_module("open_instruct.miles.datasets.recording")


def arguments(tmp_path, *, final=True, reasoning="required", responses="all"):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "workflow-model.json").write_text(json.dumps({"identity": {"source": {"path": str(checkpoint)}}}))
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    return SimpleNamespace(
        hf_checkpoint=str(checkpoint),
        olmo_core=CoreConfig(
            records_root=str(tmp_path / "records"),
            records_responses=responses,
            reward_config=str(registry),
            reward_final_answer_only=final,
            reward_reasoning_mode=reasoning,
        ),
    )


def sample():
    value = shared.item("work</think>answer", target="answer")
    value.index, value.group_index, value.rollout_id, value.reward = 0, 0, 0, 1.0
    value.weight_versions = []
    value.metadata["grading"] = {
        **grading.split_response(value.response).summary(),
        "components": {"test": {"scope": "response", "require_response": True, "gate": None}},
    }
    return value


def read_record(recorder, value):
    try:
        recorder.record_group([value], decision="kept")
    finally:
        recorder.close()
    assert recorder.counts["failed"] == 0
    rows = [
        json.loads(line)
        for path in recorder.directory.glob("records-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    return next(row for row in rows if row["kind"] == "group")


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("responses", ["off", "all"])
def test_real_recorder_persists_selection_only_in_final_mode(native, tmp_path, final, responses):
    args = arguments(tmp_path, final=final, responses=responses)
    recorder = native.create_recorder(args)
    assert isinstance(recorder, native.GradingRecorder)
    value = sample()
    selection = copy.deepcopy(value.metadata["grading"])
    row = read_record(recorder, value)
    assert ("response" in row["responses"][0]) == (responses == "all")
    base_key = native.inference_records.sha256(
        {"task": row["task_key"], "tokens": None, "protocol": recorder.protocol_sha256}
    )
    if final:
        assert row["responses"][0]["grading"] == selection
        assert row["input_key"] == native.inference_records.sha256(
            {"input_key": base_key, "grading_policy": row["grading_policy"]}
        )
        assert row["grading_policy"]["reward_reasoning_mode"] == "required"
    else:
        assert "grading_policy" not in row and "grading" not in row["responses"][0]
        assert row["input_key"] == base_key


def test_different_reasoning_contracts_have_different_evidence_keys(native, tmp_path):
    keys = []
    for mode in ("required", "none"):
        args = arguments(tmp_path, reasoning=mode)
        keys.append(read_record(native.create_recorder(args), sample())["input_key"])
    assert len(set(keys)) == 2


def test_feedback_and_persisted_rows_share_the_grading_identity(native, tmp_path):
    class Source:
        provider = SimpleNamespace(requires_feedback=False)

        def __init__(self):
            self.rows = []

        def feedback(self, rows):
            self.rows.extend(copy.deepcopy(rows))

    source = Source()
    recorder = native.FeedbackRecorder(arguments(tmp_path), source)
    row = read_record(recorder, sample())
    assert source.rows[0]["input_key"] == row["input_key"]
    assert source.rows[0]["responses"][0]["grading"] == row["responses"][0]["grading"]


def test_intact_judge_contract_cannot_pool_with_previous_extraction_contract(native, tmp_path, monkeypatch):
    keys = []
    for version in (1, 2):
        monkeypatch.setattr(grading, "SCHEMA_VERSION", version)
        row = read_record(native.create_recorder(arguments(tmp_path)), sample())
        assert row["grading_policy"]["schema_version"] == version
        keys.append(row["input_key"])
    assert keys[0] != keys[1]
