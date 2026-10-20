"""Application identity and verifier metadata passed into the MILES recorder."""

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

from miles.utils import inference_records
from record_helpers import checkpoint, make_args, sample

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.datasets import recording
from open_instruct.miles.rewards import rewards


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


def test_factory_preserves_checkpoint_lineage(tmp_path):
    args = make_args(tmp_path)
    recorder = recording.create_recorder(args)
    try:
        assert isinstance(recorder, inference_records.Recorder)
        assert recorder.lineage == recording.lineage_identity(checkpoint(tmp_path))
    finally:
        recorder.close()


def test_unprepared_checkpoint_uses_workflow_inventory(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text("{}")
    lineage = recording.lineage_identity(path)
    assert lineage["basis"] == "served_inventory"
    assert lineage["inventory_sha256"] == inference_records.sha256(recording.workflow.model_identity(path))


def test_application_summary_command_uses_miles_reader(tmp_path):
    args = make_args(tmp_path)
    recorder = recording.create_recorder(args)
    recorder.record_group([sample(0, 1.0)], decision="passed")
    recorder.close()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "open_instruct.miles",
            "records",
            "summarize",
            args.olmo_core.records_root,
            "--output",
            str(tmp_path / "summary"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["groups"] == 1
    assert json.loads((tmp_path / "summary/summary.json").read_text())["prompt_rows"] == 1
