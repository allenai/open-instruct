"""A successful process exit alone must not launch the expensive comparison."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/miles/hybrid_compare_450m.py"
spec = importlib.util.spec_from_file_location("hybrid_compare_450m", MODULE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def smoke(tmp_path):
    (tmp_path / "workflow.json").write_text(
        json.dumps({"status": "complete", "result": {"completed_rollout_ids": [0, 1]}})
    )
    root = tmp_path / "checkpoints"
    root.mkdir()
    rows = [
        {"stage": name, "passed": True}
        for name, count in (("training", 2), ("publication", 2), ("evaluation", 3), ("checkpoint", 2))
        for _ in range(count)
    ]
    (root / "driver_timing.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    for rollout in (0, 1):
        folder = root / "core" / f"rollout_{rollout:07d}"
        folder.mkdir(parents=True)
        (folder / "complete.json").write_text(json.dumps({"cursor_sha256": "abc"}))
    (tmp_path / "export-hf").mkdir()
    (tmp_path / "export-hf/.complete").touch()
    return tmp_path


def test_complete_smoke_can_open_gate(smoke):
    gate.validate_smoke(smoke)


@pytest.mark.parametrize(
    "missing", ["workflow.json", "checkpoints/core/rollout_0000001/complete.json", "export-hf/.complete"]
)
def test_missing_completion_evidence_closes_gate(smoke, missing):
    (smoke / missing).unlink()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        gate.validate_smoke(smoke)


def test_partial_training_closes_gate(smoke):
    (smoke / "workflow.json").write_text(json.dumps({"status": "complete", "result": {"completed_rollout_ids": [0]}}))
    with pytest.raises(RuntimeError, match="two learner updates"):
        gate.validate_smoke(smoke)


def test_failed_evaluation_closes_gate(smoke):
    path = smoke / "checkpoints/driver_timing.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["stage"] == "evaluation":
            row["passed"] = False
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(RuntimeError, match="evaluation"):
        gate.validate_smoke(smoke)
