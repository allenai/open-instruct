"""Scientific launch must require successful updates and retained checkpoints."""

import json

import pytest
from scripts.miles.experiments.h243_pool_batches import launch_after_gate


def receipts(tmp_path):
    root = tmp_path / "task/run/checkpoints"
    root.mkdir(parents=True)
    for rank in range(2):
        rows = [
            {"event": "optimizer", "step": step, "reduction": "token", "optimizer_skipped": False}
            for step in range(1, 5)
        ]
        (root / f"training_contract_rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for step in (2, 4):
        folder = root / "core" / f"rollout_{step - 1:07d}"
        folder.mkdir(parents=True)
        (folder / "complete.json").write_text(json.dumps({"clock": {"completed_steps": step}}))
    return root


def test_gate_requires_both_rank_updates_and_retained_checkpoints(tmp_path):
    root = receipts(tmp_path)
    assert launch_after_gate.audit(tmp_path)["checkpoint_steps"] == [2, 4]
    (root / "training_contract_rank1.jsonl").unlink()
    with pytest.raises(RuntimeError, match="both trainer"):
        launch_after_gate.audit(tmp_path)


@pytest.mark.parametrize("failure", ["skipped", "missing_update", "wrong_reduction", "missing_checkpoint"])
def test_incomplete_or_wrong_mechanics_cannot_launch(tmp_path, failure):
    root = receipts(tmp_path)
    path = root / "training_contract_rank0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if failure == "missing_update":
        rows.pop()
    elif failure == "skipped":
        rows[-1]["optimizer_skipped"] = True
    elif failure == "wrong_reduction":
        rows[-1]["reduction"] = "response"
    else:
        (root / "core/rollout_0000003/complete.json").unlink()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError):
        launch_after_gate.audit(tmp_path)
