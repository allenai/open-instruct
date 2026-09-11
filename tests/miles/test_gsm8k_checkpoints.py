"""Reject incomplete or mismatched durable boundaries in the long-run report."""

import hashlib
import json

import pytest
from scripts.miles.check_gsm8k_checkpoints import check_boundaries


@pytest.mark.parametrize("fault", [None, "cursor", "clock", "rank", "shard", "latest"])
def test_committed_boundaries(tmp_path, fault):
    for step in (100, 200):
        path = tmp_path / "core" / f"rollout_{step - 1:07d}"
        (path / "model").mkdir(parents=True)
        for name in ("model/.metadata", "model/__0.distcp", "rank_0.pt", "rank_1.pt"):
            (path / name).write_bytes(b"fixture")
        cursor = tmp_path / "rollout" / f"global_dataset_state_dict_{step - 1}.pt"
        cursor.parent.mkdir(exist_ok=True)
        cursor.write_bytes(str(step).encode())
        manifest = {
            "schema_version": 2,
            "world_size": 2,
            "expert_parallel_size": 2,
            "clock": {"completed_steps": step, "next_rollout_id": step},
            "cursor_sha256": hashlib.sha256(cursor.read_bytes()).hexdigest(),
        }
        if fault == "clock" and step == 200:
            manifest["clock"]["next_rollout_id"] = 199
        (path / "complete.json").write_text(json.dumps(manifest))
    (tmp_path / "core-latest.json").write_text(json.dumps({"rollout_id": 99 if fault == "latest" else 199}))
    if fault == "cursor":
        cursor.write_bytes(b"changed")
    elif fault == "rank":
        (path / "rank_1.pt").unlink()
    elif fault == "shard":
        (path / "model/__0.distcp").unlink()
    if fault:
        with pytest.raises(ValueError):
            check_boundaries(tmp_path, updates=200, save_interval=100)
    else:
        assert len(check_boundaries(tmp_path, updates=200, save_interval=100)["committed_boundaries"]) == 2
