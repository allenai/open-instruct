"""The launch gate must require a committed checkpoint and matching data cursor."""

import hashlib
import json

import pytest

from open_instruct.miles.execution import native_smoke


def test_smoke_rejects_changed_cursor(tmp_path):
    root = tmp_path / "checkpoints"
    path = root / "core/rollout_0000001"
    path.mkdir(parents=True)
    cursor = root / "rollout/global_dataset_state_dict_1.pt"
    cursor.parent.mkdir()
    cursor.write_bytes(b"committed cursor")
    manifest = {
        "clock": {"completed_steps": 2},
        "world_size": 2,
        "cursor_sha256": hashlib.sha256(cursor.read_bytes()).hexdigest(),
    }
    (path / "complete.json").write_text(json.dumps(manifest))
    assert native_smoke.checkpoint(tmp_path, 2) == manifest
    cursor.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="cursor differs"):
        native_smoke.checkpoint(tmp_path, 2)
