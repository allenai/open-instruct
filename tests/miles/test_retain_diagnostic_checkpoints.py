"""Retention must survive source pruning without changing the live save tree."""

import os
import shutil

import pytest
from scripts.miles.retain_diagnostic_checkpoints import retain


def checkpoint(tmp_path):
    root = tmp_path / "training"
    source = root / "iter_0000099"
    source.mkdir(parents=True)
    (source / ".metadata").write_bytes(b"metadata")
    (source / "rank.distcp").write_bytes(b"weights and optimizer")
    (root / "rollout").mkdir()
    (root / "rollout/global_dataset_state_dict_99.pt").write_bytes(b"cursor")
    (root / "latest_checkpointed_iteration.txt").write_text("99")
    return root, source, tmp_path / "diagnostic"


def test_retention_survives_native_pruning_and_is_idempotent(tmp_path):
    root, source, destination = checkpoint(tmp_path)
    result = retain(root, destination, 99)
    assert result["completed_updates"] == 100
    saved = destination / source.name
    assert os.path.samefile(source / "rank.distcp", saved / "rank.distcp")
    assert (root / "latest_checkpointed_iteration.txt").read_text() == "99"
    assert retain(root, destination, 99) == result
    shutil.rmtree(source)
    assert (saved / "rank.distcp").read_bytes() == b"weights and optimizer"
    assert retain(root, destination, 99) == result


def test_waits_for_tracker_and_cursor(tmp_path):
    root, source, destination = checkpoint(tmp_path)
    tracker = root / "latest_checkpointed_iteration.txt"
    tracker.write_text("0")
    assert retain(root, destination, 99) is None
    tracker.write_text("99")
    (root / "rollout/global_dataset_state_dict_99.pt").unlink()
    assert retain(root, destination, 99) is None
    assert not destination.exists()


def test_rejects_source_symlink_and_unrelated_destination(tmp_path):
    root, source, destination = checkpoint(tmp_path)
    (source / "link").symlink_to(source / "rank.distcp")
    with pytest.raises(ValueError, match="ordinary files"):
        retain(root, destination, 99)
    (source / "link").unlink()
    saved = destination / source.name
    saved.mkdir(parents=True)
    (saved / "rank.distcp").write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="unrelated"):
        retain(root, destination, 99)
    assert (saved / "rank.distcp").read_bytes() == b"unrelated"
