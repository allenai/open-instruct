import json

import pytest
from scripts.train.atomic_think import collect


def test_collect_only_requested_artifacts_and_never_checkpoints(tmp_path):
    root, output = tmp_path / "run", tmp_path / "result"
    audit = root / "control" / "atomic-audit"
    audit.mkdir(parents=True)
    (root / "packing-receipt.json").write_text("{}")
    (audit / "probe-step0-rank0.pt").write_bytes(b"zero")
    (audit / "probe-step512-rank0.pt").write_bytes(b"final")
    (audit / "batch-step1-rank0.json").write_text("{}")
    checkpoint = root / "control" / "step512"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"checkpoint")
    result = collect.collect(root, output, ["control"], [512])
    paths = {f["path"] for f in result["files"]}
    assert paths == {
        "packing-receipt.json",
        "control/atomic-audit/probe-step512-rank0.pt",
        "control/atomic-audit/batch-step1-rank0.json",
    }
    assert (checkpoint / "model.pt").read_bytes() == b"checkpoint"
    assert json.loads((output / "collection-receipt.json").read_text()) == result


def test_collection_rejects_source_overlap_and_symlink_escape(tmp_path):
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="separate"):
        collect.collect(root, root / "output", [], [])
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "packing-receipt.json").symlink_to(outside)
    with pytest.raises(ValueError, match="leaves"):
        collect.collect(root, tmp_path / "output", [], [])
