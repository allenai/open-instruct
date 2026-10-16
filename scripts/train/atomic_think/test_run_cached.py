import hashlib
import json

import pytest
from scripts.train.atomic_think import run_cached


def test_fresh_cached_setup_preserves_source_and_rejects_wrong_arm_or_existing_checkpoints(tmp_path, monkeypatch):
    source = tmp_path / "source"
    packing = source / "control" / "dataset-common"
    packing.mkdir(parents=True)
    for kind in ("original", "atomic"):
        (source / f"dataset-{kind}.json").write_text(json.dumps({"kind": kind}))
    cache = source / "sample-cache-original"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "kind": "original",
                    "epoch": 1,
                    "data_loader_seed": 34521,
                    "global_batch_tokens": 1048576,
                },
                "rows": [None] * 8448,
            }
        )
    )
    monkeypatch.setattr(run_cached.packed_sample_cache, "stage_cache", lambda source, local: {"copied": True})
    output = tmp_path / "new"
    result = run_cached.prepare(output, "control", source, cache, tmp_path / "local")
    assert result["copied"] and result["kind"] == "original"
    assert (output / "control" / "dataset-common").resolve() == packing
    assert (output / "dataset-original.json").read_bytes() == (source / "dataset-original.json").read_bytes()
    with pytest.raises(ValueError, match="registered manifest hash"):
        run_cached.prepare(output, "control", source, cache, tmp_path / "bad-hash", manifest_sha256="wrong")
    with pytest.raises(ValueError, match="require a registered"):
        run_cached.prepare(output, "control", source, cache, tmp_path / "missing-hash", expected_samples=188544)
    manifest = json.loads((cache / "manifest.json").read_text())
    manifest["rows"] = [None] * 188544
    (cache / "manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()
    run_cached.prepare(
        output, "control", source, cache, tmp_path / "full", expected_samples=188544, manifest_sha256=digest
    )
    with pytest.raises(ValueError, match="registered"):
        run_cached.prepare(output, "atomic-safe", source, cache, tmp_path / "bad")
    (output / "control" / "step100").mkdir()
    with pytest.raises(FileExistsError, match="existing checkpoints"):
        run_cached.prepare(output, "control", source, cache, tmp_path / "another")
