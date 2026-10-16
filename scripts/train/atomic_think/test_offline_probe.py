import json
from types import SimpleNamespace

import pytest
from scripts.train.atomic_think import offline_probe


def test_checkpoint_audit_requires_all_training_replicas_and_matching_provenance(tmp_path):
    root = tmp_path / "control" / "atomic-audit"
    root.mkdir(parents=True)
    report = {"step": 2, "mode": "control", "manifest_hash": "manifest", "model": {"module.weight": "weight-hash"}}
    for rank in range(16):
        (root / f"model-state-step2-rank{rank}.json").write_text(json.dumps(report))
    assert offline_probe.expected_state(tmp_path, "control", "control", 2, "manifest") == {"weight": "weight-hash"}
    report["model"]["module.weight"] = "different"
    (root / "model-state-step2-rank15.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="replicas disagree"):
        offline_probe.expected_state(tmp_path, "control", "control", 2, "manifest")
    with pytest.raises(ValueError, match="provenance"):
        offline_probe.expected_state(tmp_path, "control", "atomic-safe", 2, "manifest")
    (root / "model-state-step2-rank15.json").unlink()
    with pytest.raises(FileNotFoundError):
        offline_probe.expected_state(tmp_path, "control", "control", 2, "manifest")


def test_saved_atomic_tokenizer_must_preserve_tag_decoding():
    table = {"<think>": [100256], "</think>": [100270], "context": [1, 2]}
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: table[text],
        decode=lambda ids, **kwargs: "<think>" if ids == [100256] else "</think>",
    )
    docs = [{"id": "test", "text": "context", "atomic_ids": [1, 2], "original_ids": [1, 3]}]
    offline_probe.verify_tokenizer(tokenizer, docs, "atomic-safe")
    with pytest.raises(ValueError, match="frozen inputs"):
        offline_probe.verify_tokenizer(tokenizer, docs, "control")
    tokenizer.decode = lambda *args, **kwargs: ""
    with pytest.raises(ValueError, match="disappears"):
        offline_probe.verify_tokenizer(tokenizer, docs, "atomic-safe")
