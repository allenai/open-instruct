"""Negative guards for resuming saved transport evidence without republishing."""

import json

import pytest
from scripts.miles import resume_math_publication_checks as resume


def test_extended_audit_checks_every_old_hash_and_accepts_new_norm_evidence():
    old = [{"name": "expert", "parameters": {"weight": {"sha256": "expected"}}}]
    full = [
        {"name": "norm", "parameters": {"weight": {"sha256": "norm"}}},
        {"name": "expert", "class": "Experts", "parameters": {"weight": {"sha256": "expected"}}},
    ]
    assert resume.audit_subset(full, old)
    full[1]["parameters"]["weight"]["sha256"] = "corrupted"
    assert not resume.audit_subset(full, old)
    assert not resume.audit_subset(full[:1], old)
    assert not resume.audit_subset(full + full, old)


def saved(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"original weights")
    lock = tmp_path / "runtime.lock.json"
    lock.write_bytes(b"pinned runtime")
    folder = tmp_path / "saved"
    folder.mkdir()
    protocol = {
        "source": str(model),
        "precision": "bf16",
        "source_model_sha256": resume.digest(model / "model.safetensors"),
        "runtime_lock_sha256": resume.digest(lock),
        "variants": [0, 1, 2, 2, 0],
        "tolerances": resume.gate.TOLERANCES,
    }
    (folder / "protocol.json").write_text(json.dumps(protocol))
    progress = []
    scores = [{"meta_info": {"input_token_logprobs": [[None, 0]] + [[-2.0, i] for i in range(1, 40)]}}]
    result = {"outputs": [scores] * 3, "audits": [[{"name": "expert", "parameters": {"sha256": "fixed"}}]] * 3}
    (folder / "initial.json").write_text(json.dumps(result))
    for version, variant in enumerate(protocol["variants"], 1):
        progress.append(
            {
                "version": version,
                "variant": variant,
                "tensors": 23441,
                "buckets": 244,
                "forced_split_names": [
                    f"model.layers.{layer}.mlp.experts.0.{projection}.weight"
                    for layer in range(1, 16)
                    for projection in ("down_proj", "gate_proj", "up_proj")
                ],
            }
        )
        (folder / f"publication-{version}-variant-{variant}.json").write_text(json.dumps(result))
    (folder / "publication-progress.json").write_text(json.dumps(progress))
    return folder, model, lock


def test_saved_evidence_requires_all_five_exact_model_runtime_and_repeated_scores(tmp_path):
    folder, model, lock = saved(tmp_path)
    _, observed, hashes = resume.load_publications(folder, model, "bf16", lock)
    assert set(observed) == {0, 1, 2} and len(hashes) == 8
    before = dict(hashes)
    assert resume.load_publications(folder, model, "bf16", lock)[2] == before
    path = folder / "publication-4-variant-2.json"
    data = json.loads(path.read_text())
    data["outputs"][2][0]["meta_info"]["input_token_logprobs"][3][0] -= 0.01
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mismatch"):
        resume.load_publications(folder, model, "bf16", lock)


@pytest.mark.parametrize("failure", ["precision", "model", "runtime", "partial", "split", "replica"])
def test_resume_rejects_incompatible_or_incomplete_receipts(tmp_path, failure):
    folder, model, lock = saved(tmp_path)
    precision = "nvfp4" if failure == "precision" else "bf16"
    if failure == "model":
        (model / "model.safetensors").write_bytes(b"other weights")
    elif failure == "runtime":
        lock.write_bytes(b"other runtime")
    elif failure in ("partial", "split"):
        path = folder / "publication-progress.json"
        rows = json.loads(path.read_text())
        if failure == "partial":
            rows.pop()
        else:
            rows[2]["forced_split_names"].pop()
        path.write_text(json.dumps(rows))
    elif failure == "replica":
        path = folder / "publication-3-variant-2.json"
        row = json.loads(path.read_text())
        row["outputs"].pop()
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        resume.load_publications(folder, model, precision, lock)
