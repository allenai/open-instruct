"""Offline reader refuses mismatched question sets or unaudited response content."""

import csv
import json

import pytest
from scripts.miles import compare_gsm8k_generations as reader


@pytest.fixture
def extracted():
    rows = [
        {
            "input": f"Shared prompt {i}",
            "label": str(i),
            "metadata": {"prepared_sample_id": f"q{i}", "query": f"Question {i}?"},
        }
        for i in range(4)
    ]
    prepared_bytes = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    digest = reader.sha256(prepared_bytes)
    data = {"schema_version": 1, "prepared": rows, "prepared_sha256": digest, "dumps": []}
    audits = {}
    for backend in reader.BACKENDS:
        audit = {"valid": True, "backend": backend, "prepared_sha256": {"eval": digest}, "evaluation": []}
        for step in reader.STEPS:
            samples, proofs = [], []
            for i, row in enumerate(rows):
                response = f"{backend} step{step}\nAnswer: {i}\n"
                correct = int(i in ([0, 2] if backend == "core" else [1, 2]))
                samples.append(
                    {
                        "id": f"q{i}",
                        "prompt": row["input"],
                        "label": row["label"],
                        "metadata": row["metadata"],
                        "response": response,
                        "reward": correct,
                        "response_length": 20,
                        "status": "completed",
                        "weight_versions": [str(step)],
                        "prompt_tokens_sha256": reader.sha256(f"tokens{i}".encode()),
                    }
                )
                proofs.append(
                    {
                        "id": f"q{i}",
                        "correct": correct,
                        "response_tokens": 20,
                        "truncated": False,
                        "response_sha256": reader.sha256(response.encode()),
                    }
                )
            file_digest = reader.sha256(f"dump{backend}{step}".encode())
            data["dumps"].append(
                {
                    "backend": backend,
                    "completed_steps": step,
                    "file": f"/retained/{backend}/eval_{step}.pt",
                    "sha256": file_digest,
                    "samples": samples,
                }
            )
            audit["evaluation"].append(
                {
                    "completed_steps": step,
                    "valid": True,
                    "file": f"eval_{step}.pt",
                    "sha256": file_digest,
                    "policy_version": step,
                    "samples": proofs,
                }
            )
        audits[backend] = audit
    return data, audits, prepared_bytes


def validate(extracted):
    return reader.verify_generations(*extracted, expected_count=4)


def test_verifies_exact_pair_and_preserves_prepared_order(extracted, tmp_path):
    payload, report = validate(extracted)
    assert report["responses_verified"] == 48
    assert report["final_groups"] == {"core_only": ["q0"], "megatron_only": ["q1"], "both": ["q2"], "neither": ["q3"]}
    assert [row["id"] for row in payload["rows"]] == ["q0", "q1", "q2", "q3"]
    reader.write_csv(payload, tmp_path / "out.csv")
    rows = list(csv.DictReader((tmp_path / "out.csv").open()))
    assert len(rows) == 24
    assert [r["completed_steps"] for r in rows[:6]] == [str(step) for step in reader.STEPS]
    assert [r["id"] for r in rows[::6]] == ["q0", "q1", "q2", "q3"]


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("prompt", "another question", "prompt differs"),
        ("label", "wrong target", "label differs"),
        ("response", "edited response", "response SHA"),
        ("reward", 0.125, "reward mismatch"),
        ("response_length", 21, "token count"),
        ("status", "truncated", "truncation mismatch"),
        ("weight_versions", ["999"], "policy version"),
        ("prompt_tokens_sha256", "a" * 64, "prompt tokens differ"),
        ("id", "other", "question IDs differ"),
    ],
)
def test_sample_tampering_rejected(extracted, field, value, match):
    extracted[0]["dumps"][-1]["samples"][0][field] = value
    with pytest.raises(ValueError, match=match):
        validate(extracted)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_dump",
        "duplicate_dump",
        "duplicate_id",
        "file_hash",
        "prepared_hash",
        "prepared_row",
        "failed_audit",
        "missing_audit",
    ],
)
def test_incomplete_or_mismatched_provenance_rejected(extracted, kind):
    data, audits, _ = extracted
    if kind == "missing_dump":
        data["dumps"].pop()
    elif kind == "duplicate_dump":
        data["dumps"][-1] = data["dumps"][0]
    elif kind == "duplicate_id":
        data["dumps"][0]["samples"][1] = data["dumps"][0]["samples"][0]
    elif kind == "file_hash":
        data["dumps"][0]["sha256"] = "0" * 64
    elif kind == "prepared_hash":
        data["prepared_sha256"] = "0" * 64
    elif kind == "prepared_row":
        data["prepared"][0]["input"] = "changed"
    elif kind == "failed_audit":
        audits["core"]["valid"] = False
    else:
        audits["core"]["evaluation"].pop()
    with pytest.raises(ValueError):
        validate(extracted)


def test_reader_embeds_exact_text_without_script_injection(extracted):
    payload, _ = validate(extracted)
    malicious = '</script><script>window.BAD=true</script><img src="https://example.com/x">\n& <think>'
    payload["rows"][0]["generations"]["core"]["0"]["response"] = malicious
    rendered = reader.render_reader(payload)
    assert malicious not in rendered
    embedded = rendered.split('<script id="verified-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "<" not in embedded
    assert json.loads(embedded)["rows"][0]["generations"]["core"]["0"]["response"] == malicious
    assert "connect-src 'none'" in rendered
    assert "innerHTML" not in rendered
