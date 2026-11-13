import hashlib
import json

import pyarrow as pa
import pytest
from pyarrow import parquet as pq
from scripts.data.sft import export_olmo35_audit as audit


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    bundle = tmp_path / audit.BUNDLE
    extras = tmp_path / audit.EXTRAS
    bundle.parent.mkdir(parents=True)
    extras.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "shared", "filter": "cutoff"},
                {"id": "date-only", "filter": "cutoff"},
                {"id": "shared", "filter": "safety"},
                {"id": "safety-only", "filter": "safety"},
                {"id": "unrelated", "filter": "optional-cleanup"},
            ]
        ),
        bundle,
    )
    pq.write_table(pa.table({"id": [f"extra-{i}" for i in range(21)]}), extras)
    monkeypatch.setattr(audit, "FILTERS", {"dated_cutoff": ("cutoff", 2), "hidden_safety_prompt": ("safety", 2)})
    monkeypatch.setattr(audit, "EXPECTED_UNION", 24)
    monkeypatch.setattr(
        audit, "EVIDENCE", {p: hashlib.sha256((tmp_path / p).read_bytes()).hexdigest() for p in audit.EVIDENCE}
    )
    return tmp_path


def test_export_keeps_reason_overlap_and_excludes_unrelated_lists(evidence):
    output = evidence / "manifest.json"
    manifest = audit.export_manifest(evidence, output)
    assert json.loads(output.read_text()) == manifest
    assert manifest["groups"]["dated_cutoff"] == ["date-only", "shared"]
    assert manifest["groups"]["hidden_safety_prompt"] == ["safety-only", "shared"]
    assert "unrelated" not in set().union(*map(set, manifest["groups"].values()))
    assert len(manifest["groups"]["stale_identity_extra"]) == 21


def test_changed_evidence_is_rejected_before_output(evidence):
    (evidence / audit.BUNDLE).write_bytes(b"replaced evidence")
    output = evidence / "manifest.json"
    with pytest.raises(ValueError, match="Audit evidence changed"):
        audit.export_manifest(evidence, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / "manifest.json"
    output.write_text("existing")
    with pytest.raises(FileExistsError):
        audit.export_manifest(tmp_path / "missing-evidence", output)
    assert output.read_text() == "existing"


def test_unexpected_overlap_is_rejected(evidence, monkeypatch):
    monkeypatch.setattr(audit, "EXPECTED_UNION", 25)
    output = evidence / "manifest.json"
    with pytest.raises(ValueError, match="distinct release-drop IDs"):
        audit.export_manifest(evidence, output)
    assert not output.exists()
