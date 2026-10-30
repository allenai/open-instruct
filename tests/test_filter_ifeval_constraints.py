import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.data import filter_ifeval_constraints as filtering


def bundle(n=2):
    return {"instruction_id": ["count:lowercase_counting"], "kwargs": [{"N": n}]}


def miles(target=None, name="ifeval", **extra):
    return {"metadata": {"verifiers": [{"name": name, "target": target if target is not None else bundle()}]}, **extra}


def jsonl(*rows):
    return b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)


def test_filter_preserves_bytes_order_input_and_hashes(tmp_path):
    source, report, output = (tmp_path / name for name in ("input.jsonl", "report.json", "output.jsonl"))
    keep1 = (json.dumps(miles(bundle(4), id="keep-世界"), ensure_ascii=False, indent=None) + "\r\n").encode()
    keep2 = b' {"dataset" : ["math"], "ground_truth": ["42"]} '
    raw = keep1 + b"\n" + jsonl(miles(id="drop")) + keep2
    source.write_bytes(raw)
    summary = filtering.audit(source, report, output_path=output)
    expected = keep1 + b"\n" + keep2
    assert output.read_bytes() == expected
    assert source.read_bytes() == raw
    result = json.loads(report.read_text())
    assert result["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["output_sha256"] == hashlib.sha256(expected).hexdigest()
    assert result["kept_sha256"] == result["output_sha256"]
    assert summary["total_rows"] == 3
    assert summary["if_rows"] == 2
    assert summary["kept_rows"] == 2
    assert summary["dropped_rows"] == 1
    assert result["reason_counts"] == {"lowercase_counting_tiny_n": 1}
    assert result["dropped"] == [
        {"row_index": 1, "line_number": 3, "id": "drop", "reasons": ["lowercase_counting_tiny_n"]}
    ]


def test_audit_is_default_cli_and_does_not_create_output(tmp_path):
    source, report = tmp_path / "in.jsonl", tmp_path / "audit.json"
    source.write_bytes(jsonl(miles()))
    assert filtering.__file__ is not None
    script = Path(filtering.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--report", str(report)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text())["output"] is None
    assert json.loads(report.read_text())["output_sha256"] is None
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audit.json", "in.jsonl"]


def test_legacy_parallel_targets_and_scalar_name(tmp_path):
    source, report = tmp_path / "in.jsonl", tmp_path / "audit.json"
    source.write_bytes(
        jsonl(
            {"dataset": ["math", "ifeval"], "ground_truth": ["42", json.dumps([json.dumps(bundle())])]},
            {"dataset": "ifeval", "ground_truth": repr([bundle(4)])},
            {"dataset": ["math"], "ground_truth": ["not an IF bundle"]},
        )
    )
    summary = filtering.audit(source, report)
    assert summary["total_rows"] == 3
    assert summary["if_rows"] == 2
    assert summary["dropped_rows"] == 1


def test_custom_aliases_are_only_screened_when_trusted(tmp_path):
    source = tmp_path / "in.jsonl"
    source.write_bytes(jsonl(miles(name="custom_if"), miles(name="ifeval_old"), miles(name="ifeval")))
    assert filtering.audit(source, tmp_path / "default.json")["dropped_rows"] == 1
    summary = filtering.audit(source, tmp_path / "custom.json", verifier_names=["ifeval", "custom_if"])
    assert summary["dropped_rows"] == 2
    assert summary["if_rows"] == 2


def test_repeated_cli_aliases(tmp_path, capsys):
    source, report = tmp_path / "in.jsonl", tmp_path / "report.json"
    source.write_bytes(jsonl(miles(name="a"), miles(name="b"), miles()))
    assert (
        filtering.main(
            ["--input", str(source), "--report", str(report), "--verifier-name", "a", "--verifier-name", "b"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["dropped_rows"] == 2


def test_multiconstraint_and_verifier_reason_counts_deduplicate(tmp_path):
    source, report = tmp_path / "in.jsonl", tmp_path / "report.json"
    target = {"instruction_id": ["count:lowercase_counting"] * 2, "kwargs": [{"N": 1}, {"N": 2}]}
    row = miles(target)
    row["metadata"]["verifiers"].append({"name": "ifeval", "target": bundle()})
    source.write_bytes(jsonl(row))
    result = filtering.audit(source, report)
    assert result["reason_counts"] == {"lowercase_counting_tiny_n": 1}


def test_malformed_target_dropped_with_id_fallback(tmp_path):
    source, report = tmp_path / "in.jsonl", tmp_path / "report.json"
    row = miles("{")
    row["metadata"]["sample_id"] = "sample-a"
    source.write_bytes(jsonl(row))
    result = filtering.audit(source, report)
    assert result["reason_counts"] == {"malformed_target": 1}
    assert json.loads(report.read_text())["dropped"][0]["id"] == "sample-a"


@pytest.mark.parametrize(
    "row",
    [
        [],
        {},
        {"metadata": {"verifiers": []}},
        {"metadata": {"verifiers": "wrong"}},
        {"metadata": {"verifiers": [{"name": "ifeval"}]}},
        {"metadata": {"verifiers": [{"name": "", "target": {}}]}},
        {"dataset": ["ifeval", "math"], "ground_truth": [bundle()]},
        {"dataset": ["ifeval"], "ground_truth": {}},
        {"dataset": [None], "ground_truth": [{}]},
    ],
)
def test_malformed_envelope_fails_with_row_context_and_no_partial_artifacts(tmp_path, row):
    source, report, output = (tmp_path / name for name in ("input.jsonl", "report.json", "output.jsonl"))
    raw = jsonl(miles(bundle(5)), row)
    source.write_bytes(raw)
    with pytest.raises(ValueError, match=r"line 2 \(row 1\)"):
        filtering.audit(source, report, output_path=output)
    assert source.read_bytes() == raw
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.jsonl"]


@pytest.mark.parametrize("destination", ["report", "output"])
def test_refuses_overwrite(tmp_path, destination):
    source, report, output = (tmp_path / name for name in ("input.jsonl", "report.json", "output.jsonl"))
    raw = jsonl(miles())
    source.write_bytes(raw)
    existing = report if destination == "report" else output
    existing.write_bytes(b"untouched")
    with pytest.raises(ValueError, match="overwrite"):
        filtering.audit(source, report, output_path=output)
    assert existing.read_bytes() == b"untouched"
    assert source.read_bytes() == raw


@pytest.mark.parametrize(
    "collision", ["input_output", "input_report", "output_report", "symlink", "hardlink", "relative", "dangling"]
)
def test_path_alias_collisions_leave_input_untouched(tmp_path, collision):
    source, report, output = (tmp_path / name for name in ("input.jsonl", "report.json", "output.jsonl"))
    raw = jsonl(miles())
    source.write_bytes(raw)
    if collision == "input_output":
        output = source
    elif collision == "input_report":
        report = source
    elif collision == "output_report":
        output = report
    elif collision == "symlink":
        output.symlink_to(source)
    elif collision == "hardlink":
        output.hardlink_to(source)
    elif collision == "relative":
        output = source.parent / "." / source.name
    else:
        report.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="distinct|overwrite"):
        filtering.audit(source, report, output_path=output)
    assert source.read_bytes() == raw


def test_empty_input_report_is_valid_json(tmp_path):
    source, report, output = (tmp_path / name for name in ("input.jsonl", "report.json", "output.jsonl"))
    source.write_bytes(b"")
    result = filtering.audit(source, report, output_path=output)
    assert result["total_rows"] == 0
    assert json.loads(report.read_text())["dropped"] == []
    assert output.read_bytes() == b""


def test_dropped_prepared_sample_id_and_custom_id_preserved(tmp_path):
    source, report = tmp_path / "in.jsonl", tmp_path / "report.json"
    row = miles(custom_id="custom-a")
    row["metadata"]["prepared_sample_id"] = "prepared-a"
    source.write_bytes(jsonl(row))
    filtering.audit(source, report)
    record = json.loads(report.read_text())["dropped"][0]
    assert record["prepared_sample_id"] == "prepared-a"
    assert record["custom_id"] == "custom-a"


def test_scalar_legacy_label_requires_single_target(tmp_path):
    source, report = tmp_path / "in.jsonl", tmp_path / "report.json"
    source.write_bytes(jsonl({"dataset": "math", "ground_truth": ["a", "b"]}))
    with pytest.raises(ValueError, match="scalar dataset requires exactly one"):
        filtering.audit(source, report)
    assert not report.exists()


@pytest.mark.parametrize("label", ["IFEval", "IfEvAl", "ifeval"])
@pytest.mark.parametrize("as_list", [False, True])
def test_legacy_verifier_names_match_case_insensitively(tmp_path, label, as_list):
    source, report = tmp_path / "input.jsonl", tmp_path / "report.json"
    row = {"dataset": [label] if as_list else label, "ground_truth": [bundle()] if as_list else bundle()}
    source.write_bytes(jsonl(row))
    summary = filtering.audit(source, report)
    assert summary["if_rows"] == summary["dropped_rows"] == 1


def test_custom_legacy_alias_is_case_insensitive_but_miles_alias_is_exact(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_bytes(
        jsonl(
            {"dataset": "ConstraintAlias", "ground_truth": bundle()},
            miles(name="ConstraintAlias"),
            miles(name="constraintalias"),
        )
    )
    summary = filtering.audit(source, tmp_path / "report.json", verifier_names=["ConstraintAlias"])
    assert summary["if_rows"] == summary["dropped_rows"] == 2
