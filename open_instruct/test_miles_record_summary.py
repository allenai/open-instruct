"""CPU checks that record summaries keep evidence, validity, correlation and token buckets separate."""

import json

from open_instruct.miles import inference_records, record_summary
from open_instruct.test_miles_inference_records import Status, checkpoint, make_args, sample


def store(tmp_path):
    policy = checkpoint(tmp_path)
    first = inference_records.Recorder(make_args(tmp_path, run="first", policy=policy))
    second = inference_records.Recorder(make_args(tmp_path, run="second", policy=policy))
    zero = [sample(i, 0.0) for i in range(2)]
    mixed = [sample(i, float(i), status=Status.TRUNCATED if i else Status.COMPLETED) for i in range(2)]
    fractional = [sample(i, 0.25 * (i + 1), prompt="<user>other</user>") for i in range(2)]
    unknown = [sample(i, 1.0, prompt="<user>custom</user>") for i in range(2)]
    for response in unknown:
        response.metadata.pop("verifier_diagnostics")
    later = [sample(i, 1.0, versions=("3", "4")) for i in range(2)]
    first.record_group(zero, decision="filtered", reason="zero_std_0.0")
    second.record_group(mixed, decision="passed")
    second.record_disposition(mixed, accepted=True, staleness=0)
    second.record_group(fractional, decision="passed")
    second.record_group(unknown, decision="filtered", reason="zero_std_1.0")
    second.record_group(later, decision="passed")
    second.record_disposition(later, accepted=False, staleness=3)
    first.close(), second.close()
    return tmp_path / "records"


def by_scope(rows, prompt):
    task = inference_records.task_identity(sample(0, 0.0).metadata, prompt)["task_key"]
    return {row["policy_scope"]: row for row in rows if row["task_key"] == task}


def test_start_checkpoint_evidence_pools_attempts_and_keeps_distributions(tmp_path):
    rows, _, summary = record_summary.summarize(store(tmp_path))
    rows_by_scope = by_scope(rows, "<system>Be brief.</system><user>What is 6 * 7?</user>")
    start = rows_by_scope["start_checkpoint"]
    assert start["independent_units"] == 2 and start["groups"] == 2
    assert start["group_outcomes"] == {"all_zero": 1, "mixed": 1}
    assert start["valid_reward"]["histogram"] == {"0": 3, "1": 1}
    assert start["valid_reward"]["mean"] == 0.25
    assert start["truncated_responses"] == 1
    # Later-policy observations from one run are one correlated unit, kept out of the start row.
    mixed = rows_by_scope["mixed"]
    assert mixed["independent_units"] == 1 and mixed["valid_reward"]["count"] == 2
    fractional = by_scope(rows, "<user>other</user>")["start_checkpoint"]
    assert fractional["valid_reward"]["histogram"] == {"0.25": 1, "0.5": 1}
    assert summary["lineages"] and len(summary["protocols"]) == 1
    assert summary["dispositions"] == {"consumed": 1, "expired": 1}
    assert {item["path"].split("/")[-1].split("-")[0] for item in summary["input_snapshot"]} == {"manifest", "records"}


def test_unknown_validity_is_never_mixed_with_valid_evidence(tmp_path):
    rows, _, _ = record_summary.summarize(store(tmp_path))
    unknown = by_scope(rows, "<user>custom</user>")["start_checkpoint"]
    assert unknown["validity"] == {"unknown": 2}
    assert unknown["valid_reward"] == {"count": 0}
    assert unknown["unknown_validity_reward"]["count"] == 2


def test_token_account_separates_dispositions_and_truncation(tmp_path):
    _, account, _ = record_summary.summarize(store(tmp_path))
    math = account["math"]
    assert set(math) == {"consumed", "expired", "unused", "filtered:zero_std_0.0", "filtered:zero_std_1.0"}
    assert math["consumed"] == {
        "groups": 1,
        "responses": 2,
        "tokens": 201,
        "truncated_responses": 1,
        "truncated_tokens": 101,
    }
    assert math["filtered:zero_std_0.0"]["tokens"] == 201
    assert math["unused"]["groups"] == 1


def test_partial_lines_and_orphan_files_are_reported_not_read(tmp_path):
    root = store(tmp_path)
    records = next(root.rglob("records-*.jsonl"))
    with records.open("a") as stream:
        stream.write('{"schema_version": 1, "kind": "gro')
    (records.parent / "records-orphan.jsonl").write_text(records.read_text().splitlines()[0] + "\n")
    rows, _, summary = record_summary.summarize(root)
    assert any("incomplete final line" in warning for warning in summary["warnings"])
    assert any("no matching manifest" in warning for warning in summary["warnings"])
    assert summary["groups"] == 5


def test_cli_writes_all_outputs(tmp_path, capsys):
    root = store(tmp_path)
    record_summary.main(["summarize", str(root), "--output", str(tmp_path / "summary")])
    assert json.loads(capsys.readouterr().out)["groups"] == 5
    lines = (tmp_path / "summary" / "prompts.jsonl").read_text().splitlines()
    assert all(json.loads(line)["schema_version"] == 1 for line in lines)
    assert json.loads((tmp_path / "summary" / "tokens.json").read_text())["math"]
    assert json.loads((tmp_path / "summary" / "summary.json").read_text())["prompt_rows"] == len(lines)
