"""CPU checks that selection tables are conservative, frozen, verified and match recorded keys."""

import hashlib
import json

import pytest

from open_instruct.miles import inference_records, record_selection
from open_instruct.miles.config import CoreConfig
from open_instruct.miles.errors import InputError
from open_instruct.miles.run_spec import RunSpec
from open_instruct.test_miles_inference_records import checkpoint, document, make_args, sample


def prompt(name):
    return f"<user>{name}</user>"


def record(tmp_path, run, groups, *, deterministic=False, policy=None):
    args = make_args(tmp_path, run=run, policy=policy or checkpoint(tmp_path))
    args.sglang_enable_deterministic_inference = deterministic
    recorder = inference_records.Recorder(args)
    for name, rewards, versions in groups:
        responses = [sample(i, reward, prompt=prompt(name), versions=versions) for i, reward in enumerate(rewards)]
        recorder.record_group(responses, decision="passed")
    recorder.close()
    return args


def standard_store(tmp_path):
    zeros = [0.0, 0.0]
    for run in ("first", "second"):
        groups = [("hard", zeros, ("0",))] * 4 + [("solved", [1.0, 1.0], ("0",))] * 4
        groups += [("learnable", [0.0, 1.0], ("0",))] * 4 + [("late", zeros, ("3",))] * 4
        record(tmp_path, run, groups)
    record(tmp_path, "third", [("one-attempt", zeros, ("0",))] * 8)
    return tmp_path / "records"


def excluded_prompts(table):
    names = ("hard", "solved", "learnable", "late", "one-attempt")
    keys = {inference_records.task_identity(sample(0, 0.0).metadata, prompt(name))["task_key"]: name for name in names}
    return sorted(keys[item["task_key"]] for item in table["excluded"])


def test_conservative_rule_needs_observations_attempts_and_start_scope(tmp_path):
    table = record_selection.build(standard_store(tmp_path), skip=["all_zero"], readmit_fraction=0.0)
    # "one-attempt" has 16 zero observations but from one attempt; "late" is not start-checkpoint evidence.
    assert excluded_prompts(table) == ["hard"]
    [item] = table["excluded"]
    assert (item["observations"], item["units"], item["mode"]) == (16, 2, "all_zero")
    assert item["deviation_upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 16))
    assert table["rule"]["approximate_policy"] is False


def test_modes_select_constant_values(tmp_path):
    store = standard_store(tmp_path)
    assert excluded_prompts(record_selection.build(store, skip=["all_full"], readmit_fraction=0.0)) == ["solved"]
    both = record_selection.build(store, skip=["zero_variance"], readmit_fraction=0.0)
    assert excluded_prompts(both) == ["hard", "solved"]
    widened = record_selection.build(
        store, skip=["all_zero"], scopes=("start_checkpoint", "run_version"), readmit_fraction=0.0, min_units=1
    )
    assert excluded_prompts(widened) == ["hard", "late", "one-attempt"]
    assert widened["rule"]["approximate_policy"] is True


def test_deterministic_attempts_with_one_seed_are_one_unit(tmp_path):
    policy = checkpoint(tmp_path)
    for run in ("first", "second"):
        record(tmp_path, run, [("hard", [0.0, 0.0], ("0",))] * 8, deterministic=True, policy=policy)
    table = record_selection.build(tmp_path / "records", skip=["all_zero"], readmit_fraction=0.0)
    assert table["excluded"] == []


def test_readmission_is_resolved_into_the_table(tmp_path):
    store = standard_store(tmp_path)
    first = record_selection.build(store, skip=["zero_variance"], readmit_fraction=0.5, seed=3)
    again = record_selection.build(store, skip=["zero_variance"], readmit_fraction=0.5, seed=3)
    assert len(first["excluded"]) + len(first["readmitted"]) == 2
    assert [i["input_key"] for i in first["readmitted"]] == [i["input_key"] for i in again["readmitted"]]


def test_ambiguous_lineage_requires_a_choice(tmp_path):
    record(tmp_path, "first", [("hard", [0.0], ("0",))], policy=checkpoint(tmp_path, "a"))
    other = checkpoint(tmp_path, "b")
    (other / "workflow-model.json").write_text(
        json.dumps({"identity": {"source": {"path": "/weka/other", "files": []}}})
    )
    record(tmp_path, "second", [("hard", [0.0], ("0",))], policy=other)
    with pytest.raises(InputError, match="choose one with --lineage"):
        record_selection.build(tmp_path / "records", skip=["all_zero"])


def selection_args(tmp_path, table_path, digest, **changes):
    args = make_args(tmp_path, **changes)
    args.sglang_enable_deterministic_inference = False
    args.olmo_core = CoreConfig(selection_table=str(table_path), selection_sha256=digest)
    return args


def test_runtime_skips_exactly_the_recorded_keys(tmp_path):
    table = record_selection.build(standard_store(tmp_path), skip=["all_zero"], readmit_fraction=0.0)
    path = tmp_path / "table.json"
    digest = record_selection.write_table(table, path)
    selection = record_selection.Selection(selection_args(tmp_path, path, digest))
    hard = [sample(0, 0.0, prompt=prompt("hard"))]
    learnable = [sample(0, 0.0, prompt=prompt("learnable"))]
    assert not selection.keep(hard) and selection.keep(learnable)
    assert selection.skipped == {"math": 1} and selection.passed == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [("digest", "not the pinned"), ("lineage", "lineage"), ("protocol", "different sampling/verifier protocol")],
)
def test_runtime_refuses_mismatched_tables(tmp_path, change, message):
    table = record_selection.build(standard_store(tmp_path), skip=["all_zero"])
    path = tmp_path / "table.json"
    digest = record_selection.write_table(table, path)
    kwargs = {}
    if change == "digest":
        digest = hashlib.sha256(b"other").hexdigest()
    elif change == "lineage":
        other = checkpoint(tmp_path, "other")
        (other / "workflow-model.json").write_text(json.dumps({"identity": {"source": {"path": "/x", "files": []}}}))
        kwargs["policy"] = other
    else:
        kwargs["temperature"] = 0.6
    with pytest.raises(RuntimeError, match=message):
        record_selection.Selection(selection_args(tmp_path, path, digest, **kwargs))


def test_take_refills_and_stops_on_an_excluded_dataset():
    stream = iter(range(100))

    def pull(count):
        return [[next(stream)] for _ in range(count)]

    assert record_selection.take(pull, 3, lambda group: group[0] % 2 == 0, limit=10) == [[0], [2], [4]]
    with pytest.raises(RuntimeError, match="excludes the dataset"):
        record_selection.take(pull, 1, lambda group: False, limit=5)


def test_cli_prints_the_digest_to_pin(tmp_path, capsys):
    store = standard_store(tmp_path)
    record_selection.main([str(store), "--skip", "all_zero", "--output", str(tmp_path / "table.json")])
    printed = json.loads(capsys.readouterr().out)
    assert printed["sha256"] == hashlib.sha256((tmp_path / "table.json").read_bytes()).hexdigest()


def test_selection_section_pins_path_and_digest(tmp_path):
    digest = "a" * 64
    section = {"table": "table.json", "sha256": digest}
    spec = RunSpec.from_dict(document(tmp_path, selection=section), config_path=tmp_path / "run.toml")
    core = spec.compile().core
    assert (core.selection_table, core.selection_sha256) == (str(tmp_path / "table.json"), digest)
    assert spec.plan()["selection"]["sha256"] == digest
    with pytest.raises(InputError, match="needs both table and sha256"):
        RunSpec.from_dict(document(tmp_path, selection={"table": "t.json"}), config_path=tmp_path / "run.toml")
    with pytest.raises(InputError, match="64-character"):
        CoreConfig(selection_table="/t.json", selection_sha256="ABC")
