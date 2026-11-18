"""Run-specific prompt exclusion preserves validation, mixtures, and resume identity."""

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from open_instruct import test_miles_run_data, test_miles_run_spec
from open_instruct.miles.datasets import run_data
from open_instruct.miles.errors import InputError

environment = test_miles_run_data.environment


def row(text, **metadata):
    return {"input": text, "metadata": {"verifiers": [{"name": "gsm8k", "target": "42"}], **metadata}}


def prepared_input(tmp_path, rows, *, compressed=False):
    path = tmp_path / ("input.jsonl.gz" if compressed else "input.jsonl")
    raw = "".join(json.dumps(value) + "\n" for value in rows).encode()
    path.write_bytes(gzip.compress(raw) if compressed else raw)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"gsm8k": {"factory": run_data.FACTORIES["gsm8k"]}}))
    return {"prompt_data": str(path), "reward_config": str(registry)}


def prepare(environment, data, limit=10):
    return run_data.prepare_data(data, environment[0], environment[1], max_prompt_length=limit, seed=17)


@pytest.mark.parametrize("compressed", [False, True])
def test_exclude_streams_boundaries_counts_and_immutable_resume(environment, tmp_path, compressed, caplog):
    caplog.set_level("INFO", logger=run_data.logger.name)
    rows = [row("short"), row("x" * 11), row("y" * 10), row("z" * 12), row("last")]
    data = prepared_input(tmp_path, rows, compressed=compressed) | {"over_length": "exclude"}
    source_bytes = Path(data["prompt_data"]).read_bytes()
    result = prepare(environment, data)
    retained = [json.loads(line) for line in Path(result["prompt_data"]).read_text().splitlines()]
    assert [r["input"] for r in retained] == ["short", "y" * 10, "last"]
    assert [r["metadata"]["run_prompt_tokens"] for r in retained] == [5, 10, 4]
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["prompt_length"] == {
        "policy": "exclude",
        "max_prompt_length": 10,
        "scope": "training",
        "sources": {"train": {"original": 5, "excluded": 2, "retained": 3}},
    }
    assert "original=5 excluded=2 retained=3" in caplog.text
    assert Path(data["prompt_data"]).read_bytes() == source_bytes
    assert prepare(environment, data) == result
    for changed, limit in [(data | {"over_length": "fail"}, 10), (data, 11)]:
        with pytest.raises(InputError, match="contract changed"):
            prepare(environment, changed, limit)


@pytest.mark.parametrize("policy", [None, "fail"])
def test_default_still_fails(environment, tmp_path, policy):
    data = prepared_input(tmp_path, [row("short"), row("x" * 11)])
    if policy:
        data["over_length"] = policy
    with pytest.raises(InputError, match="Prompt has 11 tokens") as error:
        prepare(environment, data)
    assert 'data.over_length = "exclude"' in str(error.value)
    assert not environment[1].exists()


@pytest.mark.parametrize(
    "bad,match",
    [
        (row("x" * 11, verifiers=[]), "verifiers"),
        (row("x" * 11, prompt_token_ids=[999]), "token IDs differ"),
        (row(""), "nonempty rendered input"),
        (row("x" * 11, rendering_template_sha256="0" * 64), "template"),
    ],
)
def test_exclusion_does_not_hide_invalid_rows(environment, tmp_path, bad, match):
    data = prepared_input(tmp_path, [row("valid"), bad]) | {"over_length": "exclude"}
    with pytest.raises(InputError, match=match):
        prepare(environment, data)
    assert not environment[1].exists()


def test_evaluation_remains_strict(environment, tmp_path):
    data = prepared_input(tmp_path, [row("valid")]) | {"over_length": "exclude"}
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text(json.dumps(row("x" * 11)) + "\n")
    data["eval_prompt_data"] = ["heldout", str(evaluation)]
    with pytest.raises(InputError, match="heldout row 1: Prompt has 11 tokens") as error:
        prepare(environment, data)
    assert "data.over_length" not in str(error.value)


@pytest.mark.parametrize("mode", ["tasks", "adopt"])
def test_exclusion_for_materialized_inputs(environment, tmp_path, monkeypatch, mode):
    if mode == "tasks":
        monkeypatch.setattr(
            run_data,
            "_source_rows",
            lambda name: [{"question": text, "ground_truth": "42"} for text in ("small", "x" * 200)],
        )
        data = {"tasks": [{"task": "gsm8k", "train_count": 2}], "shuffle": False}
        limit = 60
    else:
        path = test_miles_run_data.source_manifest(tmp_path)
        artifact = path.parent / "train.jsonl"
        original = json.loads(artifact.read_text())
        large = copy.deepcopy(original)
        large["messages"][0]["content"] = "x" * 200
        large["metadata"]["prepared_sample_id"] = "long"
        artifact.write_text(json.dumps(original) + "\n" + json.dumps(large) + "\n")
        manifest = json.loads(path.read_text())
        manifest["artifacts"]["train"].update(records=2, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        path.write_text(json.dumps(manifest))
        data, limit = {"rl_manifest": str(path)}, 100
    result = prepare(environment, data | {"over_length": "exclude"}, limit)
    assert len(Path(result["prompt_data"]).read_text().splitlines()) == 1
    counts = json.loads(Path(result["manifest"]).read_text())["prompt_length"]["sources"]
    assert list(counts.values()) == [{"original": 2, "excluded": 1, "retained": 1}]


@pytest.mark.parametrize("empty", [False, True])
def test_mixture_retained_counts_preserve_fractions_and_reject_empty_source(environment, tmp_path, empty):
    source = prepared_input(tmp_path, [row("short"), row("x" * 11)])
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps(row("x" * 11 if empty else "other")) + "\n")
    data = {
        "over_length": "exclude",
        "sources": [
            {"name": "a", "fraction": 0.75, **source},
            {"name": "b", "fraction": 0.25, **(source | {"prompt_data": str(other)})},
        ],
    }
    if empty:
        with pytest.raises(InputError, match="source 'b' has no rows"):
            prepare(environment, data)
        assert not environment[1].exists()
        return
    result = prepare(environment, data)
    manifest = json.loads(Path(result["prompt_mixture"]).read_text())
    assert [s["rows"] for s in manifest["sources"]] == [1, 1]
    assert [s["fraction"] for s in manifest["sources"]] == [0.75, 0.25]
    assert all(
        len(Path(p).read_text().splitlines()) == s["rows"]
        for p, s in zip(result["train_sources"], manifest["sources"], strict=True)
    )


def test_exclusion_rejects_emptied_pool_within_retained_source(environment, tmp_path):
    rows = [row("short") | {"pool_id": "a"}, row("x" * 11) | {"pool_id": "b"}]
    source = prepared_input(tmp_path, rows)
    data = {
        "over_length": "exclude",
        "sources": [{"name": "both", **source}],
        "pools": {"a": {"label": "a"}, "b": {"label": "b"}},
        "sampling": {"policy": "weighted", "weights": {"a": 1, "b": 3}},
    }
    with pytest.raises(InputError, match="pool 'b' has no rows"):
        prepare(environment, data)
    assert not environment[1].exists()


@pytest.mark.parametrize("policy", ["exclude", "fail", "truncate", None, True])
def test_run_spec_and_direct_preparation_validate_policy(tmp_path, policy):
    data = {"over_length": policy, "tasks": [{"task": "multiplication", "train_count": 1}]}
    if policy in ("exclude", "fail"):
        spec = test_miles_run_spec.spec(tmp_path, data=data)
        assert spec.data["over_length"] == policy
        run_data.validate_data(data)
    else:
        with pytest.raises(InputError, match="data.over_length"):
            test_miles_run_spec.spec(tmp_path, data=data)
        with pytest.raises(InputError, match="data.over_length"):
            run_data.validate_data(data)


@pytest.mark.parametrize("dataset", [None, [], {}])
def test_optional_source_metadata_does_not_break_accounting(environment, tmp_path, dataset):
    data = prepared_input(tmp_path, [row("valid", source_dataset=dataset)])
    result = prepare(environment, data)
    counts = json.loads(Path(result["manifest"]).read_text())["prompt_length"]["sources"]
    assert counts == {"train": {"original": 1, "excluded": 0, "retained": 1}}


@pytest.mark.parametrize("mode", ["prompt_data", "rl_manifest"])
def test_single_input_can_exclude_entire_dataset_label(environment, tmp_path, caplog, mode):
    caplog.set_level("INFO", logger=run_data.logger.name)
    if mode == "prompt_data":
        data = prepared_input(
            tmp_path, [row("short", source_dataset="short"), row("x" * 201, source_dataset="long-docs")]
        )
    else:
        path = test_miles_run_data.source_manifest(tmp_path)
        artifact = path.parent / "train.jsonl"
        short = json.loads(artifact.read_text())
        short["metadata"]["source_dataset"] = "short"
        long = copy.deepcopy(short)
        long["messages"][0]["content"] = "x" * 201
        long["metadata"].update(source_dataset="long-docs", prepared_sample_id="long")
        artifact.write_text(json.dumps(short) + "\n" + json.dumps(long) + "\n")
        manifest = json.loads(path.read_text())
        manifest["artifacts"]["train"].update(records=2, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        path.write_text(json.dumps(manifest))
        data = {"rl_manifest": str(path)}
    data["over_length"] = "exclude"
    result = prepare(environment, data, limit=100)
    counts = json.loads(Path(result["manifest"]).read_text())["prompt_length"]["sources"]
    assert counts == {
        "short": {"original": 1, "excluded": 0, "retained": 1},
        "long-docs": {"original": 1, "excluded": 1, "retained": 0},
    }
    assert "source=long-docs original=1 excluded=1 retained=0" in caplog.text
    assert len(Path(result["prompt_data"]).read_text().splitlines()) == 1
    assert prepare(environment, data, limit=100) == result


@pytest.mark.parametrize("shared_dataset", [False, True])
def test_task_dataset_labels_are_reporting_only(environment, monkeypatch, shared_dataset):
    if shared_dataset:
        monkeypatch.setitem(run_data.TASKS, "math", (*run_data.TASKS["gsm8k"][:2], "math"))
    monkeypatch.setattr(
        run_data,
        "_source_rows",
        lambda name: [{"question": "short" if name == "gsm8k" else "x" * 201, "ground_truth": "42"}],
    )
    data = {
        "over_length": "exclude",
        "tasks": [{"task": "gsm8k", "train_count": 1}, {"task": "math", "train_count": 1}],
    }
    result = prepare(environment, data, limit=100)
    counts = json.loads(Path(result["manifest"]).read_text())["prompt_length"]["sources"]
    if shared_dataset:
        assert list(counts.values()) == [{"original": 2, "excluded": 1, "retained": 1}]
    else:
        assert counts[run_data.TASKS["math"][0]] == {"original": 1, "excluded": 1, "retained": 0}
        assert counts[run_data.TASKS["gsm8k"][0]] == {"original": 1, "excluded": 0, "retained": 1}
    assert len(Path(result["prompt_data"]).read_text().splitlines()) == 1


@pytest.mark.parametrize("mode", ["prompt_data", "rl_manifest", "tasks"])
def test_exclusion_cannot_empty_overall_training_input(environment, tmp_path, monkeypatch, mode):
    if mode == "prompt_data":
        data = prepared_input(tmp_path, [row("x" * 201, source_dataset="long-docs")])
        evaluation = tmp_path / "eval.jsonl"
        evaluation.write_text(json.dumps(row("heldout")) + "\n")
        data["eval_prompt_data"] = ["heldout", str(evaluation)]
    elif mode == "rl_manifest":
        path = test_miles_run_data.source_manifest(tmp_path)
        artifact = path.parent / "train.jsonl"
        record = json.loads(artifact.read_text())
        record["messages"][0]["content"] = "x" * 201
        record["metadata"]["source_dataset"] = "long-docs"
        artifact.write_text(json.dumps(record) + "\n")
        manifest = json.loads(path.read_text())
        manifest["artifacts"]["train"]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest))
        data = {"rl_manifest": str(path)}
    else:
        monkeypatch.setattr(run_data, "_source_rows", lambda name: [{"question": "x" * 201, "ground_truth": "42"}])
        data = {"tasks": [{"task": "gsm8k", "train_count": 1}]}
    with pytest.raises(InputError, match="Training data has no rows after prompt-length exclusion"):
        prepare(environment, data | {"over_length": "exclude"}, limit=100)
    assert not environment[1].exists()


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"source_dataset": "dataset", "source_name": "route"}, "dataset"),
        ({"source_name": "route"}, "route"),
        ({"source_dataset": None, "source_name": "route"}, "route"),
        ({"source_dataset": "", "source_name": "route"}, "route"),
        ({"source_dataset": [], "source_name": "route"}, "route"),
        ({}, "train"),
        ({"source_dataset": None, "source_name": None}, "train"),
        ({"source_dataset": {}, "source_name": []}, "train"),
        ({"source_dataset": "", "source_name": ""}, "train"),
    ],
)
def test_source_accounting_falls_back_without_requiring_metadata(environment, tmp_path, metadata, expected):
    data = prepared_input(tmp_path, [row("valid", **metadata)])
    result = prepare(environment, data)
    counts = json.loads(Path(result["manifest"]).read_text())["prompt_length"]["sources"]
    assert counts == {expected: {"original": 1, "excluded": 0, "retained": 1}}


def test_old_workbench_source_names_keep_exclusion_breakdown(environment, tmp_path, caplog):
    caplog.set_level("INFO", logger=run_data.logger.name)
    data = prepared_input(tmp_path, [row("short", source_name="c040"), row("x" * 11, source_name="lc-golong")])
    result = prepare(environment, data | {"over_length": "exclude"})
    counts = json.loads(Path(result["manifest"]).read_text())["prompt_length"]["sources"]
    assert counts == {
        "c040": {"original": 1, "excluded": 0, "retained": 1},
        "lc-golong": {"original": 1, "excluded": 1, "retained": 0},
    }
    assert "source=lc-golong original=1 excluded=1 retained=0" in caplog.text


def test_configured_source_name_takes_precedence_over_metadata(environment, tmp_path):
    source = prepared_input(tmp_path, [row("valid", source_dataset="dataset", source_name="route")])
    result = prepare(environment, {"sources": [{"name": "configured", "fraction": 1.0, **source}]})
    counts = json.loads(Path(result["manifest"]).read_text())["prompt_length"]["sources"]
    assert counts == {"configured": {"original": 1, "excluded": 0, "retained": 1}}


def test_zero_token_training_error_does_not_suggest_length_exclusion(environment, tmp_path, monkeypatch):
    data = prepared_input(tmp_path, [row("nonempty")])
    monkeypatch.setattr(environment[2], "encode", lambda *args, **kwargs: [])
    with pytest.raises(InputError, match="Prompt has 0 tokens") as error:
        prepare(environment, data)
    assert "data.over_length" not in str(error.value)
