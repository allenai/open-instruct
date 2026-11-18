"""Catch dataset/binding mismatches before an image build or GPU submission."""

import copy
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from open_instruct.miles import __main__ as cli
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import data_preflight, launch, submit


@pytest.fixture
def document(tmp_path):
    registry = tmp_path / "verifiers.json"
    registry.write_text(
        json.dumps({"quality": {"factory": "open_instruct.miles.rewards.judge_registry.NamedJudgeVerifier"}})
    )
    data = tmp_path / "train.jsonl"
    data.write_text(
        json.dumps({"metadata": {"judge_query": "Question", "verifiers": [{"name": "quality", "target": "Answer"}]}})
        + "\n"
    )
    return {
        "schema_version": 1,
        "name": "preflight",
        "model": {"source": str(tmp_path / "unavailable-model")},
        "output": {"root": str(tmp_path / "output")},
        "data": {"prompt_data": str(data), "reward_config": str(registry)},
        "judges": {"general": {"mode": "external", "model": "unused", "endpoint": "http://unused.invalid/v1"}},
        "rubrics": {"quality": {"profile": "open-instruct/general-quality"}},
        "judging": {"bindings": {"quality": {"judge": "general", "rubric": "quality"}}},
    }


def test_reject_unused_binding_even_when_present_in_registry(document):
    document["judging"]["bindings"]["unused"] = {"judge": "general", "rubric": "quality"}
    path = Path(document["data"]["reward_config"])
    registry = json.loads(path.read_text())
    registry["unused"] = registry["quality"]
    path.write_text(json.dumps(registry))
    with pytest.raises(InputError, match="Judge bindings absent.*unused"):
        data_preflight.check(RunSpec.from_dict(document))


def test_reject_missing_hashed_named_judge_binding(document):
    document["judging"]["bindings"] = {}
    document.pop("judges")
    document.pop("rubrics")
    with pytest.raises(InputError, match="Missing named judge binding.*quality"):
        data_preflight.check(RunSpec.from_dict(document))


@pytest.mark.parametrize("broken", ["query", "reference", "abstention"])
def test_reuse_runtime_query_and_target_contracts(document, broken):
    path = Path(document["data"]["prompt_data"])
    row = json.loads(path.read_text())
    if broken == "query":
        row["metadata"].pop("judge_query")
    elif broken == "reference":
        document["rubrics"]["quality"]["profile"] = "open-instruct/general-quality_ref"
        row["metadata"]["verifiers"][0]["target"] = None
    else:
        row["metadata"]["verifiers"][0]["target"] = {"kind": "nemo_abstention_v1"}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(InputError):
        data_preflight.check(RunSpec.from_dict(document))


def test_binding_used_only_by_another_sources_eval_is_valid(document, tmp_path):
    first = document.pop("data")
    second = tmp_path / "second.jsonl"
    second.write_text('{"metadata":{"verifiers":[]}}\n')
    evaluation = tmp_path / "eval.jsonl.gz"
    with gzip.open(evaluation, "wt") as stream:
        row = json.loads(Path(first["prompt_data"]).read_text())
        row["metadata"]["verifiers"][0]["name"] = "eval-judge"
        stream.write(json.dumps(row) + "\n")
    document["judging"]["bindings"]["eval-judge"] = {"judge": "general", "rubric": "quality"}
    document["data"] = {
        "sources": [
            dict(first, name="one", fraction=0.5),
            dict(
                first, prompt_data=str(second), eval_prompt_data=["heldout", str(evaluation)], name="two", fraction=0.5
            ),
        ]
    }
    result = data_preflight.check(RunSpec.from_dict(document))
    assert result["rows"] == 3
    assert result["judge_bindings"] == ["eval-judge", "quality"]


def test_local_mapping_preserves_frozen_run_paths(document, tmp_path):
    original = copy.deepcopy(document)
    original["data"] = {
        "prompt_data": "/unmounted/inputs/train.jsonl",
        "reward_config": "/unmounted/inputs/verifiers.json",
    }
    spec = RunSpec.from_dict(original)
    result = data_preflight.check(spec, [f"/unmounted/inputs={tmp_path}"])
    assert result["rows"] == 1
    assert spec.data["prompt_data"] == "/unmounted/inputs/train.jsonl"


@pytest.mark.parametrize("mapping", ["bad", "=local", "/original="])
def test_reject_malformed_mappings(document, mapping):
    with pytest.raises(InputError, match="CONFIGURED_PATH=LOCAL_PATH"):
        data_preflight.check(RunSpec.from_dict(document), [mapping])


def test_missing_file_is_not_a_successful_check(document):
    Path(document["data"]["prompt_data"]).unlink()
    with pytest.raises(InputError, match="Cannot read judge data"):
        data_preflight.check(RunSpec.from_dict(document))


def test_bad_json_has_path_and_line_number(document):
    path = Path(document["data"]["prompt_data"])
    path.write_text(path.read_text() + "{bad}\n")
    with pytest.raises(InputError, match=r"train.jsonl:2"):
        data_preflight.check(RunSpec.from_dict(document))


def test_reject_conflicting_source_registries(document, tmp_path):
    first = document.pop("data")
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"quality": {"factory": "other.Verifier"}}))
    document["data"] = {
        "sources": [
            dict(first, name="one", fraction=0.5),
            dict(first, reward_config=str(other), name="two", fraction=0.5),
        ]
    }
    with pytest.raises(InputError, match="Conflicting reward registry"):
        data_preflight.check(RunSpec.from_dict(document))


def test_unprepared_inputs_are_not_claimed_as_checked(document):
    document["data"] = {"tasks": [{"task": "gsm8k", "train_count": 2}]}
    with pytest.raises(InputError, match="requires prepared"):
        data_preflight.check(RunSpec.from_dict(document))


def test_run_checks_before_invoking_image_wrapper(document, tmp_path, monkeypatch):
    document["judging"]["bindings"]["unused"] = {"judge": "general", "rubric": "quality"}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(sys, "argv", ["miles", "run", str(path), "--check-data"])
    monkeypatch.setattr(launch, "run", lambda *_: pytest.fail("Image wrapper must not be called"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not Path(document["output"]["root"]).exists()


def test_direct_submission_checks_before_beaker(document, tmp_path, monkeypatch):
    document["judging"]["bindings"]["unused"] = {"judge": "general", "rubric": "quality"}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(sys, "argv", ["submit", "unused-image", str(path), "--check-data"])
    monkeypatch.setattr(submit.launch, "submit", lambda *_: pytest.fail("Beaker must not be called"))
    with pytest.raises(InputError, match="Judge bindings absent"):
        submit.main()


def test_validate_with_mapping_checks_without_loading_policy(document, tmp_path, monkeypatch, capsys):
    document["data"] = {
        "prompt_data": "/unmounted/inputs/train.jsonl",
        "reward_config": "/unmounted/inputs/verifiers.json",
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(
        sys, "argv", ["miles", "validate", str(path), "--check-data", "--data-path", f"/unmounted/inputs={tmp_path}"]
    )
    cli.main()
    assert '"status": "passed"' in capsys.readouterr().out
    assert not Path(document["model"]["source"]).exists()


@pytest.mark.parametrize("arguments", [["plan", "--check-data"], ["validate", "--data-path", "/remote=/local"]])
def test_reject_unsupported_flag_combinations(document, tmp_path, monkeypatch, arguments):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(sys, "argv", ["miles", arguments[0], str(path), *arguments[1:]])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_more_specific_path_mapping_wins(document, tmp_path):
    spec = RunSpec.from_dict(document)
    result = data_preflight.check(spec, [f"{tmp_path.parent}=/missing", f"{tmp_path}={tmp_path}"])
    assert result["status"] == "passed"


def test_duplicate_path_mapping_is_rejected(document, tmp_path):
    with pytest.raises(InputError, match="Duplicate --data-path"):
        data_preflight.check(RunSpec.from_dict(document), [f"{tmp_path}={tmp_path}"] * 2)


def test_preflight_works_without_site_packages(document, tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document))
    result = subprocess.run(
        [sys.executable, "-S", "-m", "open_instruct.miles", "validate", str(path), "--check-data"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert '"status": "passed"' in result.stdout
