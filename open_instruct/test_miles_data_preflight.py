"""Catch dataset/binding mismatches before an image build or GPU submission."""

import copy
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from open_instruct.miles import __main__ as cli
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.datasets import run_data
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import data_preflight, launch, submit
from open_instruct.miles.rewards import judge_data


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


def test_direct_submission_checks_before_beaker(document, tmp_path, monkeypatch, capsys):
    document["judging"]["bindings"]["unused"] = {"judge": "general", "rubric": "quality"}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(sys, "argv", ["submit", "unused-image", str(path), "--check-data"])
    monkeypatch.setattr(submit.launch, "submit", lambda *_: pytest.fail("Beaker must not be called"))
    with pytest.raises(SystemExit) as error:
        submit.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "Judge bindings absent" in stderr
    assert "Traceback" not in stderr


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


@pytest.mark.parametrize("compressed", [False, True])
def test_blank_lines_match_training_reader_behavior(document, tmp_path, compressed):
    original = Path(document["data"]["prompt_data"]).read_text()
    path = tmp_path / ("with-blanks.jsonl.gz" if compressed else "with-blanks.jsonl")
    opener = gzip.open if compressed else open
    with opener(path, "wt") as stream:
        stream.write("\n  \t\n" + original + "\n\n")
    document["data"]["prompt_data"] = str(path)
    assert data_preflight.check(RunSpec.from_dict(document))["rows"] == 1


@pytest.mark.parametrize("mode", ["symlink", "dotdot"])
def test_mapping_normalizes_configured_paths_too(document, tmp_path, mode):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    prefix = alias if mode == "symlink" else tmp_path / "unused" / ".."
    document["data"] = {"prompt_data": str(prefix / "train.jsonl"), "reward_config": str(prefix / "verifiers.json")}
    assert data_preflight.check(RunSpec.from_dict(document), [f"{prefix}={tmp_path}"])["rows"] == 1


def test_unmatched_mapping_fails_instead_of_reading_original_file(document):
    with pytest.raises(InputError, match="does not match any configured input path"):
        data_preflight.check(RunSpec.from_dict(document), ["/typo-prefix=/some-copy"])


@pytest.mark.parametrize("configured_equals,local_equals", [(True, False), (False, True), (True, True)])
def test_equals_in_mapping_paths(document, tmp_path, configured_equals, local_equals):
    prefix = Path("/unmounted/split=train" if configured_equals else "/unmounted/data")
    local = tmp_path / ("local=copy" if local_equals else "local-copy")
    local.mkdir()
    for name in ("train.jsonl", "verifiers.json"):
        (local / name).write_bytes((tmp_path / name).read_bytes())
    document["data"] = {"prompt_data": str(prefix / "train.jsonl"), "reward_config": str(prefix / "verifiers.json")}
    result = data_preflight.check(RunSpec.from_dict(document), [f"{prefix}={local}"])
    assert result["rows"] == 1
    assert result["paths"] == [str(local / "train.jsonl")]


@pytest.mark.parametrize(
    "registry",
    [
        {},
        {"quality": {}},
        {"quality": {"factory": "NoModule"}},
        {"quality": {"factory": "module.Class", "grading": {"scope": "invalid"}}},
    ],
)
def test_registry_validation_matches_preparation(document, registry):
    Path(document["data"]["reward_config"]).write_text(json.dumps(registry))
    with pytest.raises(InputError) as training:
        run_data._prepared(document["data"], {})
    with pytest.raises(InputError, match=str(training.value)):
        data_preflight.check(RunSpec.from_dict(document))


@pytest.mark.parametrize("entrypoint", ["validate", "run", "submit"])
def test_bad_topology_fails_before_scanning_rows(document, tmp_path, monkeypatch, capsys, entrypoint):
    document["launch"] = {"gpus_per_replica": 1}
    document["trainer"] = {"gpus": 1, "expert_parallel_size": 1}
    document["inference"] = {"gpus": 1, "placement_mode": "disaggregated"}
    document["judges"]["general"] = {
        "mode": "managed",
        "model": "Qwen/Qwen3-32B",
        "gpus": 2,
        "tensor_parallel_size": 2,
        "revision": "a" * 40,
        "prepared_dir": str(tmp_path / "judge"),
    }
    path = tmp_path / "bad-topology.json"
    path.write_text(json.dumps(document))
    RunSpec.from_dict(document).compile()  # Allocation validation is a separate plan() check.
    monkeypatch.setattr(judge_data, "data_rows", lambda *_: pytest.fail("must validate topology before scanning data"))
    args = (
        ["submit", "unused-image", str(path), "--check-data"]
        if entrypoint == "submit"
        else ["miles", entrypoint, str(path), "--check-data"]
    )
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        (submit.main if entrypoint == "submit" else cli.main)()
    assert error.value.code == 2
    assert "Each managed judge must fit within one replica" in capsys.readouterr().err


@pytest.mark.parametrize("existing_image", [False, True])
def test_wrapper_rejects_data_before_any_image_build_or_lookup(document, tmp_path, existing_image):
    document["judging"]["bindings"]["unused"] = {"judge": "general", "rubric": "quality"}
    config = tmp_path / "bad data.json"
    config.write_text(json.dumps(document))
    tools = tmp_path / "tools"
    tools.mkdir()
    forbidden = tmp_path / "external-command-called"
    for name in ("git", "beaker", "docker"):
        program = (
            "import sys\n"
            if name == "git"
            else f"from pathlib import Path\nPath({str(forbidden)!r}).touch()\nraise SystemExit(99)\n"
        )
        target = tools / name
        target.write_text(f"#!{sys.executable}\n" + program)
        target.chmod(0o755)
    (tools / "python").symlink_to(sys.executable)
    env = {**os.environ, "PATH": str(tools) + os.pathsep + os.defpath, "MILES_BASE_IMAGE": "unused-base"}
    env.pop("MILES_EXISTING_IMAGE", None)
    if existing_image:
        env["MILES_EXISTING_IMAGE"] = "01M2931KARFP3Y2W2FPADGRFEP"
    result = subprocess.run(
        ["/bin/bash", "scripts/train/build_image_and_launch.sh", "--miles", str(config), "--check-data"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "Judge bindings absent" in result.stderr
    assert "Traceback" not in result.stderr
    assert not forbidden.exists()
