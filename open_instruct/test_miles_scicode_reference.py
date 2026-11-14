"""SciCode reference assets must be resolved before a job is allocated."""

import hashlib
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.evaluation import evaluation, evaluation_runner
from open_instruct.miles.execution import launch, workflow

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    [
        None,
        "",
        " ",
        " test.h5",
        "test.h5",
        "~/test.h5",
        "/weka/YOUR_USERNAME/test.h5",
        "/weka/YOUR_BUCKET/test.h5",
        "/weka/${USER}/test.h5",
        "/weka/<user>/test.h5",
    ],
)
def test_unresolved_scicode_reference_rejected(path):
    task = {"task": "scicode", "kind": "external", "arguments": {"h5py_host_path": path}}
    with pytest.raises(InputError, match="explicit absolute reference-data file path"):
        evaluation.validate_scicode_reference_path({"tasks": [task]})


def test_remote_reference_does_not_need_to_exist_on_submitter():
    evaluation.validate_scicode_reference_path(
        {
            "tasks": [
                {"task": "scicode", "kind": "external", "arguments": {"h5py_host_path": "/weka/assets/test_data.h5"}}
            ]
        }
    )
    evaluation.validate_scicode_reference_path({"tasks": [{"task": "gsm8k"}]})


def example_with_placeholder():
    document = RunSpec.load(ROOT / "configs/miles/examples/tiny/three-node.toml").to_dict()
    task = next(task for task in document["evaluation"]["tasks"] if task["task"] == "scicode")
    task["arguments"]["h5py_host_path"] = "/weka/oe-adapt-default/YOUR_USERNAME/scicode/test_data.h5"
    return RunSpec.from_dict(document)


@pytest.mark.parametrize("boundary", ["launch", "execute", "evaluation"])
def test_explicit_placeholder_fails_before_launch_or_preparation(boundary, monkeypatch):
    run = example_with_placeholder()
    plan = run.plan()
    warning = next(warning for warning in plan["launch_warnings"] if "h5py_host_path" in warning)
    assert "Remove arguments.h5py_host_path to use the default" in warning
    assert "SCICODE_H5PY_HOST_PATH Beaker secret" in warning
    prepare = Mock(side_effect=AssertionError("must not prepare model"))
    monkeypatch.setattr(workflow, "prepare_model", prepare)
    with pytest.raises(InputError, match="Remove arguments.h5py_host_path"):
        if boundary == "launch":
            launch.specification("unused-image", run)
        elif boundary == "execute":
            workflow.execute(run)
        else:
            # Asset validation precedes the remaining receipt/provenance checks.
            evaluation.specification({"evaluation": run.evaluation, "tasks": run.evaluation["tasks"]})
    prepare.assert_not_called()


def test_failed_receipt_keeps_actionable_reference_error(tmp_path):
    run = example_with_placeholder()
    receipt = {"update": 0, "evaluation": run.evaluation, "tasks": run.evaluation["tasks"]}
    evaluation.submit(receipt, tmp_path / "receipt.json")
    assert receipt["status"] == "submission_failed"
    assert receipt["error"] == "InputError"
    assert "Remove arguments.h5py_host_path" in receipt["diagnostic"]


def test_submission_log_keeps_non_validation_error_type(tmp_path, monkeypatch):
    warning = Mock()
    monkeypatch.setattr(evaluation.logger, "warning", warning)
    receipt = {"update": 100, "checkpoint": str(tmp_path / "missing-snapshot")}
    evaluation.submit(receipt, tmp_path / "receipt.json")
    assert receipt["error"] == "ValueError"
    assert warning.call_args.args[2].startswith("ValueError: Submission failed")


def test_default_example_injects_secret_and_launches_without_path():
    run = RunSpec.load(ROOT / "configs/miles/examples/tiny/three-node.toml")
    task = next(task for task in run.evaluation["tasks"] if task["task"] == "scicode")
    assert "h5py_host_path" not in task["arguments"]
    assert run.evaluation["secrets"]["SCICODE_H5PY_HOST_PATH"] == "SCICODE_H5PY_HOST_PATH"
    assert not any("h5py_host_path" in warning for warning in run.plan()["launch_warnings"])
    launch.specification("unused-image", run)
    config = evaluation.runtime(run, run.compile().miles)
    receipt = {
        "evaluation": config,
        "tasks": [task],
        "runner_sha256": hashlib.sha256(Path(evaluation_runner.__file__).read_bytes()).hexdigest(),
        "training": {"name": "trial"},
        "update": 0,
    }
    document = evaluation.specification(receipt)
    assert {"name": "SCICODE_H5PY_HOST_PATH", "secret": "SCICODE_H5PY_HOST_PATH"} in document["tasks"][0]["envVars"]


def test_secret_override_and_explicit_path():
    document = RunSpec.load(ROOT / "configs/miles/examples/tiny/three-node.toml").to_dict()
    document["evaluation"]["secrets"]["SCICODE_H5PY_HOST_PATH"] = "custom-reference-secret"
    assert RunSpec.from_dict(document).evaluation["secrets"]["SCICODE_H5PY_HOST_PATH"] == "custom-reference-secret"
    document["evaluation"]["secrets"].pop("SCICODE_H5PY_HOST_PATH")
    task = next(task for task in document["evaluation"]["tasks"] if task["task"] == "scicode")
    task["arguments"]["h5py_host_path"] = "/weka/oe-training-default/test_data.h5"
    assert "SCICODE_H5PY_HOST_PATH" not in RunSpec.from_dict(document).evaluation["secrets"]
    document["evaluation"]["tasks"] = [task for task in document["evaluation"]["tasks"] if task["task"] != "scicode"]
    assert "SCICODE_H5PY_HOST_PATH" not in RunSpec.from_dict(document).evaluation["secrets"]


def test_runner_resolves_secret_into_external_command(tmp_path, monkeypatch):
    reference = tmp_path / "test_data.h5"
    reference.touch()
    monkeypatch.setenv("SCICODE_H5PY_HOST_PATH", str(reference))
    task = {"task": "scicode", "kind": "external", "arguments": {"split": "validation"}}
    identity_before = evaluation_runner.saved_task(tmp_path, task)[0]
    resolved = {
        **task,
        "arguments": {**task["arguments"], "h5py_host_path": evaluation_runner.resolve_scicode_reference(task)},
    }
    command = evaluation_runner.command({"checkpoint": "/hf", "tasks": [resolved]}, tmp_path)
    assert f'h5py_host_path="{reference}"' in command
    assert "h5py_host_path" not in task["arguments"]
    assert evaluation_runner.saved_task(tmp_path, task)[0] == identity_before
    # Explicit config wins over the environment.
    monkeypatch.setenv("SCICODE_H5PY_HOST_PATH", "/missing/secret-value")
    task["arguments"]["h5py_host_path"] = str(reference)
    assert evaluation_runner.resolve_scicode_reference(task) == str(reference)


@pytest.mark.parametrize("path", [None, "/missing/reference.h5", "relative.h5"])
def test_runner_reports_unavailable_secret_without_echoing_value(path, monkeypatch):
    monkeypatch.delenv("SCICODE_H5PY_HOST_PATH", raising=False)
    if path is not None:
        monkeypatch.setenv("SCICODE_H5PY_HOST_PATH", path)
    with pytest.raises(ValueError, match="Check the evaluator's SCICODE_H5PY_HOST_PATH Beaker secret") as error:
        evaluation_runner.resolve_scicode_reference({"task": "scicode", "arguments": {}})
    if path:
        assert path not in str(error.value)


def test_secret_resolution_preserves_completed_task_resume(tmp_path, monkeypatch):
    reference = tmp_path / "reference.h5"
    reference.touch()
    monkeypatch.setenv("SCICODE_H5PY_HOST_PATH", str(reference))
    command = Mock(return_value=["unused-evaluator"])
    monkeypatch.setattr(evaluation_runner, "command", command)
    monkeypatch.setattr(evaluation_runner.subprocess, "run", Mock())
    monkeypatch.setattr(evaluation_runner, "startup_progress", lambda path: nullcontext())
    monkeypatch.setattr(evaluation_runner, "gpu_environments", lambda receipt: (None, {}))
    monkeypatch.setattr(evaluation_runner, "completed_scores", lambda output, tasks: {"eval/scicode/score": 1.0})
    task = {"task": "scicode", "kind": "external", "arguments": {"split": "validation"}}
    receipt = {"tasks": [task]}
    evaluation_runner.run_tasks(receipt, tmp_path)
    passed_task = command.call_args.args[0]["tasks"][0]
    assert passed_task["arguments"]["h5py_host_path"] == str(reference)
    assert "h5py_host_path" not in task["arguments"]
    assert evaluation_runner.saved_task(tmp_path, task)[1] is not None
    evaluation_runner.run_tasks(receipt, tmp_path)
    assert command.call_count == 1
