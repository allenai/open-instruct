"""SciCode reference assets must be resolved before a job is allocated."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.evaluation import evaluation
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


@pytest.mark.parametrize("boundary", ["launch", "execute", "evaluation"])
def test_example_placeholder_fails_before_launch_or_preparation(boundary, monkeypatch):
    run = RunSpec.load(ROOT / "configs/miles/examples/tiny/three-node.toml")
    # Planning the template is useful before its deployment paths are filled in.
    plan = run.plan()
    assert plan["evaluation"]["milestones"]
    assert any("h5py_host_path" in warning for warning in plan["launch_warnings"])
    prepare = Mock(side_effect=AssertionError("must not prepare model"))
    monkeypatch.setattr(workflow, "prepare_model", prepare)
    with pytest.raises(InputError, match="Replace the example YOUR_USERNAME placeholder"):
        if boundary == "launch":
            launch.specification("unused-image", run)
        elif boundary == "execute":
            workflow.execute(run)
        else:
            # An unresolved asset is rejected before the receipt's other fields
            # (runner provenance, checkpoint identity) are read.
            evaluation.specification({"evaluation": run.evaluation, "tasks": run.evaluation["tasks"]})
    prepare.assert_not_called()


def test_failed_receipt_keeps_actionable_reference_error(tmp_path):
    run = RunSpec.load(ROOT / "configs/miles/examples/tiny/three-node.toml")
    receipt = {"update": 0, "evaluation": run.evaluation, "tasks": run.evaluation["tasks"]}
    evaluation.submit(receipt, tmp_path / "receipt.json")
    assert receipt["status"] == "submission_failed"
    assert receipt["error"] == "InputError"
    assert "Replace the example YOUR_USERNAME placeholder" in receipt["diagnostic"]
