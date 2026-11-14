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
        "/weka/${USER}/test.h5",
        "/weka/<user>/test.h5",
    ],
)
def test_unresolved_scicode_reference_rejected(path):
    task = {"task": "scicode", "kind": "external", "arguments": {"h5py_host_path": path}}
    with pytest.raises(InputError, match="explicit absolute reference-data file path"):
        evaluation.validate_reference_paths({"tasks": [task]})


def test_remote_reference_does_not_need_to_exist_on_submitter():
    evaluation.validate_reference_paths(
        {
            "tasks": [
                {"task": "scicode", "kind": "external", "arguments": {"h5py_host_path": "/weka/assets/test_data.h5"}}
            ]
        }
    )
    evaluation.validate_reference_paths({"tasks": [{"task": "gsm8k"}]})


@pytest.mark.parametrize("boundary", ["launch", "execute", "evaluation"])
def test_example_placeholder_fails_before_launch_or_preparation(boundary, monkeypatch):
    run = RunSpec.load(ROOT / "configs/miles/examples/tiny/three-node.toml")
    # Planning the template is useful before its deployment paths are filled in.
    assert run.plan()["evaluation"]["milestones"]
    prepare = Mock(side_effect=AssertionError("must not prepare model"))
    monkeypatch.setattr(workflow, "prepare_model", prepare)
    with pytest.raises(InputError, match="Replace the example YOUR_USERNAME placeholder"):
        if boundary == "launch":
            launch.specification("unused-image", run)
        elif boundary == "execute":
            workflow.execute(run)
        else:
            evaluation.specification({"evaluation": run.evaluation, "tasks": run.evaluation["tasks"]})
    prepare.assert_not_called()
