"""Recovery policy changes must not weaken the saved recipe identity."""

import copy
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles import workflow
from open_instruct.miles.errors import InputError


def saved_run(tmp_path, *, status="failed"):
    document = {
        "launch": {"auto_resume": False},
        "output": {"root": str(tmp_path)},
        "optimizer": {"learning_rate": 1e-6},
        "trainer": {"gpus": 2},
    }
    previous = {"spec_sha256": workflow.fingerprint(document), "status": status}
    workflow.write_json(tmp_path / "run-spec.json", document)
    workflow.write_json(tmp_path / "workflow.json", previous)
    return document, previous


def test_enable_recovery_for_existing_run_and_preserve_new_identity(tmp_path):
    document, _ = saved_run(tmp_path)
    document["launch"]["auto_resume"] = True
    spec = SimpleNamespace(
        output=document["output"], launch=document["launch"], to_dict=lambda: document, plan=lambda: {}
    )
    with workflow.run_directory(spec):
        assert json.loads((tmp_path / "run-spec.json").read_text()) == document
    assert json.loads((tmp_path / "workflow.json").read_text())["spec_sha256"] == workflow.fingerprint(document)


@pytest.mark.parametrize("section,key,value", [("optimizer", "learning_rate", 2e-6), ("trainer", "gpus", 4)])
def test_enabling_recovery_cannot_change_training_recipe(tmp_path, section, key, value):
    document, previous = saved_run(tmp_path)
    document["launch"]["auto_resume"] = True
    document[section][key] = value
    assert not workflow.recovery_configuration_matches(tmp_path, previous, document)


def test_modified_saved_spec_cannot_authorize_recipe_change(tmp_path):
    document, previous = saved_run(tmp_path)
    changed = copy.deepcopy(document)
    changed["optimizer"]["learning_rate"] = 2e-6
    workflow.write_json(tmp_path / "run-spec.json", changed)
    changed["launch"]["auto_resume"] = True
    assert not workflow.recovery_configuration_matches(tmp_path, previous, changed)


def test_recovery_does_not_restart_completed_runs(tmp_path):
    document, _ = saved_run(tmp_path, status="complete")
    document["launch"]["auto_resume"] = True
    spec = SimpleNamespace(output=document["output"], launch=document["launch"], to_dict=lambda: document)
    with pytest.raises(InputError, match="already completed"), workflow.run_directory(spec):
        pytest.fail("Completed run was entered")
