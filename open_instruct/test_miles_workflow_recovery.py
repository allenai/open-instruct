"""Recovery policy changes must not weaken the saved recipe identity."""

import copy
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.evaluation import evaluation
from open_instruct.miles.execution import workflow


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


@pytest.mark.parametrize(
    "section,key,value",
    [("optimizer", "learning_rate", 2e-6), ("trainer", "gpus", 4), ("launch", "memory", "128 GiB")],
)
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


def test_old_run_resumes_with_historical_host_memory_default(tmp_path):
    document, _ = saved_run(tmp_path)
    document["launch"].update(auto_resume=True, memory="256 GiB")
    spec = SimpleNamespace(
        output=document["output"], launch=document["launch"], to_dict=lambda: document, plan=lambda: {}
    )
    with workflow.run_directory(spec):
        assert json.loads((tmp_path / "run-spec.json").read_text()) == document
    assert json.loads((tmp_path / "workflow.json").read_text())["spec_sha256"] == workflow.fingerprint(document)


def test_memory_normalization_cannot_authenticate_a_modified_saved_spec(tmp_path):
    document, previous = saved_run(tmp_path)
    document["launch"]["memory"] = "256 GiB"
    workflow.write_json(tmp_path / "run-spec.json", document)
    document["launch"]["auto_resume"] = True
    assert not workflow.recovery_configuration_matches(tmp_path, previous, document)


def test_explicit_saved_memory_cannot_be_replaced_by_default(tmp_path):
    document, previous = saved_run(tmp_path)
    document["launch"]["memory"] = "128 GiB"
    previous["spec_sha256"] = workflow.fingerprint(document)
    workflow.write_json(tmp_path / "run-spec.json", document)
    document["launch"].update(auto_resume=True, memory="256 GiB")
    assert not workflow.recovery_configuration_matches(tmp_path, previous, document)


def test_recovery_does_not_restart_completed_runs(tmp_path):
    document, _ = saved_run(tmp_path, status="complete")
    document["launch"]["auto_resume"] = True
    spec = SimpleNamespace(output=document["output"], launch=document["launch"], to_dict=lambda: document)
    with pytest.raises(InputError, match="already completed"), workflow.run_directory(spec):
        pytest.fail("Completed run was entered")


def saved_cached_run(tmp_path):
    document, previous = saved_run(tmp_path)
    document["compiler_cache"] = {"enabled": True, "shared_root": "/shared/cache"}
    document["core"] = {"compiler_cache_root": "/shared/cache", "seed": 17}
    previous["spec_sha256"] = workflow.fingerprint(document)
    workflow.write_json(tmp_path / "run-spec.json", document)
    return document, previous


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["compiler_cache"].update(shared_root="/local/cache"),
        lambda d: d["compiler_cache"].update(enabled=False, tilelang=True),
        lambda d: d["core"].pop("compiler_cache_root"),
        lambda d: d["core"].update(compiler_cache_restore=False),
        lambda d: d.pop("compiler_cache"),
    ],
)
def test_recovery_ignores_compiler_cache_settings(tmp_path, change):
    document, previous = saved_cached_run(tmp_path)
    document["launch"]["auto_resume"] = True
    change(document)
    assert workflow.recovery_configuration_matches(tmp_path, previous, document)


def test_compiler_cache_change_cannot_hide_recipe_change(tmp_path):
    document, previous = saved_cached_run(tmp_path)
    document["compiler_cache"]["shared_root"] = "/local/cache"
    document["core"]["seed"] = 18
    assert not workflow.recovery_configuration_matches(tmp_path, previous, document)


def test_compiler_cache_change_cannot_authenticate_a_modified_saved_spec(tmp_path):
    document, previous = saved_cached_run(tmp_path)
    changed = copy.deepcopy(document)
    changed["compiler_cache"]["shared_root"] = "/local/cache"
    workflow.write_json(tmp_path / "run-spec.json", changed)
    assert not workflow.recovery_configuration_matches(tmp_path, previous, changed)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("inference", "profile", "olmo35-b300-12.5@0.8"),
        ("miles", "pause_generation_mode", "in_place"),
        ("miles", "sglang_disable_radix_cache", True),
    ],
)
def test_recovery_rejects_changed_serving_recipe(tmp_path, section, key, value):
    document, previous = saved_run(tmp_path)
    document["inference"] = {"profile": "olmo35-b300-v1"}
    document["miles"] = {"pause_generation_mode": "retract", "sglang_disable_radix_cache": False}
    previous["spec_sha256"] = workflow.fingerprint(document)
    workflow.write_json(tmp_path / "run-spec.json", document)
    workflow.write_json(tmp_path / "workflow.json", previous)
    document["launch"]["auto_resume"] = True
    document[section][key] = value
    spec = SimpleNamespace(output=document["output"], launch=document["launch"], to_dict=lambda: document)
    with pytest.raises(InputError, match="Run configuration changed"), workflow.run_directory(spec):
        pytest.fail("Changed serving settings must not enter the existing run")


@pytest.mark.parametrize(
    "invalid_change", [None, "lr", "horizon", "task", "sampling", "scoring", "task_count", "final", "saved_spec"]
)
def test_recovery_changes_artifact_cadence_only(tmp_path, invalid_change):
    document, previous = saved_run(tmp_path)
    document["training"] = {"save_interval": 25, "num_rollouts": 200}
    document["evaluation"] = {"interval": 0, "tasks": [{"task": "ifeval", "interval": 25}]}
    previous["spec_sha256"] = workflow.fingerprint(document)
    workflow.write_json(tmp_path / "run-spec.json", document)
    if invalid_change == "saved_spec":
        corrupt = copy.deepcopy(document)
        corrupt["training"]["save_interval"] = 50
        workflow.write_json(tmp_path / "run-spec.json", corrupt)
    document["training"]["save_interval"] = 100
    document["evaluation"]["interval"] = 50
    document["evaluation"]["tasks"][0]["interval"] = 50
    if invalid_change == "lr":
        document["optimizer"]["learning_rate"] = 2e-6
    elif invalid_change == "horizon":
        document["training"]["num_rollouts"] = 300
    elif invalid_change == "task":
        document["evaluation"]["tasks"][0]["task"] = "aime"
    elif invalid_change == "sampling":
        document["evaluation"]["tasks"][0]["generation"] = {"temperature": 0.5}
    elif invalid_change == "scoring":
        document["evaluation"]["tasks"][0]["scoring"] = {"judge": "different"}
    elif invalid_change == "task_count":
        document["evaluation"]["tasks"].append({"task": "aime", "interval": 50})
    elif invalid_change == "final":
        document["evaluation"]["final"] = False
    assert workflow.recovery_configuration_matches(tmp_path, previous, document) == (invalid_change is None)


@pytest.mark.parametrize("interval", [0, 50])
def test_cadence_resume_persists_normalized_spec_and_preserves_artifacts(tmp_path, interval):
    document = {
        "schema_version": 1,
        "name": "cadence-resume",
        "model": {"source": str(tmp_path / "model"), "format": "hf"},
        "output": {"root": str(tmp_path / "run")},
        "data": {"tasks": [{"task": "multiplication", "train_count": 8}]},
        "training": {"num_rollouts": 200, "save_interval": 25},
        "evaluation": {
            "mode": "background",
            "interval": 25,
            "image": "01M2XGZM2N1V4DQVMYHM52KBHZ",
            "revision": "a" * 40,
            "tasks": [{"task": "gsm8k"}, {"task": "arc_easy", "interval": 75}],
        },
        "launch": {"auto_resume": True},
    }
    original = RunSpec.from_dict(document)
    with workflow.run_directory(original) as (root, _):
        artifacts = [root / "checkpoints" / "step25", root / "evaluation" / "existing.json"]
        for artifact in artifacts:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"existing artifact")

    document["training"]["save_interval"] = 100
    document["evaluation"]["interval"] = interval
    resumed = RunSpec.from_dict(document)
    assert resumed.compile().miles["save_interval"] == 100
    assert [task["interval"] for task in resumed.evaluation["tasks"]] == [interval, 75]
    assert evaluation.groups(resumed.evaluation, 25, 200) == []
    assert bool(evaluation.groups(resumed.evaluation, 50, 200)) == (interval == 50)
    assert len(evaluation.groups(resumed.evaluation, 200, 200)[0]) == 2
    for attempt in (1, 2):
        with workflow.run_directory(resumed) as (root, state):
            assert state["attempt"] == attempt
            assert json.loads((root / "run-spec.json").read_text()) == resumed.to_dict()
            assert json.loads((root / "workflow.json").read_text())["spec_sha256"] == workflow.fingerprint(
                resumed.to_dict()
            )
            assert all(artifact.read_bytes() == b"existing artifact" for artifact in artifacts)
