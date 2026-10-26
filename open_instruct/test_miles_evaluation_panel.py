"""Post-training panel configuration, execution boundaries and completion checks."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.evaluation import evaluation, evaluation_runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def large():
    return RunSpec.load(ROOT / "configs/miles/examples/large.toml")


def test_large_panel_preserves_protocols_and_before_after_cadence(large):
    tasks = large.evaluation["tasks"]
    assert len(tasks) == len({task["task"] for task in tasks}) == 25
    assert all(not task["generation"] and not task["scoring"] for task in tasks)
    milestones = large.plan()["evaluation"]["milestones"]
    assert [point["update"] for point in milestones] == [0, 200]
    assert all(sum(map(len, point["groups"])) == 25 for point in milestones)
    assert len(milestones[0]["groups"]) == 5
    # Extending the run never introduces periodic full-panel evaluations.
    document = large.to_dict()
    document["training"]["num_rollouts"] = 2500
    extended = RunSpec.from_dict(document)
    assert [point["update"] for point in extended.plan()["evaluation"]["milestones"]] == [0, 2500]
    for update in (1, 999, 1000, 2000, 2499):
        assert evaluation.groups(extended.evaluation, update, 2500) == []
    assert sum(map(len, evaluation.groups(extended.evaluation, 2500, 2500))) == 25


def test_full_panel_submits_only_endpoints_and_does_not_resubmit_on_resume(large, tmp_path, monkeypatch):
    document = large.to_dict()
    document["output"]["root"] = str(tmp_path / "run")
    document["training"]["num_rollouts"] = 2500
    run = RunSpec.from_dict(document)
    config = evaluation.runtime(run, run.compile().miles)
    args = SimpleNamespace(background_evaluation=config)
    training = {"name": run.name, "wandb": {"mode": "offline", "id": "panel", "entity": "team", "project": "test"}}
    calls = []

    def transport(argv, **kwargs):
        assert argv[1:3] == ["-m", "open_instruct.miles.evaluation.evaluation_submit"]
        assert argv[4] == config["workspace"]
        assert kwargs["check"] and kwargs["timeout"] > 0
        calls.append(json.loads(Path(argv[3]).read_text()))
        return subprocess.CompletedProcess(argv, 0, json.dumps({"id": f"experiment-{len(calls)}"}), "")

    # Exercise real coordination, receipt writing, spec rendering and submission;
    # replace only the subprocess boundary that contacts Beaker.
    monkeypatch.setattr(evaluation.subprocess, "run", transport)
    manager = evaluation.Coordinator(config, training)
    initial = tmp_path / "initial-hf"
    manager.dispatch(0, initial)
    manager.worker.join(5)
    assert not manager.worker.is_alive()
    assert len(calls) == 5
    for update in (1, 999, 1000, 2000, 2499):
        assert evaluation.training_snapshot(args, update) is None
        manager.dispatch(update, "unused-checkpoint")
    assert len(calls) == 5
    final = Path(evaluation.training_snapshot(args, 2500))
    final.mkdir(parents=True)
    (final / ".complete").touch()
    manager.dispatch(2500, final)
    manager.worker.join(5)
    assert not manager.worker.is_alive()
    assert len(calls) == 10

    receipts = [
        json.loads(path.read_text())
        for path in (Path(run.output["root"]) / "evaluation").glob("update-*.json")
        if not path.name.endswith(".beaker.json")
    ]
    assert len(receipts) == 10
    assert {r["status"] for r in receipts} == {"submitted"}
    assert {r["experiment_id"] for r in receipts} == {f"experiment-{i}" for i in range(1, 11)}
    assert {r["update"] for r in receipts} == {0, 2500}
    for update, checkpoint in ((0, initial), (2500, final)):
        endpoint = [r for r in receipts if r["update"] == update]
        assert {r["checkpoint"] for r in endpoint} == {str(checkpoint)}
        tasks = [task for r in endpoint for task in r["tasks"]]
        assert sorted(tasks, key=lambda task: task["task"]) == sorted(
            run.evaluation["tasks"], key=lambda task: task["task"]
        )
        assert all(r["training"] == training and r["evaluation"]["revision"] == config["revision"] for r in endpoint)
    for specs in (calls[:5], calls[5:]):
        assert sorted(spec["tasks"][0]["resources"]["gpuCount"] for spec in specs) == [1, 1, 1, 2, 2]
        for spec in specs:
            job = spec["tasks"][0]
            assert job["image"] == {"beaker": config["image"]}
            assert job["context"] == {"priority": "high", "autoResume": True}
            assert job["constraints"]["cluster"] == [config["cluster"]]

    resumed = evaluation.Coordinator(config, training)
    resumed.dispatch(0, initial)
    resumed.dispatch(2500, final)
    assert resumed.worker is None
    assert len(calls) == 10


def test_panel_groups_keep_execution_requirements_separate(large):
    groups = evaluation.groups(large.evaluation, 0, 200)
    assert sorted(map(len, groups)) == [1, 1, 1, 5, 17]
    for group in groups:
        for task in group:
            assert task.get("harness", {}) == group[0].get("harness", {})
            assert task.get("harness_preset", "default") == group[0].get("harness_preset", "default")
            assert task.get("kind", "task") == group[0].get("kind", "task")


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"harness": {"provider.model": "other"}}, "frozen checkpoint"),
        ({"harness": {"provider": {"model": "other"}}}, "frozen checkpoint"),
        ({"harness": []}, "override"),
        ({"harness_preset": ""}, "nonempty"),
        ({"harness": {"auxiliary_providers.judge.api_key": "token"}}, "secrets"),
        ({"kind": "unknown"}, "kind"),
        ({"auxiliary_gpus": -1}, "auxiliary_gpus"),
        ({"auxiliary_gpus": True}, "auxiliary_gpus"),
        ({"arguments": {}}, "kind=external"),
        ({"task": "unsupported", "kind": "external"}, "Only scicode"),
        ({"task": "scicode", "kind": "external", "arguments": {"split": "test"}}, "validation"),
        ({"task": "scicode", "kind": "external", "arguments": {"split": "validation"}}, "reference data"),
    ],
)
def test_invalid_panel_settings_fail_before_launch(large, patch, message):
    config = copy.deepcopy(large.evaluation)
    config["tasks"] = [{"task": "ifeval", **patch}]
    with pytest.raises(InputError, match=message):
        evaluation.parse(config, large.launch)


def test_harness_options_precede_task_overrides(large):
    task = next(task for task in large.evaluation["tasks"] if task["task"] == "harmbench:wg_judge_thinking")
    task = {**task, "scoring": {"limit": 2}}
    args = evaluation_runner.command({"checkpoint": "/frozen/hf", "tasks": [task]}, "/results")
    index = args.index("--task")
    assert "auxiliary_providers.wg_judge.model=allenai/wildguard" in args[:index]
    assert "provider.base_url=http://127.0.0.1:30000/v1" in args[:index]
    assert "limit=2" in args[index:]


def test_scicode_uses_external_cli_with_frozen_server(large):
    task = next(task for task in large.evaluation["tasks"] if task["task"] == "scicode")
    args = evaluation_runner.command({"checkpoint": "/frozen/hf", "tasks": [task]}, "/results")
    assert args[:2] == ["olmo-eval", "run-external"]
    assert args[args.index("--model") + 1] == "/frozen/hf"
    assert args[args.index("--base-url") + 1] == "http://127.0.0.1:30000/v1"
    assert 'split="validation"' in args
    assert "--task" not in args


def test_sandbox_environment_is_only_enabled_for_code_jobs(large):
    for group in evaluation.groups(large.evaluation, 0, 200):
        receipt = {
            "evaluation": evaluation.runtime(large, large.compile().miles),
            "training": {"name": large.name},
            "update": 0,
            "tasks": group,
            "runner_sha256": hashlib.sha256(Path(evaluation_runner.__file__).read_bytes()).hexdigest(),
        }
        job = evaluation.specification(receipt)["tasks"][0]
        env = {item["name"]: item.get("value") for item in job["envVars"]}
        sandbox = group[0]["task"] in ("scicode", "livecodebench:lite")
        assert env.get("BEAKER_ALLOW_SUBCONTAINERS") == ("1" if sandbox else None)
        assert job["context"] == {"priority": "high", "autoResume": True}
        assert job["resources"]["gpuCount"] == 1 + group[0].get("auxiliary_gpus", 0)


@pytest.fixture
def scicode_result(tmp_path):
    result = {
        "name": "scicode",
        "success": True,
        "metrics": {"sub_step_accuracy": 0.0, "num_problems": 1, "num_sub_steps": 1},
        "predictions": [
            {
                "problem_id": "1",
                "total": 1,
                "step_results": [False],
                "step_codes": {"0": "bad code"},
                "step_texts": {"0": "response"},
            }
        ],
    }
    path = tmp_path / "scicode_results.json"
    path.write_text(json.dumps(result))
    evaluation_runner.write_json(
        tmp_path / "metrics.json",
        {"tasks": [{"task": "scicode", "metrics": {"sub_step_accuracy": {"external": 0.0}}}]},
    )
    return path, result


def test_scicode_accepts_scored_wrong_answers_without_model_output(scicode_result):
    path, _ = scicode_result
    assert evaluation_runner.completed_scores(path.parent, [{"task": "scicode"}]) == {
        "eval/scicode/sub_step_accuracy/external": 0.0
    }


@pytest.mark.parametrize("failure", ("problem_error", "missing_step", "empty", "count_mismatch", "failed"))
def test_scicode_rejects_infrastructure_errors_even_with_numeric_scores(scicode_result, failure):
    path, result = scicode_result
    if failure == "problem_error":
        result["predictions"][0]["error"] = "Sandbox startup failed"
    elif failure == "missing_step":
        result["predictions"][0]["step_texts"] = {}
    elif failure == "empty":
        result["predictions"] = []
    elif failure == "count_mismatch":
        result["metrics"]["num_problems"] = 2
    else:
        result["success"] = False
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="SciCode"):
        evaluation_runner.completed_scores(path.parent, [{"task": "scicode"}])


def test_mixed_group_requires_predictions_for_every_task(scicode_result, tmp_path):
    path, _ = scicode_result
    group = tmp_path / "group"
    complete = group / "complete"
    complete.mkdir(parents=True)
    (complete / path.name).write_bytes(path.read_bytes())
    missing = group / "missing"
    missing.mkdir()
    evaluation_runner.write_json(group / "task-results.json", ["complete", "missing"])
    with pytest.raises(ValueError, match="No evaluation predictions"):
        evaluation_runner.validate_predictions(group)


@pytest.mark.parametrize("mask,policy,judge", [("3,7", "3", "7"), (None, "0", "1"), ("GPU-a,GPU-b", "GPU-a", "GPU-b")])
def test_managed_judges_are_isolated_from_policy(monkeypatch, mask, policy, judge):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/training/sources")
    if mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    receipt = {"evaluation": {"gpus": 1}, "tasks": [{"auxiliary_gpus": 1}]}
    policy_env, evaluator_env = evaluation_runner.gpu_environments(receipt)
    assert policy_env["CUDA_VISIBLE_DEVICES"] == policy
    assert evaluator_env["CUDA_VISIBLE_DEVICES"] == judge
    assert "PYTHONPATH" not in evaluator_env
    assert policy_env["PYTHONPATH"] == "/training/sources"


def test_managed_judge_rejects_insufficient_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="needs 2 visible GPUs"):
        evaluation_runner.gpu_environments({"evaluation": {"gpus": 1}, "tasks": [{"auxiliary_gpus": 1}]})


def test_panel_manages_safety_judges(large):
    judges = [task for task in large.evaluation["tasks"] if task.get("auxiliary_gpus")]
    assert len(judges) == 6
    assert all(not any(key.endswith("base_url") for key in task["harness"]) for task in judges)
    strongreject = next(task for task in judges if task["task"].startswith("strongreject"))
    assert strongreject["harness"]["auxiliary_providers.sr_judge.model"] == "google/gemma-2b"
    assert strongreject["harness"]["auxiliary_providers.sr_judge.kwargs.enable_lora"] is True


@pytest.mark.parametrize("mask", ["", "-1", "0,0", "0,-1", "0,"])
def test_evaluator_rejects_hidden_duplicate_or_empty_mask_entries(monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError, match="visible GPUs"):
        evaluation_runner.gpu_environments({"evaluation": {"gpus": 1}, "tasks": [{"auxiliary_gpus": 1}]})


def test_evaluator_keeps_uuid_support_and_trims_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " MIG-a , MIG-b ")
    policy, judge = evaluation_runner.gpu_environments({"evaluation": {"gpus": 1}, "tasks": [{"auxiliary_gpus": 1}]})
    assert policy["CUDA_VISIBLE_DEVICES"] == "MIG-a"
    assert judge["CUDA_VISIBLE_DEVICES"] == "MIG-b"
