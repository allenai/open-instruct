"""Post-training panel configuration, execution boundaries and completion checks."""

import copy
import hashlib
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.evaluation import evaluation, evaluation_runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def three_node():
    return RunSpec.load(ROOT / "configs/miles/examples/three-node.toml")


def test_node_uses_periodic_native_background_evaluation():
    node = RunSpec.load(ROOT / "configs/miles/examples/node.toml")
    assert node.evaluation["mode"] == "background"
    assert [p["update"] for p in node.plan()["evaluation"]["milestones"]] == [0, 50, 100, 150, 200]
    assert {t["task"] for t in node.evaluation["tasks"]} == {"gsm8k", "gsm8k:platinum", "ifeval_ood"}
    assert "eval_prompt_data" not in node.compile().miles
    assert "eval_interval" not in node.compile().miles
    assert node.launch["secrets"]["BEAKER_TOKEN"]


def test_three_node_panel_preserves_protocols_and_before_after_cadence(three_node):
    tasks = three_node.evaluation["tasks"]
    assert len(tasks) == len({task["task"] for task in tasks}) == 25
    assert all(not task["generation"] and not task["scoring"] for task in tasks)
    milestones = three_node.plan()["evaluation"]["milestones"]
    assert [point["update"] for point in milestones] == [0, 200]
    assert all(sum(map(len, point["groups"])) == 25 for point in milestones)
    assert len(milestones[0]["groups"]) == 5
    # Extending the run never introduces periodic full-panel evaluations.
    document = three_node.to_dict()
    document["training"]["num_rollouts"] = 2500
    extended = RunSpec.from_dict(document)
    assert [point["update"] for point in extended.plan()["evaluation"]["milestones"]] == [0, 2500]
    for update in (1, 999, 1000, 2000, 2499):
        assert evaluation.groups(extended.evaluation, update, 2500) == []
    assert sum(map(len, evaluation.groups(extended.evaluation, 2500, 2500))) == 25


def test_full_panel_submits_only_endpoints_and_does_not_resubmit_on_resume(three_node, tmp_path, monkeypatch):
    document = three_node.to_dict()
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


def test_panel_groups_keep_execution_requirements_separate(three_node):
    groups = evaluation.groups(three_node.evaluation, 0, 200)
    assert sorted(map(len, groups)) == [1, 1, 1, 5, 17]
    for group in groups:
        for task in group:
            assert task.get("harness", {}) == group[0].get("harness", {})
            assert task.get("harness_preset", "default") == group[0].get("harness_preset", "default")
            assert task.get("kind", "task") == group[0].get("kind", "task")


def without_hf_token(three_node):
    document = three_node.to_dict()
    for section in ("launch", "evaluation"):
        document[section]["secrets"].pop("HF_TOKEN", None)
    return document


def test_gated_panel_requires_job_secret_even_with_local_token(three_node, monkeypatch):
    document = without_hf_token(three_node)
    monkeypatch.setenv("HF_TOKEN", "local-token-is-not-forwarded")
    with pytest.raises(InputError) as error:
        RunSpec.from_dict(document).plan()
    message = str(error.value)
    for task in document["evaluation"]["tasks"]:
        if task.get("auxiliary_gpus"):
            assert task["task"] in message
    assert "allenai/wildguard" in message and "google/gemma-2b" in message
    assert "evaluation.secrets.HF_TOKEN" in message and "launch.secrets.HF_TOKEN" in message
    assert "local-token-is-not-forwarded" not in message


@pytest.mark.parametrize("model", ["allenai/wildguard", "google/gemma-2b"])
@pytest.mark.parametrize("field", ["model", "tokenizer"])
def test_gated_judge_guard_uses_model_not_task_or_provider_name(three_node, model, field):
    document = without_hf_token(three_node)
    document["evaluation"]["tasks"] = [
        {
            "task": "custom-task",
            "harness": {
                "auxiliary_providers.custom.kind": "vllm_server",
                f"auxiliary_providers.custom.{field}": model,
            },
        }
    ]
    with pytest.raises(InputError, match="custom-task"):
        RunSpec.from_dict(document).plan()


@pytest.mark.parametrize("source", ["launch", "evaluation", "both"])
def test_gated_judge_secret_inheritance_and_override_reach_job(three_node, source):
    document = without_hf_token(three_node)
    if source in ("launch", "both"):
        document["launch"]["secrets"]["HF_TOKEN"] = "training-hf"
    if source in ("evaluation", "both"):
        document["evaluation"]["secrets"]["HF_TOKEN"] = "evaluation-hf"
    spec = RunSpec.from_dict(document)
    planned = spec.plan()["evaluation"]
    expected = "training-hf" if source == "launch" else "evaluation-hf"
    assert planned["secrets"]["HF_TOKEN"] == expected
    for group in evaluation.groups(spec.evaluation, 0, 200):
        receipt = {
            "evaluation": evaluation.runtime(spec, spec.compile().miles),
            "training": {"name": spec.name},
            "tasks": group,
            "update": 0,
            "runner_sha256": hashlib.sha256(Path(evaluation_runner.__file__).read_bytes()).hexdigest(),
        }
        job = evaluation.specification(receipt)["tasks"][0]
        assert {"name": "HF_TOKEN", "secret": expected} in job["envVars"]


@pytest.mark.parametrize("case", ["local", "public", "remote", "disabled", "no-judges"])
def test_gated_judge_guard_allows_evaluations_without_gated_downloads(three_node, case):
    document = without_hf_token(three_node)
    config = document["evaluation"]
    for task in config["tasks"]:
        harness = task.get("harness", {})
        for key in list(harness):
            if not key.endswith(".model"):
                continue
            if case == "local":
                harness[key] = "/weka/judges/downloaded-model"
            elif case == "public":
                harness[key] = "Qwen/Qwen3-8B"
            elif case == "remote":
                harness[key.removesuffix(".model") + ".base_url"] = "http://judge:8000/v1"
    if case == "disabled":
        config.update(initial=False, final=False, interval=0)
        for task in config["tasks"]:
            task["interval"] = 0
    elif case == "no-judges":
        config["tasks"] = [task for task in config["tasks"] if not task.get("auxiliary_gpus")]
    assert "HF_TOKEN" not in RunSpec.from_dict(document).plan()["evaluation"]["secrets"]


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
def test_invalid_panel_settings_fail_before_launch(three_node, patch, message):
    config = copy.deepcopy(three_node.evaluation)
    config["tasks"] = [{"task": "ifeval", **patch}]
    with pytest.raises(InputError, match=message):
        evaluation.parse(config, three_node.launch)


def test_harness_options_precede_task_overrides(three_node):
    task = next(task for task in three_node.evaluation["tasks"] if task["task"] == "harmbench:wg_judge_thinking")
    task = {**task, "scoring": {"limit": 2}}
    args = evaluation_runner.command(
        {"checkpoint": "/frozen/hf", "tasks": [task], "evaluation": three_node.evaluation}, "/results"
    )
    index = args.index("--task")
    assert "auxiliary_providers.wg_judge.model=allenai/wildguard" in args[:index]
    assert "provider.kind=sglang" in args[:index]
    assert "limit=2" in args[index:]


def test_scicode_uses_external_cli_with_frozen_server(three_node):
    task = next(task for task in three_node.evaluation["tasks"] if task["task"] == "scicode")
    args = evaluation_runner.command(
        {"checkpoint": "/frozen/hf", "tasks": [task], "evaluation": three_node.evaluation}, "/results"
    )
    assert args[:2] == ["olmo-eval", "run-external"]
    assert args[args.index("--model") + 1] == "/frozen/hf"
    assert args[args.index("--base-url") + 1] == "http://127.0.0.1:30000/v1"
    assert 'split="validation"' in args
    assert "--task" not in args


def test_sandbox_environment_is_only_enabled_for_code_jobs(three_node):
    for group in evaluation.groups(three_node.evaluation, 0, 200):
        receipt = {
            "evaluation": evaluation.runtime(three_node, three_node.compile().miles),
            "training": {"name": three_node.name},
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
def test_native_planner_sees_policy_and_judge_gpus(monkeypatch, mask, policy, judge):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/training/sources")
    if mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    receipt = {"evaluation": {"gpus": 1}, "tasks": [{"auxiliary_gpus": 1}]}
    policy_env, evaluator_env = evaluation_runner.gpu_environments(receipt)
    assert policy_env["CUDA_VISIBLE_DEVICES"] == policy
    assert evaluator_env["CUDA_VISIBLE_DEVICES"] == f"{policy},{judge}"
    assert "PYTHONPATH" not in evaluator_env
    assert policy_env["PYTHONPATH"] == "/training/sources"


def test_managed_judge_rejects_insufficient_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="needs 2 visible GPUs"):
        evaluation_runner.gpu_environments({"evaluation": {"gpus": 1}, "tasks": [{"auxiliary_gpus": 1}]})


def test_panel_manages_safety_judges(three_node):
    judges = [task for task in three_node.evaluation["tasks"] if task.get("auxiliary_gpus")]
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
    assert judge["CUDA_VISIBLE_DEVICES"] == "MIG-a,MIG-b"


def test_external_scicode_server_uses_policy_gpus_only(three_node, tmp_path, monkeypatch):
    runner = evaluation_runner
    task = next(task for task in three_node.evaluation["tasks"] if task["task"] == "scicode")
    task = {**task, "arguments": {**task["arguments"], "h5py_host_path": str(tmp_path / "data.h5")}}
    (tmp_path / "data.h5").touch()
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    for filename in ("config.json", "tokenizer_config.json"):
        (checkpoint / filename).write_text("{}")
    receipt = {"tasks": [task], "checkpoint": str(checkpoint), "evaluation": three_node.evaluation}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    policy, evaluator = runner.gpu_environments(receipt)
    assert policy["CUDA_VISIBLE_DEVICES"] == "3"
    assert evaluator["CUDA_VISIBLE_DEVICES"] == ""
    server = Mock()
    server.poll.return_value = None
    start = Mock(return_value=server)
    monkeypatch.setattr(runner.subprocess, "Popen", start)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: three_node.evaluation["revision"])
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/podman")
    monkeypatch.setattr(runner.shutil, "copytree", Mock())
    monkeypatch.setattr(runner.request, "urlopen", lambda *a, **kw: nullcontext(Mock(status=200)))
    monkeypatch.setattr(runner, "run_tasks", Mock())
    monkeypatch.setattr(runner, "completed_scores", Mock(return_value={"eval/scicode/score": 1.0}))
    monkeypatch.setattr(runner, "try_publish", Mock())
    runner.run_evaluation_locked(receipt, tmp_path)
    args = start.call_args.args[0]
    assert "sglang.launch_server" in args
    assert args[args.index("--model-path") + 1] == str(checkpoint)
    assert start.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert "--context-length" in args
    server.terminate.assert_called_once()
