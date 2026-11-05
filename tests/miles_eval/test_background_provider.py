"""Validate MILES commands against pinned olmo-eval without loading a GPU model."""

import importlib
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import tomllib

pytest.importorskip("olmo_eval", reason="Requires the pinned evaluator image", exc_type=ModuleNotFoundError)
cli = importlib.import_module("olmo_eval.cli")
factory = importlib.import_module("olmo_eval.cli.run.factory")
planner = importlib.import_module("olmo_eval.inference.gpu_planner")
tasks = importlib.import_module("olmo_eval.evals.tasks.common")
testing = importlib.import_module("click.testing")
ROOT = Path(__file__).resolve().parents[2]
runner = runpy.run_path(str(ROOT / "open_instruct/miles/evaluation/evaluation_runner.py"))
CONFIGS = {
    name: tomllib.loads((ROOT / f"configs/miles/examples/{name}.toml").read_text())["evaluation"]
    for name in ("node", "three-node")
}
CASES = [
    (name, task) for name, config in CONFIGS.items() for task in config["tasks"] if task.get("kind") != "external"
]


@pytest.mark.parametrize("gpus", [1, 2])
@pytest.mark.parametrize("name,task", CASES, ids=[f"{name}:{task['task']}" for name, task in CASES])
def test_real_cli_preserves_provider_tasks_and_gpu_allocation(name, task, gpus, monkeypatch, tmp_path):
    task = {"generation": {}, "scoring": {}, **task}
    tasks.get_task(task["task"])
    receipt = {"checkpoint": "/frozen/hf", "tasks": [task], "evaluation": {**CONFIGS[name], "gpus": gpus}}
    configs = []

    def capture(instance):
        configs.append(instance.config)
        return Mock()

    argv = runner["command"](receipt, tmp_path) + ["--dry-run", "--no-upload"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(factory.RunnerFactory, "create", capture)
    result = testing.CliRunner().invoke(cli.main, argv[1:])
    assert result.exit_code == 0, (result.output, result.exception)
    config = configs[0]
    provider = config.harness_config.provider
    assert provider.kind == "sglang"
    assert provider.model == provider.tokenizer == "/frozen/hf"
    assert provider.base_url is None
    assert provider.kwargs["model_registration"] == "olmo_sglang:register"
    assert provider.kwargs["tensor_parallel_size"] == gpus
    for key, value in CONFIGS[name]["provider_kwargs"].items():
        assert provider.kwargs[key] == value
    expected = {
        key: None if value == "null" else value for key, value in (task["generation"] | task["scoring"]).items()
    }
    assert config.task_overrides.get(task["task"], {}) == expected
    total = gpus + task.get("auxiliary_gpus", 0)
    plan = planner.GPUPlanner.from_harness_config(config.harness_config, total).plan()
    planner.validate_gpu_plan(plan, total)
    assert plan.main_workers[0].gpu_ids == list(range(gpus))
    auxiliary_ids = [gpu for allocation in plan.auxiliary.values() for gpu in allocation.gpu_ids]
    assert auxiliary_ids == list(range(gpus, total))


def test_scicode_cli_keeps_explicit_http_exception(monkeypatch, tmp_path):
    config = CONFIGS["three-node"]
    task = next(task for task in config["tasks"] if task["task"] == "scicode")
    receipt = {"checkpoint": "/frozen/hf", "tasks": [task], "evaluation": config}
    argv = runner["command"](receipt, tmp_path) + ["--dry-run", "--no-upload"]
    monkeypatch.setattr(sys, "argv", argv)
    result = testing.CliRunner().invoke(cli.main, argv[1:])
    assert result.exit_code == 0, (result.output, result.exception)
    assert argv[argv.index("--provider") + 1] == "vllm_server"
