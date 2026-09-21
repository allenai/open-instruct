"""CPU contracts for independent, lossy background evaluation."""

import asyncio
import copy
import importlib
import json
import subprocess
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from open_instruct.miles import checkpoint, evaluation, evaluation_runner, evaluation_submit
from open_instruct.miles.errors import InputError
from open_instruct.miles.run_spec import RunSpec


@pytest.fixture
def run(tmp_path):
    return RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "eval-test",
            "model": {"source": "model", "format": "hf"},
            "output": {"root": str(tmp_path / "run")},
            "data": {"tasks": [{"task": "multiplication", "train_count": 8}]},
            "training": {"num_rollouts": 6},
            "evaluation": {
                "mode": "background",
                "interval": 2,
                "image": "01M2XGZM2N1V4DQVMYHM52KBHZ",
                "revision": "a" * 40,
                "tasks": [{"task": "gsm8k"}, {"task": "arc_easy", "interval": 3}],
            },
            "launch": {"secrets": {"WANDB_API_KEY": "main-wandb", "BEAKER_TOKEN": "submitter"}},
        }
    )


def test_grouping_and_final_deduplication(run):
    config = run.evaluation
    assert evaluation.groups(config, 1, 6) == []
    assert len(evaluation.groups(config, 6, 6)) == 1
    assert [x["task"] for x in evaluation.groups(config, 6, 6)[0]] == ["gsm8k", "arc_easy"]
    config["tasks"][1]["generation"] = {"temperature": 0.5}
    assert len(evaluation.groups(config, 6, 6)) == 2
    assert len(evaluation.groups(config, 5, 5)) == 2
    config.update(initial=False, final=False)
    assert evaluation.groups(config, 0, 6) == []
    assert evaluation.groups(config, 5, 5) == []


def test_plan_and_independent_eval_tasks(run):
    plan = run.plan()["evaluation"]
    assert [x["update"] for x in plan["milestones"]] == [0, 2, 3, 4, 6]
    assert "eval-snapshots/update-00000006/hf" in plan["milestones"][-1]["snapshot"]
    assert plan["secrets"] == {"WANDB_API_KEY": "main-wandb"}
    assert plan["mounts"] == run.launch["weka_mounts"]
    assert "eval_interval" not in run.compile().miles
    assert "eval_prompt_data" not in run.compile().miles
    assert RunSpec.from_dict(run.to_dict()).evaluation == run.evaluation


@pytest.mark.parametrize(
    "patch",
    [
        {"training": {"eval_interval": 2}},
        {"miles": {"eval_num_gpus": 1}},
        {"data": {"tasks": [{"task": "multiplication", "train_count": 8, "eval_count": 1}]}},
        {"evaluation": {"mode": "background", "image": "moving-tag"}},
        {"evaluation": {"mode": "shared", "interval": 2}},
    ],
)
def test_reject_conflicts(run, patch):
    with pytest.raises(InputError):
        RunSpec.from_dict(run.to_dict() | patch)


@pytest.mark.parametrize(
    "field,value", [("interval", 0), ("gpus", True), ("submit_timeout", -1), ("revision", "main")]
)
def test_reject_invalid_config(run, field, value):
    document = run.to_dict()
    document["evaluation"][field] = value
    with pytest.raises(InputError):
        RunSpec.from_dict(document)


def coordinator(run):
    return evaluation.Coordinator(
        evaluation.runtime(run, run.compile().miles),
        {"name": run.name, "wandb": {"mode": "offline", "id": "abc", "entity": "team", "project": "test"}},
    )


def test_single_submission_resume_and_failure(run, monkeypatch):
    calls = []

    def submit(receipt, path):
        calls.append(receipt)
        receipt["status"] = "submitted"
        receipt["experiment_id"] = "experiment"
        evaluation.state.atomic_json(path, receipt)

    monkeypatch.setattr(evaluation, "submit", submit)
    first = coordinator(run)
    first.dispatch(6, "checkpoint")
    first.worker.join(2)
    first.dispatch(6, "checkpoint")
    coordinator(run).dispatch(6, "checkpoint")
    assert len(calls) == 1
    manifest = next((Path(run.output["root"]) / "evaluation").glob("update-*.json"))
    assert json.loads(manifest.read_text())["experiment_id"] == "experiment"


def test_busy_worker_drops_without_waiting(run, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def submit(*args):
        entered.set()
        release.wait(5)

    monkeypatch.setattr(evaluation, "submit", submit)
    manager = coordinator(run)
    try:
        manager.dispatch(2, "checkpoint")
        assert entered.wait(1)
        assert manager.worker.daemon
        manager.dispatch(3, "checkpoint")
        receipts = list((Path(run.output["root"]) / "evaluation").glob("update-00000003-*.json"))
        assert json.loads(receipts[0].read_text())["status"] == "skipped_busy"
    finally:
        release.set()
        manager.worker.join(2)


@pytest.mark.parametrize(
    "error",
    [subprocess.TimeoutExpired("beaker", 30), subprocess.CalledProcessError(1, "beaker"), ValueError("bad JSON")],
)
def test_submission_timeout_and_failure_are_durable(run, monkeypatch, error):
    def fail(*args, **kwargs):
        assert kwargs["timeout"] == 30
        raise error

    monkeypatch.setattr(evaluation.subprocess, "run", fail)
    checkpoint_path = evaluation.snapshot(run.output["root"], 6)
    checkpoint_path.mkdir(parents=True)
    (checkpoint_path / ".complete").touch()
    manager = coordinator(run)
    manager.dispatch(6, checkpoint_path)
    manager.worker.join(2)
    receipts = [
        p
        for p in (Path(run.output["root"]) / "evaluation").glob("update-*.json")
        if not p.name.endswith("beaker.json")
    ]
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "submission_failed"
    assert receipt["error"] == type(error).__name__
    coordinator(run).dispatch(6, "checkpoint")
    assert len(receipts) == 1


def test_publisher_uses_secondary_and_checkpoint_axis(tmp_path, monkeypatch):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"tasks": [{"task": "gsm8k", "metrics": {"accuracy": {"exact": 0.5}}}]})
    )
    writer = Mock()
    sdk = SimpleNamespace(Api=Mock(), Settings=Mock(side_effect=lambda **kw: kw), init=Mock(return_value=writer))
    monkeypatch.setattr(evaluation_runner.importlib, "import_module", lambda name: sdk)
    receipt = {
        "update": 4,
        "group_id": "a",
        "training": {"wandb": {"id": "main", "entity": "team", "project": "p", "mode": "online"}},
    }
    evaluation_runner.publish(receipt, tmp_path)
    kwargs = sdk.init.call_args.kwargs
    assert kwargs["id"] == "main" and "config" not in kwargs and "resume" not in kwargs
    assert kwargs["settings"]["x_primary"] is False
    assert kwargs["settings"]["x_update_finish_state"] is False
    writer.log.assert_called_once_with({"eval/checkpoint_update": 4, "eval/gsm8k/accuracy/exact": 0.5})
    writer.define_metric.assert_any_call("eval/*", step_metric="eval/checkpoint_update", step_sync=False)
    writer.finish.assert_called_once()


def test_offline_and_failed_upload_preserve_results(tmp_path, monkeypatch):
    original = '{"tasks":[]}'
    (tmp_path / "metrics.json").write_text(original)
    receipt = {"training": {"wandb": {"mode": "offline"}}}
    assert evaluation_runner.try_publish(receipt, tmp_path)
    assert json.loads((tmp_path / "publication.json").read_text())["status"] == "deferred"
    assert not evaluation_runner.try_publish(receipt, tmp_path, wandb_run="team/project/run")
    assert (tmp_path / "metrics.json").read_text() == original
    assert json.loads((tmp_path / "publication.json").read_text())["status"] == "failed"


def test_evaluator_command_keeps_task_overrides(run):
    config = run.evaluation
    tasks = copy.deepcopy(config["tasks"])
    tasks[0]["generation"] = {"max_tokens": 32, "temperature": 0.0}
    tasks[0]["scoring"] = {"limit": 2}
    args = evaluation_runner.command({"checkpoint": "/frozen/hf", "tasks": tasks}, "/output")
    assert args[args.index("--model") + 1] == "/frozen/hf"
    assert "max_tokens=32" in args
    assert "limit=2" in args
    assert "--save-predictions" in args


def test_driver_never_uses_shared_evaluation_or_joins_worker(run, monkeypatch):
    # Import the real driver with CPU stand-ins for the external actor runtime.
    imported = {}

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        imported[name] = value
        return value

    manager = SimpleNamespace(
        generate=SimpleNamespace(remote=AsyncMock(return_value={})), dispose=SimpleNamespace(remote=AsyncMock())
    )
    learner = SimpleNamespace(
        update_weights=AsyncMock(), train=AsyncMock(), dispose=AsyncMock(), _broadcast=AsyncMock()
    )
    placement = SimpleNamespace(
        create_placement_groups=lambda args: {"rollout": None},
        create_rollout_manager=lambda *args: (manager, 1),
        create_training_models=AsyncMock(return_value=(learner, None)),
    )
    module("miles")
    module("miles.ray", placement_group=placement)
    module("miles.ray.rollout")
    shared = Mock(side_effect=AssertionError("shared evaluator must not be constructed"))
    module("miles.ray.rollout.eval_dispatch", EvalDispatcher=shared)
    module("miles.utils", object_store=SimpleNamespace(init_instance=lambda *a, **kw: None))
    module("miles.utils.data", remove_rollout_data_refs=lambda *a: None)
    module("miles.utils.hf_config", HF_EXPORT_COMPLETE_MARKER=".complete")
    module("miles.utils.misc", should_run_periodic_action=lambda *a: False)
    module("miles.utils.tracking_utils")
    module("miles.utils.tracking_utils.tracking", finish_tracking=lambda: None, init_tracking=lambda args: None)
    module("open_instruct.miles.rolling_publication", RollingPublication=Mock())
    # Load under a private module name so monkeypatch cleanup leaves no stale driver.
    spec = importlib.util.spec_from_file_location(
        "test_background_driver", Path(evaluation.__file__).with_name("driver.py")
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setattr(driver, "stage", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(driver.throughput, "report", lambda *a: {"warnings": []})
    monkeypatch.setattr(driver.startup_cache, "prepare", lambda *a: None)
    monkeypatch.setattr(driver.startup_cache, "finish", AsyncMock())
    entered, release = threading.Event(), threading.Event()
    worker_threads = []

    def blocked_submit(receipt, path):
        worker_threads.append(threading.current_thread())
        entered.set()
        release.wait(5)
        raise RuntimeError("independent evaluator submission failed")

    monkeypatch.setattr(evaluation, "submit", blocked_submit)

    async def train_step(rollout_id, batch):
        target = evaluation.snapshot(run.output["root"], (rollout_id + 1) * 2)
        target.mkdir(parents=True)
        (target / ".complete").touch()

    learner.train.side_effect = train_step
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(publication_mode="barrier", diagnostic_interval=0),
        background_evaluation=evaluation.runtime(run, run.compile().miles),
        fully_async=False,
        offload_rollout=False,
        check_weight_update_equal=False,
        use_wandb=False,
        wandb_team="team",
        wandb_project="test",
        wandb_mode="offline",
        start_rollout_id=0,
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        global_batch_size=32,
        hf_checkpoint="initial",
        save_trigger_sentinel=None,
        save_interval=None,
        update_weights_interval=1,
        debug_exit_after_rollout=None,
    )
    try:
        result = asyncio.run(driver.train(args))
        assert entered.wait(1)
        assert worker_threads[0].is_alive()  # train returned while submission is still blocked.
        assert result["completed_rollout_ids"] == [0, 1]
        assert learner.train.await_count == 2
        skipped = list((Path(run.output["root"]) / "evaluation").glob("update-00000004-*.json"))
        assert skipped and json.loads(skipped[0].read_text())["status"] == "skipped_busy"
        shared.assert_not_called()
    finally:
        release.set()
        for thread in worker_threads:
            thread.join(2)


def test_eval_snapshots_are_outside_checkpoint_cleanup(tmp_path):
    from_checkpoint_root = tmp_path / "checkpoints"
    for update in (1, 2):
        path = from_checkpoint_root / "core" / f"rollout_{update:07d}"
        path.mkdir(parents=True)
        (path / "complete.json").write_text("{}")
    frozen = evaluation.snapshot(tmp_path, 1)
    frozen.mkdir(parents=True)
    (frozen / "model.safetensors").write_bytes(b"immutable")
    assert checkpoint.prune(from_checkpoint_root, 2, keep_last=1, keep_every=None) == [1]
    assert (frozen / "model.safetensors").read_bytes() == b"immutable"


def test_incomplete_snapshot_never_submitted(run, monkeypatch):
    network = Mock()
    monkeypatch.setattr(evaluation.subprocess, "run", network)
    manager = coordinator(run)
    manager.dispatch(6, evaluation.snapshot(run.output["root"], 6))
    manager.worker.join(2)
    network.assert_not_called()
    receipt = next((Path(run.output["root"]) / "evaluation").glob("update-*.json"))
    assert json.loads(receipt.read_text())["status"] == "submission_failed"


def test_submitter_reads_nested_beaker_workload_id(monkeypatch, capsys):
    client = SimpleNamespace(
        experiment=SimpleNamespace(
            create=Mock(return_value=SimpleNamespace(experiment=SimpleNamespace(id="accepted-experiment")))
        )
    )
    monkeypatch.setattr(evaluation_submit, "Beaker", SimpleNamespace(from_env=lambda **kwargs: nullcontext(client)))
    monkeypatch.setattr(evaluation_submit.signal, "alarm", Mock())
    monkeypatch.setattr(sys, "argv", ["submit", "spec.json", "ai2/workspace", "unique-name", "30"])
    evaluation_submit.main()
    assert json.loads(capsys.readouterr().out) == {"id": "accepted-experiment"}
    client.experiment.create.assert_called_once_with(spec="spec.json", name="unique-name")
    evaluation_submit.signal.alarm.assert_called_once_with(30)


def test_submission_diagnostics_redact_credentials(monkeypatch):
    monkeypatch.setenv("BEAKER_TOKEN", "sensitive-test-token")
    monkeypatch.setenv("WANDB_API_KEY", "sensitive-test-key")
    result = evaluation_submit.diagnostic(
        ValueError("rejected sensitive-test-token sensitive-test-key Bearer unknown-credential")
    )
    assert result["error"] == "ValueError"
    assert result["message"] == "rejected [REDACTED] [REDACTED] Bearer [REDACTED]"


def test_all_writers_define_training_and_background_axes():
    run = Mock()
    evaluation_runner.define_metrics(run)
    calls = run.define_metric.call_args_list
    groups = {call.args[0]: call.kwargs for call in calls if call.args[0].endswith("/*")}
    assert groups == {
        "train/*": {"step_metric": "train/step", "step_sync": True},
        "rollout/*": {"step_metric": "rollout/step", "step_sync": True},
        "multi_turn/*": {"step_metric": "rollout/step", "step_sync": True},
        "passrate/*": {"step_metric": "rollout/step", "step_sync": True},
        "perf/*": {"step_metric": "rollout/step", "step_sync": True},
        "eval/*": {"step_metric": "eval/checkpoint_update", "step_sync": False},
    }
