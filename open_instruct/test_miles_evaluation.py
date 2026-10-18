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
from unittest.mock import ANY, AsyncMock, Mock

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.evaluation import evaluation, evaluation_runner, evaluation_submit
from open_instruct.miles.rewards import general_judge, h080, if_gate_yes, judge_metrics


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


def test_empty_provider_outputs_reject_apparently_successful_metrics(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"tasks": [{"task": "ifeval_ood", "metrics": {"accuracy": {"exact": 0.0}}, "instances_failed": 0}]})
    )
    (tmp_path / "ifeval_ood-abc-predictions.jsonl").write_text(json.dumps({"doc_id": 0, "model_output": []}) + "\n")
    with pytest.raises(ValueError, match="Missing model outputs"):
        evaluation_runner.scores(tmp_path)


def test_prediction_validation_accepts_a_returned_empty_completion(tmp_path):
    # A real EOS-only completion is distinct from the provider returning no outputs.
    (tmp_path / "gsm8k-abc-predictions.jsonl").write_text(
        json.dumps({"doc_id": 0, "model_output": [{"text": ""}]}) + "\n"
    )
    evaluation_runner.validate_predictions(tmp_path)


def test_prediction_validation_requires_nonempty_artifacts(tmp_path):
    with pytest.raises(ValueError, match="No evaluation predictions"):
        evaluation_runner.validate_predictions(tmp_path)
    (tmp_path / "gsm8k-abc-predictions.jsonl").touch()
    with pytest.raises(ValueError, match="Empty evaluation predictions"):
        evaluation_runner.validate_predictions(tmp_path)


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


@pytest.fixture
def cpu_driver(monkeypatch):
    # Import the real driver with CPU stand-ins for the external actor runtime.
    imported = {}

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        imported[name] = value
        return value

    manager = SimpleNamespace(
        get=SimpleNamespace(remote=AsyncMock(return_value={})),
        dispose=SimpleNamespace(remote=AsyncMock()),
        save=SimpleNamespace(remote=AsyncMock()),
    )
    learner = SimpleNamespace(
        update_weights=AsyncMock(),
        train=AsyncMock(),
        dispose=AsyncMock(),
        execute_workers=AsyncMock(),
        save_model=AsyncMock(),
        finalize_checkpoint=AsyncMock(),
        export_hf=AsyncMock(),
    )
    inference = SimpleNamespace(prepare_rollout=AsyncMock(), dispose=AsyncMock())
    workers = SimpleNamespace(dispose=SimpleNamespace(remote=AsyncMock()))
    placement = SimpleNamespace(
        create_rollout_components=AsyncMock(return_value=(inference, manager, 1)),
        create_training_models=AsyncMock(return_value=(learner, None)),
        update_weights=AsyncMock(),
    )
    module("miles")
    module(
        "miles.ray", placement_group=placement, wiring=SimpleNamespace(launch_worker_manager=lambda *a, **kw: workers)
    )
    module("miles.ray.rollout")
    shared = Mock(side_effect=AssertionError("shared evaluator must not be constructed"))
    module("miles.ray.rollout.eval_dispatch", EvalDispatcher=shared)
    module("miles.utils", object_store=SimpleNamespace(init_instance=lambda *a, **kw: None))
    module(
        "miles.utils.compiler_cache",
        startup_cache=SimpleNamespace(
            prepare=Mock(), finish=AsyncMock(), configure_specs=Mock(), publish_progress=Mock()
        ),
    )
    module("miles.utils.data", remove_rollout_data_refs=lambda *a: None)
    module("miles.utils.hf_config", HF_EXPORT_COMPLETE_MARKER=".complete")
    module("miles.utils.misc", should_run_periodic_action=lambda *a: False)
    module("miles.utils.tracking_utils")
    module(
        "miles.utils.tracking_utils.tracking",
        define_step_key_metric_group=lambda *a: None,
        finish_tracking=lambda: None,
        init_tracking=lambda args: None,
        log=Mock(),
    )
    module("miles.backends.core_utils.publication.rolling_publication", RollingPublication=Mock())
    # Load under a private module name so monkeypatch cleanup leaves no stale driver.
    spec = importlib.util.spec_from_file_location(
        "test_background_driver", Path(evaluation.__file__).parents[1] / "execution/driver.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    clock = SimpleNamespace(time=Mock(return_value=1000), monotonic=Mock(return_value=0))
    monkeypatch.setattr(driver, "time", clock)
    monkeypatch.setattr(driver, "stage", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(driver.throughput, "report", lambda *a: {"warnings": []})
    monkeypatch.setattr(driver.startup_cache, "prepare", lambda *a, **kw: None)
    monkeypatch.setattr(driver.startup_cache, "finish", AsyncMock())
    return SimpleNamespace(driver=driver, manager=manager, learner=learner, clock=clock, shared=shared)


@pytest.mark.parametrize("stop_mode", ("complete", "deadline", "debug"))
def test_driver_never_uses_shared_evaluation_or_joins_worker(run, monkeypatch, stop_mode, cpu_driver):
    driver, manager, learner = cpu_driver.driver, cpu_driver.manager, cpu_driver.learner
    clock, shared = cpu_driver.clock, cpu_driver.shared
    entered, release = threading.Event(), threading.Event()
    worker_threads = []

    def blocked_submit(receipt, path):
        worker_threads.append(threading.current_thread())
        entered.set()
        release.wait(5)
        raise RuntimeError("independent evaluator submission failed")

    monkeypatch.setattr(evaluation, "submit", blocked_submit)

    async def train_step(rollout_id, batch):
        if stop_mode == "deadline":
            clock.monotonic.return_value = 2
        target = evaluation.snapshot(run.output["root"], (rollout_id + 1) * 2)
        target.mkdir(parents=True)
        (target / ".complete").touch()

    learner.train.side_effect = train_step
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            publication_mode="barrier", diagnostic_interval=0, max_run_seconds=1 if stop_mode == "deadline" else None
        ),
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
        save="native-checkpoints",
        save_interval=None,
        update_weights_interval=1,
        debug_exit_after_rollout=1 if stop_mode == "debug" else None,
    )
    try:
        result = asyncio.run(driver.train(args, export_hf="final-hf"))
        assert entered.wait(1)
        assert worker_threads[0].is_alive()  # train returned while submission is still blocked.
        expected = [0, 1] if stop_mode == "complete" else [0]
        assert result["completed_rollout_ids"] == expected
        assert learner.train.await_count == len(expected)
        assert result["stopped_for_time"] == (stop_mode == "deadline")
        if stop_mode == "deadline":
            manager.save.remote.assert_awaited_once_with(0)
            learner.save_model.assert_awaited_once_with(0, force_sync=True)
            learner.finalize_checkpoint.assert_awaited_once_with(0)
        if stop_mode == "debug":
            learner.export_hf.assert_not_awaited()
        else:
            learner.export_hf.assert_awaited_once_with(expected[-1], "final-hf")
        update = 4 if stop_mode == "complete" else 2
        skipped = list((Path(run.output["root"]) / "evaluation").glob(f"update-{update:08d}-*.json"))
        assert skipped and json.loads(skipped[0].read_text())["status"] == "skipped_busy"
        if stop_mode == "deadline":
            final = [
                json.loads(p.read_text()) for p in skipped if json.loads(p.read_text())["checkpoint"] == "final-hf"
            ]
            assert len(final) == 1
            assert {t["task"] for t in final[0]["tasks"]} == {"gsm8k", "arc_easy"}
        shared.assert_not_called()
    finally:
        release.set()
        for thread in worker_threads:
            thread.join(2)


@pytest.mark.parametrize("policy", ("raise", "zero", "zero_guarded"))
def test_driver_judge_guard_halts_only_zero_guarded_and_runs_cleanup(
    cpu_driver, monkeypatch, caplog, policy, tmp_path
):
    driver, manager, learner = cpu_driver.driver, cpu_driver.manager, cpu_driver.learner
    caplog.set_level("INFO", logger=driver.logger.name)
    metrics = judge_metrics.JudgeMetrics()
    for index in range(1000):
        metrics.arrive()
        metrics.admit()
        failed = index < 6
        metrics.finish(
            not failed,
            3 if failed else 1,
            1.0,
            1.0,
            terminal_failure=failed,
            instruction_ids=["detectable_content:postscript", "keywords:existence"],
        )
    boundary = AsyncMock(side_effect=metrics.optimizer_boundary)
    monkeypatch.setattr(driver.judge_registry, "gate_bindings", lambda: {"ifeval"})
    monkeypatch.setenv("OI_MILES_IF_GATE_FAILURE_POLICY", policy)
    monkeypatch.setattr(
        driver.judge_metrics, "start", lambda: SimpleNamespace(optimizer_boundary=SimpleNamespace(remote=boundary))
    )
    monkeypatch.setattr(driver, "EvalDispatcher", Mock(return_value=SimpleNamespace(drain=AsyncMock())))
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(publication_mode="barrier", diagnostic_interval=0),
        fully_async=False,
        offload_rollout=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        global_batch_size=32,
        save=str(tmp_path / "checkpoints"),
        save_trigger_sentinel=None,
        save_interval=None,
        update_weights_interval=1,
        eval_interval=None,
        debug_exit_after_rollout=None,
    )
    if policy == "zero_guarded":
        with pytest.raises(RuntimeError, match="zero_guarded.*6/1000.*stopping training"):
            asyncio.run(driver.train(args))
        assert learner.train.await_count == boundary.await_count == 1
        driver.startup_cache.finish.assert_awaited_once_with(args, success=False)
        # The halting update is persisted under the run's update number for a later resume.
        assert [json.loads(line) for line in (tmp_path / judge_metrics.GUARD_RECORD).read_text().splitlines()] == [
            {"run": None, "step": 1, "requests": 1000, "failures": 6, "unix": ANY}
        ]
    else:
        assert asyncio.run(driver.train(args))["completed_rollout_ids"] == [0, 1]
        assert learner.train.await_count == boundary.await_count == 2
        driver.startup_cache.finish.assert_awaited_once_with(args, success=True)
    driver.log.assert_any_call(
        args,
        {
            "judge/terminal_failures": 6,
            "judge/terminal_failure_rate_window": 0.006,
            **{
                f"judge/{kind}_by_{scope}/{key}": value
                for scope, keys in (
                    ("family", ["detectable_content", "keywords"]),
                    ("instruction", ["detectable_content:postscript", "keywords:existence"]),
                )
                for key in keys
                for kind, value in (("requests", 1000), ("terminal_failures", 6))
            },
            "judge/step": 1,
        },
        step_key="judge/step",
    )
    # smoke_check.read_log accepts a Ray/logger prefix and raw-decodes the JSON
    # immediately after this marker. Verify actual emitted lines, including the
    # failure that stops training, rather than only the actor's snapshot.
    lines = [record.getMessage() for record in caplog.records if "h087_judge_boundary " in record.getMessage()]
    emitted = json.JSONDecoder().raw_decode(lines[0].split("h087_judge_boundary ", 1)[1].lstrip())[0]
    assert emitted["step"] == 1
    assert emitted["judge_requests"] == 1000
    assert emitted["judge/terminal_failures"] == 6
    assert emitted["judge/requests_by_family/detectable_content"] == 1000
    assert emitted["judge/terminal_failures_by_family/detectable_content"] == 6
    assert emitted["judge/terminal_failures_by_instruction/detectable_content:postscript"] == 6
    manager.dispose.remote.assert_awaited_once()
    learner.dispose.assert_awaited_once()


def test_driver_rejects_inexact_guard_update_boundaries(cpu_driver, monkeypatch):
    driver = cpu_driver.driver
    monkeypatch.setattr(driver.judge_registry, "gate_bindings", lambda: {"ifeval"})
    monkeypatch.setenv("OI_MILES_IF_GATE_FAILURE_POLICY", "zero_guarded")
    args = SimpleNamespace(rollout_batch_size=8, n_samples_per_prompt=8, global_batch_size=32)
    with pytest.raises(ValueError, match="one optimizer update per collection"):
        asyncio.run(driver.train(args))
    driver.placement_group.create_rollout_components.assert_not_awaited()


def test_driver_halts_while_if_only_collection_discards_all_zero_groups(cpu_driver, monkeypatch, caplog, tmp_path):
    driver, manager, learner = cpu_driver.driver, cpu_driver.manager, cpu_driver.learner
    caplog.set_level("ERROR")
    metrics = judge_metrics.JudgeMetrics()
    boundary = AsyncMock(side_effect=metrics.optimizer_boundary)
    pending = AsyncMock(side_effect=metrics.pending_window)
    cancelled = Mock()
    discarded = []
    config = if_gate_yes.client_config(h080.SERVICE, "http://judge.invalid/v1")
    monkeypatch.setenv("OI_MILES_IF_GATE_FAILURE_POLICY", "zero_guarded")
    monkeypatch.setattr(judge_metrics, "tracker", lambda: metrics)
    monkeypatch.setattr(driver.judge_registry, "gate_bindings", lambda: {"ifeval"})
    monkeypatch.setattr(driver, "JUDGE_GUARD_POLL_SECONDS", 0.001)
    monkeypatch.setattr(
        driver.judge_metrics,
        "start",
        lambda: SimpleNamespace(
            optimizer_boundary=SimpleNamespace(remote=boundary), pending_window=SimpleNamespace(remote=pending)
        ),
    )
    monkeypatch.setattr(driver, "EvalDispatcher", Mock(return_value=SimpleNamespace(drain=AsyncMock())))

    async def request(*args):
        return "invalid"

    monkeypatch.setattr(general_judge, "async_request", request)
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(publication_mode="barrier", diagnostic_interval=0, filter_zero_std_groups=True),
        fully_async=False,
        offload_rollout=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        global_batch_size=32,
        save=str(tmp_path / "checkpoints"),
        eval_interval=None,
    )

    async def collect(rollout_id):
        try:
            while True:
                group = [
                    await if_gate_yes.score_prompt(
                        config, "user:\nx", "y", policy=if_gate_yes.failure_policy(), metadata={}, name="ifeval"
                    )
                    for _ in range(args.n_samples_per_prompt)
                ]
                assert group == [0.0] * args.n_samples_per_prompt
                if args.olmo_core.filter_zero_std_groups and min(group) == max(group):
                    discarded.append(group)
                else:
                    return group
                # Stop producing at the minimum denominator, leaving get pending.
                if metrics.judge_requests == 1000:
                    await asyncio.Event().wait()
                await asyncio.sleep(0)
        finally:
            cancelled()

    manager.get.remote.side_effect = collect

    async def run():
        with pytest.raises(RuntimeError, match="zero_guarded.*1000/1000.*pending update.*stopping training"):
            await asyncio.wait_for(driver.train(args), timeout=2)
        counters = await metrics.snapshot()
        assert counters["judge_window_updates"] == 0
        assert counters["judge_requests"] == counters["judge/terminal_failures"] == 1000

    asyncio.run(run())
    assert len(discarded) == 250
    assert pending.await_count > 0
    boundary.assert_not_awaited()
    learner.train.assert_not_awaited()
    cancelled.assert_called_once()
    driver.startup_cache.finish.assert_awaited_once_with(args, success=False)
    manager.dispose.remote.assert_awaited_once()
    learner.dispose.assert_awaited_once()


@pytest.mark.parametrize("value", (0, -1, True, float("inf"), float("nan")))
def test_wall_clock_budget_must_be_finite_and_positive(value):
    with pytest.raises(InputError):
        CoreConfig(max_run_seconds=value)


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


def test_driver_resume_refuses_to_start_without_the_runs_guard_window(cpu_driver, monkeypatch, tmp_path):
    driver = cpu_driver.driver
    monkeypatch.setattr(driver.judge_registry, "gate_bindings", lambda: {"ifeval"})
    monkeypatch.setenv("OI_MILES_IF_GATE_FAILURE_POLICY", "zero_guarded")
    placement = Mock()
    monkeypatch.setattr(driver.wiring, "launch_worker_manager", placement)
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(publication_mode="barrier", diagnostic_interval=0),
        start_rollout_id=75,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        global_batch_size=32,
        save=str(tmp_path / "checkpoints"),
    )
    with pytest.raises(RuntimeError, match="updates 26..75.*refusing to start"):
        asyncio.run(driver.train(args))
    placement.assert_not_called()


def test_driver_resume_seeds_the_runs_guard_window_before_the_first_update(cpu_driver, monkeypatch, tmp_path):
    driver, learner = cpu_driver.driver, cpu_driver.learner
    (tmp_path / "attempts.json").write_text(json.dumps([{"attempt": 0, "started_unix": 7.0}]))
    # u1 (before the resume) had 9 failures in 1,000; u2 alone has 2 in 1,000 (0.2%, passes),
    # but the run's window u1-u2 is 11/2,000 (0.55%) and must halt.
    judge_metrics.record_update(tmp_path, 7.0, 1, 1000, 9, 1.0)
    metrics = judge_metrics.JudgeMetrics()
    for index in range(1000):
        metrics.arrive()
        metrics.admit()
        metrics.finish(index >= 2, 1, 1.0, 1.0, terminal_failure=index < 2)
    monkeypatch.setattr(driver.judge_registry, "gate_bindings", lambda: {"ifeval"})
    monkeypatch.setenv("OI_MILES_IF_GATE_FAILURE_POLICY", "zero_guarded")
    monkeypatch.setattr(
        driver.judge_metrics,
        "start",
        lambda: SimpleNamespace(
            optimizer_boundary=SimpleNamespace(remote=AsyncMock(side_effect=metrics.optimizer_boundary)),
            seed_window=SimpleNamespace(remote=AsyncMock(side_effect=metrics.seed_window)),
        ),
    )
    monkeypatch.setattr(driver, "EvalDispatcher", Mock(return_value=SimpleNamespace(drain=AsyncMock())))
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(publication_mode="barrier", diagnostic_interval=0),
        fully_async=False,
        offload_rollout=False,
        check_weight_update_equal=False,
        start_rollout_id=1,
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        global_batch_size=32,
        save=str(tmp_path / "checkpoints"),
        save_trigger_sentinel=None,
        save_interval=None,
        update_weights_interval=1,
        eval_interval=None,
        debug_exit_after_rollout=None,
    )
    with pytest.raises(RuntimeError, match="11/2000 .*over 2 consecutive optimizer updates.*stopping training"):
        asyncio.run(driver.train(args))
    assert learner.train.await_count == 1
    records = [json.loads(line) for line in (tmp_path / judge_metrics.GUARD_RECORD).read_text().splitlines()]
    assert records[-1] == {"run": 7.0, "step": 2, "requests": 1000, "failures": 2, "unix": ANY}
