"""Interrupted evaluations retain completed tasks and reject incompatible reuse."""

import copy
import json
import select
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from open_instruct.miles.evaluation import evaluation_runner as runner


@pytest.fixture
def receipt(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer_config.json"):
        (checkpoint / name).write_text("{}")
    return {
        "checkpoint": str(checkpoint),
        "update": 1000,
        "group_id": "two-tasks",
        "tasks": [{"task": name, "generation": {"temperature": 0.8}, "scoring": {}} for name in ("a", "b")],
        "evaluation": {"root": str(tmp_path), "revision": "pinned", "image": "image", "gpus": 1, "server_args": []},
        "training": {"wandb": {"mode": "offline"}},
    }


def write_results(output, task):
    runner.write_json(
        output / "metrics.json", {"tasks": [{"task": task, "instances_saved": 1, "metrics": {"score": {"mean": 0.5}}}]}
    )
    (output / f"{task}-predictions.jsonl").write_text(json.dumps({"model_output": [{"text": "answer"}]}) + "\n")


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    calls = []
    state = {"fail": "b"}

    def evaluate(args, **kwargs):
        task = args[args.index("--task") + 1]
        output = Path(args[args.index("--output-dir") + 1])
        calls.append(task)
        write_results(output, task)
        if state["fail"] == task:
            # Even apparently valid partial files must not commit a failed task.
            raise subprocess.CalledProcessError(143, args)

    server = Mock()
    server.poll.return_value = None
    start = Mock(return_value=server)
    monkeypatch.setattr(runner.subprocess, "Popen", start)
    monkeypatch.setattr(runner.subprocess, "run", evaluate)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "pinned\n")
    monkeypatch.setattr(runner.request, "urlopen", lambda *a, **kw: nullcontext(Mock(status=200)))
    copytree = runner.shutil.copytree

    def copy_results(src, dst, *args, **kwargs):
        destination = tmp_path / "beaker-output" if dst == "/output/evaluation" else dst
        return copytree(src, destination, *args, **kwargs)

    monkeypatch.setattr(runner.shutil, "copytree", copy_results)
    return calls, state, start


def test_preemption_skips_completed_task_and_preserves_failed_attempt(receipt, runtime):
    calls, state, start = runtime
    output = runner.result_directory(receipt)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_evaluation(receipt)
    assert calls == ["a", "b"]
    failed_directory, completed = runner.saved_task(output, receipt["tasks"][1])
    assert completed is None
    failed = list((failed_directory / "attempts").iterdir())
    assert len(failed) == 1
    before = (failed[0] / "metrics.json").read_bytes()

    state["fail"] = None
    runner.run_evaluation(receipt)
    assert calls == ["a", "b", "b"]
    assert (failed[0] / "metrics.json").read_bytes() == before
    assert runner.scores(output) == {"eval/a/score/mean": 0.5, "eval/b/score/mean": 0.5}
    assert json.loads((output / "evaluation.json").read_text())["status"] == "complete"

    # A retry after completion neither loads a server nor repeats generation.
    runner.run_evaluation({**receipt, "experiment_id": "replacement"})
    assert start.call_count == 2
    assert calls == ["a", "b", "b"]


def test_retry_after_last_task_before_group_marker_needs_no_server(receipt, runtime):
    calls, state, start = runtime
    state["fail"] = None
    with runner.evaluation_output(receipt) as output:
        runner.run_tasks(receipt, output)
    runner.run_evaluation(receipt)
    assert calls == ["a", "b"]
    start.assert_not_called()


@pytest.mark.parametrize("change", ["checkpoint", "tasks", "revision", "worker"])
def test_resume_refuses_changed_inputs_without_overwriting_provenance(receipt, change):
    with runner.evaluation_output(receipt) as output:
        original = (output / "provenance.json").read_bytes()
    changed = copy.deepcopy(receipt)
    if change == "checkpoint":
        changed["checkpoint"] = "different-model"
    elif change == "tasks":
        changed["tasks"][0]["generation"]["temperature"] = 1.0
    else:
        changed["evaluation"][change] = "different"
    with pytest.raises(ValueError, match="different inputs"), runner.evaluation_output(changed):
        pytest.fail("Incompatible results were reused")
    assert (output / "provenance.json").read_bytes() == original


def test_resume_refuses_unowned_nonempty_directory(receipt):
    output = runner.result_directory(receipt)
    output.mkdir(parents=True)
    (output / "precious.txt").write_text("keep me")
    with pytest.raises(ValueError, match="no provenance"), runner.evaluation_output(receipt):
        pytest.fail("Unknown output was accepted")
    assert (output / "precious.txt").read_text() == "keep me"


def test_old_complete_group_is_reused_after_artifact_validation(receipt, runtime):
    calls, _, start = runtime
    with runner.evaluation_output(receipt) as output:
        for task in receipt["tasks"]:
            directory = output / task["task"]
            directory.mkdir()
            write_results(directory, task["task"])
        runner.write_json(output / "evaluation.json", {"status": "complete"})
    runner.run_evaluation(receipt)
    assert not calls
    start.assert_not_called()


def test_incomplete_group_cannot_be_skipped_just_because_marker_exists(receipt, runtime):
    calls, _, start = runtime
    with runner.evaluation_output(receipt) as output:
        write_results(output, "a")
        runner.write_json(output / "evaluation.json", {"status": "complete"})
    with pytest.raises(ValueError, match="no successful scores.*b"):
        runner.run_evaluation(receipt)
    assert not calls
    start.assert_not_called()


def test_provider_failure_count_prevents_task_completion(tmp_path):
    write_results(tmp_path, "a")
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    metrics["tasks"][0]["instances_failed"] = 1
    runner.write_json(tmp_path / "metrics.json", metrics)
    with pytest.raises(ValueError, match="No successful numeric evaluation scores"):
        runner.completed_scores(tmp_path, [{"task": "a"}])


def test_lock_excludes_other_process_and_is_released_after_kill(receipt):
    code = (
        "import json,runpy,sys,time; "
        "m=runpy.run_path(sys.argv[1]); "
        "ctx=m['evaluation_output'](json.loads(sys.argv[2])); "
        "ctx.__enter__(); print('locked', flush=True); time.sleep(60)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, runner.__file__, json.dumps(receipt)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert select.select([child.stdout], [], [], 10)[0], "Lock holder did not start"
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="Another evaluator"), runner.evaluation_output(receipt):
            pytest.fail("Concurrent evaluator acquired the lock")
    finally:
        child.kill()
        child.wait(timeout=5)
        child.stdout.close()
    with runner.evaluation_output(receipt) as output:
        assert (output / "provenance.json").is_file()
