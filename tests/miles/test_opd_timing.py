"""Timing must preserve outputs, cancellation, semaphore capacity and driver semantics."""

import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.miles import summarize_opd_timing

from open_instruct.miles.distillation import opd_gpu_monitor, opd_timing, opd_timing_driver, opd_timing_memory


def records(root):
    return [json.loads(line) for p in root.glob("events-*.jsonl") for line in p.read_text().splitlines()]


def test_disabled_does_not_write(tmp_path, monkeypatch):
    monkeypatch.delenv(opd_timing.ENV, raising=False)
    with opd_timing.stage("test"):
        pass
    assert not list(tmp_path.iterdir())


def test_teacher_wait_cancellation_preserves_capacity(tmp_path, monkeypatch):
    monkeypatch.setenv(opd_timing.ENV, str(tmp_path))
    sample = SimpleNamespace(metadata={}, index=2, group_index=1)

    async def exercise():
        semaphore = asyncio.Semaphore(0)

        async def waiting():
            async with opd_timing.teacher_slot(semaphore, sample):
                pytest.fail("must not acquire")

        task = asyncio.create_task(waiting())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert semaphore._value == 0
        semaphore.release()
        async with opd_timing.teacher_slot(semaphore, sample):
            assert semaphore._value == 0
        assert semaphore._value == 1

    asyncio.run(exercise())
    assert [r["outcome"] for r in records(tmp_path)] == ["cancelled", "completed"]


def test_await_preserves_result_and_exception(tmp_path, monkeypatch):
    monkeypatch.setenv(opd_timing.ENV, str(tmp_path))
    result = object()

    async def good():
        return result

    async def bad():
        raise RuntimeError("original failure")

    async def exercise():
        assert await opd_timing.awaited(good(), "learner_train", 3) is result
        with pytest.raises(RuntimeError, match="original failure"):
            await opd_timing.awaited(bad(), "learner_train", 4)

    asyncio.run(exercise())
    assert [r["outcome"] for r in records(tmp_path)] == ["completed", "RuntimeError"]


def test_writer_failure_does_not_mask_work(tmp_path, monkeypatch):
    bad = tmp_path / "file"
    bad.write_text("occupied")
    monkeypatch.setenv(opd_timing.ENV, str(bad))
    with opd_timing.stage("work"):
        pass


def test_driver_changes_only_await_instrumentation():
    package = importlib.util.find_spec("miles")
    if package is None:
        pytest.skip("Pinned native driver required; run in experiment image")
    source = (Path(package.origin).parents[1] / "train_async.py").read_text()
    transformed = opd_timing_driver.transform(source)

    class RemoveObservation(ast.NodeTransformer):
        def visit_Await(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_opd_timed_await"
            ):
                node.value = node.value.args[0]
            return node

    restored = RemoveObservation().visit(transformed)
    assert ast.dump(restored) == ast.dump(ast.parse(source))
    compile(transformed, "<timed-driver>", "exec")


def test_unknown_driver_fails_before_training():
    with pytest.raises(ValueError, match="Unsupported async driver"):
        opd_timing_driver.transform("async def train(args):\n    return 1\n")


def test_gpu_missing_data_stays_missing_and_roles_are_explicit():
    rows = opd_gpu_monitor.parse(
        "0, GPU-abc, 90, 20, 1000, 2000, 180\n7, GPU-def, [N/A], 0, 500, 2000, 90", {"trainer": [0], "teacher": [7]}
    )
    assert rows[0]["role"] == "trainer" and rows[1]["role"] == "teacher"
    assert rows[1]["utilization.gpu"] is None


def test_concurrent_request_seconds_are_not_wall_time():
    report = summarize_opd_timing.summarize(
        [
            dict(stage="student_request", started_unix=0, seconds=10, outcome="completed"),
            dict(stage="student_request", started_unix=5, seconds=10, outcome="cancelled"),
        ]
    )
    stage = report["stages"]["student_request"]
    assert stage["summed_seconds"] == 20 and stage["observed_union_seconds"] == 15
    assert stage["outcomes"]["cancelled"] == 1


def test_memory_hook_preserves_optimizer_and_records_each_step(monkeypatch, tmp_path):
    monkeypatch.setenv(opd_timing.ENV, str(tmp_path))
    resets = []
    cuda = opd_timing_memory.torch.cuda
    for name, value in [
        ("current_device", 0),
        ("max_memory_allocated", 10),
        ("max_memory_reserved", 20),
        ("memory_allocated", 5),
        ("memory_reserved", 15),
        ("mem_get_info", (80, 100)),
    ]:
        monkeypatch.setattr(cuda, name, lambda *args, v=value: v)
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda device: resets.append(device))
    calls = []
    optimizer = SimpleNamespace(step=lambda value: calls.append(value) or ("original", value))
    for rollout in [0, 1]:
        opd_timing_memory.before_train_step(SimpleNamespace(rank=2), rollout, 0, None, optimizer, None)
        assert optimizer.step(rollout) == ("original", rollout)
    assert calls == [0, 1] and resets == [0, 0]
    rows = records(tmp_path)
    assert [r["rollout_id"] for r in rows] == [0, 1]
    assert all(r["peak_allocated_bytes"] == 10 and r["rank"] == 2 for r in rows)
