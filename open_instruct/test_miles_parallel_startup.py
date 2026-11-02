"""Startup readiness and failure propagation need no GPU or external service."""

import asyncio
import contextlib
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_instruct.miles.execution import cluster, startup


def test_parallel_join_really_overlaps_and_joins_cancelled_sibling():
    async def scenario(fail):
        entered = [asyncio.Event(), asyncio.Event()]
        finished = []

        async def work(i):
            entered[i].set()
            try:
                await entered[1 - i].wait()
                if fail and i == 0:
                    raise ValueError("initialization failed")
                if fail:
                    await asyncio.Event().wait()
                return i
            finally:
                finished.append(i)

        if fail:
            with pytest.raises(ValueError, match="initialization failed"):
                await startup.join(work(0), work(1))
        else:
            assert await startup.join(work(0), work(1)) == [0, 1]
        assert sorted(finished) == [0, 1]

    for fail in (False, True):
        asyncio.run(asyncio.wait_for(scenario(fail), 1))


def test_cancelling_startup_joins_all_work_before_returning():
    async def scenario():
        started, finished = asyncio.Event(), asyncio.Event()

        async def worker():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                finished.set()

        task = asyncio.create_task(startup.join(worker()))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(asyncio.wait_for(scenario(), 1))


@pytest.mark.parametrize("state", ["ready", "timeout", "failed", "delayed", "manual"])
def test_managed_barrier_and_manual_launch(tmp_path, monkeypatch, state):
    monkeypatch.setenv(startup.REPORT_DIR_ENV, str(tmp_path))
    cluster.write(tmp_path / "startup.json", {"deadline_unix": time.time() + (2 if state == "delayed" else -1)})
    if state in {"ready", "failed"}:
        cluster.write(tmp_path / "ray-ready.json", {})
    if state == "failed":
        cluster.write(tmp_path / "failed-1.json", {"error": "judge crashed"})
    if state == "manual":
        monkeypatch.delenv(startup.REPORT_DIR_ENV)

    async def scenario():
        if state == "timeout":
            with pytest.raises(TimeoutError, match="Startup barrier ray"):
                await startup.wait_ready("ray")
        elif state == "failed":
            with pytest.raises(RuntimeError, match="judge crashed"):
                await startup.wait_ready("ray")
        elif state == "delayed":
            task = asyncio.create_task(startup.wait_ready("ray"))
            await asyncio.sleep(0)
            assert not task.done()
            cluster.write(tmp_path / "ray-ready.json", {})
            await task
        else:
            await startup.wait_ready("ray")

    asyncio.run(scenario())


def supervisor(tmp_path):
    spec = SimpleNamespace(launch={"coordination": {"startup_timeout": 30, "heartbeat_timeout": 30}})
    return cluster.Supervisor(spec, tmp_path, 0, 1)


def test_judge_loading_does_not_count_as_liveness_failure(tmp_path, monkeypatch):
    owner = supervisor(tmp_path)
    owner.pending_health["judge"] = "http://localhost/health"
    http = Mock(side_effect=OSError("loading"))
    monkeypatch.setattr(cluster.request, "urlopen", http)
    owner.probe_health()
    assert owner.health_failures == {}
    assert not (tmp_path / "judge-ready-judge.json").exists()
    http.side_effect = lambda *a, **k: contextlib.nullcontext(SimpleNamespace(status=200))
    owner.probe_health()
    assert not owner.pending_health
    assert "judge" in owner.health
    assert (tmp_path / "judge-ready-judge.json").is_file()
    http.side_effect = OSError("lost judge")
    owner.probe_health()
    owner.probe_health()
    with pytest.raises(RuntimeError, match="three consecutive"):
        owner.probe_health()


@pytest.mark.parametrize("status", [503, None])
def test_pending_judge_has_a_deadline(tmp_path, monkeypatch, status):
    owner = supervisor(tmp_path)
    owner.started -= 31
    owner.pending_health["judge"] = "http://localhost/health"
    http = (
        Mock(side_effect=OSError("loading"))
        if status is None
        else Mock(return_value=contextlib.nullcontext(SimpleNamespace(status=status)))
    )
    monkeypatch.setattr(cluster.request, "urlopen", http)
    with pytest.raises(TimeoutError, match="Judge judge"):
        owner.probe_health()


def test_finite_startup_tasks_are_supervised_and_owned(tmp_path, monkeypatch):
    owner = supervisor(tmp_path)
    process = SimpleNamespace(poll=lambda: 0, returncode=0)
    monkeypatch.setattr(owner, "start", lambda *a, **k: process)
    owner.start_task("canaries", [], {})
    owner.check()  # A completed startup task is not a crashed long-lived service.
    process.returncode = 1
    process.poll = lambda: 1
    with pytest.raises(RuntimeError, match="canaries exited with 1"):
        owner.check()
    stopped = []
    monkeypatch.setattr(cluster, "terminate", stopped.append)
    monkeypatch.setattr(owner.thread, "join", lambda **k: None)
    owner.close()
    assert stopped == [process]


def test_judge_qualification_waits_for_http_without_waiting_for_ray(tmp_path, monkeypatch):
    spec = SimpleNamespace(judges={"judging": {"bindings": {"reward": {}}}})
    monkeypatch.setattr(cluster.judging, "registry", lambda _: {"judges": {"judge": {"mode": "managed"}}})
    cluster.write(tmp_path / "startup.json", {"deadline_unix": time.time() + 2})
    cluster.write(tmp_path / "judge-ready-judge.json", {})

    async def probe(output):
        assert not (tmp_path / "ray-ready.json").exists()
        assert not (tmp_path / "judges-ready.json").exists()

    monkeypatch.setattr(cluster.judge_registry, "probe", probe)
    cluster.probe_judges(spec, tmp_path)
    assert (tmp_path / "judges-ready.json").exists()
    stages = [json.loads(line)["stage"] for line in (tmp_path / "driver_timing.jsonl").read_text().splitlines()]
    assert stages == ["judge_http_readiness", "judge_canaries"]


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_workflow_preparation_overlaps_readiness_only_for_supported_driver(tmp_path, algorithm):
    spec = SimpleNamespace(training={"algorithm": algorithm})
    events = []

    def start_task(name, command, env):
        events.append(name)
        assert env[startup.REPORT_DIR_ENV] == str(tmp_path)
        assert cluster.read(tmp_path / "registry.json") == {"endpoint": "known"}
        assert not (tmp_path / "judges-ready.json").exists()

    def wait(predicate):
        assert not predicate()
        cluster.write(tmp_path / "ray-ready.json", {})
        cluster.write(tmp_path / "judges-ready.json", {})
        assert predicate()
        events.append("wait")

    def run_child(name, command, env):
        assert name == "driver"
        assert (tmp_path / "judges-ready.json").exists() == (algorithm != "grpo")
        events.append("prepare")

    owner = SimpleNamespace(
        root=tmp_path,
        rank=0,
        started=time.monotonic(),
        startup_timeout=30,
        start_task=start_task,
        run_child=run_child,
        wait=wait,
    )
    cluster.start_workflow(spec, "config.json", owner, {}, {"node": {"ray_gpus": 2}}, {"endpoint": "known"})
    assert events == ["ray-readiness", "judge-readiness"] + (["wait"] if algorithm != "grpo" else []) + ["prepare"]


def test_process_is_not_leaked_if_start_receipt_fails(tmp_path, monkeypatch):
    owner = supervisor(tmp_path)
    process = SimpleNamespace(pid=123)
    stopped = []
    monkeypatch.setattr(cluster.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(cluster, "write", Mock(side_effect=OSError("disk full")))
    monkeypatch.setattr(cluster, "terminate", stopped.append)
    try:
        with pytest.raises(OSError, match="disk full"):
            owner.start_task("probe", [], {})
        assert stopped == [process]
    finally:
        for stream in owner.logs:
            stream.close()
