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
    monkeypatch.setattr(cluster.time, "time", lambda: owner.startup_deadline + 1)
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
    stages = [
        json.loads(line)["stage"]
        for line in (tmp_path / "judge-readiness" / "driver_timing.jsonl").read_text().splitlines()
    ]
    assert stages == ["judge_http_readiness", "judge_canaries"]


def test_workflow_preparation_overlaps_readiness_without_resetting_deadline(tmp_path):
    cluster.write(tmp_path / "startup.json", {"deadline_unix": 123})
    events = []

    def start_task(name, command, env):
        events.append(name)
        assert env[startup.REPORT_DIR_ENV] == str(tmp_path)
        assert cluster.read(tmp_path / "registry.json") == {"endpoint": "known"}
        assert not (tmp_path / "judges-ready.json").exists()

    def run_child(name, command, env):
        assert name == "driver"
        assert not (tmp_path / "judges-ready.json").exists()
        assert startup.deadline(tmp_path) == 123
        events.append("prepare")

    owner = SimpleNamespace(root=tmp_path, rank=0, start_task=start_task, run_child=run_child)
    cluster.start_workflow("config.json", owner, {}, {"node": {"ray_gpus": 2}}, {"endpoint": "known"})
    assert events == ["ray-readiness", "judge-readiness", "prepare"]


def test_late_replica_driver_and_judge_share_deadline_but_cleanup_gets_own_budget(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(cluster.time, "time", lambda: clock.now)
    monkeypatch.setattr(cluster.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    first = supervisor(tmp_path)
    clock.now += 20
    late = cluster.Supervisor(first.spec, tmp_path, 1, 2)
    assert first.startup_deadline == late.startup_deadline == startup.deadline(tmp_path) == 1030
    clock.now = 1031
    for owner in (first, late):
        with pytest.raises(TimeoutError, match="Cluster startup"):
            owner.wait(lambda: False)
    late.pending_health["judge"] = "http://localhost/health"
    monkeypatch.setattr(cluster.request, "urlopen", Mock(side_effect=OSError("loading")))
    with pytest.raises(TimeoutError, match="Judge judge"):
        late.probe_health()
    monkeypatch.setenv(startup.REPORT_DIR_ENV, str(tmp_path))
    with pytest.raises(TimeoutError, match="Startup barrier ray"):
        asyncio.run(startup.wait_ready("ray"))
    with pytest.raises(TimeoutError):
        cluster.probe_judges(SimpleNamespace(judges={}), tmp_path)
    late.wait(lambda: clock.now >= 1033, timeout=5)
    assert clock.now == 1033


def test_missing_deadline_publication_is_bounded(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(cluster.time, "time", lambda: clock.now)
    monkeypatch.setattr(cluster.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    spec = SimpleNamespace(launch={"coordination": {"startup_timeout": 2, "heartbeat_timeout": 30}})
    with pytest.raises(TimeoutError, match="Replica zero"):
        cluster.Supervisor(spec, tmp_path, 1, 2)
    assert clock.now == 1002


@pytest.mark.parametrize("ready", [True, False])
def test_ray_probe_uses_remaining_attempt_budget_and_separate_timeline(tmp_path, monkeypatch, ready):
    clock = SimpleNamespace(now=1028.0)
    monkeypatch.setattr(cluster.time, "time", lambda: clock.now)
    monkeypatch.setattr(cluster.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    cluster.write(tmp_path / "startup.json", {"deadline_unix": 1030})
    cluster.write(tmp_path / "expected-ray.json", {"node": 2})
    nodes = [{"Alive": True, "NodeManagerAddress": "node", "Resources": {"GPU": 2}}]
    ray = SimpleNamespace(init=Mock(), shutdown=Mock(), nodes=Mock(return_value=nodes if ready else []))
    monkeypatch.setattr(cluster.importlib, "import_module", lambda name: ray)
    monkeypatch.setenv("RAY_ADDRESS", "node:1234")
    if ready:
        cluster.probe(tmp_path)
        assert (tmp_path / "ray-ready.json").is_file()
    else:
        with pytest.raises(TimeoutError, match="Ray replicas"):
            cluster.probe(tmp_path)
        assert clock.now == 1030
        assert not (tmp_path / "ray-ready.json").exists()
    ray.shutdown.assert_called_once()
    monkeypatch.setattr(cluster.judging, "registry", lambda _: {"judges": {}})
    # The judge timeline cannot append to (or overwrite) the Ray timeline.
    ray_log = tmp_path / "ray-readiness" / "driver_timing.jsonl"
    recorded = ray_log.read_bytes()
    clock.now = 1029
    cluster.probe_judges(SimpleNamespace(judges={"judging": {"bindings": {}}}), tmp_path)
    assert ray_log.read_bytes() == recorded
    assert (tmp_path / "judge-readiness" / "driver_timing.jsonl").is_file()
    assert not (tmp_path / "driver_timing.jsonl").exists()


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
