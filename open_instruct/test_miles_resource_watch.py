"""Verify retained evidence captures memory pressure and survives actor death."""

from open_instruct.miles.infrastructure import resource_watch


def test_memory_snapshot_identifies_actor_without_logging_arguments(tmp_path):
    proc = tmp_path / "proc"
    actor = proc / "815"
    actor.mkdir(parents=True)
    (actor / "cmdline").write_bytes(b"ray::RolloutExecutor\0private-argument")
    (actor / "status").write_text("VmRSS:\t123 kB\nThreads:\t4\nOther:\tprivate\n")
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.events").write_text("oom 1\noom_kill 1\n")
    data = resource_watch.memory_snapshot(proc, cgroup)
    assert data["coordinators"] == [{"pid": 815, "VmRSS": "123 kB", "Threads": "4"}]
    assert "oom_kill 1" in data["cgroup"]["memory.events"]
    assert "private" not in str(data)


def test_preserve_worker_tail_after_process_is_gone(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "worker-id-job-815.err").write_bytes(b"old-data-DEATH")
    (logs / "worker-id-job-999.err").write_bytes(b"unrelated")
    (logs / "raylet.err").write_bytes(b"RAY-DEATH")
    monkeypatch.setattr(resource_watch, "_LIMIT", 5)
    output = tmp_path / "output"
    output.mkdir()
    resource_watch.preserve_logs(output, {815}, logs)
    target = output / "ray-worker-diagnostics"
    assert (target / "worker-id-job-815.err").read_bytes() == b"DEATH"
    assert (target / "raylet.err").read_bytes() == b"DEATH"
    assert not (target / "worker-id-job-999.err").exists()
