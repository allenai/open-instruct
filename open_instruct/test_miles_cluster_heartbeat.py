"""Missing heartbeat reads must not erase evidence that a peer joined."""

from types import SimpleNamespace

import pytest

from open_instruct.miles.execution import cluster


def supervisor(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(cluster.time, "time", lambda: clock.now)
    monkeypatch.setattr(cluster.time, "monotonic", lambda: clock.now)
    spec = SimpleNamespace(launch={"coordination": {"startup_timeout": 20, "heartbeat_timeout": 10}})
    return cluster.Supervisor(spec, tmp_path, 0, 2), clock


def test_missing_read_of_joined_peer_uses_last_heartbeat(tmp_path, monkeypatch):
    worker, clock = supervisor(tmp_path, monkeypatch)
    clock.now += 30  # Already past startup, as in the live failure.
    for rank in range(2):
        cluster.write(tmp_path / f"heartbeat-{rank}.json", {"time": clock.now})
    worker.check_peer_heartbeats()
    (tmp_path / "heartbeat-1.json").unlink()
    clock.now += 2
    worker.check_peer_heartbeats()
    cluster.write(tmp_path / "heartbeat-1.json", {"time": clock.now})
    worker.check_peer_heartbeats()
    assert worker.peer_heartbeats[1] == clock.now


def test_missing_peer_still_expires_without_deadline_extension(tmp_path, monkeypatch):
    worker, clock = supervisor(tmp_path, monkeypatch)
    for rank in range(2):
        cluster.write(tmp_path / f"heartbeat-{rank}.json", {"time": clock.now})
    worker.check_peer_heartbeats()
    (tmp_path / "heartbeat-1.json").unlink()
    for elapsed in (4, 8, 11):
        clock.now = 1000 + elapsed
        cluster.write(tmp_path / "heartbeat-0.json", {"time": clock.now})
        if elapsed <= 10:
            worker.check_peer_heartbeats()
        else:
            with pytest.raises(RuntimeError, match="Replica 1 heartbeat expired"):
                worker.check_peer_heartbeats()


def test_unseen_peer_still_has_startup_deadline(tmp_path, monkeypatch):
    worker, clock = supervisor(tmp_path, monkeypatch)
    cluster.write(tmp_path / "heartbeat-0.json", {"time": clock.now})
    worker.check_peer_heartbeats()
    clock.now += 21
    cluster.write(tmp_path / "heartbeat-0.json", {"time": clock.now})
    with pytest.raises(TimeoutError, match="Replica 1 did not rendezvous"):
        worker.check_peer_heartbeats()


def test_present_stale_heartbeat_still_fails(tmp_path, monkeypatch):
    worker, clock = supervisor(tmp_path, monkeypatch)
    cluster.write(tmp_path / "heartbeat-0.json", {"time": clock.now})
    cluster.write(tmp_path / "heartbeat-1.json", {"time": clock.now - 11})
    with pytest.raises(RuntimeError, match="Replica 1 heartbeat expired"):
        worker.check_peer_heartbeats()
