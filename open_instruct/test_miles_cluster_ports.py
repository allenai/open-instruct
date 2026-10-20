"""Local startup selects distinct ports without leaking temporary sockets."""

import contextlib
import socket
from types import SimpleNamespace

import pytest

from open_instruct.miles.execution import cluster


@pytest.mark.parametrize("count", [3, 6, 8])
@pytest.mark.parametrize("fail", [False, True])
def test_free_ports_hold_entire_batch_and_close_on_exit(monkeypatch, count, fail):
    streams = []

    class Socket(socket.socket):
        def bind(self, address):
            # No earlier selection may be released while another is being made.
            assert all(stream.fileno() >= 0 for stream in streams)
            streams.append(self)
            if fail and len(streams) == count:
                raise OSError("injected bind failure")
            return super().bind(address)

    monkeypatch.setattr(cluster.socket, "socket", Socket)
    if fail:
        with pytest.raises(OSError, match="injected bind failure"):
            cluster.free_ports(count)
    else:
        ports = cluster.free_ports(count)
        assert len(set(ports)) == count
        assert all(0 < port < 65536 for port in ports)
    assert len(streams) == count
    assert all(stream.fileno() == -1 for stream in streams)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("judge_count", [0, 2])
def test_startup_assigns_one_port_batch_to_ray_and_local_judges(tmp_path, monkeypatch, rank, judge_count):
    addresses = ["10.0.0.1", "10.0.0.2"]
    judges = {f"judge-{i}": [str(i + 1)] for i in range(judge_count)}
    spec = SimpleNamespace(
        output={"root": str(tmp_path)},
        launch={"auto_resume": False, "coordination": {"startup_timeout": 30}},
        judges={},
    )
    node = {"ray_gpus": 1}
    commands = {}
    batches = []

    class StartedRay(Exception):
        pass

    class Supervisor:
        def __init__(self, *args):
            self.thread = SimpleNamespace(start=lambda: None)
            self.health = {}

        def wait(self, predicate):
            assert predicate()

        def start(self, name, command, env):
            commands[name] = command
            if name == "ray":
                raise StartedRay

        def close(self):
            pass

    def allocate(count):
        ports = list(range(20000, 20000 + count))
        batches.append(ports)
        return ports

    monkeypatch.setattr(cluster.RunSpec, "load", lambda path: spec)
    monkeypatch.setattr(cluster.topology, "plan", lambda spec: {"replicas": 2, "gpus_per_replica": 1 + judge_count})
    monkeypatch.setattr(cluster.topology, "assign", lambda *args: dict.fromkeys(addresses, node))
    monkeypatch.setattr(cluster.topology, "devices", lambda *args: (["0"], judges))
    monkeypatch.setattr(cluster.rendezvous, "join", lambda *args: tmp_path)
    monkeypatch.setattr(cluster.socket, "gethostbyname", lambda name: addresses[rank])
    monkeypatch.setattr(cluster.signal, "signal", lambda *args: None)
    monkeypatch.setattr(cluster, "Supervisor", Supervisor)
    monkeypatch.setattr(cluster, "free_ports", allocate)
    monkeypatch.setattr(
        cluster.judging, "registry", lambda spec: {"judges": {name: {"mode": "managed"} for name in judges}}
    )
    monkeypatch.setattr(cluster.judge_server, "command", lambda service, port: ["judge", "--port", str(port)])
    monkeypatch.setattr(
        cluster.request, "urlopen", lambda *args, **kwargs: contextlib.nullcontext(SimpleNamespace(status=200))
    )
    monkeypatch.setenv("OI_MILES_REPLICA_RANK", str(rank))
    monkeypatch.setenv("OI_MILES_REPLICA_COUNT", "2")
    monkeypatch.setenv("OI_MILES_LAUNCH_ID", "test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(1 + judge_count)))
    for index, address in enumerate(addresses):
        cluster.write(tmp_path / f"node-{index}.json", {"address": address})
    if rank == 1:
        cluster.write(tmp_path / "head.json", {"address": f"{addresses[0]}:19000"})

    with pytest.raises(StartedRay):
        cluster.run("unused.toml")

    assert len(batches) == 1
    selected = []
    for command in commands.values():
        selected.extend(int(command[i + 1]) for i, flag in enumerate(command) if flag.endswith("port"))
    assert len(selected) == (6 if rank == 0 else 3) + judge_count
    assert sorted(selected) == batches[0]
    for name in judges:
        port = int(commands["judge-" + name][-1])
        assert cluster.read(tmp_path / f"judge-{name}.json")["endpoint"] == f"http://{addresses[rank]}:{port}/v1"
    if rank == 0:
        ray = commands["ray"]
        assert cluster.read(tmp_path / "head.json")["address"] == f"{addresses[0]}:{ray[ray.index('--port') + 1]}"
