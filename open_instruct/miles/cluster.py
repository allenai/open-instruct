"""Own a replicated Ray cluster and fixed-weight judges for one committed run.

All coordination lives under a unique submission UUID on shared WEKA. No Beaker
credential reaches training. Beaker propagates failure/preemption; heartbeats
also bound peer failures and startup hangs from inside the allocation.
"""

import argparse
import asyncio
import contextlib
import importlib
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from open_instruct.miles import judge_registry, judge_server, judging, topology
from open_instruct.miles.run_spec import RunSpec


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def free_port():
    with socket.socket() as stream:
        stream.bind(("0.0.0.0", 0))
        return stream.getsockname()[1]


def terminate(process):
    if process is None:
        return
    # Even a finished group leader may have surviving grandchildren.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=15)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=15)


def require_layout(nodes, expected):
    live = [node for node in nodes if node.get("Alive")]
    actual = {node["NodeManagerAddress"]: node.get("Resources", {}).get("GPU", 0) for node in live}
    if len(live) != len(actual) or actual != expected:
        raise RuntimeError(f"Ray ownership differs from planned per-node GPU pools: {actual} != {expected}")


class Supervisor:
    def __init__(self, spec, root, rank, count):
        self.spec, self.root, self.rank, self.count = spec, root, rank, count
        self.started = time.monotonic()
        self.startup_timeout = spec.launch["coordination"]["startup_timeout"]
        self.heartbeat_timeout = spec.launch["coordination"]["heartbeat_timeout"]
        self.children = []
        self.logs = []
        self.health = {}
        self.health_failures = {}
        self.next_health = 0.0
        self.log_offsets = {}
        self.stopping = threading.Event()
        self.failed = None
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def heartbeat(self):
        while not self.stopping.is_set():
            try:
                write(self.root / f"heartbeat-{self.rank}.json", {"time": time.time(), "rank": self.rank})
                for rank in range(self.count):
                    path = self.root / f"heartbeat-{rank}.json"
                    if path.exists():
                        if time.time() - read(path)["time"] > self.heartbeat_timeout:
                            raise RuntimeError(f"Replica {rank} heartbeat expired")
                    elif time.monotonic() - self.started > self.startup_timeout:
                        raise TimeoutError(f"Replica {rank} did not rendezvous")
                for _, process in self.children:
                    if not (self.root / "complete.json").exists() and process.poll() is not None:
                        raise RuntimeError(f"Managed process {process.args[:3]} exited with {process.returncode}")
                if not (self.root / "complete.json").exists() and time.monotonic() >= self.next_health:
                    self.probe_health()
                    self.next_health = time.monotonic() + 10
            except BaseException as error:
                self.failed = error
                return
            self.stopping.wait(2)

    def probe_health(self):
        for name, url in tuple(self.health.items()):
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status != 200:
                        raise RuntimeError("Unhealthy HTTP response")
                self.health_failures[name] = 0
            except (OSError, RuntimeError) as error:
                self.health_failures[name] = self.health_failures.get(name, 0) + 1
                write(
                    self.root / f"judge-health-{self.rank}.json",
                    {"failures": self.health_failures, "error": str(error)},
                )
                if self.health_failures[name] >= 3:
                    raise RuntimeError(f"Judge {name}: three consecutive liveness failures") from error

    def relay_logs(self):
        for stream in self.logs:
            with Path(stream.name).open("rb") as reader:
                reader.seek(self.log_offsets.get(stream.name, 0))
                raw = reader.read(16384)
                self.log_offsets[stream.name] = reader.tell()
            if raw:
                print(f"[{Path(stream.name).name}] " + raw.decode(errors="replace"), end="", flush=True)

    def check(self):
        self.relay_logs()
        if self.failed:
            raise self.failed
        failures = list(self.root.glob("failed-*.json"))
        if failures:
            raise RuntimeError(f"Peer failure: {read(failures[0])}")

    def wait(self, predicate, *, timeout=None):
        deadline = time.monotonic() + (self.startup_timeout if timeout is None else timeout)
        while True:
            self.check()
            result = predicate()
            if result:
                return result
            if time.monotonic() > deadline:
                raise TimeoutError("Cluster startup/readiness deadline expired")
            time.sleep(1)

    def start(self, label, command, env, *, monitor=True):
        log = (self.root / f"{label}-{self.rank}.log").open("w")
        self.logs.append(log)
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if monitor:
            self.children.append((label, process))
        return process

    def run_child(self, label, command, env):
        process = self.start(label, command, env, monitor=False)
        try:
            while process.poll() is None:
                self.check()
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(
                    f"{label} exited with {process.returncode}; see {self.root}/{label}-{self.rank}.log"
                )
        finally:
            terminate(process)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=5)
        for _, process in reversed(self.children):
            terminate(process)
        for stream in self.logs:
            stream.close()


def run(path):
    spec = RunSpec.load(path)
    layout = topology.plan(spec)
    if (Path(spec.output["root"]) / "workflow.json").exists() and not spec.launch["auto_resume"]:
        raise RuntimeError(
            "This run directory already has a workflow; choose a new output.root before starting services"
        )
    rank = int(os.environ.get("OI_MILES_REPLICA_RANK", os.environ.get("BEAKER_REPLICA_RANK", "0")))
    count = int(os.environ.get("OI_MILES_REPLICA_COUNT", os.environ.get("BEAKER_REPLICA_COUNT", "1")))
    if count != layout["replicas"] or rank not in range(count):
        raise RuntimeError("Beaker replica topology differs from the submitted run")
    attempt = os.environ["OI_MILES_LAUNCH_ID"]
    root = Path(spec.output["root"]) / "cluster" / attempt
    root.mkdir(parents=True, exist_ok=True)
    address = socket.gethostbyname(os.environ.get("BEAKER_NODE_HOSTNAME", socket.gethostname()))
    supervisor = Supervisor(spec, root, rank, count)

    def interrupt(signum, frame):
        raise RuntimeError(f"Replica received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    supervisor.thread.start()
    try:
        write(root / f"node-{rank}.json", {"address": address, "rank": rank})
        paths = [root / f"node-{index}.json" for index in range(count)]
        supervisor.wait(lambda: all(p.is_file() for p in paths))
        assignments = topology.assign(layout, [read(p)["address"] for p in paths])
        node = assignments[address]
        print(f"Replica {rank} address={address} assignment={node}; reports={root}", flush=True)
        head = min(assignments, key=ipaddress.IPv4Address)
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mask:
            visible = mask.split(",")
        else:
            result = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True)
            visible = [str(index) for index, _ in enumerate(result.strip().splitlines())]
        if any(not device.isdecimal() for device in visible):
            raise RuntimeError("The pinned MILES serving adapter requires numeric CUDA_VISIBLE_DEVICES")
        if len(visible) != layout["gpus_per_replica"]:
            raise RuntimeError("Visible devices differ from the Beaker GPU allocation")
        ray_devices, judges = topology.devices(node, visible)
        write(
            root / f"placement-{rank}.json",
            {"address": address, "head": head, "ray_devices": ray_devices, "judge_devices": judges, "layout": layout},
        )
        env = dict(os.environ)
        registry = judging.registry(spec.judges)
        health = {}
        for name, devices in judges.items():
            service = registry["judges"][name]
            port = free_port()
            service["endpoint"] = f"http://{address}:{port}/v1"
            command = judge_server.command(service, port)
            judge_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(devices))
            # The fixed judge is a standard Qwen model, not an Olmo extension.
            judge_env.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
            judge_env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "false"
            supervisor.start("judge-" + name, command, judge_env)
            health[name] = f"http://127.0.0.1:{port}/health"
            write(root / f"judge-{name}.json", service)
        for name, url in health.items():

            def ready(url=url):
                try:
                    with urllib.request.urlopen(url, timeout=3) as response:
                        return response.status == 200
                except OSError:
                    return False

            supervisor.wait(ready)
            supervisor.health[name] = url
            write(root / f"judge-ready-{name}.json", {"endpoint": registry["judges"][name]["endpoint"]})
        for name, service in registry["judges"].items():
            if service["mode"] == "managed":
                supervisor.wait(lambda name=name: (root / f"judge-ready-{name}.json").exists())
                registry["judges"][name] = read(root / f"judge-{name}.json")
        env[judging.REGISTRY_ENV] = json.dumps(registry)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(ray_devices)
        env["RAY_NODE_IP_ADDRESS"] = address
        if address == head:
            port = free_port()
            write(root / "head.json", {"address": f"{head}:{port}"})
        supervisor.wait(lambda: (root / "head.json").exists())
        ray_address = read(root / "head.json")["address"]
        ray_command = [
            "ray",
            "start",
            "--block",
            "--node-ip-address",
            address,
            "--num-gpus",
            str(node["ray_gpus"]),
            "--num-cpus",
            "32",
            "--disable-usage-stats",
        ]
        ray_command += [
            "--dashboard-agent-listen-port",
            str(free_port()),
            "--dashboard-agent-grpc-port",
            str(free_port()),
            "--runtime-env-agent-port",
            str(free_port()),
        ]
        if address == head:
            ray_command += [
                "--head",
                "--port",
                str(port),
                "--include-dashboard=false",
                "--ray-client-server-port",
                str(free_port()),
                "--dashboard-port",
                str(free_port()),
            ]
        else:
            ray_command += ["--address", ray_address]
        print(f"Replica {rank}: starting Ray with {node['ray_gpus']} policy GPUs", flush=True)
        supervisor.start("ray", ray_command, env)
        env["RAY_ADDRESS"] = ray_address
        if address == head:
            # Run readiness/probes in a child so heartbeats and failure monitoring
            # continue while Ray or HTTP operations block.
            write(root / "expected-ray.json", {ip: n["ray_gpus"] for ip, n in assignments.items()})
            supervisor.run_child(
                "readiness",
                [sys.executable, "-m", "open_instruct.miles.cluster", str(path), "--probe", str(root)],
                env,
            )
            write(root / "registry.json", registry)
            print(f"Cluster ready; starting Core workflow. Driver log: {root}/driver-{rank}.log", flush=True)
            supervisor.run_child("driver", [sys.executable, "-m", "open_instruct.miles", "train", str(path)], env)
            write(root / "complete.json", {"status": "complete", "time": time.time()})
        else:
            # No startup timeout on the training phase; Beaker's run timeout is
            # authoritative and the peer heartbeat still bounds lost processes.
            while not (root / "complete.json").exists():
                supervisor.check()
                time.sleep(2)
        # All replicas acknowledge completion before the head stops Ray.
        write(root / f"done-{rank}.json", {"status": "complete"})
        supervisor.wait(lambda: all((root / f"done-{i}.json").exists() for i in range(count)), timeout=60)
    except BaseException as error:
        write(root / f"failed-{rank}.json", {"error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        supervisor.close()
        write(root / f"cleanup-{rank}.json", {"complete": True})


def probe(spec, root):
    ray = importlib.import_module("ray")
    ray.init(address=os.environ["RAY_ADDRESS"])
    expected = read(root / "expected-ray.json")
    deadline = time.monotonic() + spec.launch["coordination"]["startup_timeout"]
    try:
        while True:
            nodes = ray.nodes()
            if len([n for n in nodes if n.get("Alive")]) >= len(expected):
                require_layout(nodes, expected)
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Ray replicas did not join")
            time.sleep(2)
        write(root / "ray-layout.json", {"expected": expected, "actual": nodes})
    finally:
        ray.shutdown()
    if spec.judges["judging"]["bindings"]:
        asyncio.run(judge_registry.probe(root / "judge-canaries.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--probe", type=Path)
    args = parser.parse_args()
    if args.probe:
        probe(RunSpec.load(args.config), args.probe)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
