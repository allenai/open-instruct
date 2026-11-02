"""Manage the Ray cluster and fixed-weight judge services inside a Beaker allocation.
Each replica joins a fresh attempt directory on shared WEKA, starts its assigned
processes and reports readiness, completion or failure to its peers. Supervision
and heartbeats bound startup hangs and peer failures alongside Beaker's propagation
settings, while the leader runs the training workflow without Beaker credentials.
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
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib import request

from open_instruct.miles.configuration import inference_profiles, topology, validation
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.execution import rendezvous, startup
from open_instruct.miles.execution.timing import stage
from open_instruct.miles.infrastructure import gpu_hardware, infra_timeouts
from open_instruct.miles.rewards import judge_registry, judge_server, judging


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def free_ports(count):
    """Select distinct local ports together, then release them for service startup.

    Holding every socket until selection finishes prevents duplicates within this
    batch. Other processes can still claim a port after these sockets close.
    """
    ports = []
    with contextlib.ExitStack() as sockets:
        for _ in range(count):
            stream = sockets.enter_context(socket.socket())
            stream.bind(("0.0.0.0", 0))
            ports.append(stream.getsockname()[1])
    return ports


def terminate(process):
    if process is None:
        return
    # Even a finished group leader may have surviving grandchildren.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=infra_timeouts.seconds(15))
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=infra_timeouts.seconds(15))


def require_layout(nodes, expected):
    live = [node for node in nodes if node.get("Alive")]
    actual = {node["NodeManagerAddress"]: node.get("Resources", {}).get("GPU", 0) for node in live}
    if len(live) != len(actual) or actual != expected:
        raise RuntimeError(f"Ray ownership differs from planned per-node GPU pools: {actual} != {expected}")


class Supervisor:
    def __init__(self, spec, root, rank, count):
        self.spec, self.root, self.rank, self.count = spec, root, rank, count
        self.started = time.monotonic()
        self.startup_timeout = infra_timeouts.seconds(spec.launch["coordination"]["startup_timeout"])
        self.heartbeat_timeout = infra_timeouts.seconds(spec.launch["coordination"]["heartbeat_timeout"])
        self.heartbeat_warning_at = {}
        self.children = []
        self.startup_tasks = []
        self.pending_health = {}
        self.logs = []
        self.health = {}
        self.health_failures = {}
        self.peer_heartbeats = {}
        self.next_health = 0.0
        self.log_offsets = {}
        self.stopping = threading.Event()
        self.failed = None
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def heartbeat(self):
        while not self.stopping.is_set():
            try:
                write(self.root / f"heartbeat-{self.rank}.json", {"time": time.time(), "rank": self.rank})
                self.check_peer_heartbeats()
                for _, process in self.children:
                    if not (self.root / "complete.json").exists() and process.poll() is not None:
                        raise RuntimeError(f"Managed process {process.args[:3]} exited with {process.returncode}")
                if not (self.root / "complete.json").exists() and time.monotonic() >= self.next_health:
                    self.probe_health()
                    self.next_health = time.monotonic() + (2 if self.pending_health else 10)
            except BaseException as error:
                self.failed = error
                return
            self.stopping.wait(2)

    def check_peer_heartbeats(self):
        now = time.time()
        for rank in range(self.count):
            # A missing directory entry during shared-filesystem publication is
            # not evidence that an already joined peer never rendezvoused.
            # Retain its last observed heartbeat, without extending its deadline.
            try:
                observed = read(self.root / f"heartbeat-{rank}.json")["time"]
            except FileNotFoundError:
                observed = self.peer_heartbeats.get(rank)
            else:
                self.peer_heartbeats[rank] = observed
            if observed is None:
                if time.monotonic() - self.started > self.startup_timeout:
                    raise TimeoutError(f"Replica {rank} did not rendezvous")
            else:
                age = now - observed
                if age > min(
                    30, self.spec.launch["coordination"]["heartbeat_timeout"]
                ) and now >= self.heartbeat_warning_at.get(rank, 0):
                    infra_timeouts.logger.warning(
                        "Replica heartbeat stale: observer=%s replica=%s age_s=%.1f deadline_s=%s",
                        self.rank,
                        rank,
                        age,
                        self.heartbeat_timeout,
                    )
                    self.heartbeat_warning_at[rank] = now + 30
                if age > self.heartbeat_timeout:
                    raise RuntimeError(f"Replica {rank} heartbeat expired")

    def probe_health(self):
        # Judges load on their own GPUs while Ray and the workflow start. Only
        # promote a service into steady-state liveness checks once HTTP is ready.
        for name, url in tuple(self.pending_health.items()):
            try:
                with request.urlopen(url, timeout=infra_timeouts.seconds(3)) as response:
                    if response.status != 200:
                        raise OSError(f"Judge HTTP status {response.status}")
            except OSError as error:
                if time.monotonic() - self.started > self.startup_timeout:
                    raise TimeoutError(f"Judge {name} did not become ready before the startup deadline") from error
                continue
            self.health[name] = url
            self.pending_health.pop(name)
            write(self.root / f"judge-ready-{name}.json", {"ready_unix": time.time()})
        for name, url in tuple(self.health.items()):
            try:
                with request.urlopen(url, timeout=infra_timeouts.seconds(5)) as response:
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
        for label, process in self.startup_tasks:
            if process.poll() not in (None, 0):
                raise RuntimeError(f"Startup task {label} exited with {process.returncode}; see {self.root}")
        failures = list(self.root.glob("failed-*.json"))
        if failures:
            raise RuntimeError(f"Peer failure: {read(failures[0])}")

    def wait(self, predicate, *, timeout=None):
        deadline = time.monotonic() + (self.startup_timeout if timeout is None else infra_timeouts.seconds(timeout))
        started = time.monotonic()
        next_warning = started + 30
        while True:
            if time.monotonic() >= next_warning:
                infra_timeouts.logger.warning(
                    "Cluster readiness still waiting: replica=%s elapsed_s=%.1f deadline_s=%.1f",
                    self.rank,
                    time.monotonic() - started,
                    deadline - started,
                )
                next_warning = time.monotonic() + 30
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
        try:
            write(self.root / f"process-{label}-{self.rank}.json", {"started_unix": time.time(), "pid": process.pid})
        except BaseException:
            terminate(process)
            raise
        if monitor:
            self.children.append((label, process))
        return process

    def start_task(self, label, command, env):
        """Own a finite startup subprocess without treating success as a crash."""
        process = self.start(label, command, env, monitor=False)
        self.startup_tasks.append((label, process))
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
        for _, process in reversed(self.children + self.startup_tasks):
            terminate(process)
        for stream in self.logs:
            stream.close()


def run(path):
    spec = RunSpec.load(path)
    ray_cpus = validation.ray_cpus(os.environ)
    compiled = spec.compile()
    layout = topology.plan(spec, compiled)
    if (Path(spec.output["root"]) / "workflow.json").exists() and not spec.launch["auto_resume"]:
        raise RuntimeError(
            "This run directory already has a workflow; choose a new output.root before starting services"
        )
    rank = int(os.environ.get("OI_MILES_REPLICA_RANK", os.environ.get("BEAKER_REPLICA_RANK", "0")))
    count = int(os.environ.get("OI_MILES_REPLICA_COUNT", os.environ.get("BEAKER_REPLICA_COUNT", "1")))
    if count != layout["replicas"] or rank not in range(count):
        raise RuntimeError("Beaker replica topology differs from the submitted run")
    attempt = os.environ["OI_MILES_LAUNCH_ID"]
    root = rendezvous.join(
        Path(spec.output["root"]) / "cluster" / attempt,
        rank,
        count,
        infra_timeouts.seconds(spec.launch["coordination"]["startup_timeout"]),
    )
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
        inventory = gpu_hardware.inventory()
        visible = [device["id"] for device in inventory]
        if len(visible) != layout["gpus_per_replica"]:
            raise RuntimeError("Visible devices differ from the Beaker GPU allocation")
        ray_devices, judges = topology.devices(node, visible)
        inventory_report = gpu_hardware.rollout_inventory(node, inventory)
        profile = compiled.inference_profile
        if node["rollout_gpus"]:
            inference_profiles.validate_hardware(profile, inventory_report["hardware"])
        write(root / f"inference-profile-{rank}.json", {"profile": profile, "address": address, **inventory_report})
        # Three Ray agent ports everywhere; the head needs three more Ray ports.
        ports = iter(free_ports(len(judges) + (6 if address == head else 3)))
        write(
            root / f"placement-{rank}.json",
            {
                "address": address,
                "head": head,
                "ray_devices": ray_devices,
                "judge_devices": judges,
                "layout": layout,
                "ray_cpus": ray_cpus,
            },
        )
        env = dict(os.environ)
        registry = judging.registry(spec.judges)
        for name, devices in judges.items():
            service = registry["judges"][name]
            port = next(ports)
            service["endpoint"] = f"http://{address}:{port}/v1"
            command = judge_server.command(service, port)
            judge_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(devices))
            # The fixed judge is a standard Qwen model, not an Olmo extension.
            judge_env.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
            judge_env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "false"
            supervisor.start("judge-" + name, command, judge_env)
            supervisor.pending_health[name] = f"http://127.0.0.1:{port}/health"
            write(root / f"judge-{name}.json", service)
        for name, service in registry["judges"].items():
            if service["mode"] == "managed":
                supervisor.wait(lambda name=name: (root / f"judge-{name}.json").exists())
                registry["judges"][name] = read(root / f"judge-{name}.json")
        env[judging.REGISTRY_ENV] = json.dumps(registry)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(ray_devices)
        env["RAY_NODE_IP_ADDRESS"] = address
        if address == head:
            port = next(ports)
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
            str(ray_cpus),
            "--disable-usage-stats",
        ]
        ray_command += [
            "--dashboard-agent-listen-port",
            str(next(ports)),
            "--dashboard-agent-grpc-port",
            str(next(ports)),
            "--runtime-env-agent-port",
            str(next(ports)),
        ]
        if address == head:
            ray_command += [
                "--head",
                "--port",
                str(port),
                "--include-dashboard=false",
                "--ray-client-server-port",
                str(next(ports)),
                "--dashboard-port",
                str(next(ports)),
            ]
        else:
            ray_command += ["--address", ray_address]
        print(f"Replica {rank}: starting Ray with {node['ray_gpus']} policy GPUs", flush=True)
        supervisor.start("ray", ray_command, env)
        env["RAY_ADDRESS"] = ray_address
        env["OI_MILES_INFERENCE_REPORT_DIR"] = str(root)
        if address == head:
            start_workflow(spec, path, supervisor, env, assignments, registry)
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


def start_workflow(spec, path, supervisor, env, assignments, registry):
    """Start preparation immediately, with separate Ray and judge readiness gates."""
    root, rank = supervisor.root, supervisor.rank
    # Endpoint discovery is complete; readiness is deliberately separate.
    # The driver can prepare model/data while Ray and judges initialize.
    write(root / "expected-ray.json", {ip: n["ray_gpus"] for ip, n in assignments.items()})
    write(root / "registry.json", registry)
    write(
        root / "startup.json",
        {"deadline_unix": time.time() + max(0, supervisor.startup_timeout - (time.monotonic() - supervisor.started))},
    )
    env[startup.REPORT_DIR_ENV] = str(root)
    for label, option in (("ray-readiness", "--probe"), ("judge-readiness", "--probe-judges")):
        supervisor.start_task(
            label, [sys.executable, "-m", "open_instruct.miles.execution.cluster", str(path), option, str(root)], env
        )
    if getattr(spec, "training", {}).get("algorithm", "grpo") != "grpo":
        # Other drivers retain their established all-services-ready boundary.
        supervisor.wait(lambda: (root / "ray-ready.json").exists() and (root / "judges-ready.json").exists())
    print(f"Startup branches launched; preparing Core workflow. Driver log: {root}/driver-{rank}.log", flush=True)
    supervisor.run_child("driver", [sys.executable, "-m", "open_instruct.miles", "train", str(path)], env)


def probe(spec, root):
    ray = importlib.import_module("ray")
    ray.init(address=os.environ["RAY_ADDRESS"])
    expected = read(root / "expected-ray.json")
    deadline = time.monotonic() + infra_timeouts.seconds(spec.launch["coordination"]["startup_timeout"])
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
    write(root / "ray-ready.json", {"ready_unix": time.time()})


def probe_judges(spec, root):
    async def qualify():
        registry = judging.registry(spec.judges)
        paths = [
            root / f"judge-ready-{name}.json"
            for name, service in registry["judges"].items()
            if service["mode"] == "managed"
        ]
        with stage(SimpleNamespace(save=root), "judge_http_readiness"):
            while not all(path.is_file() for path in paths):
                await asyncio.sleep(0.2)
        if spec.judges["judging"]["bindings"]:
            with stage(SimpleNamespace(save=root), "judge_canaries"):
                await judge_registry.probe(root / "judge-canaries.json")
        write(root / "judges-ready.json", {"ready_unix": time.time()})

    remaining = read(root / "startup.json")["deadline_unix"] - time.time()
    asyncio.run(asyncio_wait(qualify(), remaining))


async def asyncio_wait(operation, timeout):
    # This absolute deadline already includes the infrastructure multiplier.
    return await asyncio.wait_for(operation, timeout=max(0, timeout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument("--probe", type=Path)
    probes.add_argument("--probe-judges", type=Path)
    args = parser.parse_args()
    if args.probe:
        with stage(SimpleNamespace(save=args.probe), "ray_cluster_readiness"):
            probe(RunSpec.load(args.config), args.probe)
    elif args.probe_judges:
        probe_judges(RunSpec.load(args.config), args.probe_judges)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
