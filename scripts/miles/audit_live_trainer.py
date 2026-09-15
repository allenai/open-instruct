"""Bounded read-only GPU/process/stack snapshots of an existing owned RL run."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from open_instruct.miles.run_spec import RunSpec


def node_snapshot():
    """Runs as a zero-GPU diagnostic task on each existing Ray node."""
    result = {"host": socket.gethostname(), "time_unix": time.time()}
    fields = "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm"
    for name, command in {
        "gpus": ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
        "processes": ["ps", "-eo", "pid,ppid,comm,pcpu,etimes"],
    }.items():
        p = subprocess.run(command, capture_output=True, text=True, timeout=15)
        result[name] = {"code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    actors = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            title = (directory / "cmdline").read_bytes().split(b"\0", 1)[0].decode(errors="replace")
        except OSError:
            continue
        if "ray::OLMoCoreTrainRayActor" in title:
            actors.append(int(directory.name))
    result["trainer_pids"] = actors
    result["stacks"] = {}
    executable = shutil.which("py-spy")
    if executable:
        for pid in actors[:8]:
            try:
                p = subprocess.run([executable, "dump", "--pid", str(pid)], capture_output=True, text=True, timeout=20)
                result["stacks"][str(pid)] = {"code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
            except subprocess.TimeoutExpired:
                result["stacks"][str(pid)] = {"error": "stack collection timed out"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    spec = RunSpec.load(args.config)
    root = Path(spec.output["root"])
    output = Path("/output/live-trainer")
    output.mkdir(parents=True, exist_ok=True)
    for path in (root / "checkpoints").glob("*.jsonl"):
        if path.stat().st_size <= 64 * 1024 * 1024:
            shutil.copyfile(path, output / path.name)
    heads = sorted((root / "cluster").glob("*/head.json"), key=lambda p: p.stat().st_mtime)
    if not heads:
        raise RuntimeError("No active-run Ray head manifest")
    address = json.loads(heads[-1].read_text())["address"]
    local = socket.gethostbyname(os.environ.get("BEAKER_NODE_HOSTNAME", socket.gethostname()))
    ray.init(address=address, _node_ip_address=local, log_to_driver=False)
    try:
        nodes = [node for node in ray.nodes() if node["Alive"]]
        function = ray.remote(num_cpus=0, num_gpus=0)(node_snapshot)
        for index in range(2):
            futures = [
                function.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
                ).remote()
                for node in nodes
            ]
            records = ray.get(futures, timeout=180)
            (output / f"snapshot-{index}.json").write_text(json.dumps(records, indent=2))
            for record in records:
                print(
                    json.dumps(
                        {"host": record["host"], "gpus": record["gpus"], "trainer_pids": record["trainer_pids"]}
                    ),
                    flush=True,
                )
            if index == 0:
                time.sleep(10)
    finally:
        ray.shutdown()
    print("LIVE_TRAINER_AUDIT_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
