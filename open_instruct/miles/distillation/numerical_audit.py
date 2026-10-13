"""Isolate repeated-scoring, process, graph, and request-history differences."""

import asyncio
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from urllib import request

from open_instruct.miles.distillation import panel, teacher


def compare(left, right):
    if not left or len(left) != len(right):
        raise ValueError("Comparison requires equal nonempty token arrays")
    delta = [a - b for a, b in zip(left, right, strict=True)]
    if not all(math.isfinite(x) for x in delta):
        raise ValueError("Nonfinite score difference")
    absolute = sorted(abs(x) for x in delta)
    return {
        "tokens": len(delta),
        "signed_mean": sum(delta) / len(delta),
        "mae": sum(absolute) / len(delta),
        "max": absolute[-1],
        "p50": absolute[len(delta) // 2],
        "p99": absolute[min(len(delta) - 1, int(len(delta) * 0.99))],
        "exact_fraction": sum(x == 0 for x in delta) / len(delta),
    }


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


async def measure(endpoints, saved, root, mode):
    prefix, tokens = saved["prefix"], saved["tokens"]
    limit = asyncio.Semaphore(1)
    records = {}

    async def score(label, server, sequence=tokens):
        start = time.monotonic()
        values = await panel.score(endpoints[server] + "/generate", prefix, sequence, limit)
        records[label] = values
        append(
            root / "scores.jsonl",
            {
                "mode": mode,
                "label": label,
                "server": server,
                "prefix": prefix,
                "tokens": sequence,
                "scores": values,
                "seconds": time.monotonic() - start,
            },
        )
        return values

    # Both processes first see the same scoring workload, before any long decode.
    for repeat in range(3):
        for server in range(2):
            await score(f"cold-{server}-{repeat}", server)
    for server in range(2):
        await score(f"short-{server}", server, tokens[:256])
    result = await asyncio.to_thread(
        panel.post,
        endpoints[0] + "/generate",
        {
            "input_ids": prefix,
            "return_logprob": True,
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": 1024,
                "sampling_seed": 130064,
                "skip_special_tokens": False,
            },
        },
    )
    entries = result["meta_info"]["output_token_logprobs"]
    generated = [v[1] for v in entries]
    decode = [v[0] for v in entries]
    append(
        root / "scores.jsonl",
        {"mode": mode, "label": "decode", "server": 0, "prefix": prefix, "tokens": generated, "scores": decode},
    )
    for repeat in range(2):
        for server in range(2):
            await score(f"after-decode-{server}-{repeat}", server)
    fresh = [await score(f"generated-prefill-{i}", i, generated) for i in range(2)]
    comparisons = {}
    for left, right in [
        ("cold-0-0", "cold-1-0"),
        ("cold-0-0", "cold-0-1"),
        ("cold-1-0", "cold-1-1"),
        ("cold-0-1", "cold-0-2"),
        ("cold-1-1", "cold-1-2"),
        ("cold-0-0", "after-decode-0-0"),
        ("cold-1-0", "after-decode-1-0"),
        ("after-decode-0-0", "after-decode-1-0"),
        ("short-0", "short-1"),
    ]:
        comparisons[f"{left} vs {right}"] = compare(records[left], records[right])
    comparisons["decode vs generated-prefill-0"] = compare(decode, fresh[0])
    comparisons["decode vs generated-prefill-1"] = compare(decode, fresh[1])
    comparisons["saved-prefill-0 vs cold-0-0"] = compare(saved["same_server_prefill"], records["cold-0-0"])
    result = {"mode": mode, "comparisons": comparisons}
    append(root / "summary.jsonl", result)
    print(json.dumps(result), flush=True)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    root = Path(spec["output"]["root"])
    root.mkdir(parents=True, exist_ok=False)
    source = Path(spec["launch"]["env"]["OI_MILES_OPD_AUDIT_INPUT"])
    saved = json.loads(source.read_text())
    if saved["same_checkpoint"] != spec["model"]["source"]:
        raise ValueError("Audit must use the original calibration weights")
    (root / "input.json").write_bytes(source.read_bytes())
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "model": saved["same_checkpoint"],
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "runtime": json.loads(Path("/opt/core-rl/runtime/miles/runtime.lock.json").read_text()),
            },
            indent=2,
        )
    )
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) != 2:
        raise ValueError("Expected two GPUs")
    service = {
        "source": saved["same_checkpoint"],
        "tensor_parallel_size": 1,
        "max_context_length": 32768,
        "concurrency": 8,
    }
    for mode in ("default-graphs", "default-eager", "core-eager"):
        servers, handles, endpoints = [], [], []
        try:
            with ExitStack() as stack:
                ports = []
                for _ in devices:
                    sock = stack.enter_context(socket.socket())
                    sock.bind(("0.0.0.0", 0))
                    ports.append(sock.getsockname()[1])
            for index, (device, port) in enumerate(zip(devices, ports, strict=True)):
                cmd = teacher.command(service, port)
                if mode == "default-graphs":
                    cmd.remove("--disable-cuda-graph")
                    cmd.extend(["--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", "8"])
                cmd.extend(["--cuda-graph-backend-prefill", "disabled", "--random-seed", "130064"])
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": device}
                if mode == "core-eager":
                    env["OLMO_SGLANG_CORE_COMPAT"] = "1"
                handle = (root / f"{mode}-{index}.log").open("w")
                handles.append(handle)
                append(
                    root / "commands.jsonl",
                    {
                        "mode": mode,
                        "server": index,
                        "command": cmd,
                        "compat": env.get("OLMO_SGLANG_CORE_COMPAT", "auto"),
                        "device": device,
                    },
                )
                servers.append(
                    subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                )
                endpoints.append(f"http://127.0.0.1:{port}")
            deadline = time.monotonic() + 900
            pending = set(endpoints)
            while pending:
                if any(p.poll() is not None for p in servers):
                    raise RuntimeError(f"{mode} server exited; inspect logs")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{mode} startup timeout")
                for endpoint in list(pending):
                    try:
                        with request.urlopen(endpoint + "/health", timeout=2):
                            pending.remove(endpoint)
                    except OSError:
                        pass
                if pending:
                    time.sleep(2)
            asyncio.run(measure(endpoints, saved, root, mode))
        finally:
            for process in servers:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for process in servers:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for handle in handles:
                handle.close()


if __name__ == "__main__":
    main()
