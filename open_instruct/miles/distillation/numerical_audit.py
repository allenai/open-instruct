"""Isolate repeated-scoring, process, graph, and request-history differences."""

import asyncio
import hashlib
import importlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tarfile
import time
from contextlib import ExitStack
from pathlib import Path
from urllib import request

from open_instruct.miles.distillation import alignment, panel, teacher


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
        marker = root / f"{mode}-trace-{server}" / "capture"
        capture = os.environ.get("OI_MILES_OPD_AUDIT_TRACE") == "1" and label == f"cold-{server}-1"
        if capture:
            marker.touch()
        try:
            values = await panel.score(endpoints[server] + "/generate", prefix, sequence, limit)
        finally:
            marker.unlink(missing_ok=True)
        if capture and not (marker.parent / "weights.json").exists():
            raise RuntimeError("Component tracing did not capture loaded weights")
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
    extra_pairs = []
    if os.environ.get("OI_MILES_OPD_AUDIT_NATIVE_PROFILE") == "1":
        # Fixed tokens isolate implementation differences without another generation pass.
        for length in (4096, 8192):
            sequence = (tokens * math.ceil(length / len(tokens)))[:length]
            for server in range(2):
                await score(f"length-{length}-{server}", server, sequence)
            extra_pairs.append((f"length-{length}-0", f"length-{length}-1"))
        sequences = [tokens, (tokens * 4)[:4096]]
        for server in range(2):
            batch = await asyncio.to_thread(
                panel.post,
                endpoints[server] + "/generate",
                {
                    "input_ids": [prefix + seq for seq in sequences],
                    "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                    "return_logprob": True,
                    "logprob_start_len": 0,
                },
            )
            if not isinstance(batch, list) or len(batch) != len(sequences):
                raise ValueError("Batched scoring returned the wrong result count")
            for index, (result, seq) in enumerate(zip(batch, sequences, strict=True)):
                label = f"batch-{index}-{server}"
                values = alignment.extract_scores(result, prefix + seq, len(seq))
                records[label] = values
                append(
                    root / "scores.jsonl",
                    {
                        "mode": mode,
                        "label": label,
                        "server": server,
                        "prefix": prefix,
                        "tokens": seq,
                        "scores": values,
                    },
                )
        extra_pairs.extend((f"batch-{i}-0", f"batch-{i}-1") for i in range(len(sequences)))
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
        *extra_pairs,
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
    tuning_path = os.environ.get("OI_MILES_OPD_AUDIT_TUNING")
    if tuning_path:
        (root / "requested-tuning.json").write_bytes(Path(tuning_path).read_bytes())
    native_profile = os.environ.get("OI_MILES_OPD_AUDIT_NATIVE_PROFILE") == "1"
    if native_profile and not tuning_path:
        raise ValueError("Native profile qualification needs recorded tuning choices")
    tracing = os.environ.get("OI_MILES_OPD_AUDIT_TRACE") == "1" or bool(tuning_path)
    modes = ("rounding-graphs", "rounding-eager", "off-eager", "core-eager")
    requested_modes = os.environ.get("OI_MILES_OPD_AUDIT_MODES")
    if requested_modes:
        requested_modes = tuple(requested_modes.split(","))
        if not set(requested_modes) <= set(modes):
            raise ValueError("Unknown audit mode")
        modes = requested_modes
    for mode in modes:
        print(f"Starting numerical audit mode {mode}; tracing={tracing}", flush=True)
        profile_path, profile_digest = None, None
        if native_profile:
            kernel_profile = importlib.import_module("olmo_sglang.kernel_profile")

            replay = root / f"{mode}-replay.json"
            replay.write_text(json.dumps(json.loads(Path(tuning_path).read_text())[mode]))
            profile_path = root / f"{mode}-kernel-profile.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "olmo_sglang.kernel_profile",
                    "import-replay",
                    str(replay),
                    str(profile_path),
                    "--model",
                    saved["same_checkpoint"],
                    "--arithmetic",
                    "full" if mode.startswith("core") else "rounding",
                ],
                check=True,
            )
            profile_digest = kernel_profile.digest(json.loads(profile_path.read_text()))
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
                cmd.remove("--disable-cuda-graph")
                cmd.extend(["--cuda-graph-backend-decode", "full" if mode.endswith("graphs") else "disabled"])
                if mode.endswith("graphs"):
                    cmd.extend(["--cuda-graph-max-bs-decode", "8"])
                cmd.extend(["--cuda-graph-backend-prefill", "disabled", "--random-seed", "130064"])
                if tracing:
                    cmd.extend(
                        [
                            "--forward-hooks",
                            json.dumps(
                                [
                                    {
                                        "name": "numerical-boundaries",
                                        "target_modules": ["model"],
                                        "hook_factory": "open_instruct.miles.distillation.audit_trace:factory",
                                        "config": {
                                            "root": str(root / f"{mode}-trace-{index}"),
                                            "base": "model",
                                            "tuning_path": None if native_profile else tuning_path,
                                            "tuning_mode": mode,
                                        },
                                    },
                                    {
                                        "name": "numerical-head",
                                        "target_modules": ["logits_processor"],
                                        "hook_factory": "open_instruct.miles.distillation.audit_trace:head_factory",
                                        "config": {"root": str(root / f"{mode}-trace-{index}")},
                                    },
                                ]
                            ),
                        ]
                    )
                env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": device,
                    "OLMO_SGLANG_REQUIRE_CORE_ROUNDING": "1",
                    "OLMO_SGLANG_DIAGNOSTIC_OVERRIDE_REASON": f"numerical audit mode {mode}; no training",
                }
                env["OLMO_SGLANG_CORE_COMPAT"] = {"rounding": "rounding", "off": "0", "core": "1"}[mode.split("-")[0]]
                if native_profile:
                    env.update(
                        {
                            "OLMO_SGLANG_KERNEL_PROFILE": str(profile_path),
                            "OLMO_SGLANG_KERNEL_PROFILE_MODE": "strict",
                            "OLMO_SGLANG_KERNEL_PROFILE_SHA256": profile_digest,
                            "OLMO_SGLANG_KERNEL_PROFILE_REPORT": str(
                                root / f"{mode}-trace-{index}" / "kernel-profile-usage.json"
                            ),
                        }
                    )
                handle = (root / f"{mode}-{index}.log").open("w")
                handles.append(handle)
                append(
                    root / "commands.jsonl",
                    {
                        "mode": mode,
                        "server": index,
                        "command": cmd,
                        "compat": env.get("OLMO_SGLANG_CORE_COMPAT", "auto"),
                        "diagnostic_override_reason": env["OLMO_SGLANG_DIAGNOSTIC_OVERRIDE_REASON"],
                        "kernel_profile_sha256": profile_digest,
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
            if tracing:
                with tarfile.open(f"/output/{mode}-traces.tar.gz", "w:gz") as archive:
                    for index in range(2):
                        archive.add(root / f"{mode}-trace-{index}", arcname=f"trace-{index}")
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
