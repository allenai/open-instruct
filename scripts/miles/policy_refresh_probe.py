"""Measure stock retract + GPU publication + fresh prefill on live KDA requests.

No optimizer is run: the sender holds a controlled changed checkpoint on GPU.
Original sampling logprobs/routes are retained independently of rescoring.
"""

import argparse
import concurrent.futures
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import requests
import torch
from miles.backends.fsdp_utils import update_weight_utils
from miles.utils import distributed_utils
from safetensors import safe_open
from torch import distributed as dist
from transformers import AutoTokenizer

URL = "http://127.0.0.1:31000"
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=32)


def rpc(endpoint, payload=None):
    response = requests.post(URL + "/" + endpoint, json=payload or {}, timeout=600)
    response.raise_for_status()
    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
        return {"body": response.text}
    if isinstance(result, dict) and result.get("success") is False:
        raise RuntimeError(f"{endpoint} rejected: {result}")
    return result


class Stream:
    def __init__(self, payload):
        self.payload = dict(payload, stream=True)
        self.latest = None
        self.first = None
        self.received = []
        self.changed = threading.Event()
        self.started = time.perf_counter()

    def run(self):
        with requests.post(URL + "/generate", json=self.payload, stream=True, timeout=900) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                self.latest = json.loads(line[6:])
                now = time.perf_counter()
                self.first = self.first or now
                self.received.append([now, self.latest["meta_info"]["completion_tokens"]])
                self.changed.set()
        self.finished = time.perf_counter()
        return self.latest


def wait_tokens(streams, count, futures):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        for future in futures:
            if future.done():
                future.result()
                raise RuntimeError("A response finished before the requested interruption")
        if all(s.latest and s.latest["meta_info"]["completion_tokens"] >= count for s in streams):
            return
        time.sleep(0.01)
    raise TimeoutError("No progress to interruption boundary")


class Publisher:
    def __init__(self, root):
        torch.cuda.set_device(0)
        self.weights = []
        for shard in sorted(root.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cuda:0") as handle:
                names = handle.keys()
                self.weights.extend((name, handle.get_tensor(name)) for name in names)
        if not self.weights:
            raise ValueError("No safetensors weights found")
        self.changed = []
        # A controlled, deliberately visible perturbation of attention and router
        # weights. This is not claimed to represent a particular RL optimizer step.
        for name, tensor in self.weights:
            if name.startswith("model.layers.0.") and ("q_proj.weight" in name or name.endswith("gate.weight")):
                tensor.mul_(1.015625)
                self.changed.append(name)
        if not self.changed:
            raise ValueError("Fixture has no selected attention/router weights")
        self.bytes = sum(t.numel() * t.element_size() for _, t in self.weights)
        self.group_name = "policy-refresh"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        pending = POOL.submit(
            rpc,
            "init_weights_update_group",
            {
                "master_address": "127.0.0.1",
                "master_port": port,
                "rank_offset": 1,
                "world_size": 2,
                "group_name": self.group_name,
                "backend": "nccl",
            },
        )
        self.group = distributed_utils.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=2,
            rank=0,
            group_name=self.group_name,
            timeout=timedelta(seconds=120),
        )
        pending.result(timeout=150)

    def publish(self, version):
        torch.cuda.set_device(0)
        started = time.perf_counter()
        rpc("begin_weight_update")
        buckets, size, batch = 0, 0, []

        def send():
            flat = update_weight_utils.FlattenedTensorBucket(named_tensors=batch).get_flattened_tensor()
            pending = POOL.submit(
                rpc,
                "update_weights_from_distributed",
                {
                    "names": [n for n, _ in batch],
                    "shapes": [list(t.shape) for _, t in batch],
                    "dtypes": [str(t.dtype).removeprefix("torch.") for _, t in batch],
                    "group_name": self.group_name,
                    "weight_version": str(version),
                    "load_format": "flattened_bucket",
                    "flush_cache": False,
                },
            )
            dist.broadcast(flat, 0, group=self.group, async_op=True).wait()
            pending.result(timeout=150)

        for name, tensor in self.weights:
            nbytes = tensor.numel() * tensor.element_size()
            if size and size + nbytes > 1024**3:
                send()
                buckets += 1
                size, batch = 0, []
            batch.append((name, tensor))
            size += nbytes
        if batch:
            send()
            buckets += 1
        rpc("end_weight_update")
        delivered = time.perf_counter()
        rpc("flush_cache")
        return {
            "transfer_seconds": delivered - started,
            "flush_seconds": time.perf_counter() - delivered,
            "bytes": self.bytes,
            "buckets": buckets,
        }


def request(prompt, rid, new_tokens, temperature=0):
    return {
        "rid": rid,
        "input_ids": prompt,
        "sampling_params": {"temperature": temperature, "max_new_tokens": new_tokens, "ignore_eos": True},
        "return_logprob": True,
        "logprob_start_len": 0,
        "return_routed_experts": True,
    }


def run_case(root, publisher, prompt, label, batch, length, cut, version, refresh):
    rpc("flush_cache")
    streams = [Stream(request(prompt, f"{label}-{i}", length)) for i in range(batch)]
    futures = [POOL.submit(s.run) for s in streams]
    wait_tokens(streams, cut, futures)
    boundary = time.perf_counter()
    pause_seconds = 0
    if refresh:
        rpc("pause_generation", {"mode": "retract"})
        paused = time.perf_counter()
        pause_seconds = paused - boundary
        publication = publisher.publish(version)
        ready = time.perf_counter()
        rpc("continue_generation")
        resumed = time.perf_counter()
    else:
        for future in futures:
            future.result(timeout=900)
        paused = time.perf_counter()
        publication = publisher.publish(version)
        ready = resumed = time.perf_counter()
    results = [future.result(timeout=900) for future in futures]
    report = {
        "label": label,
        "refresh": refresh,
        "batch": batch,
        "prompt_tokens": len(prompt),
        "output_tokens_per_response": length,
        "publication": publication,
        "drain_wait_seconds": 0 if refresh else paused - boundary,
        "pause_including_instrumentation_seconds": pause_seconds,
        "boundary_to_weights_ready_seconds": ready - boundary,
        "request_batch_seconds": max(s.finished for s in streams) - min(s.started for s in streams),
        "responses": [],
    }
    for stream, result in zip(streams, results):
        rid = stream.payload["rid"]
        meta = result["meta_info"]
        output = result["output_ids"]
        assert len(output) == length and meta["finish_reason"]["type"] == "length", meta
        assert len(meta["output_token_logprobs"]) == length, "lost/duplicated behavior logprobs"
        entry = {"rid": rid, "version_spans": meta.get("weight_versions"), "retractions": meta.get("retraction_count")}
        if refresh:
            before = json.loads((root / "trace" / f"{rid}-0.json").read_text())
            kept = len(before["output_ids"])
            assert 0 < kept < length
            assert output[:kept] == before["output_ids"], "prefix changed"
            assert [x[0] for x in meta["output_token_logprobs"][:kept]] == before["behavior_logprobs"], (
                "behavior overwritten"
            )
            assert [x[1] for x in meta["output_token_logprobs"][:kept]] == before["behavior_token_ids"]
            after = [(stamp, n) for stamp, n in stream.received if stamp >= resumed and n > kept]
            assert after, "no continuation after refresh"
            entry.update(
                retained_old_tokens=kept,
                current_policy_fraction=(length - kept) / length,
                resume_to_first_new_token_seconds=after[0][0] - resumed,
                instrumentation_seconds=before["instrumentation_seconds"],
            )
            # Independent empty-cache prefill of EXACT retained prefix. Compare
            # greedy continuation and teacher-forced old-prefix scores separately.
            rpc("flush_cache")
            reference = rpc("generate", request(prompt + output[:kept], rid + "-reference", min(16, length - kept)))
            count = len(reference["output_ids"])
            ref_lp = [x[0] for x in reference["meta_info"]["output_token_logprobs"]]
            got_lp = [x[0] for x in meta["output_token_logprobs"][kept : kept + count]]
            entry["fresh_prefill_token_matches"] = sum(
                a == b for a, b in zip(reference["output_ids"], output[kept : kept + count])
            )
            entry["fresh_prefill_tokens_compared"] = count
            entry["fresh_prefill_max_logprob_error"] = float(np.max(np.abs(np.array(ref_lp) - got_lp)))
            # The reference input now contains the OLD sampled tokens, scored
            # under current weights. Preserve both; never relabel the old draw.
            scores = reference["meta_info"]["input_token_logprobs"][len(prompt) : len(prompt) + kept]
            assert len(scores) == kept
            assert [x[1] for x in scores] == output[:kept]
            drift = np.array([x[0] for x in scores]) - before["behavior_logprobs"]
            entry["old_prefix_rescore_mean_abs_logprob_delta"] = float(np.abs(drift).mean())
            entry["old_prefix_rescore_max_abs_logprob_delta"] = float(np.abs(drift).max())
            entry["old_behavior_routes_saved"] = before["routes_shape"]
            (root / f"{rid}-reference.json").write_text(json.dumps(reference))
        report["responses"].append(entry)
        (root / f"{rid}.json").write_text(json.dumps(result))
    (root / f"{label}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--radix", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    model = args.model or root / "tiny"
    if args.model is None:
        spec = importlib.util.spec_from_file_location(
            "fixture", "/opt/core-rl/sources/olmo-sglang/tools/create_tiny_parity_checkpoint.py"
        )
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        fixture.build_checkpoint(model, profile="hero-hybrid-moe", max_position_embeddings=6144)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    env = dict(
        os.environ, SGLANG_EXTERNAL_MODEL_PACKAGE="olmo_sglang.models", POLICY_REFRESH_TRACE=str(root / "trace")
    )
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model),
        "--port",
        "31000",
        "--base-gpu-id",
        "1",
        "--tp-size",
        "1",
        "--trust-remote-code",
        "--context-length",
        "6144",
        "--mem-fraction-static",
        "0.65",
        "--max-running-requests",
        "16",
        "--max-total-tokens",
        "65536",
        "--max-mamba-cache-size",
        "32",
        "--chunked-prefill-size",
        "8192",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--skip-server-warmup",
        "--enable-return-routed-experts",
        "--weight-version",
        "0",
        "--stream-interval",
        "1",
    ]
    if not args.radix:
        command.append("--disable-radix-cache")
    log = (root / "server.log").open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited {process.returncode}; see server.log")
            try:
                if requests.get(URL + "/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Server startup exceeded 20 minutes")
        publisher = Publisher(model)
        # Warm kernels and transport outside measured cases. Subsequent version
        # changes use this SAME changed checkpoint to isolate state-refresh cost.
        prompt = (
            [5 + i % 30 for i in range(16)]
            if args.model is None
            else tokenizer.encode(
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": "A shop has 120 apples and sells 37. Explain step by step how many remain.",
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                add_special_tokens=False,
            )
        )
        rpc("generate", request(prompt, "warm", 32))
        publisher.publish(1)
        reports = []
        for i, (label, p, batch, length, cut, refresh) in enumerate(
            [
                ("drain-short", prompt, 4, 256, 32, False),
                ("refresh-short", prompt, 4, 256, 32, True),
                ("refresh-long", (prompt * (1024 // len(prompt) + 1))[:1024], 4, 512, 128, True),
            ],
            start=2,
        ):
            # Change attention weights between versions; affects recurrent state.
            for name, tensor in publisher.weights:
                if name in publisher.changed:
                    tensor.mul_(1.015625 if refresh else 1.0)
            reports.append(run_case(root, publisher, p, label, batch, length, cut, i, refresh))
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "model": str(model),
                    "radix": args.radix,
                    "gpu": torch.cuda.get_device_name(0),
                    "changed_parameters": publisher.changed,
                    "scope": "serving/publication prototype; no optimizer or RL quality claim",
                    "cases": reports,
                },
                indent=2,
            )
        )
    finally:
        os.killpg(process.pid, signal.SIGTERM) if process.poll() is None else None
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        log.close()
        POOL.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
