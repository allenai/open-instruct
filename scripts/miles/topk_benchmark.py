"""Paired, blocked standalone SGLang output-top-k benchmark.

The client runs on the server host. Network residual includes HTTP scheduling and
loopback transfer, not a remote network estimate. Request CPU timings are nested
inside end-to-end latency; GPU stream spans overlap and are reported separately.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import importlib
import json
import mmap
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

MODES = [("none", 0, False), ("selected", 0, False)] + [
    (f"{fmt}_{k}", k, fmt == "flat") for k in (32, 64, 128) for fmt in ("nested", "flat")
]


def payload(ids, mode, seed, tokens, temperature=1.0):
    name, k, flat = mode
    return {
        "input_ids": ids,
        "sampling_params": {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "sampling_seed": seed,
            "max_new_tokens": tokens,
            "ignore_eos": True,
        },
        "return_logprob": name != "none",
        "logprob_start_len": -1,
        "top_logprobs_num": k,
        "return_text_in_logprobs": False,
        "return_flat_raw_output_top_logprobs": flat,
        "return_flat_raw_top_logprobs_b64": flat,
        "stream": False,
    }


def unpack(meta, k, flat, tokens):
    if not k:
        return None, None
    if flat:
        assert meta["output_top_logprobs_shape"] == [tokens, k]
        assert meta["output_top_logprobs_null_prefix"] == 0
        assert meta["output_top_logprobs_val_flat_b64_dtype"] == "float32"
        assert meta["output_top_logprobs_idx_flat_b64_dtype"] == "int32"
        values = np.frombuffer(
            base64.b64decode(meta["output_top_logprobs_val_flat_b64"], validate=True), dtype="<f4"
        ).reshape(tokens, k)
        indices = np.frombuffer(
            base64.b64decode(meta["output_top_logprobs_idx_flat_b64"], validate=True), dtype="<i4"
        ).reshape(tokens, k)
        assert not meta.get("output_top_logprobs")
    else:
        rows = meta["output_top_logprobs"]
        assert len(rows) == tokens and all(len(row) == k for row in rows)
        values = np.asarray([[v[0] for v in row] for row in rows], dtype=np.float32)
        indices = np.asarray([[v[1] for v in row] for row in rows], dtype=np.int32)
    return values, indices


def request(client, ids, mode, seed, tokens, temperature=1.0, retain=False):
    started = time.perf_counter()
    body = json.dumps(payload(ids, mode, seed, tokens, temperature)).encode()
    packed = time.perf_counter()
    with client.stream("POST", "/generate", content=body, headers={"content-type": "application/json"}) as response:
        headers_received = time.perf_counter()
        raw = response.read()
        received = time.perf_counter()
        response.raise_for_status()
        serialization_s = float(response.headers["x-benchmark-json-seconds"])
    data = json.loads(raw)
    parsed = time.perf_counter()
    meta = data["meta_info"]
    arrays = unpack(meta, mode[1], mode[2], tokens)
    unpacked = time.perf_counter()
    assert meta["completion_tokens"] == tokens, meta
    assert len(meta.get("input_top_logprobs", [])) <= 1
    if mode[0] != "none":
        assert len(meta["output_token_logprobs"]) == tokens
    if arrays[0] is not None:
        assert not np.isnan(arrays[0]).any()
        assert (arrays[0] <= 1e-5).all()
        assert (arrays[1] >= 0).all()
    row = {
        "mode": mode[0],
        "prompt_tokens": len(ids),
        "completion_tokens": tokens,
        "client_request_pack_s": packed - started,
        "http_headers_wait_s": headers_received - packed,
        "body_read_s": received - headers_received,
        "json_parse_s": parsed - received,
        "array_unpack_s": unpacked - parsed,
        "total_s": unpacked - started,
        "server_e2e_s": meta["e2e_latency"],
        "server_pack_s": meta.get("benchmark_pack_s", 0.0),
        "server_json_s": serialization_s,
        "response_bytes": len(raw),
        "first_token_latency_s": meta.get("first_token_latency"),
        "decode_throughput": meta.get("decode_throughput"),
        "text_sha256": hashlib.sha256(data["text"].encode()).hexdigest(),
    }
    # This is a residual, not pure network time. Keep signed values to expose timing inconsistencies.
    row["http_residual_s"] = received - packed - row["server_e2e_s"] - serialization_s
    row["server_other_s"] = row["server_e2e_s"] - row["server_pack_s"]
    return (row, data["text"], arrays) if retain else row


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def summarize(rows, blocks):
    result = {}
    for name, _, _ in MODES:
        selected = [r for r in rows if r["mode"] == name]
        times = [r["total_s"] for r in selected]
        total = sum(times)
        bs = [b for b in blocks if b["mode"] == name]
        result[name] = {
            "requests": len(selected),
            "mean_s": statistics.mean(times),
            "p50_s": float(np.quantile(times, 0.5)),
            "p95_s": float(np.quantile(times, 0.95)),
            "tokens_per_s": sum(r["completion_tokens"] for r in selected) / sum(b["seconds"] for b in bs),
            "mean_response_bytes": statistics.mean(r["response_bytes"] for r in selected),
            "mean_stage_s": {
                key: statistics.mean(r[key] for r in selected)
                for key in (
                    "server_e2e_s",
                    "server_pack_s",
                    "server_json_s",
                    "server_other_s",
                    "body_read_s",
                    "json_parse_s",
                    "array_unpack_s",
                    "http_residual_s",
                )
            },
            "latency_fraction": {
                key: sum(r[key] for r in selected) / total
                for key in (
                    "client_request_pack_s",
                    "server_other_s",
                    "server_pack_s",
                    "server_json_s",
                    "http_residual_s",
                    "json_parse_s",
                    "array_unpack_s",
                )
            },
        }
    baseline = result["none"]
    for value in result.values():
        value["mean_latency_overhead_pct"] = 100 * (value["mean_s"] / baseline["mean_s"] - 1)
        value["throughput_loss_pct"] = 100 * (1 - value["tokens_per_s"] / baseline["tokens_per_s"])
    return result


def prepare_prompts(args):
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = (
        importlib.import_module("datasets")
        .load_dataset("techarb/gsm8k-cleaner", revision="3a4e9e3e600ea2854a6d2d0483721b3cd68580ce", split="train")
        .shuffle(seed=17)
    )
    prompts = []
    for item in dataset:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"]}], tokenize=True, add_generation_prompt=True
        )
        if len(ids) <= 768:
            prompts.append({"question": item["question"], "input_ids": ids})
        if len(prompts) == args.prompts + 32:
            break
    assert len(prompts) == args.prompts + 32
    path = args.output / "prompts.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in prompts))
    return prompts


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    assert args.prompts % args.block_size == 0
    prompts = prepare_prompts(args)
    control_path = Path("/tmp/topk-benchmark-control")
    control_path.write_bytes(bytes(256))
    with control_path.open("r+b") as handle:
        control = mmap.mmap(handle.fileno(), 256)
    env = {**os.environ, "TOPK_BENCH_CONTROL": str(control_path), "TOPK_BENCH_OUTPUT": str(args.output)}
    server_log = open(args.output / "server.log", "w")  # noqa: SIM115 - closed in finally after child exit
    server = subprocess.Popen(
        [sys.executable, "-m", "scripts.miles.topk_server", "--model", args.model, "--output", str(args.output)],
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    rows, blocks = [], []
    try:
        with httpx.Client(
            base_url="http://127.0.0.1:30000",
            timeout=900,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=32),
        ) as client:
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; see server.log")
                try:
                    if client.get("/health_generate", timeout=10).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("SGLang startup exceeded 20 minutes")
                time.sleep(5)
            write_json(args.output / "server-info.json", client.get("/get_server_info").json())
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:

                def block(mode, batch, name, seed_offset=0, measured=False):
                    control[:] = name.encode().ljust(256, b"\0")
                    t = time.perf_counter()
                    futures = [
                        executor.submit(request, client, p["input_ids"], mode, 17000 + i + seed_offset, args.tokens)
                        for i, p in batch
                    ]
                    collected = []
                    for (i, _), future in zip(batch, futures):
                        collected.append({**future.result(), "prompt_index": i, "block": name})
                    elapsed = time.perf_counter() - t
                    print(
                        json.dumps({"block": name, "mode": mode[0], "requests": len(batch), "seconds": elapsed}),
                        flush=True,
                    )
                    if measured:
                        rows.extend(collected)
                        blocks.append({"block": name, "mode": mode[0], "seconds": elapsed})
                        with (args.output / "requests.jsonl").open("a") as f:
                            for row in collected:
                                f.write(json.dumps(row) + "\n")
                        write_json(args.output / "blocks.json", blocks)
                    return collected

                for mode in MODES:
                    block(mode, list(enumerate(prompts[-32:])), f"warmup-{mode[0]}")
                # Controlled greedy requests check that both wire formats preserve values.
                parity = []
                for k in (32, 64, 128):
                    a = request(client, prompts[0]["input_ids"], (f"nested_{k}", k, False), 17, 32, 0, True)
                    b = request(client, prompts[0]["input_ids"], (f"flat_{k}", k, True), 17, 32, 0, True)
                    assert a[1] == b[1]
                    np.testing.assert_array_equal(a[2][1], b[2][1])
                    np.testing.assert_allclose(a[2][0], b[2][0], atol=1e-6, rtol=0)
                    parity.append({"k": k, "passed": True})
                write_json(args.output / "parity.json", parity)
                for repeat in range(args.repeats):
                    for offset in range(0, args.prompts, args.block_size):
                        modes = list(MODES)
                        random.Random(1700 + repeat * args.prompts + offset).shuffle(modes)
                        batch = list(enumerate(prompts[offset : offset + args.block_size], start=offset))
                        for mode in modes:
                            block(mode, batch, f"r{repeat}-b{offset}-{mode[0]}", repeat * args.prompts, True)
                        write_json(args.output / "summary.json", summarize(rows, blocks))
                control[:] = b"finished".ljust(256, b"\0")
                write_json(args.output / "completed.json", {"requests": len(rows), "status": "complete"})
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        control.close()
        server_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
