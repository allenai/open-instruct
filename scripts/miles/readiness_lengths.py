"""Bounded synthetic long-input capacity probe; not an RL or retrieval benchmark."""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path

import httpx
from transformers import AutoTokenizer

MODEL = "/weka/oe-training-default/robertb/olmo-miles/checkpoints/olmoe3-kda-1.2b-dolci-think-sft-65536-router-bf16-autocast-v2-hf"


def prompt_tokens(tokenizer, length, identity):
    """Keep chat boundaries intact while sizing only the synthetic filler."""
    marker = f"The secret number for this document is {731 + identity}."
    messages = [{"role": "user", "content": "Read the following records.\n<FILLER>\nReturn only the secret number."}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    before, after = rendered.split("<FILLER>")
    head = tokenizer.encode(before, add_special_tokens=False)
    tail = tokenizer.encode(after, add_special_tokens=False)
    needle = tokenizer.encode("\n" + marker + "\n", add_special_tokens=False)
    filler = tokenizer.encode("Ordinary record: the blue box contains paper and pencils.\n", add_special_tokens=False)
    capacity = length - len(head) - len(tail) - len(needle)
    if capacity < 0 or not filler:
        raise ValueError("Prompt budget cannot hold the retrieval instruction")
    body = (filler * ((capacity + len(filler) - 1) // len(filler)))[:capacity]
    midpoint = len(body) // 2
    result = head + body[:midpoint] + needle + body[midpoint:] + tail
    if len(result) != length:
        raise AssertionError("Incorrect synthetic input length")
    return result


def gpu_snapshot():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    return {"time": time.time(), "csv": result.stdout.strip()}


def exercise(output, contexts, prefill, kv_tokens=131072, max_running=8):
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    report = {
        "model": MODEL,
        "contexts": contexts,
        "chunked_prefill_size": prefill,
        "kv_tokens": kv_tokens,
        "max_running": max_running,
        "limits": "Synthetic exact-length input; short greedy output. No backward, replay, weight publication, natural-task score or sustained long decoding coverage. Memory samples are device-wide at 1s intervals, not allocator peaks.",
        "cases": [],
        "passed": False,
    }
    report_path = output / "lengths.json"
    for context in contexts:
        folder = output / str(context)
        folder.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            "31000",
            "--tp-size",
            "1",
            "--context-length",
            str(context),
            "--max-total-tokens",
            str(kv_tokens),
            "--mem-fraction-static",
            "0.6",
            "--max-running-requests",
            str(max_running),
            "--max-mamba-cache-size",
            str(2 * max_running),
            "--chunked-prefill-size",
            str(prefill),
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--cuda-graph-backend-decode",
            "full",
            "--cuda-graph-max-bs-decode",
            str(max_running),
            "--cuda-graph-backend-prefill",
            "disabled",
            "--disable-radix-cache",
            "--trust-remote-code",
            "--skip-server-warmup",
        ]
        case = {"context": context, "command": command, "groups": [], "passed": False}
        report["cases"].append(case)
        process = None
        stop = threading.Event()

        def sample_memory(folder=folder, stop=stop):
            with (folder / "memory.jsonl").open("w") as stream:
                while not stop.is_set():
                    try:
                        stream.write(json.dumps(gpu_snapshot()) + "\n")
                        stream.flush()
                    except (OSError, subprocess.SubprocessError) as error:
                        stream.write(json.dumps({"error": str(error)}) + "\n")
                    stop.wait(1)

        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
        started = time.monotonic()
        try:
            with (folder / "server.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with httpx.Client(base_url="http://127.0.0.1:31000", timeout=900) as client:
                deadline = started + 1200
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited with {process.returncode}")
                    try:
                        if client.get("/health", timeout=5).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server startup exceeded 1200s")
                    time.sleep(2)
                case["startup_seconds"] = time.monotonic() - started
                # Every simultaneous prompt + its reserved output fits the explicit KV pool.
                maximum = min(max_running, kv_tokens // context)
                concurrencies = list(dict.fromkeys([1, 1, min(2, maximum), maximum]))
                # Separate first-shape compilation from a warmed identical-capacity request.
                concurrencies.insert(1, 1)
                for group_index, concurrency in enumerate(concurrencies):
                    inputs = [prompt_tokens(tokenizer, context - 256, i) for i in range(concurrency)]
                    barrier = threading.Barrier(concurrency)

                    def generate(item, barrier=barrier):
                        identity, tokens = item
                        barrier.wait(timeout=30)
                        begin = time.monotonic()
                        response = client.post(
                            "/generate",
                            json={
                                "input_ids": tokens,
                                "return_logprob": True,
                                "sampling_params": {"temperature": 0, "max_new_tokens": 128},
                            },
                        )
                        response.raise_for_status()
                        payload = response.json()
                        meta = payload["meta_info"]
                        logprobs = meta["output_token_logprobs"]
                        if meta["prompt_tokens"] != len(tokens) or not logprobs:
                            raise ValueError("Serving input length mismatch or empty output")
                        if any(not math.isfinite(row[0]) for row in logprobs):
                            raise ValueError("Nonfinite output log probabilities")
                        return {
                            "seconds": time.monotonic() - begin,
                            "input_tokens": len(tokens),
                            "input_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                            "expected_number": str(731 + identity),
                            "text": payload["text"],
                            "meta_info": meta,
                        }

                    begin = time.monotonic()
                    with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        responses = list(pool.map(generate, enumerate(inputs)))
                    group = {
                        "index": group_index,
                        "concurrency": concurrency,
                        "started_at": time.time() - (time.monotonic() - begin),
                        "seconds": time.monotonic() - begin,
                        "responses": responses,
                    }
                    case["groups"].append(group)
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        json.dumps({"context": context, "concurrency": concurrency, "seconds": group["seconds"]}),
                        flush=True,
                    )
            case["passed"] = True
        except Exception as error:
            case["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            stop.set()
            monitor.join(timeout=15)
            report["passed"] = len(report["cases"]) == len(contexts) and all(c["passed"] for c in report["cases"])
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768, 65536])
    parser.add_argument("--chunked-prefill-size", type=int, default=2048)
    parser.add_argument("--kv-tokens", type=int, default=131072)
    parser.add_argument("--max-running", type=int, default=8)
    args = parser.parse_args()
    if any(c not in (16384, 32768, 65536) for c in args.contexts):
        parser.error("Only the bounded 16K/32K/64K sweep is supported")
    if args.kv_tokens < max(args.contexts) or args.max_running not in (8, 16, 32):
        parser.error("KV pool must hold a full context; admission must be 8, 16 or 32")
    exercise(args.output, args.contexts, args.chunked_prefill_size, args.kv_tokens, args.max_running)
