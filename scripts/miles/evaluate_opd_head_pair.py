"""Evaluate two completed OPD exports with identical BF16 SGLang serving settings.

Reuse exact prompt token IDs from a captured native evaluation. Run checkpoints
sequentially on one GPU, with the same order, concurrency and seed. This controls
serving precision, not all batch-dependent floating-point variation.
"""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from scripts.miles import compare_opd_evaluations

from open_instruct import logger_utils
from open_instruct.miles.distillation import opd_rewards as rewards

logger = logger_utils.setup_logger(__name__)


def load_prompts(path):
    rows = []
    seen = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["dataset"], row["prompt"])
        if key in seen:
            raise ValueError("Duplicate evaluation prompt")
        seen.add(key)
        length = row["response_length"]
        if not 0 < length < len(row["tokens"]):
            raise ValueError("Capture must have an unambiguous prompt/response boundary")
        rows.append({**row, "input_ids": row["tokens"][:-length]})
    if not rows:
        raise ValueError("Empty evaluation panel")
    return sorted(rows, key=lambda row: (row["dataset"], row["sample_index"]))


def server_command(checkpoint, tokenizer):
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(checkpoint),
        "--tokenizer-path",
        str(tokenizer),
        "--host",
        "127.0.0.1",
        "--port",
        "31000",
        "--tp-size",
        "1",
        "--trust-remote-code",
        "--random-seed",
        "42",
        "--context-length",
        "18432",
        "--mem-fraction-static",
        "0.6",
        "--max-running-requests",
        "128",
        "--max-total-tokens",
        "2359296",
        "--chunked-prefill-size",
        "16384",
        "--disable-radix-cache",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--disable-flashinfer-autotune",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        "128",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--skip-server-warmup",
    ]


async def generate(rows, registry, destination):
    semaphore = asyncio.Semaphore(128)
    results = [None] * len(rows)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:31000", timeout=1800) as client:

        async def one(index, row):
            async with semaphore:
                response = await client.post(
                    "/generate",
                    json={
                        "input_ids": row["input_ids"],
                        "sampling_params": {
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "max_new_tokens": 16384,
                            "skip_special_tokens": False,
                        },
                    },
                )
                response.raise_for_status()
                payload = response.json()
                meta = payload["meta_info"]
                finish = meta["finish_reason"]["type"]
                if finish not in ("stop", "length"):
                    raise RuntimeError(f"Unexpected generation finish: {meta['finish_reason']}")
                sample = SimpleNamespace(
                    response=payload["text"],
                    tokens=[],
                    response_length=meta["completion_tokens"],
                    prompt=row["prompt"],
                    metadata={"verifiers": [{"name": "math", "target": row["label"], "weight": 1.0}]},
                )
                reward = await rewards.score(sample, str(registry))
                results[index] = {
                    **{key: row[key] for key in ("dataset", "sample_index", "prompt", "label")},
                    "input_ids": row["input_ids"],
                    "response": sample.response,
                    "response_length": sample.response_length,
                    "reward": reward,
                    "status": "truncated" if finish == "length" else "completed",
                    "verifier_metadata": sample.metadata,
                }
                # Each append is synchronous in this event loop; retain partial work on failure.
                with destination.open("a") as stream:
                    stream.write(json.dumps(results[index]) + "\n")
                if sum(result is not None for result in results) % 64 == 0:
                    logger.info("Completed %d/%d prompts", sum(r is not None for r in results), len(rows))

        await asyncio.gather(*(one(index, row) for index, row in enumerate(rows)))
    return results


def evaluate(checkpoint, tokenizer, rows, registry, output, name):
    if not (checkpoint / ".complete").is_file():
        raise ValueError(f"Incomplete checkpoint: {checkpoint}")
    command = server_command(checkpoint, tokenizer)
    (output / f"{name}-command.json").write_text(json.dumps(command, indent=2))
    environment = dict(os.environ)
    environment.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
    started = time.monotonic()
    with (output / f"{name}-server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see {name}-server.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 15 minutes")
                try:
                    if httpx.get("http://127.0.0.1:31000/health", timeout=5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(5)
            generation_start = time.monotonic()
            rows_out = asyncio.run(generate(rows, registry, output / f"{name}.jsonl"))
            return {
                "checkpoint": str(checkpoint),
                "head": "bf16",
                "questions": len(rows_out),
                "generation_and_scoring_seconds": time.monotonic() - generation_start,
                "total_seconds": time.monotonic() - started,
                "response_tokens": sum(row["response_length"] for row in rows_out),
            }
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "treatment", "tokenizer", "reference", "registry", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    rows = load_prompts(options.reference)
    summary = {
        "reference_sha256": hashlib.sha256(options.reference.read_bytes()).hexdigest(),
        "registry_sha256": hashlib.sha256(options.registry.read_bytes()).hexdigest(),
        "arms": {},
    }
    for name in ("control", "treatment"):
        summary["arms"][name] = evaluate(
            getattr(options, name), options.tokenizer, rows, options.registry, options.output, name
        )
        (options.output / "progress.json").write_text(json.dumps(summary, indent=2))
    summary["paired"] = compare_opd_evaluations.compare(
        options.output / "control.jsonl", options.output / "treatment.jsonl"
    )
    (options.output / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Common evaluation complete: %s", options.output / "summary.json")


if __name__ == "__main__":
    main()
