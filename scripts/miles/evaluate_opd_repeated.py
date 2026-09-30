"""Common FP32-head serving evaluation, with repeated pass@1 on small math sets."""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from scripts.miles import evaluate_opd_head_pair

from open_instruct import logger_utils
from open_instruct.miles.distillation import opd_rewards as rewards

logger = logger_utils.setup_logger(__name__)
SMALL_SETS = {"math_aime_2025", "math_brumo_2025"}


def requests(rows, repeats=16, sampled_datasets=None):
    sampled_datasets = SMALL_SETS if sampled_datasets is None else set(sampled_datasets)
    if not sampled_datasets <= {row["dataset"] for row in rows} | SMALL_SETS:
        raise ValueError("Requested sampled dataset is absent from the panel")
    result = []
    for mode, count in (("greedy", 1), ("sampled", repeats)):
        for repeat in range(count):
            for row in rows:
                if mode == "sampled" and row["dataset"] not in sampled_datasets:
                    continue
                # Same question/repetition gets the same seed in every arm. It does
                # not guarantee identical trajectories or eliminate numerical variance.
                identity = f"opd-math-v1:{row['question_id']}:{mode}:{repeat}"
                seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little") % (2**31)
                result.append({**row, "mode": mode, "repeat": repeat, "seed": seed})
    return result


async def generate_all(work, registry, output):
    semaphore = asyncio.Semaphore(128)
    completed = 0
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=128)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:31000", timeout=1800, limits=limits) as client:

        async def one(row):
            nonlocal completed
            async with semaphore:
                response = await client.post(
                    "/generate",
                    json={
                        "input_ids": row["input_ids"],
                        "sampling_params": {
                            "temperature": 0.0 if row["mode"] == "greedy" else 1.0,
                            "top_p": 1.0,
                            "top_k": -1,
                            "min_p": 0.0,
                            "max_new_tokens": 16384,
                            "sampling_seed": row["seed"],
                            "skip_special_tokens": False,
                        },
                    },
                )
                response.raise_for_status()
                payload = response.json()
                meta = payload["meta_info"]
                finish = meta["finish_reason"]["type"]
                if finish not in ("stop", "length"):
                    raise RuntimeError(f"Unexpected finish for {row['question_id']}: {meta['finish_reason']}")
                sample = SimpleNamespace(
                    response=payload["text"],
                    tokens=[],
                    response_length=meta["completion_tokens"],
                    prompt=row["prompt"],
                    metadata={"verifiers": [{"name": "math", "target": row["label"], "weight": 1.0}]},
                )
                reward = await rewards.score(sample, str(registry))
                record = {
                    **row,
                    "response": sample.response,
                    "response_length": sample.response_length,
                    "reward": reward,
                    "status": "truncated" if finish == "length" else "completed",
                    "verifier_metadata": sample.metadata,
                }
                with output.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                completed += 1
                if completed % 64 == 0:
                    logger.info("Completed %d/%d responses", completed, len(work))

        await asyncio.gather(*(one(row) for row in work))


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compatible_checkpoint(root, destination):
    """Rename the legacy text-wrapper prefix without changing any tensor bytes."""
    headers = {}
    for path in sorted(root.glob("*.safetensors")):
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            headers[path.name] = json.loads(stream.read(size))
    config = json.loads((root / "config.json").read_text())
    architectures = config.get("architectures") or []
    if not architectures or not all(name.endswith("ForCausalLM") for name in architectures):
        return root, {}
    prefix = "model.language_model."
    if not any(name.startswith(prefix) for header in headers.values() for name in header):
        return root, {}
    if len(headers) != 1 or (root / "model.safetensors.index.json").exists():
        raise ValueError("Legacy prefix conversion currently requires a single safetensors file")
    destination.mkdir(parents=True)
    for path in root.glob("*.json"):
        shutil.copyfile(path, destination / path.name)
    audit = {}
    for filename, header in headers.items():
        renamed = {}
        mapping = {}
        for name, entry in header.items():
            target = "model." + name.removeprefix(prefix) if name.startswith(prefix) else name
            if target in renamed:
                raise ValueError(f"Checkpoint rename collision: {target}")
            renamed[target] = entry
            if target != name:
                mapping[name] = target
        encoded = json.dumps(renamed, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        digest = hashlib.sha256()
        with (root / filename).open("rb") as source, (destination / filename).open("wb") as target:
            size = struct.unpack("<Q", source.read(8))[0]
            source.seek(8 + size)
            target.write(struct.pack("<Q", len(encoded)))
            target.write(encoded)
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
        with (destination / filename).open("rb") as target:
            target.seek(8 + len(encoded))
            if hashlib.file_digest(target, "sha256").hexdigest() != digest.hexdigest():
                raise ValueError("Checkpoint tensor payload changed during prefix conversion")
        audit[filename] = {"renamed_keys": mapping, "tensor_payload_sha256": digest.hexdigest()}
    return destination, audit


def server_command(root, tokenizer):
    command = evaluate_opd_head_pair.server_command(root, tokenizer)
    command.extend(["--dtype", "bfloat16", "--enable-fp32-lm-head", "--enable-deterministic-inference"])
    return command


def resume_responses(previous, destination, work, provenance):
    """Retain validated complete answers after preemption; regenerate missing keys."""
    old = json.loads((previous / "provenance.json").read_text())
    for field in (
        "arm",
        "checkpoint",
        "panel_sha256",
        "repeats",
        "head",
        "sampling_temperature",
        "response_cap",
        "request_order_sha256",
    ):
        if old[field] != provenance[field]:
            raise ValueError(f"Resume provenance mismatch: {field}")
    keys = {(row["question_id"], row["mode"], row["repeat"]): row for row in work}
    seen = set()
    source = previous / "responses.jsonl"
    lines = source.read_text().splitlines()
    dropped_partial = False
    with destination.open("x") as target:
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                dropped_partial = True
                continue
            key = (row["question_id"], row["mode"], row["repeat"])
            if key in seen or key not in keys:
                raise ValueError("Unknown or duplicated completed response in resume data")
            if any(row[field] != value for field, value in keys[key].items()):
                raise ValueError("Resume prompt, sampling seed or question identity changed")
            if row["status"] not in ("completed", "truncated") or not isinstance(row["response"], str):
                raise ValueError("Resume contains an unsuccessful response")
            seen.add(key)
            target.write(json.dumps(row) + "\n")
    provenance["resume"] = {
        "source_sha256": file_digest(source),
        "retained": len(seen),
        "dropped_partial_last_line": dropped_partial,
    }
    return [row for row in work if (row["question_id"], row["mode"], row["repeat"]) not in seen]


def run(options):
    options.output.mkdir(parents=True, exist_ok=True)
    destination = options.output / "responses.jsonl"
    if destination.exists():
        raise ValueError("Choose a fresh output directory; existing responses must not be overwritten")
    manifest = json.loads((options.panel / "manifest.json").read_text())
    if file_digest(options.panel / "panel.jsonl") != manifest["panel_sha256"]:
        raise ValueError("Evaluation panel changed after preparation")
    if file_digest(options.panel / "verifiers.json") != manifest["verifiers_sha256"]:
        raise ValueError("Verifier registry changed after preparation")
    checkpoint = manifest["checkpoints"][options.arm]
    root = Path(checkpoint["path"])
    for name, identity in checkpoint["files"].items():
        if file_digest(root / name) != identity["sha256"]:
            raise ValueError(f"Checkpoint changed after preparation: {name}")
    rows = [json.loads(line) for line in (options.panel / "panel.jsonl").read_text().splitlines()]
    work = requests(rows, options.repeats, options.sampled_dataset)
    converted_root = Path(tempfile.mkdtemp(prefix="opd-aime-checkpoint-")) / "model"
    serving_root, conversion = compatible_checkpoint(root, converted_root)
    command = server_command(serving_root, manifest["config"]["tokenizer"])
    provenance = {
        "arm": options.arm,
        "checkpoint": checkpoint,
        "checkpoint_conversion": conversion,
        "panel_sha256": manifest["panel_sha256"],
        "repeats": options.repeats,
        "sampled_datasets": sorted(SMALL_SETS if options.sampled_dataset is None else options.sampled_dataset),
        "expected_responses": len(work),
        "head": "fp32",
        "command": command,
        "sampling_temperature": 1.0,
        "response_cap": 16384,
        "request_order_sha256": hashlib.sha256(
            json.dumps([(row["question_id"], row["mode"], row["repeat"], row["seed"]) for row in work]).encode()
        ).hexdigest(),
    }
    if options.resume_from is not None:
        work = resume_responses(options.resume_from, destination, work, provenance)
        logger.info("Retained %d answers; generating %d missing answers", provenance["resume"]["retained"], len(work))
    (options.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    environment = dict(os.environ)
    environment.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
    started = time.monotonic()
    with (options.output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {process.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 15 minutes")
                try:
                    if httpx.get("http://127.0.0.1:31000/health", timeout=5).status_code == 200:
                        if "not found in params_dict" in (options.output / "server.log").read_text():
                            raise ValueError("Server skipped checkpoint parameters; refusing to score")
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(5)
            generation_started = time.monotonic()
            asyncio.run(generate_all(work, options.panel / "verifiers.json", destination))
            actual = sum(1 for _ in destination.open())
            if actual != provenance["expected_responses"]:
                raise ValueError(f"Incomplete evaluation: {actual}/{provenance['expected_responses']}")
            (options.output / "complete.json").write_text(
                json.dumps(
                    {
                        "responses": actual,
                        "total_seconds": time.monotonic() - started,
                        "generation_and_scoring_seconds": time.monotonic() - generation_started,
                        "responses_sha256": file_digest(destination),
                    },
                    indent=2,
                )
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--sampled-dataset", action="append", help="Repeat this dataset; defaults to AIME and BRUMO")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run(args)
