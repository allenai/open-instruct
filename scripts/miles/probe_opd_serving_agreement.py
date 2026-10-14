"""Frozen-token scoring and bounded greedy panels for SGLang and Kevin's vLLM."""

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def parse_input_scores(meta, tokens, response_length):
    entries = meta["input_token_logprobs"][-response_length:]
    if len(entries) != response_length or [int(x[1]) for x in entries] != tokens[-response_length:]:
        raise ValueError("Scored token IDs do not align with the frozen response")
    values = [float(x[0]) for x in entries]
    return values


async def sglang_work(options):
    limits = httpx.Limits(max_connections=128, max_keepalive_connections=0)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:31000", timeout=1800, limits=limits) as client:
        for wave, concurrency in (
            [] if options.full_repeat_only else [("score-single", 1), ("score-batch-r0", 16), ("score-batch-r1", 16)]
        ):
            semaphore = asyncio.Semaphore(concurrency)

            async def score(row, semaphore=semaphore, wave=wave):
                async with semaphore:
                    response = await client.post(
                        "/generate",
                        json=dict(
                            input_ids=row["tokens"],
                            sampling_params=dict(max_new_tokens=0, temperature=0.0),
                            return_logprob=True,
                            logprob_start_len=0,
                        ),
                    )
                    response.raise_for_status()
                    meta = response.json()["meta_info"]
                    values = parse_input_scores(meta, row["tokens"], row["response_length"])
                    append(options.output / "scores.jsonl", dict(id=row["id"], wave=wave, logprobs=values))

            await asyncio.gather(*(score(row) for row in options.score_rows))
            response = await client.post("/flush_cache")
            response.raise_for_status()

        async def generate_wave(rows, wave, concurrency, cap):
            semaphore = asyncio.Semaphore(concurrency)

            async def one(rank, row):
                async with semaphore:
                    response = await client.post(
                        "/generate",
                        json=dict(
                            input_ids=row["input_ids"],
                            sampling_params=dict(
                                temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=cap, skip_special_tokens=False
                            ),
                            return_logprob=True,
                            logprob_start_len=-1,
                        ),
                    )
                    response.raise_for_status()
                    value = response.json()
                    meta = value["meta_info"]
                    if meta["finish_reason"]["type"] not in ("stop", "length"):
                        raise ValueError(meta["finish_reason"])
                    ids = [int(x[1]) for x in meta["output_token_logprobs"]]
                    assert len(ids) == meta["completion_tokens"]
                    append(
                        options.output / "generations.jsonl",
                        dict(
                            id=row["id"],
                            rank=rank,
                            wave=wave,
                            response=value["text"],
                            output_ids=ids,
                            finish=meta["finish_reason"]["type"],
                        ),
                    )

            await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
            response = await client.post("/flush_cache")
            response.raise_for_status()

        if not options.full_repeat_only:
            await generate_wave(options.eval_rows, "accuracy", 128, 16384)
        panel = options.eval_rows[:8]
        for wave, rows, concurrency in [
            ("repeat-r0", panel, 8),
            ("repeat-r1", panel, 8),
            ("reversed", panel[::-1], 8),
            ("single", panel, 1),
            ("crowded", panel * (2 if options.full_repeat_only else 16), 128),
        ]:
            await generate_wave(rows, wave, concurrency, 16384 if options.full_repeat_only else 1024)


def run_sglang(options):
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        options.checkpoint,
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
    if options.backend == "sglang-stable":
        command += ["--enable-deterministic-inference"]
    (options.output / "command.json").write_text(json.dumps(command, indent=2))
    env = dict(os.environ)
    env.pop("SGLANG_EXTERNAL_MODEL_PACKAGE", None)
    with (options.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Server exited during startup")
                try:
                    response = httpx.get("http://127.0.0.1:31000/health", timeout=3)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Server readiness")
            asyncio.run(sglang_work(options))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def vllm_worker_provenance(worker):
    model = worker.model_runner.get_model()
    language_model = getattr(model, "language_model", model)
    head = getattr(language_model, "lm_head", None)
    compute = getattr(language_model, "compute_logits", None)
    return dict(
        model_class=type(model).__module__ + "." + type(model).__name__,
        language_model_class=type(language_model).__module__ + "." + type(language_model).__name__,
        head_dtype=str(head.weight.dtype) if head is not None else None,
        patch_marker=bool(getattr(type(model), "_open_instruct_lm_head_fp32_patch", False)),
        language_model_patch_marker=bool(getattr(type(language_model), "_open_instruct_lm_head_fp32_patch", False)),
        compute_logits_module=getattr(compute, "__module__", None),
        compute_logits_name=getattr(compute, "__qualname__", None),
    )


async def run_vllm(options):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Kevin uses AsyncLLM inside Ray, which forces a spawned EngineCore despite
    # the legacy environment switch. Reproduce that process boundary: a patch
    # applied only in the parent must not silently be installed in the child.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    vllm = importlib.import_module("vllm")
    patch = importlib.import_module("open_instruct.vllm_utils")
    patch.patch_vllm_qwen3_5_lm_head_fp32()
    kwargs = dict(
        model=options.checkpoint,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=18432,
        max_num_seqs=128,
        gpu_memory_utilization=0.6,
        seed=42,
        distributed_executor_backend="uni",
        enable_prefix_caching=True,
        generation_config="vllm",
        disable_cascade_attn=True,
        mamba_ssm_cache_dtype="float32",
        gdn_prefill_backend="triton",
        language_model_only=True,
    )
    (options.output / "command.json").write_text(json.dumps(kwargs, indent=2))
    engine = vllm.AsyncLLMEngine.from_engine_args(vllm.AsyncEngineArgs(**kwargs))
    try:
        provenance = await engine.collective_rpc(vllm_worker_provenance)
    except Exception as error:
        provenance = dict(error=repr(error))
        logger.exception("Worker provenance failed; continue numerical measurements")
    (options.output / "worker-provenance.json").write_text(json.dumps(provenance, indent=2))
    logger.info("Actual inference worker: %s", provenance)

    async def generate(rows, params, key, wave):
        async def one(rank, row):
            last = None
            async for value in engine.generate(
                dict(prompt_token_ids=row[key]), params, request_id=f"{wave}-{row['id']}-{rank}"
            ):
                last = value
            if last is None:
                raise ValueError("Missing generation output")
            return last

        return await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))

    params = vllm.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0, detokenize=False)
    for wave, bs in (
        [] if options.full_repeat_only else [("score-single", 1), ("score-batch-r0", 16), ("score-batch-r1", 16)]
    ):
        for start in range(0, len(options.score_rows), bs):
            rows = options.score_rows[start : start + bs]
            outputs = await generate(rows, params, "tokens", wave)
            for row, value in zip(rows, outputs, strict=True):
                assert list(value.prompt_token_ids) == row["tokens"]
                items = value.prompt_logprobs[-row["response_length"] :]
                values = [
                    float(entry[token].logprob)
                    for token, entry in zip(row["tokens"][-row["response_length"] :], items, strict=True)
                ]
                append(options.output / "scores.jsonl", dict(id=row["id"], wave=wave, logprobs=values))
        await engine.reset_prefix_cache()

    async def generate_wave(rows, wave, bs, cap):
        params = vllm.SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=cap, skip_special_tokens=False)
        for start in range(0, len(rows), bs):
            batch = rows[start : start + bs]
            outputs = await generate(batch, params, "input_ids", wave)
            for rank, (row, value) in enumerate(zip(batch, outputs, strict=True), start):
                item = value.outputs[0]
                if item.finish_reason not in ("stop", "length"):
                    raise ValueError(item.finish_reason)
                append(
                    options.output / "generations.jsonl",
                    dict(
                        id=row["id"],
                        rank=rank,
                        wave=wave,
                        response=item.text,
                        output_ids=list(item.token_ids),
                        finish=item.finish_reason,
                    ),
                )
        await engine.reset_prefix_cache()

    if not options.full_repeat_only:
        await generate_wave(options.eval_rows, "accuracy", 128, 16384)
    panel = options.eval_rows[:8]
    for wave, rows, bs in [
        ("repeat-r0", panel, 8),
        ("repeat-r1", panel, 8),
        ("reversed", panel[::-1], 8),
        ("single", panel, 1),
        ("crowded", panel * (2 if options.full_repeat_only else 16), 128),
    ]:
        await generate_wave(rows, wave, bs, 16384 if options.full_repeat_only else 1024)
    engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["sglang-baseline", "sglang-stable", "vllm"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--full-repeat-only",
        action="store_true",
        help="Repeat eight complete answers to measure correctness stability",
    )
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=False)
    options.score_rows = json.loads((options.panel / "score-panel.json").read_text())
    options.eval_rows = json.loads((options.panel / "eval-panel.json").read_text())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in options.panel.glob("*.json")}
    (options.output / "inputs.json").write_text(json.dumps(hashes, indent=2))
    if options.backend == "vllm":
        asyncio.run(run_vllm(options))
    else:
        run_sglang(options)
    (options.output / "complete.json").write_text(json.dumps(dict(backend=options.backend, complete=True)))


if __name__ == "__main__":
    main()
