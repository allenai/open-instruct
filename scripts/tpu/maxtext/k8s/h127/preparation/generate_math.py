"""Sample H127's frozen MATH panel without forcing any reasoning-body token."""

import argparse
import asyncio
import gzip
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import sglang
from olmo_sglang import core_compat, register
from transformers import AutoTokenizer

PANEL_SHA = "96d18a968061fa88aae38ad1dd69ac297345db5ada3994522b6ca5de034dfecc"
RUNTIME_HASHES = {
    "models/olmo3_moe.py": "d65778d3ef89edc5533b720825fc637f2f737c4fb2f66f5eebadbb8c82ae6e8b",
    "core_compat.py": "584044aab407020c28ad6d4d57bf4c48d832ac4a9127984b2e31d080a0d76f03",
}
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": -1,
    "max_new_tokens": 32768,
    "stop_token_ids": [100257, 100265],
    "skip_special_tokens": True,
    "no_stop_trim": False,
}


def write(path, value):
    payload = json.dumps(value, allow_nan=False).encode()
    if path.suffix == ".gz":
        payload = gzip.compress(payload, mtime=0)
    temporary = path.with_suffix(".building")
    temporary.write_bytes(payload)
    temporary.replace(path)


async def generate(engine, tokenizer, panel, args, fingerprint):
    semaphore = asyncio.Semaphore(24)
    completed = []

    async def one(row):
        path = args.output / f"generation-{row['id']}.json.gz"
        prompt_hash = hashlib.sha256(json.dumps(row["input_ids"], separators=(",", ":")).encode()).hexdigest()
        async with semaphore:
            if path.exists():
                result = json.loads(gzip.decompress(path.read_bytes()))
                assert result["fingerprint"] == fingerprint and result["qualified_id"] == row["qualified_id"]
                assert result["prompt_sha256"] == prompt_hash and result["sampling"] == SAMPLING
            else:
                seed = int(hashlib.sha256(("h127-math/" + row["qualified_id"]).encode()).hexdigest()[:8], 16) % (2**31)
                started = time.monotonic()
                out = await engine.async_generate(
                    input_ids=row["input_ids"], sampling_params={**SAMPLING, "sampling_seed": seed}
                )
                body, meta = out["output_ids"], out["meta_info"]
                assert meta["prompt_tokens"] == len(row["input_ids"])
                assert meta["completion_tokens"] == len(body) <= 32768
                assert meta["finish_reason"]["type"] in ("stop", "length")
                result = {
                    "qualified_id": row["qualified_id"],
                    "id": row["id"],
                    "arm": args.arm,
                    "step": args.step,
                    "fingerprint": fingerprint,
                    "prompt_sha256": prompt_hash,
                    "input_ids": row["input_ids"],
                    "body_ids": body,
                    "text": tokenizer.decode(body, skip_special_tokens=True),
                    "text_with_special_tokens": tokenizer.decode(body, skip_special_tokens=False),
                    "finish_reason": meta["finish_reason"],
                    "sampling": SAMPLING,
                    "requested_seed": seed,
                    "seconds": time.monotonic() - started,
                }
                write(path, result)
            completed.append(
                {
                    "file": path.name,
                    "qualified_id": row["qualified_id"],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            write(args.output / "progress.json", {"completed": len(completed), "expected": len(panel)})
            if len(completed) % 10 == 0:
                print("MATH-GENERATED", len(completed), "of", len(panel), flush=True)

    await asyncio.gather(*(one(row) for row in panel))
    return sorted(completed, key=lambda r: r["qualified_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("atomic-original", "atomic-presentation-balanced", "infra-smoke"), required=True
    )
    parser.add_argument("--step", type=int, choices=(50, 2048), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--pilot-items", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    assert (args.step == 50) == (args.arm == "infra-smoke")
    assert args.step != 50 or args.pilot_items is not None
    assert hashlib.sha256(args.panel.read_bytes()).hexdigest() == PANEL_SHA
    with gzip.open(args.panel, "rt") as stream:
        panel = [json.loads(line) for line in stream]
    assert len(panel) == len({r["qualified_id"] for r in panel}) == 1000
    assert all(r["input_ids"][-1] == 100256 for r in panel)
    assert 0 <= args.shard < args.shards
    panel = panel[args.shard :: args.shards]
    if args.pilot_items is not None:
        assert 1 <= args.pilot_items <= 24
        panel = panel[: args.pilot_items]
    assert core_compat.mode() == "rounding"
    runtime = Path(core_compat.__file__).parent
    for name, sha in RUNTIME_HASHES.items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == sha, name
    receipt = json.loads((args.model / "export-receipt.json").read_text())
    with (args.model / "model.safetensors").open("rb") as stream:
        model_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert model_sha == receipt["model_safetensors_sha256"]
    metadata_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.model.iterdir()
        if p.is_file() and p.suffix in (".json", ".py", ".jinja")
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": model_sha,
                "metadata": metadata_hashes,
                "panel": PANEL_SHA,
                "sampling": SAMPLING,
                "runtime": RUNTIME_HASHES,
                "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "arm": args.arm,
                "step": args.step,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    assert tokenizer.decode([100270], skip_special_tokens=True) == "</think>"
    register()
    engine = sglang.Engine(
        model_path=str(args.model),
        tokenizer_path=str(args.model),
        trust_remote_code=True,
        dtype="bfloat16",
        tp_size=1,
        context_length=65536,
        mem_fraction_static=0.7,
        max_running_requests=24,
        chunked_prefill_size=1024,
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        disable_radix_cache=True,
        attention_backend="triton",
        moe_runner_backend="triton",
        mamba_ssm_dtype="float32",
        random_seed=34521,
        sampling_defaults="openai",
        disable_overlap_schedule=True,
    )
    try:
        assert set(engine.tokenizer_manager.model_config.hf_eos_token_id) == {100257, 100265}
        startup = time.monotonic() - started
        rows = engine.loop.run_until_complete(generate(engine, tokenizer, panel, args, fingerprint))
        write(
            args.output / "generation-complete.json",
            {
                "arm": args.arm,
                "step": args.step,
                "shard": args.shard,
                "shards": args.shards,
                "pilot_items": args.pilot_items,
                "qualification_only": args.step == 50,
                "fingerprint": fingerprint,
                "rows": rows,
                "startup_seconds": startup,
                "total_seconds": time.monotonic() - started,
                "model_sha256": model_sha,
                "export_receipt": receipt,
                "sampling": SAMPLING,
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "sglang", "olmo-sglang")},
                "seed_note": "Per-question seeds requested; identical random draws across models are not assumed.",
            },
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
