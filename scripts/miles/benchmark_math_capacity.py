"""Bounded, same-device serving-capacity pilot; synthetic outputs are not accuracy samples."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.miles import qualify_math_nvfp4_kv

GIB = 1024**3


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def tensor_bytes(value):
    if isinstance(value, list):
        return sum(tensor_bytes(item) for item in value)
    return value["bytes"] if value else 0


def capacity_options(original, concurrency):
    if concurrency not in (16, 32, 64, 128):
        raise ValueError("Pilot capacity must be 16, 32, 64 or 128")
    options = dict(original)
    options.update(
        context_length=18434,
        max_running_requests=concurrency,
        # Page-round every full request, plus one page per admitted request.
        max_total_tokens=max(393216, concurrency * (18448 + 16)),
        max_mamba_cache_size=max(128, concurrency),
        cuda_graph_max_bs_decode=max(64, concurrency),
    )
    return options


def next_capacity(previous, options, concurrency):
    """Fail closed without physical pool/free-memory measurements; retain transient reserve."""
    state = previous["internal_states"][0]
    telemetry = state["capacity_pilot"]
    target = capacity_options(options, concurrency)
    old_tokens = state["memory_usage"]["token_capacity"]
    extra_pool = telemetry["full_kv_bytes"] * max(0, target["max_total_tokens"] / old_tokens - 1)
    # Graph/workspace growth is unknown before capture. Reserve at least 32GiB,
    # 15% of device memory, AND another measured peak above idle for the new size.
    workspace = max(0, telemetry["peak_allocated_bytes"] - telemetry["allocated_bytes"])
    reserve = max(32 * GIB, 0.15 * telemetry["total_bytes"]) + workspace
    required = extra_pool + reserve
    return {
        "allowed": telemetry["free_bytes"] > required,
        "free_bytes": telemetry["free_bytes"],
        "extra_full_kv_bytes": extra_pool,
        "reserve_bytes": reserve,
        "next_options": target,
        "scope": "Conservative pre-capture estimate, not proof of memory fit or stability",
    }


def audited_scheduler(*args, **kwargs):
    scheduler = importlib.import_module("sglang.srt.managers.scheduler")
    torch = importlib.import_module("torch")
    original = scheduler.Scheduler.get_internal_state

    def state(instance, request):
        result = original(instance, request)
        pool = instance.tp_worker.model_runner.token_to_kv_pool
        full = pool.full_kv_pool
        recurrent = pool.mamba_pool.mamba_cache
        free, total = torch.cuda.mem_get_info()
        buffers = {
            key: qualify_math_nvfp4_kv.tensor_info(getattr(full, key, None))
            for key in ("k_buffer", "v_buffer", "native_k_scale_buffer", "native_v_scale_buffer")
        }
        result.internal_state["capacity_pilot"] = {
            "free_bytes": free,
            "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "full_kv_bytes": sum(tensor_bytes(value) for value in buffers.values()),
            "full_kv_buffers": buffers,
            "recurrent_temporal": qualify_math_nvfp4_kv.tensor_info(recurrent.temporal),
            "recurrent_conv": qualify_math_nvfp4_kv.tensor_info(recurrent.conv),
            "backends": qualify_math_nvfp4_kv.backend_info(instance.tp_worker.model_runner.attn_backend),
        }
        torch.cuda.reset_peak_memory_stats()
        return result

    scheduler.Scheduler.get_internal_state = state
    return scheduler.run_scheduler_process(*args, **kwargs)


def generate(engine, tokens, concurrency):
    start = time.monotonic()
    first, last, counts, metadata = {}, {}, {}, {}
    for chunk in engine.generate(
        input_ids=[tokens] * concurrency,
        sampling_params={"temperature": 1.0, "top_p": 0.95, "top_k": -1, "max_new_tokens": 16384, "ignore_eos": True},
        stream=True,
    ):
        now = time.monotonic() - start
        for item in chunk if isinstance(chunk, list) else [chunk]:
            index, meta = item["index"], item["meta_info"]
            count = meta.get("completion_tokens", 0)
            if count > counts.get(index, 0):
                first.setdefault(index, now)
                last[index] = now
                counts[index] = count
            metadata[index] = meta
    seconds = time.monotonic() - start
    if set(counts) != set(range(concurrency)) or set(counts.values()) != {16384}:
        raise ValueError(f"Incomplete full-length capacity probe: {counts}")
    retractions = [meta.get("num_retractions") for meta in metadata.values()]
    return {
        "seconds": seconds,
        "output_tokens_per_second": sum(counts.values()) / seconds,
        "completion_counts": counts,
        "ttft_seconds": first,
        "request_latency_seconds": last,
        "decode_seconds_per_token": {i: (last[i] - first[i]) / 16383 for i in counts},
        "num_retractions": sum(retractions) if all(type(n) is int for n in retractions) else None,
        "final_metadata": metadata,
    }


def single(args, options):
    os.environ["OLMO_SGLANG_CORE_COMPAT"] = args.compat
    importlib.import_module("olmo_sglang").register()
    torch = importlib.import_module("torch")
    if torch.cuda.device_count() != 1 or "B300" not in torch.cuda.get_device_name(0):
        raise ValueError("Capacity pilot requires exactly one approved B300 GPU")
    engine_type = importlib.import_module("sglang.srt.entrypoints.engine").Engine
    engine_type.run_scheduler_process_func = staticmethod(audited_scheduler)
    engine = engine_type(**options)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        initial = engine.get_server_info()
        save(args.output / "initial-state.json", initial)
        state = initial["internal_states"][0]
        if (
            state["effective_max_running_requests_per_dp"] < args.single
            or state["memory_usage"]["token_capacity"] < args.single * 18434
        ):
            raise ValueError("Resolved engine capacity cannot hold all full-length requests")
        tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(options["model_path"])
        sentence = tokenizer.encode("Solve the following mathematics problem step by step. ", add_special_tokens=False)
        tokens = (sentence * (2048 // len(sentence) + 1))[:2048]
        engine.flush_cache()
        save(args.output / "warmup.json", generate(engine, tokens, args.single))
        rows = []
        for repeat in range(3):
            engine.flush_cache()
            before = engine.get_server_info()  # Also resets allocator peaks before timing.
            row = {"repeat": repeat, "before": before, **generate(engine, tokens, args.single)}
            row["after"] = engine.get_server_info()
            save(args.output / f"repeat-{repeat}.json", row)
            rows.append(row)
        last_state = rows[-1]["after"]
        telemetry = last_state["internal_states"][0]["capacity_pilot"]
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            telemetry[key] = max(row["after"]["internal_states"][0]["capacity_pilot"][key] for row in rows)
        save(
            args.output / "complete.json",
            {
                "options": options,
                "synthetic_only": True,
                "accuracy_validation": False,
                "repeats": 3,
                "stable": all(r["num_retractions"] == 0 for r in rows),
                "mean_output_tokens_per_second": sum(r["output_tokens_per_second"] for r in rows) / 3,
                "last_state": last_state,
                "device": str(torch.cuda.get_device_properties(0)),
                "job_id": os.environ.get("BEAKER_JOB_ID"),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "timing": "streaming after identical full-workload warmup; load/compile/warmup/flush/telemetry excluded",
            },
        )
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compat", choices=("auto", "off"), required=True)
    parser.add_argument("--single", type=int, choices=(16, 32, 64, 128))
    args = parser.parse_args()
    options = json.loads(args.options.read_text())
    if args.single:
        single(args, capacity_options(options, args.single))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "options": options,
        "compat": args.compat,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    identity_path = args.output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Capacity pilot provenance changed")
    save(identity_path, identity)
    if (args.output / "pilot-complete.json").exists():
        return
    # A preemption can move the job to a different device. Preserve the interrupted
    # attempt; all paired capacities must be measured within one allocation.
    attempt = args.output / f"attempt-{os.environ.get('BEAKER_JOB_ID', str(os.getpid()))}"
    attempt.mkdir(exist_ok=False)
    previous = None
    records = []
    for concurrency in (16, 32, 64, 128):
        if previous:
            decision = next_capacity(previous["last_state"], options, concurrency)
            save(attempt / f"admission-{concurrency}.json", decision)
            if not decision["allowed"]:
                break
        folder = attempt / str(concurrency)
        command = [
            sys.executable,
            "-m",
            "scripts.miles.benchmark_math_capacity",
            "--options",
            str(args.options),
            "--output",
            str(folder),
            "--compat",
            args.compat,
            "--single",
            str(concurrency),
        ]
        with (attempt / f"{concurrency}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            save(
                attempt / "stopped.json",
                {"reason": "pilot process failed; inspect retained log", "concurrency": concurrency},
            )
            raise RuntimeError(f"Capacity pilot {concurrency} failed; no higher capacity attempted")
        current = json.loads((folder / "complete.json").read_text())
        records.append(current)
        if not current["stable"] or (
            previous and current["mean_output_tokens_per_second"] <= previous["mean_output_tokens_per_second"] * 1.02
        ):
            save(
                attempt / "stopped.json",
                {"reason": "unknown/retracted requests or throughput saturation", "concurrency": concurrency},
            )
            break
        previous = current
    stable = [record for record in records if record["stable"]]
    best = max(stable, key=lambda record: record["mean_output_tokens_per_second"]) if stable else None
    save(
        args.output / "pilot-complete.json",
        {
            "scope": "Synthetic capacity pilot only; real AIME batching validation still required",
            "attempt": str(attempt),
            "best_measured_stable_options": best["options"] if best else None,
            "best_mean_output_tokens_per_second": best["mean_output_tokens_per_second"] if best else None,
            "measured_capacities": [record["options"]["max_running_requests"] for record in records],
        },
    )


if __name__ == "__main__":
    main()
