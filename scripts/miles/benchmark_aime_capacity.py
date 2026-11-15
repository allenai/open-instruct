"""Instrument the pinned evaluator for a representative, warmed AIME capacity pilot.

The registration callback keeps the native model registration and adds scheduler
memory telemetry and Engine.generate timings. Warmup outputs are discarded before
the harness receives any predictions. This pilot does not replace full AIME30x32.
"""

import argparse
import copy
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.miles import benchmark_math_capacity

from open_instruct.miles.evaluation import evaluation_runner


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, default=str) + "\n")


def register():
    importlib.import_module("olmo_sglang").register()
    engine_type = importlib.import_module("sglang.srt.entrypoints.engine").Engine
    if getattr(engine_type, "_aime_capacity_instrumented", False):
        return
    engine_type._aime_capacity_instrumented = True
    engine_type.run_scheduler_process_func = staticmethod(benchmark_math_capacity.audited_scheduler)
    original = engine_type.generate
    warmed = set()

    def generate(engine, *args, **kwargs):
        path = Path(os.environ["OLMO_AIME_CAPACITY_TELEMETRY"]) / f"generation-{os.getpid()}.jsonl"
        if kwargs.get("stream"):
            raise ValueError("AIME harness instrumentation requires non-streaming native provider")
        if id(engine) not in warmed:
            warm = copy.deepcopy(kwargs)
            # Deterministic warmup consumes the same expanded workload and full cap;
            # it is excluded from saved requests, predictions, metrics and timing.
            warm["sampling_params"] = {**warm["sampling_params"], "temperature": 0.0}
            start = time.monotonic()
            original(engine, *args, **warm)
            append(path, {"phase": "warmup", "seconds": time.monotonic() - start})
            warmed.add(id(engine))
        engine.flush_cache()
        before = engine.get_server_info()
        start = time.monotonic()
        outputs = original(engine, *args, **kwargs)
        seconds = time.monotonic() - start
        after = engine.get_server_info()
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("No native outputs in representative AIME batch")
        metadata = [item["meta_info"] for item in outputs]
        counts = [item["completion_tokens"] for item in metadata]
        append(
            path,
            {
                "phase": "measured",
                "seconds": seconds,
                "output_tokens_per_second": sum(counts) / seconds,
                "requests": len(outputs),
                "completion_counts": counts,
                "sampling_params": kwargs["sampling_params"],
                "metadata": [{key: value for key, value in item.items() if "logprob" not in key} for item in metadata],
                "before": before,
                "after": after,
            },
        )
        return outputs

    engine_type.generate = generate


def command(receipt, output, capacity):
    if capacity not in (16, 32, 64, 128):
        raise ValueError("Unmeasured capacity")
    if len(receipt["tasks"]) != 1:
        raise ValueError("Keep AIME years separate")
    task = receipt["tasks"][0]
    if (
        task["generation"]
        != {"temperature": 1.0, "top_p": 0.95, "max_tokens": 16384, "num_samples": 32, "stop_sequences": []}
        or task["scoring"]["seed"] != 17
    ):
        raise ValueError("Changed AIME sampling protocol")
    options = receipt["evaluation"]["provider_kwargs"]
    if options["max_running_requests"] != capacity or options["random_seed"] != 17:
        raise ValueError("Engine capacity or seed mismatch")
    return [
        *evaluation_runner.command(receipt, output),
        "-o",
        "provider.kwargs.model_registration=scripts.miles.benchmark_aime_capacity:register",
        "-o",
        f"batching.chunk_size={capacity}",
        "-o",
        f"batching.max_batch_sequences={capacity}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["output"])
    root.mkdir(parents=True, exist_ok=True)
    # Same allocation for all serving settings. Preserve interrupted attempts rather
    # than combining measurements from different devices into a paired comparison.
    attempt = root / f"attempt-{os.environ.get('BEAKER_JOB_ID', 'dry-run')}"
    attempt.mkdir(exist_ok=args.dry_run)
    for case in manifest["cases"]:
        output = attempt / case["name"]
        output.mkdir(exist_ok=args.dry_run)
        receipt = case["receipt"]
        revision = subprocess.check_output(["git", "-C", "/opt/olmo-eval", "rev-parse", "HEAD"], text=True).strip()
        if revision != receipt["evaluation"]["revision"]:
            raise ValueError("Pinned evaluator revision mismatch")
        argv = command(receipt, output / "results", case["capacity"])
        env = {**os.environ, "OLMO_AIME_CAPACITY_TELEMETRY": str(output)}
        with (output / "evaluator.log").open("w") as log:
            subprocess.run(
                [*argv, *(["--dry-run"] if args.dry_run else [])],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not args.dry_run:
            evaluation_runner.validate_predictions(output / "results")
            (output / "complete.json").write_text(json.dumps({"case": case, "status": "complete"}, indent=2) + "\n")
    (attempt / "pilot-complete.json").write_text(
        json.dumps({"dry_run": args.dry_run, "cases": len(manifest["cases"])})
    )


if __name__ == "__main__":
    main()
