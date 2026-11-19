"""Same-B300 queued-throughput experiment; fixed work is not an accuracy evaluation.

A manifest freezes options and tokenized input. Each backend runs in a fresh
process; failed cases retain logs and never contribute a throughput measurement.
"""

import argparse
import hashlib
import importlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def audit_scheduler(*args, **kwargs):
    scheduler = importlib.import_module("sglang.srt.managers.scheduler")
    original = scheduler.Scheduler.__init__

    def init(instance, *a, **kw):
        original(instance, *a, **kw)
        runner = instance.tp_worker.model_runner
        modules = []
        for name, layer in runner.model.named_modules():
            method = getattr(layer, "quant_method", None)
            if method is not None:
                modules.append(
                    {
                        "name": name,
                        "layer_class": type(layer).__name__,
                        "method": type(method).__name__,
                        "runner": str(getattr(getattr(method, "runner", None), "runner_backend", None)),
                        "parameters": {
                            k: {"dtype": str(v.dtype), "shape": list(v.shape)}
                            for k, v in layer.named_parameters(recurse=False)
                        },
                    }
                )
        torch = importlib.import_module("torch")
        triton_configs = {}
        for name, layer in runner.model.named_modules():
            method = getattr(layer, "quant_method", None)
            backend = str(getattr(getattr(method, "runner", None), "runner_backend", None))
            if name.endswith(".experts") and backend.endswith(".TRITON"):
                config_module = importlib.import_module(
                    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config"
                )
                quant_config = getattr(method, "quant_config", None)
                fp8 = "fp8" in type(method).__name__.lower()
                triton_configs = {
                    str(batch): config_module.try_get_optimal_moe_config(
                        tuple(layer.w13_weight.shape),
                        tuple(layer.w2_weight.shape),
                        runner.model.config.num_experts_per_tok,
                        "fp8_w8a8" if fp8 else None,
                        batch,
                        block_shape=getattr(quant_config, "weight_block_size", None) if fp8 else None,
                    )
                    for batch in [128, 256, 512, 1024]
                }
                break
        save(
            Path(os.environ["QUANT_BENCH_OUTPUT"]) / "loaded-audit.json",
            {
                "modules": modules,
                "server_args": vars(instance.server_args),
                "attention_backend": type(runner.attn_backend).__name__,
                "gpu": torch.cuda.get_device_name(),
                "free_total_bytes": torch.cuda.mem_get_info(),
                "triton_moe_configs": triton_configs,
                "moe_config_dir": os.environ.get("SGLANG_MOE_CONFIG_DIR"),
                "core_compat_mode": getattr(runner.model, "core_compat_mode", None),
            },
        )

    scheduler.Scheduler.__init__ = init
    # Use the production KDA policy installer, with a per-case descriptor binding.
    return importlib.import_module("open_instruct.miles.execution.kda_dispatch").run_scheduler_process(*args, **kwargs)


def validate_outputs(outputs, requests, output_tokens):
    if not isinstance(outputs, list) or len(outputs) != requests:
        raise ValueError("Missing responses")
    counts = [o["meta_info"]["completion_tokens"] for o in outputs]
    if any(n != output_tokens for n in counts):
        raise ValueError("Fixed-token workload truncated")
    return sum(counts)


def measure(engine, inputs, output_tokens):
    engine.flush_cache()
    start = time.monotonic()
    outputs = engine.generate(
        input_ids=inputs,
        sampling_params={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        stream=False,
    )
    seconds = time.monotonic() - start
    tokens = validate_outputs(outputs, len(inputs), output_tokens)
    return {
        "seconds": seconds,
        "output_tokens": tokens,
        "requests": len(inputs),
        "output_tokens_per_second": tokens / seconds,
        "requests_per_second": len(inputs) / seconds,
        "output_digest": hashlib.sha256(
            json.dumps([o.get("output_ids", o.get("text")) for o in outputs]).encode()
        ).hexdigest(),
    }


def single(manifest, case, output):
    output.mkdir(parents=True, exist_ok=False)
    os.environ["QUANT_BENCH_OUTPUT"] = str(output)
    os.environ["OLMO_SGLANG_CORE_COMPAT"] = "auto"
    if case.get("moe_config_dir"):
        os.environ["SGLANG_MOE_CONFIG_DIR"] = case["moe_config_dir"]
    torch = importlib.import_module("torch")
    if torch.cuda.device_count() != 1 or "B300" not in torch.cuda.get_device_name():
        raise ValueError("Requires exactly one B300")
    model = Path(case["options"]["model_path"])
    if case["options"].get("moe_runner_backend") == "flashinfer_trtllm":
        raise ValueError("Use flashinfer_trtllm_routed to preserve Olmo custom routing and restored weight scale")
    if case["precision"] == "nvfp4" and not model.exists():
        subprocess.run(
            [
                sys.executable,
                "/opt/core-rl/sources/miles/tools/convert_core_hf_quantized.py",
                "--model-dir",
                manifest["source_model"],
                "--save-dir",
                str(model),
                "--format",
                "nvfp4",
                "--device",
                "cuda",
            ],
            check=True,
        )
    policy = json.loads(Path(case["policy"]).read_text())
    policy["model_config_sha256"] = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    for key in policy["serving_scope"]:
        if key in case["options"]:
            policy["serving_scope"][key] = case["options"][key]
    save(output / "kda-policy.json", policy)
    os.environ["OI_MILES_KDA_POLICY"] = str(output / "kda-policy.json")
    save(
        output / "device.json",
        {
            "properties": str(torch.cuda.get_device_properties(0)),
            "nvidia_smi": subprocess.check_output(["nvidia-smi", "-q"], text=True),
        },
    )
    importlib.import_module("olmo_sglang").register()
    engine_type = importlib.import_module("sglang.srt.entrypoints.engine").Engine
    engine_type.run_scheduler_process_func = staticmethod(
        importlib.import_module("scripts.miles.benchmark_quant_backend").audit_scheduler
    )
    started = time.monotonic()
    engine = engine_type(**case["options"])
    save(output / "startup.json", {"seconds": time.monotonic() - started, "case": case})
    try:
        audit = json.loads((output / "loaded-audit.json").read_text())
        experts = [m for m in audit["modules"] if m["name"].endswith(".experts")]
        if len(experts) != 15:
            raise ValueError("Expected 15 routed expert layers")
        for layer in experts:
            method = layer["method"].lower()
            expected = case["precision"]
            if (expected == "fp8" and "fp8" not in method) or (expected == "nvfp4" and "fp4" not in method):
                raise ValueError("Requested expert quantization was not loaded")
            if expected == "bf16" and any(k in method for k in ["fp4", "fp8"]):
                raise ValueError("Quantized BF16 baseline")
        save(output / "server-info.json", engine.get_server_info())
        inputs = json.loads(Path(manifest["inputs"]).read_text())
        rows = []
        for workload in manifest["workloads"]:
            batch = inputs[: workload["requests"]]
            if len(batch) != workload["requests"]:
                raise ValueError("Insufficient distinct prompts")
            warm = measure(engine, batch, workload["output_tokens"])
            save(output / (workload["name"] + "-warmup.json"), warm)
            for repeat in range(manifest["repeats"]):
                row = {
                    "case": case["name"],
                    "workload": workload["name"],
                    "repeat": repeat,
                    "wall_time": time.time(),
                    **measure(engine, batch, workload["output_tokens"]),
                }
                rows.append(row)
                save(output / f"{workload['name']}-{repeat}.json", row)
                print(json.dumps(row), flush=True)
        save(output / "complete.json", {"rows": rows, "case": case, "accuracy_evaluated": False})
        if manifest.get("profile"):
            # Separate untimed workload: profiling must never affect the speed rows.
            engine.flush_cache()
            engine.start_profile(
                output_dir=str(output / "profile"),
                num_steps=10,
                activities=["CPU", "GPU"],
                profile_by_stage=True,
                profile_stages=["decode"],
                record_shapes=True,
            )
            measure(engine, inputs[:1024], 64)
            save(output / "profile-complete.json", {"timed": False})
    finally:
        engine.shutdown()


def run_case(manifest_path, case, attempt, timeout):
    output = attempt / case["name"]
    cmd = [
        sys.executable,
        "-m",
        "scripts.miles.benchmark_quant_backend",
        str(manifest_path),
        "--case",
        case["name"],
        "--output",
        str(output),
    ]
    with (attempt / (case["name"] + ".log")).open("w") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    status = {"case": case["name"], "exit_code": code, "completed": (output / "complete.json").exists()}
    save(attempt / (case["name"] + "-status.json"), status)
    return status


def report(attempt, manifest):
    results = []
    for case in manifest["cases"]:
        p = attempt / case["name"] / "complete.json"
        if p.exists():
            rows = json.loads(p.read_text())["rows"]
            for workload in manifest["workloads"]:
                rates = [r["output_tokens_per_second"] for r in rows if r["workload"] == workload["name"]]
                results.append(
                    {
                        "case": case["name"],
                        "workload": workload["name"],
                        "median_tps": statistics.median(rates),
                        "min_tps": min(rates),
                        "max_tps": max(rates),
                    }
                )
    save(attempt / "summary.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.case:
        case = next(c for c in manifest["cases"] if c["name"] == args.case)
        single(manifest, case, args.output)
        return
    root = Path(manifest["output"])
    attempt = root / ("attempt-" + os.environ["BEAKER_JOB_ID"])
    attempt.mkdir(parents=True, exist_ok=False)
    save(attempt / "manifest.json", manifest)
    for case in manifest["cases"]:
        status = run_case(args.manifest, case, attempt, manifest["case_timeout_seconds"])
        print(json.dumps(status), flush=True)
        report(attempt, manifest)
        if case["name"] == "bf16" and not status["completed"]:
            raise RuntimeError("Baseline failed; do not spend compute on unmatched cases")
    save(
        attempt / "screen-complete.json",
        {"finished_at": time.time(), "scope": "Backend screening; inspect failed case logs"},
    )


if __name__ == "__main__":
    main()
