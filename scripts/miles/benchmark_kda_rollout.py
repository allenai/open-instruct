"""Bounded matched KDA policy comparison through native MILES rollout components.

The driver reuses MILES sample admission, group generation, Ray HTTP workers,
response/candidate decoding and application rewards. No learner or publication is
simulated. Each invocation runs one precision's pinned/native pair on one GPU.
"""

import argparse
import asyncio
import contextlib
import copy
import hashlib
import importlib
import json
import math
import os
import signal
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


def resolve(target):
    module, attribute, wrapper = target.rsplit(".", 2)
    return getattr(getattr(importlib.import_module(module), attribute), wrapper)


def config_dict(config):
    return {key: getattr(config, key, None) for key in ("kwargs", "num_warps", "num_stages", "num_ctas", "maxnreg")}


def audit_scheduler(*args, **kwargs):
    dispatch = importlib.import_module("open_instruct.miles.execution.kda_dispatch")
    publication = importlib.import_module("open_instruct.miles.execution.quantized_publication")
    scheduler = importlib.import_module("sglang.srt.managers.scheduler")
    original = scheduler.Scheduler.get_internal_state
    manifest = json.loads(Path(os.environ["BENCH_POLICY_REFERENCE"]).read_text())

    def state(instance, request):
        result = original(instance, request)
        targets = []
        for entry in manifest["entries"]:
            tuner = resolve(entry["target"])
            installed = getattr(tuner.run, "_oi_kda_policy", None)
            expected = os.environ.get("OI_MILES_KDA_POLICY")
            if bool(installed) != bool(expected):
                raise ValueError("Actual KDA installation differs from requested condition")
            targets.append(
                {
                    "target": entry["target"],
                    "installed_policy": installed,
                    "cache": {repr(k): config_dict(v) for k, v in tuner.cache.items()},
                }
            )
        model = importlib.import_module("olmo_sglang.models.olmo3_moe").Olmo3MoeForCausalLM
        if not getattr(model.load_weights, "_oi_nvfp4_publication_context", False):
            raise ValueError("NVFP4 loader-context fix missing")
        torch = importlib.import_module("torch")
        result.internal_state["kda_benchmark"] = {
            "targets": targets,
            "loader_context_fix": True,
            "server_args": vars(instance.server_args),
            "gpu_name": torch.cuda.get_device_name(),
            "free_total_bytes": torch.cuda.mem_get_info(),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        return result

    scheduler.Scheduler.get_internal_state = state
    # BOTH conditions run the same scheduler wrapper; install() is a no-op when
    # OI_MILES_KDA_POLICY is absent. Keep the quantizer context fix in both.
    publication.install_nvfp4_publication_context()
    return dispatch.run_scheduler_process(*args, **kwargs)


def serve():
    importlib.import_module("olmo_sglang.kda.backend").register_olmo_kda_backend()
    importlib.import_module("open_instruct.miles.execution.quantized_publication").install_nvfp4_publication_context()
    http = importlib.import_module("sglang.srt.entrypoints.http_server")
    original = http.launch_server

    def launch(args, **kwargs):
        return original(args, run_scheduler_process_func=audit_scheduler, **kwargs)

    http.launch_server = launch
    importlib.import_module("runpy").run_module("sglang.launch_server", run_name="__main__")


def native_args(plan, output):
    argv = json.loads(Path(plan).read_text())["argv"]
    original = sys.argv
    try:
        sys.argv = ["kda-rollout-benchmark", *argv]
        args = importlib.import_module("miles.utils.arguments").parse_args()
    finally:
        sys.argv = original
    # Rollout-only topology. Serving precision, capacity, graphs, head and pools
    # remain from the production parser; no trainer process is constructed.
    args.rollout_num_gpus = 1
    args.num_gpus_per_node = 1
    args.sglang_router_ip = "127.0.0.1"
    args.sglang_router_port = 31000
    args.save = str(output)
    args.load = None
    return args


def server_options(args):
    module = importlib.import_module("miles.backends.sglang_utils.sglang_engine")
    options = module._compute_server_args(
        args,
        node_rank=0,
        dist_init_addr=None,
        nccl_port=31002,
        host="127.0.0.1",
        port=31000,
        worker_type="regular",
        disaggregation_bootstrap_port=None,
        base_gpu_id=0,
        engine_info_bootstrap_port=None,
        sglang_overrides=None,
        num_gpus_per_engine=1,
        gated_launch_port=None,
        random_seed=17,
    )
    assert options["max_running_requests"] == 32 and options["max_total_tokens"] == 590848
    assert options["enable_fp32_lm_head"] and args.rollout_top_logprobs_num == 20
    assert args.rollout_temperature == args.rollout_top_p == 1 and args.rollout_top_k == -1
    return options


def select_inputs(path, output):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    # Frozen deterministic selection spans the real prompt distribution. Preserve
    # native rendered prompts, labels, metadata and exact saved tokenizer IDs.
    rows.sort(key=lambda row: (len(row["metadata"]["prompt_token_ids"]), row["metadata"]["prepared_sample_id"]))
    selected = [rows[round(i * (len(rows) - 1) / 7)] for i in range(8)]
    save(output, selected)
    return selected


async def measure(args, rows, output, *, length, repeat, warmup=False):
    common = importlib.import_module("miles.rollout.inference_rollout.inference_rollout_common")
    types = importlib.import_module("miles.utils.types")
    scheduler = importlib.import_module("miles.rollout.submission_scheduler").make_submission_scheduler(
        args, default="sample"
    )
    state = common.GenerateState(args)
    for row in rows:
        actual = state.tokenizer.encode(row["input"], add_special_tokens=False)
        if actual != row["metadata"]["prompt_token_ids"]:
            raise ValueError("Prepared prompt tokenizer IDs differ from the production tokenizer")
    state.sampling_params["max_new_tokens"] = length or 16384
    if length:
        state.sampling_params["ignore_eos"] = True
    times, group_times = {}, []
    generate = state.generate_function
    reward = common.async_rm

    async def timed_generate(request):
        times[request.sample.index]["admitted"] = time.monotonic()
        result = await generate(request)
        times[request.sample.index]["response_processed"] = time.monotonic()
        return result

    async def timed_reward(a, sample, **kwargs):
        start = time.monotonic()
        result = await reward(a, sample, **kwargs)
        times[sample.index]["reward_seconds"] = time.monotonic() - start
        return result

    state.generate_function = timed_generate
    common.async_rm = timed_reward
    active, completed, next_group = {}, [], 0
    group_count = 4 if length else 2
    # The production pending budget remains 128; the finite fixed window has
    # 64 requests and the natural 16K-cap probe 32. Engine active cap stays 32.
    start = time.monotonic()
    try:
        while active or next_group < group_count:
            while next_group < group_count and scheduler.has_capacity(pending_groups=len(active), group_budget=8):
                row = rows[round(next_group * (len(rows) - 1) / (group_count - 1))]
                group = []
                submitted = time.monotonic()
                for offset in range(16):
                    index = next_group * 16 + offset
                    sample = types.Sample(
                        index=index,
                        group_index=next_group,
                        prompt=row["input"],
                        label=row["label"],
                        metadata=copy.deepcopy(row["metadata"]),
                    )
                    times[index] = {"submitted": submitted, "prompt_tokens": len(row["metadata"]["prompt_token_ids"])}
                    group.append(sample)
                scheduler.on_submit([group])
                task = asyncio.create_task(
                    common.generate_and_rm_group(
                        state,
                        group,
                        state.sampling_params.copy(),
                        evaluation=False,
                        sample_done_callback=scheduler.sample_done_callback,
                    )
                )
                active[task] = (next_group, submitted)
                next_group += 1
            done, _ = await scheduler.wait_for_progress(set(active))
            for task in done:
                index, submitted = active.pop(task)
                group = task.result()
                now = time.monotonic()
                group_times.append(
                    {
                        "group": index,
                        "submitted": submitted - start,
                        "assembled": now - start,
                        "tail_seconds": now - max(times[s.index]["response_processed"] for s in group),
                    }
                )
                for sample in group:
                    count = sample.response_length
                    if length and count != length:
                        raise ValueError(f"Fixed length mismatch: {sample.index}: {count} != {length}")
                    if (
                        not count
                        or len(sample.rollout_log_probs) != count
                        or not all(map(math.isfinite, sample.rollout_log_probs))
                    ):
                        raise ValueError("Empty/nonfinite/incomplete behavior logprobs")
                    candidates = sample.rollout_topk_log_probs
                    if candidates is None or tuple(candidates.shape) != (count, 20):
                        raise ValueError("Requested candidate20 missing")
                    valid = sample.rollout_topk_token_ids >= 0
                    if not importlib.import_module("numpy").isfinite(candidates[valid]).all():
                        raise ValueError("Nonfinite valid candidate logprobs")
                    record = times[sample.index]
                    record.update(
                        index=sample.index,
                        output_tokens=count,
                        reward=sample.reward,
                        status=sample.status.value,
                        assembled=now,
                    )
                    completed.append(record)
        seconds = time.monotonic() - start
    finally:
        common.async_rm = reward
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
    result = {
        "warmup": warmup,
        "repeat": repeat,
        "fixed_output_length": length,
        "seconds": seconds,
        "output_tokens": sum(row["output_tokens"] for row in completed),
        "responses": len(completed),
        "groups": group_times,
        "samples": completed,
        "prompt_lengths": [len(row["metadata"]["prompt_token_ids"]) for row in rows],
        "time_origin_monotonic": start,
    }
    result["output_tokens_per_second"] = result["output_tokens"] / seconds
    result["responses_per_second"] = len(completed) / seconds
    save(output, result)
    return result


async def sample_metrics(connection, output):
    with (output / "engine-timeline.jsonl").open("a") as stream:
        while True:
            row = {"unix": time.time(), "monotonic": time.monotonic()}
            try:
                response = await connection.get("http://127.0.0.1:31000/metrics", timeout=10)
                response.raise_for_status()
                row["metrics"] = response.text
            except Exception as error:
                row["error"] = repr(error)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            await asyncio.sleep(1)


async def client(args, rows, output, deadline):
    http = importlib.import_module("httpx")
    native_http = importlib.import_module("miles.utils.http_utils")
    ray = importlib.import_module("ray")
    ray.init(num_cpus=24, num_gpus=0, include_dashboard=False, log_to_driver=False)
    native_http.init_http_client(args)
    async with http.AsyncClient(timeout=1800) as connection:

        async def snapshot(name):
            response = await connection.get("http://127.0.0.1:31000/get_server_info")
            response.raise_for_status()
            save(output / name, response.json())

        sampler = asyncio.create_task(sample_metrics(connection, output))
        try:
            await snapshot("initial-state.json")
            await measure(args, rows, output / "warmup-fixed.json", length=4096, repeat=0, warmup=True)
            await snapshot("warmed-state.json")
            for repeat in range(2):
                await measure(args, rows, output / f"fixed-{repeat}.json", length=4096, repeat=repeat)
                await snapshot(f"fixed-{repeat}-state.json")
            # A real length-varying 16K-cap workload is separate from fixed-token
            # ratios. It exercises production stopping/reward behavior, not accuracy.
            remaining = deadline - time.monotonic() - 30
            if remaining > 120:
                try:
                    await asyncio.wait_for(
                        measure(args, rows, output / "natural.json", length=None, repeat=0),
                        timeout=min(300, remaining),
                    )
                except asyncio.TimeoutError:
                    save(
                        output / "natural-incomplete.json",
                        {"reason": "Optional real-length window exceeded its bounded allowance"},
                    )
            else:
                save(
                    output / "natural-incomplete.json",
                    {"reason": "Fixed measurements prioritized within the two-hour budget"},
                )
            await snapshot("final-state.json")
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
    ray.shutdown()
    if native_http._http_client is not None:
        await native_http._http_client.aclose()
    native_http._http_client = None
    native_http._post_actors = []
    native_http._distributed_post_enabled = False


def run_condition(args, condition, rows, output, policy, deadline):
    output.mkdir(parents=True, exist_ok=False)
    options = server_options(args)
    save(output / "requested-server-options.json", options)
    save(output / "effective-miles-args.json", vars(args))
    cache = Path("/tmp/kda-benchmark") / f"{os.getpid()}-{condition}"
    cache.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    for key in ("OI_CORE_STARTUP_CACHE", "OI_MILES_KDA_POLICY"):
        environment.pop(key, None)
    environment.update(
        BENCH_POLICY_REFERENCE=str(policy),
        TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        XDG_CACHE_HOME=str(cache / "xdg"),
        CUDA_CACHE_PATH=str(cache / "cuda"),
    )
    if condition == "pinned":
        environment["OI_MILES_KDA_POLICY"] = str(policy)
    save(
        output / "cache-provenance.json",
        {"fresh": True, "shared_restore": False, "paths": {k: v for k, v in environment.items() if "CACHE" in k}},
    )
    argv = importlib.import_module("miles.backends.sglang_utils.server_args_utils").server_args_to_argv(options)
    command = [sys.executable, str(Path(__file__).resolve()), "--server", *argv]
    save(output / "server-command.json", command)
    http = importlib.import_module("httpx")
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}: {output / 'server.log'}")
                try:
                    response = http.get("http://127.0.0.1:31000/health", timeout=5)
                    if response.status_code == 200:
                        break
                except http.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Bounded startup deadline exceeded")
            asyncio.run(
                asyncio.wait_for(client(args, rows, output, deadline), timeout=max(1, deadline - time.monotonic()))
            )
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    if "--server" in sys.argv:
        sys.argv.remove("--server")
        serve()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", nargs=2, choices=("pinned", "native"), default=["pinned", "native"])
    parser.add_argument("--seconds", type=int, default=3500)
    parser.add_argument("--preflight", action="store_true")
    parsed = parser.parse_args()
    assert set(parsed.order) == {"native", "pinned"}
    parsed.output.mkdir(parents=True, exist_ok=True)
    args = native_args(parsed.plan, parsed.output)
    options = server_options(args)
    save(
        parsed.output / "preflight.json",
        {
            "passed": True,
            "options": options,
            "plan_sha256": hashlib.sha256(parsed.plan.read_bytes()).hexdigest(),
            "policy_sha256": hashlib.sha256(parsed.policy.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    if parsed.preflight:
        return
    torch = importlib.import_module("torch")
    assert torch.cuda.device_count() == 1 and "B300" in torch.cuda.get_device_name()
    save(
        parsed.output / "gpu.json",
        {
            "nvidia_smi": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name,pci.bus_id", "--format=csv,noheader"], text=True
            )
        },
    )
    rows = select_inputs(args.prompt_data, parsed.output / "inputs.json")
    deadline = time.monotonic() + parsed.seconds
    for condition in parsed.order:
        args.save = str(parsed.output / condition)
        condition_deadline = min(deadline, time.monotonic() + parsed.seconds / 2)
        run_condition(args, condition, rows, parsed.output / condition, parsed.policy, condition_deadline)
    save(parsed.output / "completed.json", {"passed": True, "finished_unix": time.time()})


if __name__ == "__main__":
    main()
