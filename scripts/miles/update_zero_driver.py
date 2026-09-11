"""Probe original MILES startup before/after publication without any optimizer call."""

import asyncio
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import ray
import requests
from miles.ray import placement_group
from miles.utils import arguments, object_store
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def verify_original_sources(output):
    manifest = json.loads((output / "manifest.json").read_text())
    expected = manifest["original_recipe"]["module_sha256"]
    if not expected:
        raise ValueError("Original source manifest has no module hashes")
    observed = {}
    for name, checksum in expected.items():
        module = importlib.import_module(name)
        source = Path(module.__file__)
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        observed[name] = {"path": str(source), "sha256": actual, "expected_sha256": checksum}
        write_json(output / "original-source-verification.json", observed)
        if actual != checksum:
            raise ValueError(f"Original source hash mismatch for {name}")
    return observed


def core_arguments(root, output):
    # Import the configuration from the original image, never a current checkout.
    gsm8k_parity = importlib.import_module("scripts.miles.gsm8k_parity")
    config = gsm8k_parity.configuration(root)
    config.miles.update(
        num_rollout=0,
        use_wandb=False,
        wandb_mode="disabled",
        save=str(output / "metrics"),
        save_debug_rollout_data=str(output / "unused-rollouts/{rollout_id}.pt"),
        wandb_dir=str(output / "unused-wandb"),
    )
    sys.argv = [sys.argv[0], *config.arguments()]
    return arguments.parse_args()


def phase_probe(url, phase, inputs, output):
    trace = Path(os.environ["OI_UPDATE_ZERO_TRACE_DIR"])
    marker = trace / "capture-request.json"
    records = []
    # An unarmed pass provides a control for the observer's effect on results.
    cases = [(case, False, "control") for case in inputs["cases"]]
    cases += [(case, True, "capture") for case in inputs["cases"]]
    cases += [(inputs["cases"][0], True, "repeat")]
    try:
        for case, armed, suffix in cases:
            capture_id = f"{phase}-{case['case_id']}-{suffix}"
            if marker.exists():
                marker.unlink()
            if armed:
                write_json(
                    marker,
                    dict(phase=phase, case_id=case["case_id"], capture_id=capture_id, input_ids=case["input_ids"]),
                )
            payload = {
                "input_ids": case["input_ids"],
                "sampling_params": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 32,
            }
            print(f"UPDATE_ZERO_REQUEST_BEGIN {capture_id} armed={armed} tokens={len(case['input_ids'])}", flush=True)
            started = time.monotonic()
            response = requests.post(url + "/generate", json=payload, timeout=600)
            response.raise_for_status()
            value = response.json()
            write_json(output / f"{capture_id}-response.json", value)
            print(f"UPDATE_ZERO_REQUEST_END {capture_id} seconds={time.monotonic() - started:.3f}", flush=True)
            captures = sorted(str(path.relative_to(output)) for path in trace.glob(f"worker-*/{capture_id}.json"))
            if armed and len(captures) != 1:
                raise RuntimeError(f"Expected one TP1 exact-prefix capture for {capture_id}, got {captures}")
            records.append(
                dict(
                    capture_id=capture_id,
                    armed=armed,
                    elapsed_seconds=time.monotonic() - started,
                    request=payload,
                    capture_metadata=captures,
                )
            )
            write_json(output / f"{phase}-requests.json", records)
    finally:
        marker.unlink(missing_ok=True)
    return records


async def probe(args, inputs, output):
    if args.num_rollout != 0 or args.use_critic or args.fully_async or args.offload_rollout:
        raise ValueError("Only a synchronous resident actor, zero updates, and no critic are supported")
    if not args.check_weight_update_equal:
        raise ValueError("The full initial serving-weight comparison must remain enabled")
    write_json(output / "resolved-arguments.json", vars(args))
    groups = placement_group.create_placement_groups(args)
    manager = learner = None
    try:
        object_store.init_instance(args, contribute_segment=False)
        init_tracking(args)
        # The ordinary helper resets serving tensors immediately after its snapshot.
        # Delay those exact operations until after the direct-HF control requests.
        args.check_weight_update_equal = False
        try:
            manager, _ = placement_group.create_rollout_manager(args, groups["rollout"])
        finally:
            args.check_weight_update_equal = True
        engines = await manager.get_updatable_engines_and_lock.remote()
        if len(engines.rollout_engines) != 1:
            raise ValueError("Expected one resident TP1 inference engine")
        engine = engines.rollout_engines[0]
        topology = await engine.get_topology_info.remote()
        write_json(output / "serving-topology.json", topology)
        write_json(output / "serving-resolved.json", await engine.get_server_info.remote())
        await asyncio.to_thread(phase_probe, topology["url"], "hf", inputs, output)

        await manager.check_weights.remote(action="snapshot")
        await manager.check_weights.remote(action="reset_tensors", skip_list=args.check_weight_update_skip_list)
        learner, critic = await placement_group.create_training_models(args, groups, manager)
        if critic is not None:
            raise ValueError("Unexpected critic")
        await learner.update_weights()
        comparison = await manager.check_weights.remote(
            action="compare",
            allow_quant_error=False,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )
        write_json(output / "initial-weight-comparison.json", comparison)
        await asyncio.to_thread(phase_probe, topology["url"], "published", inputs, output)
        write_json(
            output / "probe-complete.json",
            dict(
                completed=True,
                optimizer_calls=0,
                initial_publications=1,
                phases=["hf", "published"],
                cases=len(inputs["cases"]),
                input_sha256=hashlib.sha256(Path(os.environ["OI_UPDATE_ZERO_INPUTS"]).read_bytes()).hexdigest(),
                interpretation="Exact-prefix prefill observations; does not reproduce historical decode batch scheduling.",
            ),
        )
    finally:
        # Complete every cleanup action even if an earlier disposal fails.
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        if learner is not None:
            try:
                await asyncio.wait_for(learner.dispose(), timeout=180)
            except Exception as error:
                cleanup_errors.append(error)
        if manager is not None:
            try:
                await asyncio.wait_for(manager.dispose.remote(), timeout=180)
            except Exception as error:
                cleanup_errors.append(error)
        try:
            finish_tracking()
        except Exception as error:
            cleanup_errors.append(error)
        write_json(
            output / "cleanup.json",
            dict(completed=not cleanup_errors, errors=[repr(error) for error in cleanup_errors]),
        )
        if cleanup_errors and primary_error is None:
            raise RuntimeError("Update-zero cleanup failed: " + repr(cleanup_errors)) from cleanup_errors[0]


def main():
    output = Path(os.environ["OI_UPDATE_ZERO_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(Path(os.environ["OI_UPDATE_ZERO_INPUTS"]).read_text())
    if len(inputs["cases"]) != 4 or any(not 1 <= len(case["input_ids"]) <= 1024 for case in inputs["cases"]):
        raise ValueError("Expected four frozen prefixes, each between 1 and 1024 tokens")
    backend = os.environ["OI_UPDATE_ZERO_BACKEND"]
    if backend not in ("core", "megatron"):
        raise ValueError("Unknown update-zero backend")
    if backend == "core":
        verify_original_sources(output)
    args = (
        core_arguments(Path(os.environ["OI_UPDATE_ZERO_ROOT"]), output)
        if backend == "core"
        else arguments.parse_args()
    )
    runtime_env = json.loads(os.environ.get("OI_UPDATE_ZERO_RAY_ENV", "{}"))
    runtime_env.setdefault("env_vars", {}).update(
        {name: value for name, value in os.environ.items() if name.startswith("OI_UPDATE_ZERO_")}
    )
    runtime_env["env_vars"]["PYTHONPATH"] = os.environ["PYTHONPATH"]
    runtime_env["env_vars"]["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "olmo_sglang.models"
    ray.init(num_gpus=3, num_cpus=16, include_dashboard=False, object_store_memory=1024**3, runtime_env=runtime_env)
    try:
        asyncio.run(probe(args, inputs, output))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
