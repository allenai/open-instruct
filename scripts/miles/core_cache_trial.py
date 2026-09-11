"""Same fixed native Core update through cold and restored private compiler caches."""

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from open_instruct.miles import compiler_cache as cache
from open_instruct.miles.config import CoreConfig, RunConfig


def expected_config(root):
    """Freeze the existing EP contract settings; worker checks the actual constructor."""
    return RunConfig(
        CoreConfig(
            expert_parallel_size=1,
            attention_backend="torch",
            max_sequence_length=128,
            activation_checkpointing=False,
            diagnostic_interval=1,
            router_aux_loss_weight=0.01,
            router_z_loss_weight=1e-5,
        ),
        dict(
            hf_checkpoint=str(root / "hf"),
            global_batch_size=4,
            rollout_batch_size=1,
            n_samples_per_prompt=4,
            num_rollout=2,
            actor_num_gpus_per_node=1,
            debug_train_only=True,
            save=str(root / "ep1-combined-ac0"),
            rollout_global_dataset=True,
            prompt_data=str(root / "prompts.jsonl"),
            lr=1e-4,
            clip_grad=1e9,
            use_rollout_routing_replay=True,
            use_miles_router=True,
        ),
    )


def config_document(config):
    sections = dataclasses.asdict(config)
    return (
        "\n".join(
            f"[{section}]\n"
            + "\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items() if value is not None)
            for section, values in sections.items()
        )
        + "\n"
    )


def tensor_comparison(left, right):
    if set(left) != set(right) or not left:
        raise ValueError("Missing or different tensor inventory")
    result = {"exact": True, "tensor_count": len(left), "different_tensors": [], "max_abs_error": 0.0}
    for name, reference in left.items():
        actual = right[name]
        if reference.shape != actual.shape or reference.dtype != actual.dtype:
            raise ValueError(f"Tensor shape/dtype differs: {name}")
        if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
            raise ValueError(f"Non-finite tensor: {name}")
        error = float((reference.double() - actual.double()).abs().max()) if reference.numel() else 0.0
        result["max_abs_error"] = max(result["max_abs_error"], error)
        if not torch.equal(reference, actual):
            result["different_tensors"].append({"name": name, "max_abs_error": error})
            result["exact"] = False
    return result


def flatten_scores(records):
    values = {}
    for call, record in enumerate(records):
        for index, tensor in enumerate(record["scores"]):
            values[f"call{call}/sample{index}"] = tensor
    if not values:
        raise ValueError("No scored log-probabilities captured")
    return values


def compare(root):
    report = {
        "schema_version": 1,
        "passed": False,
        "failures": [],
        "comparisons": {},
        "scope": "Native Core EP1 fixed combined update: exact numerical invariance and observed Triton reuse",
        "tolerance": "Exact tensor equality; no inherited EP2 or cross-backend tolerance",
        "limitations": [
            "One tiny update per process",
            "No serving, Ray, remote worker propagation or full SFT",
            "Other cache families reported individually, not presumed warm",
        ],
    }
    try:
        cached = {name: json.loads((root / f"{name}-cache.json").read_text()) for name in ("cold", "restored")}
        if any(value["status"] != "completed" or value["child_returncode"] != 0 for value in cached.values()):
            raise ValueError("Both child updates must complete successfully")
        if cached["cold"]["fingerprint"] != cached["restored"]["fingerprint"]:
            raise ValueError("Cold/restored runtime fingerprints differ")
        if cached["cold"]["private_local_root"] == cached["restored"]["private_local_root"]:
            raise ValueError("Arms reused a mutable cache directory")
        report["fingerprint"] = cached["cold"]["fingerprint"]
        report["cache_reports"] = cached
        observed = {name: json.loads((root / name / "observation.json").read_text()) for name in cached}
        for name, value in observed.items():
            if (
                value["status"] != "completed"
                or value["completed_optimizer_steps"] != 1
                or not value["configuration_matches"]
            ):
                raise ValueError(f"{name}: wrong optimizer/configuration contract")
        if observed["cold"]["input_inventory"] != observed["restored"]["input_inventory"]:
            raise ValueError("Input checkpoint or prompts differ")
        report["observations"] = observed
        payload = {name: torch.load(root / name / "observation.pt", weights_only=True) for name in cached}
        for name, value in payload.items():
            if [record["use_replay"] for record in value["scores"]] != [False, True, True]:
                raise ValueError(f"{name}: expected route-capture, replay scoring, and actor scoring")
            if any(
                len(record["scores"]) != 4
                or any(tensor.shape != (3 + index,) for index, tensor in enumerate(record["scores"]))
                for record in value["scores"]
            ):
                raise ValueError(f"{name}: missing sample or shifted response log-probabilities")
        for section in ("initial_model", "final_model"):
            report["comparisons"][section] = tensor_comparison(payload["cold"][section], payload["restored"][section])
        if [record["use_replay"] for record in payload["cold"]["scores"]] != [
            record["use_replay"] for record in payload["restored"]["scores"]
        ]:
            raise ValueError("Scoring call/replay order differs")
        report["comparisons"]["scored_log_probabilities"] = tensor_comparison(
            flatten_scores(payload["cold"]["scores"]), flatten_scores(payload["restored"]["scores"])
        )
        states = {name: torch.load(root / name / "stress-ep1-rank0.pt", weights_only=True) for name in cached}
        for section in ("before", "after", "state"):
            report["comparisons"][section] = tensor_comparison(states["cold"][section], states["restored"][section])
        routes = {name: torch.load(root / name / "routes.pt", weights_only=True) for name in cached}
        report["comparisons"]["routing_ids"] = tensor_comparison(
            {str(i): value for i, value in enumerate(routes["cold"])},
            {str(i): value for i, value in enumerate(routes["restored"])},
        )
        for name, value in report["comparisons"].items():
            if not value["exact"]:
                report["failures"].append(f"Numerical mismatch: {name}")
        restored_triton = next(value for value in cached["restored"]["restore"] if value["family"] == "triton")
        cold_writes = observed["cold"]["triton"]["put_calls"]
        warm_writes = observed["restored"]["triton"]["put_calls"]
        warm_hits = observed["restored"]["triton"]["group_hits"]
        report["triton_reuse"] = {
            "cold_put_calls": cold_writes,
            "restored_put_calls": warm_writes,
            "restored_group_hits": warm_hits,
        }
        if restored_triton["status"] != "hit" or warm_hits <= 0 or not 0 <= warm_writes < cold_writes:
            report["failures"].append("Actual Triton reuse was not demonstrated by both restore and compiler activity")
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as error:
        report["failures"].append(f"{type(error).__name__}: {error}")
    report["passed"] = not report["failures"]
    (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def run(root, *, image, shared_root, source_root=Path("/opt/core-rl")):
    root.mkdir(parents=True, exist_ok=False)
    bootstrap_env = cache.local_environment(root / "bootstrap-cache")
    environment = {**os.environ, **bootstrap_env}
    subprocess.run(
        [sys.executable, str(source_root / "tests/miles/ep_contract.py"), "bootstrap", str(root / "fixture")],
        env=environment,
        check=True,
    )
    for name in ("cold", "restored"):
        shutil.copytree(root / "fixture", root / name)
        (root / name / "expected.toml").write_text(config_document(expected_config(root / name)))
    before = cache.inventory(root / "fixture")
    common = [
        sys.executable,
        "-m",
        "scripts.miles.compiler_cache_run",
        "--publish",
        "--shared-root",
        str(shared_root),
        "--image",
        image,
        "--runtime-lock",
        str(source_root / "build/runtime/miles/runtime.lock.json"),
        "--hf-config",
        str(root / "fixture/hf/config.json"),
        "--run-config",
        str(root / "cold/expected.toml"),
        "--source",
        f"olmo-core={source_root / 'sources/olmo-core/src'}",
        "--source",
        f"miles={source_root / 'sources/miles/miles'}",
        "--source",
        f"open-instruct={source_root / 'open_instruct'}",
        "--source",
        f"olmo-sglang={source_root / 'sources/olmo-sglang/src'}",
        "--source",
        f"qualification={source_root / 'tests/miles'}",
        "--source",
        f"driver={source_root / 'scripts/miles'}",
    ]
    try:
        for name in ("cold", "restored"):
            command = [
                *common,
                "--mode",
                "cold" if name == "cold" else "restore",
                "--report",
                str(root / f"{name}-cache.json"),
                "--",
                "torchrun",
                "--nnodes=1",
                "--master-addr=127.0.0.1",
                "--master-port=29500",
                "--nproc-per-node=1",
                str(source_root / "tests/miles/cache_core_contract.py"),
                str(root / name),
            ]
            with (root / f"{name}.log").open("xb") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        compare(root)
        raise
    if cache.inventory(root / "fixture") != before:
        raise ValueError("Canonical fixture mutated")
    if not compare(root)["passed"]:
        raise RuntimeError("Core cold/restored cache qualification failed; see comparison.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "compare"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--shared-root", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        if not args.image or args.shared_root is None:
            parser.error("run requires --image and --shared-root")
        run(args.root, image=args.image, shared_root=args.shared_root)
    elif not compare(args.root)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
