"""Profile full-model weight publication: transport versus engine load, by bucket size.

Runs inside the built runtime on three B300 GPUs with the frozen GSM8K parity
checkpoint: two EP2 trainer ranks and one resident SGLang engine, zero optimizer
updates. After the ordinary initial publication it republishes the unchanged
weights several times at each bucket size, so every publication is the same
29,669-tensor workload and only the bucket size varies. Each publication record
carries a per-bucket split between the NCCL broadcast wait and the wait for the
engine's response, plus tensor and byte counts, so engine time can be regressed
on tensor count against bytes across buckets. The full serving-weight equality
check runs at the end to show the sweep loaded nothing incorrectly.
"""

import asyncio
import dataclasses
import importlib
import json
import os
import sys
from pathlib import Path

import ray
from miles.ray import placement_group
from miles.utils import arguments, object_store
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

BUCKET_SIZES = {
    "256MiB": 256 * 1024**2,
    "512MiB": 512 * 1024**2,
    "1GiB": 1024**3,
    "2GiB": 2 * 1024**3,
    "4GiB": 4 * 1024**3,
}
REPEATS = 3


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Parsed arguments carry dataset config objects; record them by repr rather than fail.
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=repr))


def core_arguments(root, output):
    gsm8k_parity = importlib.import_module("scripts.miles.gsm8k_parity")
    config = gsm8k_parity.configuration(root)
    config.miles.update(
        num_rollout=1,  # validation floor; the profile never calls train
        wandb_mode="disabled",
        save=str(output / "metrics"),
        save_debug_rollout_data=str(output / "unused-rollouts/{rollout_id}.pt"),
        wandb_dir=str(output / "unused-wandb"),
    )
    config.miles.pop("use_wandb", None)
    experts = os.environ.get("OI_PUBLICATION_PROFILE_EXPERTS", "per_expert")
    config = dataclasses.replace(config, core=dataclasses.replace(config.core, expert_publication=experts))
    sys.argv = [sys.argv[0], *config.arguments()]
    return arguments.parse_args()


def _fit(points):
    """Least squares for engine seconds = a * tensors + b * gigabytes + c over bucket points."""
    n = len(points)
    if n < 3:
        return None
    # Normal equations for three unknowns, solved without numpy so tests stay light.
    columns = [[p["tensors"] for p in points], [p["bytes"] / 1e9 for p in points], [1.0] * n]
    target = [p["engine_seconds"] for p in points]
    ata = [[sum(a * b for a, b in zip(ci, cj, strict=True)) for cj in columns] for ci in columns]
    atb = [sum(a * b for a, b in zip(ci, target, strict=True)) for ci in columns]
    # Gaussian elimination.
    m = [row + [rhs] for row, rhs in zip(ata, atb, strict=True)]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda r: abs(m[r][i]))
        m[i], m[pivot] = m[pivot], m[i]
        if abs(m[i][i]) < 1e-12:
            return None
        for r in range(3):
            if r != i:
                factor = m[r][i] / m[i][i]
                m[r] = [a - factor * b for a, b in zip(m[r], m[i], strict=True)]
    a, b, c = (m[i][3] / m[i][i] for i in range(3))
    predicted = [a * p["tensors"] + b * p["bytes"] / 1e9 + c for p in points]
    mean = sum(target) / n
    ss_res = sum((t - q) ** 2 for t, q in zip(target, predicted, strict=True))
    ss_tot = sum((t - mean) ** 2 for t in target) or 1.0
    return {
        "seconds_per_tensor": a,
        "seconds_per_gigabyte": b,
        "intercept_seconds": c,
        "r_squared": 1 - ss_res / ss_tot,
        "points": n,
    }


def summarize(publications):
    """Group publication records by bucket size and attribute their time."""
    by_size = {}
    for record in publications:
        details = record.get("bucket_details") or []
        if not details:
            continue
        entry = by_size.setdefault(
            int(record["buffer_bytes"]),
            {"publications": 0, "total_seconds": [], "broadcast_seconds": [], "engine_seconds": [], "buckets": []},
        )
        entry["publications"] += 1
        entry["total_seconds"].append(record["total_seconds"])
        entry["broadcast_seconds"].append(record["broadcast_seconds"])
        entry["engine_seconds"].append(record["engine_seconds"])
        entry["buckets"].append(record["buckets"])
    summary = {}
    all_points = []
    for size, entry in sorted(by_size.items()):
        summary[str(size)] = {
            "publications": entry["publications"],
            "buckets_per_publication": entry["buckets"][0],
            "mean_total_seconds": sum(entry["total_seconds"]) / entry["publications"],
            "mean_broadcast_seconds": sum(entry["broadcast_seconds"]) / entry["publications"],
            "mean_engine_seconds": sum(entry["engine_seconds"]) / entry["publications"],
        }
    for record in publications:
        for detail in record.get("bucket_details") or []:
            all_points.append(detail)
    return {
        "by_buffer_bytes": summary,
        "engine_seconds_regression": _fit(all_points),
        "broadcast_gigabytes_per_second": (
            sum(p["bytes"] for p in all_points) / 1e9 / sum(p["broadcast_seconds"] for p in all_points)
            if all_points and sum(p["broadcast_seconds"] for p in all_points) > 0
            else None
        ),
        "interpretation": (
            "engine_seconds is the wait after the broadcast completes until the engine responds "
            "(reconstruct plus load_weights); broadcast_seconds covers request dispatch, engine "
            "receive posting, and the NCCL transfer. seconds_per_tensor versus seconds_per_gigabyte "
            "attributes engine time to per-tensor work versus bytes."
        ),
    }


async def profile(args, output):
    if args.fully_async or args.offload_rollout or args.colocate:
        raise ValueError("Only the synchronous disaggregated resident configuration is supported")
    if not args.check_weight_update_equal:
        raise ValueError("The full serving-weight comparison must remain enabled")
    write_json(output / "resolved-arguments.json", vars(args))
    groups = placement_group.create_placement_groups(args)
    manager = learner = None
    try:
        object_store.init_instance(args, contribute_segment=False)
        init_tracking(args)
        manager, _ = placement_group.create_rollout_manager(args, groups["rollout"])
        learner, critic = await placement_group.create_training_models(args, groups, manager)
        if critic is not None:
            raise ValueError("Unexpected critic")
        # Ordinary initial publication (cold caches), checked against the engine's
        # startup snapshot of the HF checkpoint, then the sweep on unchanged weights.
        await learner.update_weights()
        comparison = await manager.check_weights.remote(
            action="compare",
            allow_quant_error=False,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )
        write_json(output / "initial-weight-comparison.json", comparison)
        schedule = []
        for label, size in BUCKET_SIZES.items():
            await learner._broadcast("configure_publication", size)
            for repeat in range(REPEATS):
                await learner.update_weights()
                schedule.append({"label": label, "buffer_bytes": size, "repeat": repeat})
        comparison = await manager.check_weights.remote(
            action="compare",
            allow_quant_error=False,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )
        write_json(output / "final-weight-comparison.json", comparison)
        publications = [
            json.loads(line) for line in (output / "metrics" / "publication.jsonl").read_text().splitlines()
        ]
        write_json(
            output / "profile.json",
            {
                "schedule": schedule,
                "initial_publication": {k: v for k, v in publications[0].items() if k != "bucket_details"},
                "summary": summarize(publications[1:]),
                "publications": publications,
            },
        )
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        if learner is not None:
            # Retire the trainer-to-engine NCCL group collectively before either side
            # is disposed, as the training driver does; disposal otherwise hangs.
            try:
                await asyncio.wait_for(learner._broadcast("close_weight_transport"), timeout=60)
            except Exception as error:
                cleanup_errors.append(error)
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
            output / "cleanup.json", {"completed": not cleanup_errors, "errors": [repr(e) for e in cleanup_errors]}
        )
        if cleanup_errors and primary_error is None:
            raise RuntimeError("Publication profile cleanup failed: " + repr(cleanup_errors)) from cleanup_errors[0]


def main():
    root = Path(os.environ["OI_PUBLICATION_PROFILE_ROOT"])
    output = Path(os.environ["OI_PUBLICATION_PROFILE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    args = core_arguments(root, output)
    runtime_env = {
        "env_vars": {
            name: value
            for name, value in os.environ.items()
            if name.startswith(("OI_PUBLICATION_PROFILE_", "NCCL_"))
            or name in ("PYTHONPATH", "SGLANG_EXTERNAL_MODEL_PACKAGE")
        }
    }
    ray.init(num_gpus=3, num_cpus=16, include_dashboard=False, object_store_memory=1024**3, runtime_env=runtime_env)
    try:
        asyncio.run(profile(args, output))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
