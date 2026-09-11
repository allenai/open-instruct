"""Awaited driver stage timings; async generation wait excludes producer overlap."""

import json
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def stage(args, name, rollout_id=None, *, details=None):
    started = time.perf_counter()
    wall = time.time()
    passed = False
    try:
        yield
        passed = True
    finally:
        if getattr(args, "save", None):
            root = Path(args.save)
            root.mkdir(parents=True, exist_ok=True)
            with (root / "driver_timing.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            stage=name,
                            rollout_id=rollout_id,
                            started_unix=wall,
                            seconds=time.perf_counter() - started,
                            passed=passed,
                            **({"details": details} if details is not None else {}),
                        )
                    )
                    + "\n"
                )


def evaluation_stage(args, rollout_id, *, initial=False):
    """Shared-engine dispatch awaits eval; snapshot dispatch only submits work."""
    snapshots = getattr(args, "eval_uses_snapshots", False)
    keys = (
        "rollout_num_gpus",
        "rollout_num_gpus_per_engine",
        "sglang_server_concurrency",
        "sglang_max_running_requests",
        "sglang_max_total_tokens",
        "sglang_max_mamba_cache_size",
        "sglang_cuda_graph_max_bs_decode",
        "sglang_mem_fraction_static",
        "sglang_context_length",
        "sglang_disable_radix_cache",
    )
    return stage(
        args,
        "evaluation_dispatch" if snapshots else "evaluation",
        rollout_id,
        details={
            "phase": "initial" if initial else "periodic",
            "scope": "snapshot_submission" if snapshots else "blocking_shared_engine_evaluation",
            "configured_serving": {key: getattr(args, key, None) for key in keys},
        },
    )


@contextmanager
def startup_stage(args, name, *, device=None):
    """Per-rank initialization intervals; synchronize only explicitly GPU stages."""
    started = time.perf_counter()
    wall = time.time()
    passed = False
    try:
        yield
        if device is not None:
            device.synchronize()
        passed = True
    finally:
        if getattr(args, "save", None):
            root = Path(args.save)
            root.mkdir(parents=True, exist_ok=True)
            rank = getattr(args, "rank", 0)
            with (root / f"startup_rank{rank}.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(stage=name, started_unix=wall, seconds=time.perf_counter() - started, passed=passed)
                    )
                    + "\n"
                )
