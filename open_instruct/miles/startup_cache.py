"""Triton cache lifecycle for MILES-owned Core trainer and serving Ray workers.

The setup hook has no training imports. Ray calls it before deserializing the
actor, and each logical worker restores into a fresh, node-local directory.
"""

import dataclasses
import importlib
import json
import os
import re
import shutil
import socket
import tempfile
import time
import uuid
from importlib import util
from pathlib import Path
from typing import Any, cast

from scripts.miles import compiler_cache_run as probes

from open_instruct import logger_utils
from open_instruct.miles import compiler_cache as cache

logger = logger_utils.setup_logger(__name__)
ENV = "OI_CORE_STARTUP_CACHE"
DEFAULT_SHARED = "/weka/oe-training-default/open-instruct-compiler-cache/tmp-7d"


def prepare(args):
    """Freeze source/config identity on the driver without initializing CUDA."""
    args.olmo_core_startup_cache = None
    if not args.olmo_core.compiler_cache:
        return
    shared = Path(args.olmo_core.compiler_cache_root or DEFAULT_SHARED)
    if args.olmo_core.compiler_cache_root is None and not Path("/weka/oe-training-default").is_dir():
        logger.info("Core compiler cache: no default WEKA mount; using ordinary local compiler caches")
        return
    cache.validate_shared_root(shared)
    root = Path(__file__).resolve().parents[2]
    lock_path = root / "build/runtime/miles/runtime.lock.json"
    if not lock_path.exists():
        lock_path = root / "runtime/miles/runtime.lock.json"
    lock = json.loads(lock_path.read_text())
    sources = {"open-instruct": cache.source_identity(root / "open_instruct")}
    for name, package in [("olmo-core", "olmo_core"), ("miles", "miles"), ("olmo-sglang", "olmo_sglang")]:
        spec = util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError(f"Cannot fingerprint installed source: {package}")
        sources[name] = cache.source_identity(Path(next(iter(spec.submodule_search_locations))))
    core = dataclasses.asdict(args.olmo_core)
    for key in tuple(core):
        if key.startswith("compiler_cache"):
            del core[key]
    # Include shape, precision and execution switches; exclude labels and paths.
    miles = {
        key: value
        for key, value in vars(args).items()
        if key.startswith("sglang_")
        or key
        in {
            "actor_num_nodes",
            "actor_num_gpus_per_node",
            "num_gpus_per_node",
            "rollout_num_gpus",
            "rollout_num_gpus_per_engine",
            "micro_batch_size",
            "global_batch_size",
            "rollout_batch_size",
            "n_samples_per_prompt",
            "rollout_max_response_len",
            "rollout_max_prompt_len",
            "rollout_max_context_len",
            "colocate",
            "offload_rollout",
            "use_rollout_routing_replay",
            "use_rollout_logprobs",
            "calculate_per_token_loss",
            "update_weight_transfer_mode",
        }
    }
    for key in tuple(miles):
        if any(secret in key for secret in ("api_key", "password", "secret", "access_token", "auth_token")):
            del miles[key]
    report_dir = shared / "runs" / uuid.uuid4().hex
    report_dir.mkdir(parents=True)
    args.olmo_core_startup_cache = dict(
        shared=str(shared),
        report_dir=str(report_dir),
        image=lock["base_image"]["docker_id"],
        runtime_lock=lock,
        sources=sources,
        model_config=json.loads((Path(args.hf_checkpoint) / "config.json").read_text()),
        run_config={"core": core, "miles": miles},
        restore=args.olmo_core.compiler_cache_restore,
        observe=args.olmo_core.compiler_cache_diagnostics,
    )
    logger.info("Core compiler cache: Triton enabled; worker reports in %s", report_dir)


def worker_runtime_env(args, slot, env_vars):
    """Explicit per-actor environment, independent of Ray cluster inheritance."""
    env_vars = dict(env_vars)
    if "OI_MILES_JUDGE_REGISTRY" in os.environ:
        env_vars["OI_MILES_JUDGE_REGISTRY"] = os.environ["OI_MILES_JUDGE_REGISTRY"]
    policy = getattr(args, "olmo_core_startup_cache", None)
    if not policy:
        return {"env_vars": env_vars}
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", slot):
        raise ValueError("Invalid cache worker slot")
    hook = "open_instruct.miles.startup_cache.setup_worker"
    runtime_env = {
        "env_vars": {**env_vars, ENV: json.dumps({**policy, "slot": slot})},
        "worker_process_setup_hook": hook,
    }
    # Ray 2.58 translates this option for ray.init(), but not actor.options().
    # Use the pinned Ray helper to encode its worker-bootstrap environment key.
    setup_hook = importlib.import_module("ray._private.runtime_env.setup_hook")
    return setup_hook.export_setup_func_module(runtime_env, hook)


def observe_triton(local):
    """Qualification-only compiler activity; no changes to kernel choices."""
    module = importlib.import_module("triton.runtime.cache")
    original_group = module.FileCacheManager.get_group
    original_put = module.FileCacheManager.put

    def record(event):
        with (local / f"activity-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": event}) + "\n")

    def get_group(self, *args, **kwargs):
        value = original_group(self, *args, **kwargs)
        record("group_hit" if value else "group_miss")
        return value

    def put(self, *args, **kwargs):
        value = original_put(self, *args, **kwargs)
        record("put")
        return value

    manager = cast(Any, module.FileCacheManager)
    manager.get_group = get_group
    manager.put = put


def setup_worker():
    """Restore before actor imports, with identity probed on its actual node."""
    raw = os.environ.get(ENV)
    if not raw:
        return
    policy = json.loads(raw)
    started = time.monotonic()
    local = Path(tempfile.mkdtemp(prefix="core-triton-", dir="/tmp"))
    (local / "triton").mkdir()
    probes.validate_local_cache_controls(os.environ)
    os.environ["TRITON_CACHE_DIR"] = str(local / "triton")
    ray = importlib.import_module("ray")
    report = dict(
        slot=policy["slot"],
        local=str(local),
        pid=os.getpid(),
        host=socket.gethostname(),
        node_id=ray.get_runtime_context().get_node_id(),
        shared=policy["shared"],
    )
    try:
        hardware = probes.toolchain(dict(os.environ))
        settings = {**policy["run_config"], "worker_slot": policy["slot"]}
        key, identity = cache.fingerprint(
            image=policy["image"],
            runtime_lock=policy["runtime_lock"],
            sources=policy["sources"],
            model_config=policy["model_config"],
            run_config=settings,
            toolchain=hardware,
            compiler_env=probes.compiler_environment(os.environ),
        )
        report.update(fingerprint=key, identity=identity)
        report["restore"] = (
            cache.restore(Path(policy["shared"]), local, key, "triton")
            if policy["restore"]
            else {"family": "triton", "status": "cold"}
        )
        report["restored_files"] = len(cache.inventory(local / "triton"))
    except Exception as error:
        # Compilation is still available locally; a cache failure is not a run failure.
        report["restore"] = {"family": "triton", "status": "unavailable", "reason": str(error)}
    report["setup_seconds"] = time.monotonic() - started
    report_path = Path(policy["report_dir"]) / f"{policy['slot']}-{uuid.uuid4().hex}.json"
    report_path.write_bytes(cache.encoded(report))
    if policy["observe"]:
        observe_triton(local)
        # SGLang uses spawn: Python patches in the Ray coordinator do not carry
        # into its scheduler. This diagnostic-only hook observes those children.
        hook = local / "observer"
        hook.mkdir()
        (hook / "sitecustomize.py").write_text(
            "import os\nfrom pathlib import Path\n"
            "from open_instruct.miles.startup_cache import observe_triton\n"
            "observe_triton(Path(os.environ['OI_CORE_CACHE_OBSERVE_ROOT']))\n"
        )
        os.environ["OI_CORE_CACHE_OBSERVE_ROOT"] = str(local)
        os.environ["PYTHONPATH"] = str(hook) + os.pathsep + os.environ.get("PYTHONPATH", "")
    logger.info("Core compiler cache worker %s: %s", policy["slot"], report["restore"])


def publish_worker(report):
    """Runs on the original node after successful training and worker disposal."""
    local = Path(report["local"])
    if not local.name.startswith("core-triton-") or local.parent != Path("/tmp"):
        raise ValueError("Unexpected worker cache directory")
    result = dict(slot=report["slot"], restore=report["restore"], setup_seconds=report["setup_seconds"])
    try:
        result["files_after"] = len(cache.inventory(local / "triton"))
        activity = {}
        for path in local.glob("activity-*.jsonl"):
            for line in path.read_text().splitlines():
                event = json.loads(line)["event"]
                activity[event] = activity.get(event, 0) + 1
        result["activity"] = activity
        result["publish"] = cache.publish(Path(report["shared"]), local, report["fingerprint"], "triton")
    except Exception as error:
        result["publish"] = {"status": "rejected", "reason": str(error)}
    else:
        shutil.rmtree(local)
    return result


async def finish(args, *, success):
    """Publish only after the driver's successful teardown, on each owning node."""
    policy = getattr(args, "olmo_core_startup_cache", None)
    if not policy:
        return
    report = {"success": success, "workers": [], "report_dir": policy["report_dir"]}
    if success:
        ray = importlib.import_module("ray")
        strategies = importlib.import_module("ray.util.scheduling_strategies")
        asyncio = importlib.import_module("asyncio")
        for path in sorted(Path(policy["report_dir"]).glob("*.json")):
            worker = json.loads(path.read_text())
            if "fingerprint" not in worker:
                report["workers"].append(worker)
                continue
            try:
                task = (
                    ray.remote(num_cpus=0)(publish_worker)
                    .options(
                        scheduling_strategy=strategies.NodeAffinitySchedulingStrategy(worker["node_id"], soft=False)
                    )
                    .remote(worker)
                )
                result = await asyncio.wait_for(task, timeout=120)
            except Exception as error:
                result = {"slot": worker["slot"], "publish": {"status": "unavailable", "reason": str(error)}}
            report["workers"].append(result)
    if getattr(args, "save", None):
        target = Path(args.save) / "compiler-cache.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(cache.encoded(report))
    logger.info("Core compiler cache completion: %s", json.dumps(report))
