"""Opt-in exact-key native KDA dispatch, shared by rollout and reference servers.

The manifest is a serving policy, not evidence of numerical qualification. No
kernel, autotuner configuration list, tensor or dependency is modified. Unlisted
keys fail before launch so a new precision/shape cannot silently autotune.
"""

import ast
import hashlib
import importlib
import inspect
import json
import os
from functools import wraps
from pathlib import Path

from . import quantized_publication

RUNTIME_LOCK = Path(os.environ.get("OI_MILES_KDA_RUNTIME_LOCK", "/opt/core-rl/sources/runtime.lock.json"))

TARGETS = (
    "fla.ops.kda.chunk_intra_token_parallel.chunk_kda_fwd_kernel_intra_token_parallel.fn",
    "fla.ops.kda.gate.kda_gate_chunk_cumsum_vector_kernel.fn",
)
OPTIONAL_TARGETS = ("fla.ops.kda.chunk_intra.chunk_kda_fwd_kernel_inter_solve_fused.fn",)


def validate_targets(entries):
    targets = [entry["target"] for entry in entries]
    if (
        len(targets) != len(set(targets))
        or not set(TARGETS).issubset(targets)
        or not set(targets).issubset((*TARGETS, *OPTIONAL_TARGETS))
    ):
        raise ValueError("KDA policy must cover both required native targets and only supported optional targets once")


def runtime_key(tuner, args, kwargs):
    all_args = {**dict(zip(tuner.arg_names, args)), **kwargs}
    selected = {name: value for name, value in all_args.items() if name in tuner.arg_names}
    key = [selected[name] for name in tuner.keys if name in selected]
    key.extend(str(value.dtype) for value in selected.values() if hasattr(value, "dtype"))
    return tuple(key)


def prepare(tuner, entry, policy_hash):
    """Validate before installing any wrappers; CPU-testable without Triton."""
    if hashlib.sha256(inspect.getsource(type(tuner).run).encode()).hexdigest() != entry["run_sha256"]:
        raise ValueError("KDA policy autotuner source mismatch")
    if list(tuner.keys) != entry["key_names"]:
        raise ValueError("KDA policy key schema mismatch")
    configs = {}
    for key_text, details in entry["configs"].items():
        key = ast.literal_eval(key_text)
        if not isinstance(key, tuple) or key in configs:
            raise ValueError("KDA policy requires distinct tuple keys")
        matches = [c for c in tuner.configs if all(getattr(c, name, None) == value for name, value in details.items())]
        if len(matches) != 1:
            raise ValueError("KDA policy configuration not uniquely present in native runtime")
        configs[key] = matches[0]
    if not configs:
        raise ValueError("Empty KDA dispatch policy")
    original = tuner.run
    announced = set()

    @wraps(original)
    def run(*args, **kwargs):
        key = runtime_key(tuner, args, kwargs)
        if key not in configs:
            raise ValueError(f"Unsupported KDA policy key for {entry['target']}: {key!r}")
        config = configs[key]
        if key not in announced:
            print(
                json.dumps({"kda_policy_sha256": policy_hash, "target": entry["target"], "key": repr(key)}), flush=True
            )
            announced.add(key)
        tuner.cache[key] = config
        result = original(*args, **kwargs)
        if tuner.cache.get(key) is not config or tuner.best_config is not config:
            raise ValueError("Native KDA launch did not retain policy configuration")
        return result

    run._oi_kda_policy = policy_hash
    return run


def install(path=None, *, server_args=None, gpu_id=None):
    """Install inside each scheduler process, before loading/capture/publication.

    Operators must retain the immutable manifest with the run. This intentionally
    does no per-forward logging, synchronization, tracing or tensor inspection.
    """
    path = path or os.environ.get("OI_MILES_KDA_POLICY")
    if not path:
        return None
    raw = Path(path).read_bytes()
    policy_hash = hashlib.sha256(raw).hexdigest()
    policy = json.loads(raw)
    if policy.get("version") != 1 or policy.get("device_capability") not in ([10, 0], [10, 3]):
        raise ValueError("Unsupported KDA policy version/device")
    torch = importlib.import_module("torch")
    if gpu_id is None:
        raise ValueError("KDA policy requires the scheduler GPU assignment")
    torch.cuda.set_device(gpu_id)
    actual_capability = tuple(torch.cuda.get_device_capability(gpu_id))
    expected_capability = tuple(policy["device_capability"])
    if actual_capability != expected_capability:
        raise ValueError(
            f"KDA policy device mismatch: expected {expected_capability}, actual {actual_capability}, GPU {gpu_id}"
        )
    if hashlib.sha256(RUNTIME_LOCK.read_bytes()).hexdigest() != policy["runtime_lock_sha256"]:
        raise ValueError("KDA policy runtime lock mismatch")
    if server_args is None:
        raise ValueError("KDA policy requires actual serving arguments")
    config = Path(server_args.model_path) / "config.json"
    if hashlib.sha256(config.read_bytes()).hexdigest() != policy["model_config_sha256"]:
        raise ValueError("KDA policy model configuration mismatch")
    for name, expected in policy["serving_scope"].items():
        if getattr(server_args, name, None) != expected:
            raise ValueError(f"Unsupported KDA policy serving scope: {name}")
    entries = policy["entries"]
    validate_targets(entries)
    installs = []
    for entry in entries:
        module_name, attribute, wrapper = entry["target"].rsplit(".", 2)
        module = importlib.import_module(module_name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != entry["module_sha256"]:
            raise ValueError("KDA native kernel source mismatch")
        tuner = getattr(getattr(module, attribute), wrapper)
        installed = getattr(tuner.run, "_oi_kda_policy", None)
        if installed == policy_hash:
            continue
        if installed:
            raise ValueError("Cannot replace an installed KDA dispatch policy")
        installs.append((tuner, prepare(tuner, entry, policy_hash)))
    for tuner, run in installs:
        tuner.run = run
    return policy_hash


def run_scheduler_process(*args, **kwargs):
    install(
        server_args=args[0] if args else kwargs.get("server_args"),
        gpu_id=args[2] if len(args) > 2 else kwargs.get("gpu_id"),
    )
    quantized_publication.install_nvfp4_publication_context()
    scheduler = importlib.import_module("sglang.srt.managers.scheduler")
    return scheduler.run_scheduler_process(*args, **kwargs)


def register():
    """Native evaluator model-registration hook with the same scheduler policy."""
    importlib.import_module("olmo_sglang").register()
    if os.environ.get("OI_MILES_KDA_POLICY"):
        engine = importlib.import_module("sglang.srt.entrypoints.engine").Engine
        engine.run_scheduler_process_func = staticmethod(run_scheduler_process)
