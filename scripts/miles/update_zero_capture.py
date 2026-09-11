"""Opt-in diagnostic hooks for exact teacher-forced SGLang prefill requests.

Call install_import_hook() in each scheduler before external-model import, or
attach_capture(model, root) immediately after model construction. The driver
atomically writes ROOT/capture-request.json using arm_capture() before its /generate RPC.
No model weights, kernels, routing choices, or outputs are modified.
"""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import re
import sys
from pathlib import Path

import torch
from triton.runtime.autotuner import Autotuner

TRACE_ENV = "OI_UPDATE_ZERO_TRACE_DIR"
MODEL_MODULE = "olmo_sglang.models.olmo3_moe"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_arm(arm):
    phase, ids, positions = arm["phase"], arm["input_ids"], arm["positions"]
    for name in ("capture_id", "case_id"):
        if not isinstance(arm[name], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", arm[name]):
            raise ValueError(f"Invalid {name}")
    if not isinstance(phase, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", phase):
        raise ValueError("Capture phase must be a simple filename component")
    if not ids or len(ids) > 16384 or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Capture requires 1..16384 nonnegative integer input IDs")
    if not positions or len(positions) > 144 or positions != sorted(set(positions)):
        raise ValueError("Capture requires 1..144 unique ordered token positions")
    if any(type(pos) is not int or not 0 <= pos < len(ids) for pos in positions):
        raise ValueError("Capture position outside fixed prefix")
    return arm


def request_positions(input_ids):
    return sorted(set(range(min(16, len(input_ids)))) | set(range(max(0, len(input_ids) - 128), len(input_ids))))


def arm_capture(root, phase, case_id, capture_id, input_ids):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arm = validate_arm(
        {
            "phase": phase,
            "case_id": case_id,
            "capture_id": capture_id,
            "input_ids": input_ids,
            "positions": request_positions(input_ids),
        }
    )
    atomic_json(root / "capture-request.json", arm)
    return {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        "return_logprob": True,
        "logprob_start_len": 0,
    }


def source_manifest():
    records = {}
    for name, module in list(sys.modules.items()):
        if name.startswith(
            (
                "olmo_sglang",
                "sglang.srt.models",
                "sglang.srt.layers",
                "fla.",
                "scripts.miles.update_zero_capture",
                "update_zero_capture",
            )
        ):
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).is_file():
                records[name] = {"file": filename, "sha256": digest(Path(filename).read_bytes())}
    return records


def snapshot_autotune_configs():
    """Read populated tuner decisions without running, compiling, or retuning kernels.

    This follows olmo-miles' evaluation.determinism_trace implementation. Empty
    caches are explicitly absent, so an empty report is not proof of equal choices.
    """
    result, seen = {}, set()
    for module_name, module in sorted(list(sys.modules.items())):
        if module is None or not module_name.startswith(
            ("fla.", "sglang.kernels.ops.attention.fla.", "sglang.srt.batch_invariant_ops.")
        ):
            continue
        for symbol, candidate in sorted(list(vars(module).items())):
            current = candidate
            for _ in range(8):
                if isinstance(current, Autotuner):
                    break
                current = getattr(current, "fn", None)
                if current is None:
                    break
            if not isinstance(current, Autotuner) or id(current) in seen:
                continue
            seen.add(id(current))
            if not current.cache:
                continue
            result[module_name + "." + symbol] = {
                repr(key): {
                    "kwargs": dict(config.kwargs),
                    **{
                        name: getattr(config, name, None)
                        for name in ("num_warps", "num_stages", "num_ctas", "maxnreg")
                    },
                }
                for key, config in current.cache.items()
            }
    return result


def autotune_policy():
    mode = getattr(sys.modules.get("fla.ops.utils.cache"), "FLA_CACHE_MODE", None)
    return {
        "cache_results": getattr(sys.modules.get("fla.utils._config"), "FLA_CACHE_RESULTS", None),
        "cache_mode": getattr(mode, "value", None),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("FLA_", "TRITON_"))
            or key
            in (
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "CUDA_MODULE_LOADING",
                "CUDA_DEVICE_MAX_CONNECTIONS",
            )
        },
        "interpretation": "Observed populated autotuner caches after this forward; includes earlier warmup choices and does not identify every kernel actually launched by this request.",
    }


def selected_tensor(value, token_count, positions):
    """Preserve dtype; reject tensors whose token axis cannot be established."""
    if not isinstance(value, torch.Tensor):
        return None
    if value.ndim and value.shape[0] == token_count:
        return value.detach()[positions].cpu().clone()
    if value.ndim >= 2 and value.shape[:2] == (1, token_count):
        return value.detach()[0, positions].cpu().clone()
    return None


def tensor_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def route_record(logits, topk_ids, topk_weights):
    """Save actual assignments and row-wise k/k+1 margins without recasting storage."""
    if logits.ndim != 2 or topk_ids.ndim != 2 or logits.shape[0] != topk_ids.shape[0]:
        raise ValueError("Router capture requires aligned two-dimensional rows")
    k = topk_ids.shape[1]
    if not 0 < k < logits.shape[1] or not torch.isfinite(logits).all():
        raise ValueError("Router logits must be finite with 0 < k < expert count")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("Routing weights/IDs shape mismatch")
    ordered = torch.argsort(logits.float(), dim=-1, descending=True, stable=True)
    scores = logits.float().gather(1, ordered)
    canonical = ordered[:, :k]
    return {
        "logits": logits,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "logits_dtype": str(logits.dtype),
        "ids_dtype": str(topk_ids.dtype),
        "weights_dtype": str(topk_weights.dtype),
        "canonical_stable_topk_ids": canonical,
        "boundary_margin": scores[:, k - 1] - scores[:, k],
        "canonical_set_matches": (torch.sort(canonical, dim=-1).values == torch.sort(topk_ids, dim=-1).values).all(-1),
    }


def attach_capture(model, root):
    """Attach after construction; only an armed, exact, full prefill can save data.

    CUDA graph capture and decode graph replay are outside this measurement.
    The caller must disable prefix/radix reuse and send one unchunked prefix.
    Missing layers/routes fail the capture rather than yielding a partial proof.
    """
    if getattr(model, "_oi_prefill_capture_installed", False):
        return
    root = Path(root)
    destination = root / f"worker-{os.getpid()}"
    destination.mkdir(parents=True, exist_ok=True)
    state = {"active": None}
    layers, route_names, handles = [], [], []
    selected_modules = {}
    for name, module in model.named_modules():
        if re.fullmatch(r"model.layers.\d+", name):
            layers.append(name)

            def capture_input(_module, args, kwargs, *, name=name):
                active = state["active"]
                if active is not None:
                    value = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
                    tensor = selected_tensor(value, len(active["input_ids"]), active["positions"])
                    if tensor is not None:
                        active["activations"][name + ".input"] = tensor

            handles.append(module.register_forward_pre_hook(capture_input, with_kwargs=True))
        if name.endswith(".mlp"):

            def capture_mlp_input(_module, args, kwargs, *, name=name):
                active = state["active"]
                if active is not None:
                    value = kwargs.get("hidden_states", args[0] if args else None)
                    tensor = selected_tensor(value, len(active["input_ids"]), active["positions"])
                    if tensor is not None:
                        active["activations"][name + ".input"] = tensor

            handles.append(module.register_forward_pre_hook(capture_mlp_input, with_kwargs=True))
        if name.endswith(".mlp.topk"):
            route_names.append(name)
        if name in ("model.embed_tokens", "model.embed_norm", "model.norm") or (
            name.startswith("model.layers.")
            and (
                name.count(".") == 2
                or name.endswith(
                    (
                        ".self_attn",
                        ".mlp",
                        ".pre_attention_layernorm",
                        ".post_attention_layernorm",
                        ".pre_feedforward_layernorm",
                        ".post_feedforward_layernorm",
                        ".latent_down_proj",
                        ".latent_up_proj",
                    )
                )
            )
        ):
            selected_modules[name] = module

            def capture(_module, _args, output, *, name=name):
                active = state["active"]
                if active is not None:
                    tensor = selected_tensor(tensor_output(output), len(active["input_ids"]), active["positions"])
                    if tensor is not None:
                        active["activations"][name] = tensor

            handles.append(module.register_forward_hook(capture))
        if name.endswith(".mlp.topk"):

            def capture_topk(_module, args, kwargs, output, *, name=name):
                active = state["active"]
                if active is None:
                    return
                logits = kwargs.get("router_logits", args[1] if len(args) > 1 else None)
                count = len(active["input_ids"])
                positions = list(range(count))
                values = [
                    selected_tensor(value, count, positions)
                    for value in (logits, output.topk_ids, output.topk_weights)
                ]
                if any(value is None for value in values):
                    raise ValueError("Router outputs do not match exact prefill token axis")
                active["routes"][name] = route_record(*values)

            handles.append(module.register_forward_hook(capture_topk, with_kwargs=True))
    if not layers or not route_names:
        raise ValueError("No decoder layers or actual SGLang topk modules found")
    original = model.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        state["active"] = None
        ids = kwargs.get("input_ids", args[0] if args else None)
        armed = root / "capture-request.json"
        capturing = isinstance(ids, torch.Tensor) and ids.is_cuda and torch.cuda.is_current_stream_capturing()
        if not capturing and isinstance(ids, torch.Tensor) and armed.is_file():
            arm = json.loads(armed.read_text())
            arm["positions"] = request_positions(arm["input_ids"])
            arm = validate_arm(arm)
            already_saved = (destination / f"{arm['capture_id']}.pt").exists()
            if (
                not already_saved
                and list(ids.shape) == [len(arm["input_ids"])]
                and ids.detach().cpu().tolist() == arm["input_ids"]
            ):
                state["active"] = {**arm, "activations": {}, "routes": {}}
        try:
            result = original(*args, **kwargs)
            active = state["active"]
            if active is not None:
                expected_activations = (
                    set(layers)
                    | {name + ".input" for name in layers}
                    | {
                        name
                        for name in selected_modules
                        if name.endswith((".self_attn", ".mlp")) or name in ("model.embed_tokens", "model.norm")
                    }
                )
                missing = expected_activations - active["activations"].keys()
                missing_routes = set(route_names) - active["routes"].keys()
                if missing or missing_routes:
                    raise ValueError(
                        f"Incomplete prefill capture: layers={sorted(missing)}, routes={sorted(missing_routes)}"
                    )
                logits = getattr(result, "next_token_logits", None)
                if isinstance(logits, torch.Tensor):
                    active["next_token_logits"] = logits.detach().cpu().clone()
                    values, indices = torch.topk(
                        active["next_token_logits"].float(), min(32, logits.shape[-1]), dim=-1
                    )
                    active["next_token_topk"] = {"values": values, "ids": indices, "logits_dtype": str(logits.dtype)}
                active["parameter_dtypes"] = {
                    name: {"shape": list(p.shape), "dtype": str(p.dtype)} for name, p in model.named_parameters()
                }
                active["module_types"] = {
                    name: type(module).__module__ + "." + type(module).__qualname__
                    for name, module in selected_modules.items()
                }
                active["controls"] = {
                    "torch": str(torch.__version__),
                    "cuda": torch.version.cuda,
                    "grad_enabled": torch.is_grad_enabled(),
                    "tf32": torch.backends.cuda.matmul.allow_tf32,
                    "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    "float32_matmul_precision": torch.get_float32_matmul_precision(),
                    "cuda_graph_capturing": False,
                    "pid": os.getpid(),
                }
                active["sources"] = source_manifest()
                active["autotune_configs"] = snapshot_autotune_configs()
                active["autotune_policy"] = autotune_policy()
                path = destination / f"{active['capture_id']}.pt"
                temporary = path.with_suffix(".tmp")
                torch.save(active, temporary)
                temporary.replace(path)
                atomic_json(
                    path.with_suffix(".json"),
                    {
                        "phase": active["phase"],
                        "case_id": active["case_id"],
                        "capture_id": active["capture_id"],
                        "sha256": digest(path.read_bytes()),
                        "input_ids_sha256": digest(json.dumps(active["input_ids"]).encode()),
                        "positions": active["positions"],
                        "layers": layers,
                        "route_modules": route_names,
                        "activation_names": list(active["activations"]),
                        "controls": active["controls"],
                        "sources": active["sources"],
                        "autotune_configs": active["autotune_configs"],
                        "autotune_policy": active["autotune_policy"],
                    },
                )
            return result
        finally:
            state["active"] = None

    model.forward = forward
    model._oi_prefill_capture_installed = True
    model._oi_prefill_capture_handles = handles


def _patch_model(module):
    cls = module.Olmo3MoeForCausalLM
    if getattr(cls, "_oi_prefill_constructor_patched", False):
        return
    original = cls.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        attach_capture(self, Path(os.environ[TRACE_ENV]))

    cls.__init__ = initialize
    cls._oi_prefill_constructor_patched = True


class CaptureLoader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        creator = getattr(self.delegate, "create_module", None)
        return creator(spec) if creator else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        _patch_model(module)


class CaptureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != MODEL_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find required external model {fullname}")
        spec.loader = CaptureLoader(spec.loader)
        sys.meta_path.remove(self)
        return spec


def install_import_hook():
    """Run from existing worker setup or diagnostic sitecustomize before model import."""
    if not os.environ.get(TRACE_ENV):
        return
    if MODEL_MODULE in sys.modules:
        _patch_model(sys.modules[MODEL_MODULE])
    elif not any(isinstance(finder, CaptureFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, CaptureFinder())
