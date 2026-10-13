"""Capture module boundaries and loaded-weight hashes for frozen scoring audits."""

import hashlib
import json
import sys
from pathlib import Path

import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return [tensors(x) for x in value if isinstance(x, (torch.Tensor, tuple, list))]
    return None


def tuning_configs():
    found = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("fla.ops.") or module is None:
            continue
        for key, value in list(vars(module).items()):
            for _ in range(5):
                cache = getattr(value, "cache", None)
                if isinstance(cache, dict) and cache:
                    entries = {repr(k): str(v) for k, v in cache.items() if hasattr(v, "num_warps")}
                    if entries:
                        found[f"{name}.{key}"] = entries
                        break
                value = getattr(value, "fn", None)
                if value is None:
                    break
    return found


def factory(config):
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    installed = False

    def active():
        return (root / "capture").exists()

    def after(model, args, output):
        nonlocal installed
        if not installed:
            for name, module in model.named_modules():
                if config.get("base"):
                    name = config["base"] + ("." + name if name else "")
                boundary = name.startswith("model.layers.") and name.count(".") == 3
                first_layer_detail = name.startswith("model.layers.0.self_attn.")
                if not (boundary or first_layer_detail or name in {"model.embed_norm", "model.norm"}):
                    continue
                state = {}

                def before(m, inputs, state=state):
                    if active():
                        state["inputs"] = tensors(inputs)

                def capture(m, inputs, result, name=name, state=state):
                    if active():
                        state["outputs"] = tensors(result)
                        torch.save(state, root / (name + ".pt"))
                    state.clear()

                module.register_forward_pre_hook(before)
                module.register_forward_hook(capture)
            installed = True
        if active():
            weights = {}
            for name, parameter in model.named_parameters():
                raw = parameter.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
                weights[name] = {
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            (root / "weights.json").write_text(json.dumps(weights))
            (root / "tuning.json").write_text(json.dumps(tuning_configs(), indent=2))
            (root / "device.json").write_text(
                json.dumps({"name": torch.cuda.get_device_name(), "compat": getattr(model, "core_compat_mode", None)})
            )
        return output

    return after


def head_factory(config):
    root = Path(config["root"])

    def after(module, args, output):
        if (root / "capture").exists():
            weights = {}
            for name, parameter in args[2].named_parameters():
                raw = parameter.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
                weights[name] = {
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            (root / "head-weights.json").write_text(json.dumps(weights))
        return output

    return after
