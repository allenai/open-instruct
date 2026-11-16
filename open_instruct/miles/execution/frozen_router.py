"""Freeze only Core routed-expert router parameters before parallelization."""

import functools
import json
from pathlib import Path


def router_parameters(model):
    return {
        f"{name}.{key}": param
        for name, module in model.named_modules()
        if name.split(".")[-1] == "routed_experts_router"
        for key, param in module.named_parameters()
    }


def freeze(model):
    parameters = router_parameters(model)
    if not parameters:
        raise ValueError("core.freeze_router requires Core routed_experts_router modules")
    for param in parameters.values():
        param.requires_grad_(False)
    return {name: {"shape": list(param.shape), "numel": param.numel()} for name, param in parameters.items()}


def install(backend):
    """The actor-local hook receives the meta model before Core builds DDP/optim."""
    original = backend.build_train_module
    if getattr(original, "_oi_freeze_router", False):
        return

    @functools.wraps(original)
    def build(args, *, common, **kwargs):
        if not getattr(args.olmo_core, "freeze_router", False):
            return original(args, common=common, **kwargs)
        expected = freeze(common["model"])
        module = original(args, common=common, **kwargs)
        actual = router_parameters(module.model)
        if set(actual) != set(expected) or any(param.requires_grad for param in actual.values()):
            raise RuntimeError("Core model materialization changed the frozen router parameter set")
        owned = {id(param) for group in module.optim.param_groups for param in group["named_params"].values()}
        if any(id(param) in owned for param in actual.values()):
            raise RuntimeError("Frozen router parameter was included in optimizer groups")
        receipt = {
            "frozen_parameters": expected,
            "parameter_tensors": len(expected),
            "parameter_elements": sum(item["numel"] for item in expected.values()),
            "optimizer_excludes_router": True,
            "frozen_before_parallelization_and_optimizer": True,
        }
        path = Path(args.save) / "frozen-router-validation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        return module

    build._oi_freeze_router = True
    backend.build_train_module = build
