"""Fail-closed restoration checks for the registered H077 SFT continuation."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from olmo_core.distributed import checkpoint as core_checkpoint
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.optim import LinearWithWarmup


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(equal_tree(v, right[k]) for k, v in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(equal_tree(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def verify_trainer_state(expected, actual):
    if expected.keys() != actual.keys():
        raise ValueError("Restored trainer state has different fields")
    for key in expected:
        if key == "max_steps":
            if actual[key] <= expected["global_step"]:
                raise ValueError("Continuation must extend beyond the restored step")
        elif key == "callbacks":
            # The one-time promoted-row check has empty state and is deliberately
            # absent on resume. Never discard a callback's nonempty saved state.
            before = {k: v for k, v in expected[key].items() if v}
            after = {k: v for k, v in actual[key].items() if v}
            if not equal_tree(before, after):
                raise ValueError("Restored callback state differs")
        elif not equal_tree(expected[key], actual[key]):
            raise ValueError(f"Restored trainer state differs: {key}")


def verify_next_lr(module, trainer, horizon):
    scheduler = module.scheduler
    if (
        not isinstance(scheduler, LinearWithWarmup)
        or horizon != 11768
        or scheduler.t_max != horizon
        or scheduler.warmup != 353
        or scheduler.alpha_f != 0.0
        or str(scheduler.units) != "steps"
    ):
        raise ValueError("Continuation changed the registered LR schedule")
    step = trainer.global_step + 1
    expected = 5e-5 * (step / 353 if step < 353 else max(0.0, 1 - (step - 353) / (horizon - 353)))
    values = []
    for group in module.optim.param_groups:
        if not math.isclose(float(group[scheduler.initial_lr_field]), 5e-5, rel_tol=1e-7, abs_tol=0):
            raise ValueError("Continuation changed the initial learning rate")
        copied = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in group.items()
            if key in (scheduler.lr_field, scheduler.initial_lr_field)
        }
        value = float(scheduler.set_lr(copied, SimpleNamespace(global_step=step, max_steps=trainer.max_steps)))
        if not math.isclose(value, expected, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError("Incorrect learning rate for the first resumed update")
        values.append(value)
    return values


def verify(trainer, checkpoint, mode, horizon, tensor_hash):
    """Called immediately after the native trainer load, before dry-run or updates."""
    path = Path(checkpoint)
    rank, step = get_rank(), trainer.global_step
    if get_world_size() != 16 or step <= 0 or path.name != f"step{step}":
        raise ValueError("Resume qualification requires a native 16-rank training checkpoint")
    source = path.parent / "atomic-audit" / f"state-step{step}-rank{rank}.json"
    expected = json.loads(source.read_text())
    if expected["step"] != step or len(expected["model"]) != 415 or len(expected["optimizer"]) != 1660:
        raise ValueError("Incomplete source model/optimizer audit")
    native_path = path / "train" / f"rank{rank}.pt"
    # These are our own retained native checkpoints, including NumPy RNG state.
    native = torch.load(native_path, map_location="cpu", weights_only=False)
    actual_trainer = trainer.state_dict()
    verify_trainer_state(native, actual_trainer)
    audit_state = native["callbacks"]["atomic_audit"]
    if audit_state["mode"] != mode or audit_state["horizon"] != horizon or not audit_state["offline"]:
        raise ValueError("Resume changed intervention or offline protocol")
    module = trainer.train_module
    if {name: tensor_hash(value) for name, value in module.model.state_dict().items()} != expected["model"]:
        raise ValueError("Restored model differs from the saved training audit")
    if {name: tensor_hash(value) for name, value in module.optim.states.items()} != expected["optimizer"]:
        raise ValueError("Restored FP32 masters, moments or optimizer steps differ")
    aux = {module.optim.LOSSES_STATE_DICT_KEY: [], module.optim.GRAD_NORMS_STATE_DICT_KEY: []}
    # Match the native writer. Torch's generic filesystem reader cannot read
    # Core's storage metadata in the pinned training runtime.
    core_checkpoint.load_state_dict(path / "model_and_optim", aux)
    actual_aux = {
        module.optim.LOSSES_STATE_DICT_KEY: [float(v.detach().cpu()) for v in module.optim._losses],
        module.optim.GRAD_NORMS_STATE_DICT_KEY: [float(v.detach().cpu()) for v in module.optim._grad_norms],
    }
    if not equal_tree(aux, actual_aux):
        raise ValueError("Restored rolling optimizer statistics differ")
    next_lr = verify_next_lr(module, trainer, horizon)
    batch = trainer.data_loader[step]
    batch_hashes = {key: tensor_hash(value) for key, value in batch.items() if isinstance(value, torch.Tensor)}
    if set(batch_hashes) != {"input_ids", "label_mask", "index", "doc_lens"}:
        raise ValueError("Unexpected first resumed batch fields")
    # Verification itself must not advance data position or any RNG stream.
    verify_trainer_state(native, trainer.state_dict())
    return {
        "checkpoint": str(path),
        "step": step,
        "rank": rank,
        "mode": mode,
        "horizon": horizon,
        "source_state_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "native_trainer_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "model_tensors": len(expected["model"]),
        "optimizer_tensors": len(expected["optimizer"]),
        "trainer_state_including_rng_exact": True,
        "optimizer_auxiliary_state_exact": True,
        "saved_max_steps": native["max_steps"],
        "new_max_steps": trainer.max_steps,
        "next_step": step + 1,
        "next_lr": next_lr,
        "expected_first_batch": batch_hashes,
    }
