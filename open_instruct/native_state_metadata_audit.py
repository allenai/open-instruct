"""Bounded CPU metadata readability of our own native ZeRO-3 checkpoint.

Uses mmap for tensor storage and never restores a model/optimizer or scans full tensors.
Trusted native pickle deserialization is confined to the explicitly opted-in run root.
"""

import hashlib
import json
import re
from pathlib import Path

import torch

MAX_FILE_BYTES = 16 * 1024**3
MAX_TOTAL_BYTES = 64 * 1024**3
MAX_KEYS = 10000


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid native {name}")
    return value


def dictionary(value, name):
    if not isinstance(value, dict) or not 1 <= len(value) <= MAX_KEYS:
        raise ValueError(f"Invalid bounded native {name}")
    if any(not isinstance(key, (str, int)) for key in value):
        raise ValueError(f"Invalid native {name} keys")
    return value


def checked_path(root, name):
    path = root / name
    if Path(name).name != name or path.is_symlink() or not path.is_file():
        raise ValueError("Missing or linked native checkpoint file")
    return path


def file_stat(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_metadata(state, rank, world_size, training_step):
    dictionary(state, "model state")
    if integer(state.get("rank"), "rank") != rank:
        raise ValueError("Native client rank mismatch")
    if integer(state.get("training_step"), "client training step") != training_step:
        raise ValueError("Native client step mismatch")
    if integer(state.get("dp_world_size"), "DP world size", 1) != world_size:
        raise ValueError("Native DP world size mismatch")
    integer(state.get("skipped_steps"), "skipped steps")
    if state["skipped_steps"] != 0:
        raise ValueError("Native checkpoint records skipped steps")
    global_steps = integer(state.get("global_steps"), "global steps", 1)
    module = dictionary(state.get("module"), "module")
    shapes = state.get("param_shapes")
    if not isinstance(shapes, list) or not 1 <= len(shapes) <= MAX_KEYS:
        raise ValueError("Native parameter shapes missing")
    shape_rows = []
    for group in shapes:
        dictionary(group, "parameter shape group")
        for name, shape in group.items():
            if not isinstance(name, str) or not isinstance(shape, (tuple, list, torch.Size)):
                raise ValueError("Invalid native parameter shape")
            dims = [integer(dim, "parameter dimension") for dim in shape]
            shape_rows.append((name, dims))
            if len(shape_rows) > MAX_KEYS:
                raise ValueError("Parameter shape count exceeds bound")
    loader = dictionary(state.get("dataloader_state"), "dataloader cursor")
    if integer(loader.get("training_step"), "dataloader training step") != training_step:
        raise ValueError("Dataloader cursor mismatch")
    prep = dictionary(state.get("data_prep_actor_state"), "preparation cursor")
    if integer(prep.get("last_consumed_step"), "last consumed step") != training_step - 1:
        raise ValueError("Preparation consumed cursor mismatch")
    integer(prep.get("training_step"), "preparation training step")
    dictionary(prep.get("iter_dataloader_state"), "preparation iterator cursor")
    rng = dictionary(state.get("rng_states"), "RNG state")
    if not {"torch_cpu_rng_state", "numpy_rng_state", "python_rng_state"} <= set(rng):
        raise ValueError("Native RNG state incomplete")
    return {
        "rank": rank,
        "training_step": training_step,
        "global_steps": global_steps,
        "skipped_steps": state["skipped_steps"],
        "dp_world_size": world_size,
        "episode": integer(state.get("episode"), "episode"),
        "num_total_tokens": integer(state.get("num_total_tokens"), "total tokens"),
        "module_keys": len(module),
        "parameter_shape_keys": len(shape_rows),
        "parameter_shape_sha256": hashlib.sha256(json.dumps(sorted(shape_rows)).encode()).hexdigest(),
        "cursor": {
            "dataloader_training_step": loader["training_step"],
            "dataloader_keys": sorted(map(str, loader)),
            "prep_training_step": prep["training_step"],
            "last_consumed_step": prep["last_consumed_step"],
            "iterator_keys": sorted(map(str, prep["iter_dataloader_state"])),
        },
        "rng_keys": sorted(map(str, rng)),
    }


def optimizer_metadata(state, world_size):
    outer = dictionary(state, "optimizer file")
    zero = dictionary(outer.get("optimizer_state_dict"), "ZeRO optimizer")
    if int(zero.get("zero_stage", -1)) != 3 or zero.get("partition_count") != world_size:
        raise ValueError("Native ZeRO-3 partition metadata mismatch")
    if zero.get("overflow") is not False:
        raise ValueError("Native optimizer overflow not explicitly false")
    optimizer = dictionary(zero.get("optimizer_state_dict"), "Adam optimizer")
    states = dictionary(optimizer.get("state"), "Adam moments")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or not 1 <= len(groups) <= MAX_KEYS:
        raise ValueError("Native Adam parameter groups missing")
    flat = zero.get("fp32_flat_groups")
    if (
        not isinstance(flat, list)
        or not 1 <= len(flat) <= MAX_KEYS
        or any(not isinstance(t, torch.Tensor) for t in flat)
    ):
        raise ValueError("Native fp32 optimizer partitions missing")
    steps = []
    for value in states.values():
        dictionary(value, "Adam moment entry")
        if not isinstance(value.get("exp_avg"), torch.Tensor) or not isinstance(value.get("exp_avg_sq"), torch.Tensor):
            raise ValueError("Native Adam moments missing")
        step = value.get("step")
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or step.device.type != "cpu":
                raise ValueError("Native Adam scalar step invalid")
            step = step.item()
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not 0 < step <= 1000000:
            raise ValueError("Native Adam scalar step invalid")
        steps.append(float(step))
    return {
        "zero_stage": 3,
        "partition_count": world_size,
        "overflow": False,
        "optimizer_keys": sorted(map(str, zero)),
        "adam_entries": len(states),
        "adam_param_groups": len(groups),
        "adam_step_min": min(steps),
        "adam_step_max": max(steps),
        "fp32_partition_numel": [tensor.numel() for tensor in flat],
        "full_moment_finiteness_checked": False,
    }


def inspect(checkpoint_root, world_size, training_step):
    integer(world_size, "world size", 1)
    integer(training_step, "training step", 1)
    if world_size > 8:
        raise ValueError("Native metadata audit limited to one eight-GPU node")
    root = Path(checkpoint_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid native checkpoint root")
    latest = checked_path(root, "latest")
    if latest.stat().st_size > 256:
        raise ValueError("Native latest tracker exceeds bound")
    tag = latest.read_text().strip()
    if not re.fullmatch(r"global_step[0-9]+", tag):
        raise ValueError("Invalid native checkpoint tag")
    folder = root / tag
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Missing or linked native tag directory")
    model_names = [f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt" for rank in range(world_size)]
    optimizer_names = [f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(world_size)]
    expected = set(model_names + optimizer_names)
    if {file.name for file in folder.glob("*.pt")} != expected:
        raise ValueError("Native checkpoint shard set mismatch")
    paths = [latest, *(checked_path(folder, name) for name in sorted(expected))]
    before = {str(path.relative_to(root)): file_stat(path) for path in paths}
    if (
        any(value["size"] > MAX_FILE_BYTES for value in before.values())
        or sum(value["size"] for value in before.values()) > MAX_TOTAL_BYTES
    ):
        raise ValueError("Native checkpoint byte bound exceeded")
    models, optimizers = [], []
    for rank, (model_name, optimizer_name) in enumerate(zip(model_names, optimizer_names, strict=True)):
        # Own opt-in run checkpoint, native pickle format. No non-mmap fallback.
        model_state = torch.load(folder / model_name, map_location="cpu", weights_only=False, mmap=True)
        models.append(model_metadata(model_state, rank, world_size, training_step))
        del model_state
        optimizer_state = torch.load(folder / optimizer_name, map_location="cpu", weights_only=False, mmap=True)
        optimizers.append({"rank": rank, **optimizer_metadata(optimizer_state, world_size)})
        del optimizer_state
    if len({model["global_steps"] for model in models}) != 1 or tag != f"global_step{models[0]['global_steps']}":
        raise ValueError("Native tag/global counter mismatch")
    if len({model["parameter_shape_sha256"] for model in models}) != 1:
        raise ValueError("Native rank parameter shape mismatch")
    if any(model["cursor"] != models[0]["cursor"] for model in models):
        raise ValueError("Native rank cursor mismatch")
    after = {str(path.relative_to(root)): file_stat(path) for path in paths}
    if before != after or latest.read_text().strip() != tag:
        raise ValueError("Native checkpoint changed during metadata read")
    return {
        "status": "native-metadata-readable",
        "tag": tag,
        "training_step": training_step,
        "world_size": world_size,
        "models": models,
        "optimizers": optimizers,
        "inventory_before": before,
        "inventory_after": after,
        "restore_verified": False,
        "full_tensor_finiteness_checked": False,
        "limits": "Own native ZeRO-3 pickle/mmap CPU metadata readability only; not model/optimizer/cursor/RNG restore, accepted-update or full-weight equality proof.",
    }
