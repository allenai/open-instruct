"""Synchronous native Core checkpoints committed with the MILES data cursor."""

import hashlib
import json
import random
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist

from open_instruct import logger_utils
from open_instruct.miles import models
from open_instruct.miles.state import PolicyClock, atomic_json

logger = logger_utils.setup_logger(__name__)


def checkpoint_path(root, rollout_id):
    return Path(root) / "core" / f"rollout_{rollout_id:07d}"


def prepare_checkpoint_path(path):
    """Preserve an interrupted save while allowing the same rollout to be retried."""
    if path.exists():
        if (path / "complete.json").exists():
            raise FileExistsError(f"Refusing to replace a committed checkpoint: {path}")
        path.rename(path.with_name(f"{path.name}.incomplete-{uuid.uuid4().hex}"))
    path.mkdir(parents=True)


def save(actor, rollout_id):
    path = checkpoint_path(actor.args.save, rollout_id)
    actor._agree(lambda: prepare_checkpoint_path(path) if dist.get_rank() == 0 else None)
    save_metrics = models.save_native(actor.train_module, path / "model")
    if save_metrics is not None and actor.args.olmo_core.checkpoint_profile:
        atomic_json(path / f"save_metrics_rank_{dist.get_rank()}.json", save_metrics)
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "scheduler": actor.lr_scheduler.state_dict(),
        },
        path / f"rank_{dist.get_rank()}.pt",
    )
    dist.barrier()
    if dist.get_rank() == 0:
        atomic_json(
            path / "pending.json",
            {
                "schema_version": 2,
                "world_size": dist.get_world_size(),
                "expert_parallel_size": actor.args.olmo_core.expert_parallel_size,
                "clock": actor.clock.as_dict(),
                "model_config": actor.model_config.as_config_dict(),
                "hf_config": actor.hf_config.to_dict(),
            },
        )


def finalize(actor, rollout_id):
    path = checkpoint_path(actor.args.save, rollout_id)
    if dist.get_rank() == 0:
        pending = json.loads((path / "pending.json").read_text())
        cursor = Path(actor.args.save) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        if not actor.args.rollout_global_dataset or not cursor.is_file():
            raise RuntimeError("A resumable Core checkpoint requires the completed MILES global-dataset cursor")
        pending["cursor_sha256"] = hashlib.sha256(cursor.read_bytes()).hexdigest()
        atomic_json(path / "complete.json", pending)
        atomic_json(Path(actor.args.save) / "core-latest.json", {"rollout_id": rollout_id})
        core = actor.args.olmo_core
        try:
            removed = prune(
                actor.args.save, rollout_id, keep_last=core.checkpoint_keep_last, keep_every=core.checkpoint_keep_every
            )
        except OSError as error:
            # Retention is housekeeping; a failed delete must not fail a committed checkpoint.
            logger.warning("Checkpoint retention could not remove an older checkpoint: %s", error)
        else:
            if removed:
                logger.info("Checkpoint retention removed committed rollouts %s; latest is %d", removed, rollout_id)


def committed_rollout_ids(root):
    """Rollout ids of committed checkpoints under root, ascending; interrupted saves are ignored."""
    base = Path(root) / "core"
    ids = []
    if base.is_dir():
        for entry in base.iterdir():
            match = re.fullmatch(r"rollout_(\d{7})", entry.name)
            if match and (entry / "complete.json").exists():
                ids.append(int(match.group(1)))
    return sorted(ids)


def retained_rollout_ids(ids, latest_id, *, keep_last, keep_every):
    """The newest keep_last committed checkpoints, every keep_every-th completed update, and the latest.

    keep_last=None keeps everything. keep_every counts completed updates, so with
    zero-based rollout ids the checkpoint after update 50 is rollout 49.
    """
    keep = {latest_id}
    if keep_last is None:
        return set(ids) | keep
    keep.update(ids[-keep_last:])
    if keep_every:
        keep.update(rollout_id for rollout_id in ids if (rollout_id + 1) % keep_every == 0)
    return keep


def prune(root, latest_id, *, keep_last, keep_every):
    """Delete committed checkpoints outside the retention set; returns the removed rollout ids."""
    ids = committed_rollout_ids(root)
    keep = retained_rollout_ids(ids, latest_id, keep_last=keep_last, keep_every=keep_every)
    removed = []
    for rollout_id in ids:
        if rollout_id in keep:
            continue
        shutil.rmtree(checkpoint_path(root, rollout_id))
        cursor = Path(root) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        if cursor.is_file():
            cursor.unlink()
        removed.append(rollout_id)
    return removed


def resume_manifest(root):
    latest = Path(root) / "core-latest.json"
    if not latest.exists():
        raise FileNotFoundError(f"No committed Core RL checkpoint: {latest}")
    rollout_id = json.loads(latest.read_text())["rollout_id"]
    path = checkpoint_path(root, rollout_id)
    manifest = json.loads((path / "complete.json").read_text())
    cursor = Path(root) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
    if hashlib.sha256(cursor.read_bytes()).hexdigest() != manifest["cursor_sha256"]:
        raise ValueError("Rollout cursor differs from the committed checkpoint boundary")
    return path, manifest


def validate_topology(manifest, world_size, expert_parallel_size):
    """Reject ambiguous legacy topology and all unimplemented resharding."""
    schema = manifest.get("schema_version")
    saved_world = manifest.get("world_size")
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError(f"Unsupported Core RL checkpoint schema: {schema!r}")
    if type(saved_world) is not int or saved_world < 1:
        raise ValueError("Invalid saved trainer world_size")
    if schema == 1:
        if saved_world != 1:
            raise ValueError(
                "Legacy schema-1 multi-rank checkpoints do not record expert_parallel_size. "
                "Verify the original launch configuration and explicitly migrate the manifest "
                "to schema 2 with its saved expert_parallel_size; automatic inference is disabled."
            )
        saved_ep = 1
    else:
        saved_ep = manifest.get("expert_parallel_size")
        if type(saved_ep) is not int or saved_ep < 1 or saved_world % saved_ep:
            raise ValueError("Invalid or missing saved expert_parallel_size in schema-2 checkpoint")
    if (saved_world, saved_ep) != (world_size, expert_parallel_size):
        raise ValueError(
            "Core RL resume requires the saved trainer topology: "
            f"saved world_size={saved_world}, expert_parallel_size={saved_ep}; "
            f"requested world_size={world_size}, expert_parallel_size={expert_parallel_size}. "
            "Changing trainer topology during resume is not implemented."
        )


def _restore_preflight(actor):
    # Read and validate rank-local state before any native checkpoint collective.
    path, manifest = resume_manifest(actor.args.load)
    validate_topology(manifest, dist.get_world_size(), actor.args.olmo_core.expert_parallel_size)
    if comparable_model_config(manifest["model_config"]) != comparable_model_config(
        actor.model_config.as_config_dict()
    ):
        raise ValueError("Core model configuration differs from the saved architecture")
    clock = PolicyClock.from_dict(manifest["clock"])
    state = torch.load(path / f"rank_{dist.get_rank()}.pt", map_location="cpu", weights_only=False)
    for field in ("scheduler", "python", "numpy", "torch", "cuda"):
        if field not in state:
            raise ValueError(f"Missing rank checkpoint field: {field}")
    return path, state, clock


def restore(actor):
    if not actor.args.load:
        return
    path, state, clock = actor._agree(lambda: _restore_preflight(actor))
    models.load_native(actor.train_module, path / "model")
    actor.lr_scheduler.load_state_dict(state["scheduler"])
    actor.clock = clock
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def comparable_model_config(value, *, parent=None):
    """Canonicalize JSON layer indices and validated SwiGLU execution selection.

    Old manifests omit this field. Retain the original config in new manifests for
    provenance while allowing a static/dynamic rollback without changing weights.
    """
    if isinstance(value, list):
        return [comparable_model_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if parent == "block_overrides":
            # JSON object keys become strings on disk. YaRN supplies integer
            # per-layer overrides in memory; compare without weakening geometry.
            if type(key) is int and key >= 0:
                key = str(key)
            elif not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                raise ValueError("Invalid saved block_overrides layer index")
            if key in result:
                raise ValueError("Duplicate block_overrides layer index")
        if parent == "routed_experts" and key == "row_specialization":
            if item not in ("static", "dynamic"):
                raise ValueError("Invalid saved routed-expert row_specialization")
            continue
        result[key] = comparable_model_config(item, parent=key)
    return result
