"""Synchronous native Core checkpoints committed with the MILES data cursor."""

import hashlib
import json
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist

from open_instruct.miles import models
from open_instruct.miles.state import PolicyClock, atomic_json


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
    models.save_native(actor.train_module, path / "model")
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
                "schema_version": 1,
                "world_size": dist.get_world_size(),
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


def restore(actor):
    if not actor.args.load:
        return
    path, manifest = resume_manifest(actor.args.load)
    if manifest["schema_version"] != 1 or manifest["world_size"] != dist.get_world_size():
        raise ValueError("Core RL resume currently requires the saved trainer topology")
    if manifest["model_config"] != actor.model_config.as_config_dict():
        raise ValueError("Core model configuration differs from the saved architecture")
    models.load_native(actor.train_module, path / "model")
    state = torch.load(path / f"rank_{dist.get_rank()}.pt", map_location="cpu", weights_only=False)
    actor.lr_scheduler.load_state_dict(state["scheduler"])
    actor.clock = PolicyClock.from_dict(manifest["clock"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
