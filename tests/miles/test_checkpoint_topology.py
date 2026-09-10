"""Reject ambiguous resume topology before any native checkpoint collectives."""

import json
import random
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from open_instruct.miles import actor, checkpoint
from open_instruct.miles.state import PolicyClock


def _manifest(schema=2, world=2, ep=2):
    document = {
        "schema_version": schema,
        "world_size": world,
        "clock": PolicyClock(completed_steps=1, published_step=0, next_rollout_id=1).as_dict(),
        "model_config": {"test": "unchanged"},
    }
    if schema == 2:
        document["expert_parallel_size"] = ep
    return document


def _state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.get_rng_state(),
        "scheduler": {"last_epoch": 1},
    }


def _actor(root, ep):
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(load=str(root), save=str(root), olmo_core=SimpleNamespace(expert_parallel_size=ep))
    worker.model_config = SimpleNamespace(as_config_dict=lambda: {"test": "unchanged"})
    worker.hf_config = SimpleNamespace(to_dict=lambda: {})
    worker.clock = PolicyClock()
    worker.train_module = object()
    worker.lr_scheduler = SimpleNamespace(load_state_dict=mock.Mock(), state_dict=lambda: {"last_epoch": 0})
    return worker


@pytest.mark.parametrize("saved_ep,requested_ep", [(2, 1), (1, 2)])
def test_same_world_changed_ep_rejected_before_native_load(saved_ep, requested_ep, tmp_path, monkeypatch):
    worker = _actor(tmp_path, requested_ep)
    worker._agree = lambda operation: operation()
    monkeypatch.setattr(checkpoint, "resume_manifest", lambda root: (tmp_path, _manifest(ep=saved_ep)))
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 2)
    native = mock.Mock()
    monkeypatch.setattr(checkpoint.models, "load_native", native)
    with pytest.raises(ValueError, match="saved trainer topology"):
        checkpoint.restore(worker)
    native.assert_not_called()
    assert worker.clock.completed_steps == 0


@pytest.mark.parametrize("schema,world,ep", [(2, 2, 1), (2, 2, 2), (2, 1, 1), (1, 1, 1)])
def test_same_topology_and_unambiguous_legacy_restore(schema, world, ep, tmp_path, monkeypatch):
    worker = _actor(tmp_path, ep)
    events = []

    def agree(operation):
        result = operation()
        events.append("preflight-agreed")
        return result

    worker._agree = agree
    monkeypatch.setattr(checkpoint, "resume_manifest", lambda root: (tmp_path, _manifest(schema, world, ep)))
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: world)
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.torch, "load", lambda *args, **kwargs: _state())
    monkeypatch.setattr(checkpoint.models, "load_native", lambda *args: events.append("native-loaded"))
    monkeypatch.setattr(checkpoint.torch.cuda, "set_rng_state", lambda value: None)
    checkpoint.restore(worker)
    assert events == ["preflight-agreed", "native-loaded"]
    assert worker.clock.completed_steps == worker.clock.next_rollout_id == 1
    worker.lr_scheduler.load_state_dict.assert_called_once_with({"last_epoch": 1})


def test_legacy_multirank_requires_explicit_migration(tmp_path, monkeypatch):
    worker = _actor(tmp_path, 2)
    worker._agree = lambda operation: operation()
    monkeypatch.setattr(checkpoint, "resume_manifest", lambda root: (tmp_path, _manifest(schema=1)))
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 2)
    native = mock.Mock()
    monkeypatch.setattr(checkpoint.models, "load_native", native)
    with pytest.raises(ValueError, match="explicitly migrate"):
        checkpoint.restore(worker)
    native.assert_not_called()


@pytest.mark.parametrize("saved_ep", [None, True, 0, 3])
def test_schema_two_requires_valid_saved_ep(saved_ep):
    with pytest.raises(ValueError, match="saved expert_parallel_size"):
        checkpoint.validate_topology(_manifest(ep=saved_ep), 2, 2)


def test_save_persists_schema_two_expert_degree(tmp_path, monkeypatch):
    worker = _actor(tmp_path, 2)
    worker._agree = lambda operation: operation()
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)
    monkeypatch.setattr(checkpoint.models, "save_native", lambda *args: None)
    monkeypatch.setattr(checkpoint.torch.cuda, "get_rng_state", torch.get_rng_state)
    checkpoint.save(worker, 0)
    manifest = json.loads((checkpoint.checkpoint_path(tmp_path, 0) / "pending.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["world_size"] == manifest["expert_parallel_size"] == 2


def _rank_local_failure(rank, rendezvous, root, fault):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        worker = _actor(root, 2)
        manifest = _manifest(ep=1 if rank == 1 and fault == "topology" else 2)
        native = mock.Mock(side_effect=AssertionError("native collectives reached"))
        loaded_state = mock.Mock(return_value=_state())
        if rank == 1 and fault == "read":
            loaded_state.side_effect = OSError("rank-local checkpoint read failure")
        with (
            mock.patch.object(checkpoint, "resume_manifest", return_value=(root, manifest)),
            mock.patch.object(checkpoint.torch, "load", loaded_state),
            mock.patch.object(checkpoint.models, "load_native", native),
            mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=dist.group.WORLD),
            pytest.raises(RuntimeError, match="rank 1.*(topology|OSError)"),
        ):
            checkpoint.restore(worker)
        native.assert_not_called()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fault", ["topology", "read"])
def test_rank_local_resume_failure_rejected_collectively(fault, tmp_path):
    mp.spawn(_rank_local_failure, args=(str(tmp_path / "rdzv"), tmp_path, fault), nprocs=2, join=True)
