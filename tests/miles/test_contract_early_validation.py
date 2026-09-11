"""Reject rank-local bad schedules and routes before distributed model execution."""

import contextlib
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn

from open_instruct.miles import actor
from open_instruct.miles.state import PolicyClock


def _rollout(count):
    return {
        "tokens": [torch.arange(5) for _ in range(count)],
        "total_lengths": [5] * count,
        "response_lengths": [2] * count,
        "loss_masks": [torch.ones(2) for _ in range(count)],
        "rewards": [1.0] * count,
        "weight_versions": [["0"] for _ in range(count)],
    }


def _worker(rank, rendezvous, fault):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
        worker.args = SimpleNamespace(
            global_batch_size=4,
            olmo_core=SimpleNamespace(max_sequence_length=16, max_policy_lag=0),
            use_rollout_routing_replay=fault != "schedule",
        )
        worker.clock = PolicyClock()
        worker.clock.published()
        worker.ref_module = None
        model = nn.Module()
        block = nn.Module()
        block.add_module("routed_experts_router", MoERouterConfigV2(d_model=8, num_experts=4, top_k=2).build())
        model.add_module("blocks", nn.ModuleList([block]))
        worker.train_module = SimpleNamespace(model=model, _miles_model_backend="moe")
        # Both schedules contain complete local optimizer batches. Only their
        # cross-rank disagreement should reject them before the first EP forward.
        rollout = _rollout(4 if fault == "schedule" and rank == 1 else 2)
        if fault != "schedule":
            rollout["rollout_routed_experts"] = [torch.tensor([[[0, 1]]] * 4) for _ in range(2)]
            if rank == 1:
                if fault == "route_shape":
                    rollout["rollout_routed_experts"][1] = torch.tensor([[[0, 1]]] * 3)
                elif fault == "route_ids":
                    rollout["rollout_routed_experts"][1][0, 0, 1] = 4
                elif fault == "route_dtype":
                    rollout["rollout_routed_experts"][1] = rollout["rollout_routed_experts"][1].float() + 0.5
                elif fault == "route_none":
                    rollout["rollout_routed_experts"][1] = None
                elif fault == "route_layers":
                    rollout["rollout_routed_experts"][1] = torch.empty(4, 0, 2, dtype=torch.long)
                else:
                    raise AssertionError(fault)
        expected = "same number" if fault == "schedule" else "replay|Replay"
        with (
            mock.patch.object(actor.distributed_utils, "get_gloo_group", return_value=dist.group.WORLD),
            mock.patch.object(actor.miles_data, "get_rollout_data", return_value=(rollout, contextlib.nullcontext())),
            mock.patch.object(worker, "_score", side_effect=AssertionError("model forward reached")) as score,
            pytest.raises((ValueError, RuntimeError), match=expected),
        ):
            worker.train(0, None)
        score.assert_not_called()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "fault", ["schedule", "route_shape", "route_ids", "route_dtype", "route_layers", "route_none"]
)
def test_rank_local_validation_rejects_before_first_forward(fault, tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rdzv"), fault), nprocs=2, join=True)
