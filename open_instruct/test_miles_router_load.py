"""Router balance is measured per layer and per replica, using actual dispatch counts."""

import json
from datetime import timedelta

import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from open_instruct.miles import router_load


def test_replica_imbalance_is_visible_when_global_load_is_uniform():
    # DP2/EP2: each pair of source ranks routes entirely to a different expert.
    counts = torch.tensor([[[4, 0]], [[4, 0]], [[0, 4]], [[0, 4]]])
    result = router_load.measurements(["layer0"], counts, ep_degree=2)
    summary = result["summary"]
    assert summary["moe/max_expert_load"] == 8
    assert summary["moe/load_cv_max"] == 0
    assert summary["moe/dead_experts"] == 0
    assert summary["moe/replica_load_cv_max"] == 1
    assert summary["moe/replica_max_mean_load_ratio"] == 2
    assert summary["moe/replica_dead_experts_max_per_layer"] == 1
    assert result["layers"]["layer0"]["assignments"] == 16
    json.dumps(result, allow_nan=False)


def test_layers_remain_distinct_and_empty_counts_are_finite():
    counts = torch.tensor([[[4, 0], [0, 4], [0, 0]]])
    result = router_load.measurements(["first", "second", "empty"], counts, ep_degree=1)
    assert result["summary"]["moe/dead_experts"] == 4  # expert-layer pairs
    assert result["summary"]["moe/dead_experts_max_per_layer"] == 2
    assert result["summary"]["moe/load_cv_mean"] == pytest.approx(2 / 3)
    assert result["layers"]["empty"]["load_cv"] == 0
    json.dumps(result, allow_nan=False)


def test_snapshot_is_detached_and_cannot_mutate_native_counters():
    model = torch.nn.Module()
    model.routed_experts_router = torch.nn.Module()
    router = model.routed_experts_router
    router.num_experts = 3
    router.batch_size_per_expert = torch.tensor([5.0, 0.0, 1.0])
    local = router_load.snapshot(model)
    router.batch_size_per_expert.zero_()
    result = router_load.collect(local, ep_degree=1)
    assert result["summary"]["moe/max_expert_load"] == 5
    assert result["summary"]["moe/dead_experts"] == 1
    assert not local[1].requires_grad
    assert router_load.snapshot(torch.nn.Linear(2, 2)) is None
    assert router_load.collect(None, ep_degree=1) is None


@pytest.mark.parametrize("counts,ep", [(torch.zeros(3, 1, 2), 2), (torch.tensor([[[-1, 2]]]), 1)])
def test_invalid_counts_fail_explicitly(counts, ep):
    with pytest.raises(ValueError):
        router_load.measurements(["layer"], counts, ep)


def _collect_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        local = (["layer"], torch.tensor([[4, 0] if rank == 0 else [0, 4]]))
        for ep_degree in (1, 2):
            result = router_load.collect(local, ep_degree)
            reference = router_load.measurements(["layer"], torch.tensor([[[4, 0]], [[0, 4]]]), ep_degree)
            assert result == reference
            json.dumps(result, allow_nan=False)
    finally:
        dist.destroy_process_group()


def test_distributed_histograms_count_each_source_once(tmp_path):
    mp.spawn(_collect_worker, args=(str(tmp_path / "group"),), nprocs=2, join=True)
