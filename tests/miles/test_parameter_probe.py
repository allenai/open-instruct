"""Probe persistent optimizer shards across FSDP's temporary parameter views.

Run with pytest locally, or torchrun --standalone --nproc_per_node=2 -m pytest
-q tests/miles/test_parameter_probe.py for real two-GPU sharding.
"""

import os
from datetime import timedelta

import pytest
import torch
from torch import distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from open_instruct.miles import contract


def test_probe_uses_optimizer_parameters_when_model_exposes_temporary_views():
    model = nn.Linear(8, 4, bias=False)
    parameters = list(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    persistent = model.weight
    before = persistent.detach().clone()
    # A no-gradient forward can leave FSDP's temporary full parameter visible.
    temporary = nn.Parameter(persistent.detach().clone())
    model.weight = temporary
    probe = contract.ParameterProbe(parameters)
    model.weight = persistent
    model(torch.ones(2, 8)).sum().backward()
    expected_gradient = persistent.grad.norm().item()
    optimizer.step()
    temporary.untyped_storage().resize_(0)
    gradients = probe.gradients()["dense"]
    updates = probe.updates()["dense"]
    assert gradients["missing_parameters"] == 0
    assert gradients["local_l2"] == pytest.approx(expected_gradient)
    assert updates["sampled_update_l2"] == pytest.approx((persistent - before).norm().item())
    assert updates["sampled_update_l2"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("reshard_after_forward", [True, False])
def test_probe_across_real_fsdp_scoring_backward_and_optimizer(tmp_path, reshard_after_forward):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    init_method = "env://" if "RANK" in os.environ else f"file://{tmp_path}/rendezvous"
    dist.init_process_group(
        "nccl", init_method=init_method, rank=rank, world_size=world, timeout=timedelta(seconds=120)
    )
    try:
        torch.manual_seed(17)
        model = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 8)).cuda()
        fully_shard(model[0], reshard_after_forward=reshard_after_forward)
        fully_shard(model[2], reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        parameters = list(model.named_parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        assert all(isinstance(p, DTensor) for _, p in parameters)
        assert {id(p) for _, p in parameters} == {id(p) for g in optimizer.param_groups for p in g["params"]}
        for step in range(3):
            inputs = torch.randn(4 + step, 16, device="cuda") + rank
            with torch.no_grad():
                scored = model(inputs).clone()
            probe = contract.ParameterProbe(parameters)
            before = {name: p.to_local().detach().clone() for name, p in parameters}
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            torch.testing.assert_close(output.detach(), scored, rtol=0, atol=0)
            output.square().mean().backward()
            gradients = probe.gradients()["dense"]
            assert gradients["missing_parameters"] == 0
            expected_grad = sum(float(p.grad.to_local().double().square().sum()) for _, p in parameters) ** 0.5
            assert gradients["local_l2"] == pytest.approx(expected_grad, rel=1e-6)
            assert expected_grad > 0
            optimizer.step()
            updates = probe.updates()["dense"]
            expected_update = (
                sum(float((p.to_local().detach() - before[name]).double().square().sum()) for name, p in parameters)
                ** 0.5
            )
            # All shards have <=256 entries, so the probe measures every entry.
            assert updates["sampled_update_l2"] == pytest.approx(expected_update, rel=1e-6)
            assert expected_update > 0
    finally:
        dist.destroy_process_group()
