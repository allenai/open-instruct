from types import SimpleNamespace

import pytest
import torch
from torch import nn

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.execution import frozen_router


class Model(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.attention = nn.Linear(4, 4, device=device)
        self.routed_experts_router = nn.Linear(4, 2, bias=False, device=device)
        self.experts = nn.Linear(4, 2, device=device)

    def forward(self, x):
        x = self.attention(x)
        return (self.routed_experts_router(x).softmax(-1) * self.experts(x)).sum(-1)


def test_freeze_preserves_input_gradients_and_updates_other_weights():
    torch.manual_seed(17)
    model = Model()
    frozen_router.freeze(model)
    before = {name: p.clone() for name, p in model.named_parameters()}
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    model(torch.randn(8, 4)).square().mean().backward()
    assert model.routed_experts_router.weight.grad is None
    assert model.attention.weight.grad.abs().sum() > 0
    optim.step()
    for name, param in model.named_parameters():
        if "routed_experts_router" in name:
            assert torch.equal(param, before[name])
        else:
            assert not torch.equal(param, before[name])


@pytest.mark.parametrize("wrapped", [False, True])
def test_hook_freezes_meta_model_before_materialization_and_optimizer(tmp_path, wrapped):
    model = Model("meta")

    def build(args, *, common):
        assert not common["model"].routed_experts_router.weight.requires_grad
        common["model"].to_empty(device="cpu")
        groups = [{"named_params": {n: p for n, p in model.named_parameters() if p.requires_grad}}]
        materialized = nn.Module() if wrapped else model
        if wrapped:
            materialized.add_module("module", model)
        return SimpleNamespace(model=materialized, optim=SimpleNamespace(param_groups=groups))

    backend = SimpleNamespace(build_train_module=build)
    frozen_router.install(backend)
    installed = backend.build_train_module
    frozen_router.install(backend)
    assert installed is backend.build_train_module
    backend.build_train_module(
        SimpleNamespace(olmo_core=CoreConfig(freeze_router=True), save=str(tmp_path)), common={"model": model}
    )
    assert (tmp_path / "frozen-router-validation.json").exists()


def test_missing_router_rejected():
    with pytest.raises(ValueError, match="routed_experts_router"):
        frozen_router.freeze(nn.Linear(4, 4))
