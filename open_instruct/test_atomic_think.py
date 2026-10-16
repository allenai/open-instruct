"""Mechanism checks for H077's interventions and observational probes."""

import json
from types import SimpleNamespace

import pytest
import torch
from olmo_core.distributed.utils import unhide_from_torch
from olmo_core.nn.moe.v2.router import MoERouterV2
from torch import distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from open_instruct import atomic_think, dataset_transformation, olmo_core_utils


@pytest.mark.parametrize("mode", atomic_think.MODES)
def test_intervention_preserves_everything_except_intended_masters(tmp_path, mode):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        original = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 13 + 0.00013
        param = torch.nn.Parameter(original.bfloat16())
        master = distribute_tensor(original.flatten(), mesh, [Shard(0)])
        moment = distribute_tensor(torch.ones(40), mesh, [Shard(0)])
        optimizer = SimpleNamespace(
            states={"embedding.main": master, "embedding.moment": moment},
            param_groups=[{"named_params": {"embedding": param}}],
        )
        model = SimpleNamespace(embeddings=SimpleNamespace(weight=param), tie_word_embeddings=True)
        rows = [dataset_transformation.PromotedToken(content="<think>", token_id=9, source_ids=(0, 1, 5))]
        tokenizer = SimpleNamespace(promoted_reserved_slot_tokens=rows if mode.startswith("atomic-") else [])
        atomic_think.initialize_arm(
            SimpleNamespace(model=model, optim=optimizer),
            tokenizer,
            mode,
            tmp_path,
            olmo_core_utils.sync_promoted_token_optimizer_rows,
        )
        actual = master.full_tensor().reshape_as(original)
        if mode in ("control", "atomic-safe"):
            torch.testing.assert_close(actual[:9], original[:9], rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual[:9], original[:9].bfloat16().float(), rtol=0, atol=0)
            assert not torch.equal(actual[:9], original[:9])
        torch.testing.assert_close(moment.full_tensor(), torch.ones(40), rtol=0, atol=0)
        assert json.loads((tmp_path / "initialization-rank0.json").read_text())["mode"] == mode
    finally:
        dist.destroy_process_group()


def test_probe_restores_real_router_state_and_rng_even_on_exception():
    router = MoERouterV2(d_model=4, num_experts=8, top_k=2, bias_gamma=0.001, jitter_eps=0.1)
    model = torch.nn.Module()
    model.add_module("routed_experts_router", router)
    assert isinstance(router.score_bias, torch.Tensor)
    router.score_bias.copy_(torch.arange(8) / 100)
    router.batch_size_per_expert.add_(2)
    before = router.batch_size_per_expert.clone()
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="injected"), atomic_think.preserved_probe_state(model):
        assert not router.training
        router.batch_size_per_expert.add_(17)
        router.score_bias.add_(1)
        router._z_loss = None
        torch.rand(3)
        raise RuntimeError("injected")
    assert router.training and model.training
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(router.batch_size_per_expert, before)
    assert torch.equal(router.score_bias, torch.arange(8) / 100)
    assert router._z_loss is None
    assert torch.equal(unhide_from_torch(router._batch_size_per_expert), before)


def test_route_capture_is_observational_and_reports_biased_margin():
    router = MoERouterV2(d_model=4, num_experts=8, top_k=2, bias_gamma=0.001)
    model = torch.nn.Module()
    model.add_module("routed_experts_router", router)
    assert isinstance(router.score_bias, torch.Tensor)
    router.score_bias.copy_(torch.arange(8) / 8)
    x = torch.randn(1, 5, 4)
    with torch.no_grad():
        expected = router(x, False)
        with atomic_think.capture_routes(model) as (routes, stats):
            actual = router(x, False)
        for left, right in zip(expected, actual):
            if left is not None:
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        scores = router.get_expert_logits(x.float()).float().softmax(-1) + router.score_bias
        sorted_scores = scores.sort(descending=True).values
        torch.testing.assert_close(
            stats["routed_experts_router"]["selection_margin"], sorted_scores[..., 1] - sorted_scores[..., 2]
        )
    assert torch.equal(routes["routed_experts_router"], expected[1].int())
    assert "get_top_k" not in router.__dict__
    assert not router._forward_hooks


def test_resume_rejects_changed_arm_or_schedule():
    callback = atomic_think.AtomicAuditCallback(mode="atomic-safe", horizon=11768)
    for state in ({"mode": "control", "horizon": 11768}, {"mode": "atomic-safe", "horizon": 512}):
        with pytest.raises(ValueError, match="Resume changed"):
            callback.load_state_dict(state)


@pytest.mark.parametrize("tensor_lr", [False, True])
def test_state_audit_serializes_learning_rate_without_mutating_training(tmp_path, tensor_lr):
    model = torch.nn.Linear(4, 2)
    lr = torch.tensor(5e-5) if tensor_lr else 5e-5
    master = model.weight.detach().clone()
    optimizer = SimpleNamespace(states={"weight.main": master}, param_groups=[{"lr": lr}])
    callback = atomic_think.AtomicAuditCallback(output_dir=str(tmp_path), state_steps=[3])
    callback.trainer = SimpleNamespace(global_step=3, train_module=SimpleNamespace(model=model, optim=optimizer))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    callback.post_step()
    report = json.loads((tmp_path / "state-step3-rank0.json").read_text())
    assert report["lr"] == [float(lr)]
    assert report["step"] == 3
    assert report["optimizer"]["weight.main"] == atomic_think.tensor_hash(master)
    assert optimizer.param_groups[0]["lr"] is lr
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
