"""Backend-only replay and early rejection of unsupported dense trainer options."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from transformers import AutoModelForCausalLM, Olmo3Config

from open_instruct.miles import actor, checkpoint, models, moe_models
from open_instruct.miles.config import CoreConfig


@pytest.mark.parametrize("replay,ep,error", [(True, 1, "replay"), (False, 2, "Expert parallelism")])
def test_dense_options_rejected_before_loading_weights_or_using_cuda(tmp_path, replay, ep, error):
    hf = Olmo3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention", "full_attention"],
    )
    hf.save_pretrained(tmp_path)
    args = SimpleNamespace(
        hf_checkpoint=str(tmp_path),
        seed=17,
        use_rollout_routing_replay=replay,
        olmo_core=CoreConfig(attention_backend="torch", expert_parallel_size=ep),
    )
    with (
        mock.patch.object(
            AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("weights loaded")
        ) as load,
        mock.patch.object(torch.cuda, "current_device", side_effect=AssertionError("CUDA touched")) as device,
        pytest.raises(ValueError, match=error),
    ):
        models.build_train_module(args)
    load.assert_not_called()
    device.assert_not_called()


def test_actor_disabled_replay_does_not_load_any_backend(monkeypatch):
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(use_rollout_routing_replay=False)
    with (
        mock.patch.object(models, "_backend", side_effect=AssertionError("backend loaded")),
        worker._replay_context(SimpleNamespace(), {}),
    ):
        pass


def test_standard_context_rejects_replay_even_for_external_module():
    with pytest.raises(ValueError, match="standard dense trainer"):
        models.replay_context(SimpleNamespace(_miles_model_backend="standard"), {}, enabled=True)


def test_actor_moe_replay_preserves_router_gradients_and_cleans_up():
    torch.manual_seed(19)
    router = MoERouterConfigV2(d_model=8, num_experts=4, top_k=2).build()
    torch.nn.init.normal_(router.weight, std=0.1)
    block = torch.nn.Module()
    block.add_module("routed_experts_router", router)
    model = torch.nn.Module()
    model.add_module("blocks", torch.nn.ModuleList([block]))
    module = SimpleNamespace(model=model, _miles_model_backend="moe")
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(use_rollout_routing_replay=True)
    routes = torch.tensor([[[0, 2]], [[1, 3]], [[2, 3]]])
    batch = {"tokens": torch.zeros(1, 4, dtype=torch.long), "rollout_routed_experts": [routes]}
    x = torch.randn(1, 4, 8, requires_grad=True)
    with worker._replay_context(module, batch):
        weights, actual, _, _ = router(x, scores_only=False)
        torch.testing.assert_close(actual[0, :3], routes[:, 0])
        (weights * torch.tensor([1.0, 2.0])).sum().backward()
    assert router.weight.grad is not None and torch.isfinite(router.weight.grad).all()
    assert router.weight.grad.abs().sum() > 0
    assert router.replay_expert_indices is None
    with pytest.raises(RuntimeError, match="injected"), worker._replay_context(module, batch):
        raise RuntimeError("injected")
    assert router.replay_expert_indices is None


def test_checkpoint_row_mode_rollback_and_legacy_manifest():
    legacy = {"block": {"routed_experts": {"hidden_size": 128}, "router": {"top_k": 2}}}
    static = {
        "block": {"routed_experts": {"hidden_size": 128, "row_specialization": "static"}, "router": {"top_k": 2}}
    }
    dynamic = {
        "block": {"routed_experts": {"hidden_size": 128, "row_specialization": "dynamic"}, "router": {"top_k": 2}}
    }
    assert checkpoint.comparable_model_config(legacy) == checkpoint.comparable_model_config(static)
    assert checkpoint.comparable_model_config(static) == checkpoint.comparable_model_config(dynamic)
    dynamic["block"]["router"]["top_k"] = 3
    assert checkpoint.comparable_model_config(static) != checkpoint.comparable_model_config(dynamic)
    dynamic["block"]["routed_experts"]["row_specialization"] = "invalid"
    with pytest.raises(ValueError, match="row_specialization"):
        checkpoint.comparable_model_config(dynamic)


def test_row_mode_initialization_covers_named_blocks_and_overrides():
    blocks = [SimpleNamespace(routed_experts=SimpleNamespace(row_specialization="static")) for _ in range(3)]
    config = SimpleNamespace(block={"kda": blocks[0], "attention": blocks[1]}, block_overrides={7: blocks[2]})
    moe_models.prepare_model_config(config, SimpleNamespace(layer_types=[]), CoreConfig(row_specialization="dynamic"))
    assert all(block.routed_experts.row_specialization == "dynamic" for block in blocks)
