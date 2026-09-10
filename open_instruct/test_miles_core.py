"""Numerical checks against real Core modules and HF reference models."""

import contextlib
from importlib import import_module

import pytest
import torch
from olmo_core.nn.hf import convert
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, LlamaConfig, Olmo2Config, Qwen2Config, Qwen3Config

from open_instruct.miles.config import CoreConfig

# The normal SFT/DPO environment intentionally keeps its existing Core pin.
# These numerical tests run when the separate patched Core runtime is supplied.
replay_routes = pytest.importorskip("olmo_core.nn.moe.v2.replay").replay_routes
train_batch_with_loss = pytest.importorskip("olmo_core.train.train_module.transformer.objective").train_batch_with_loss
core_models = import_module("open_instruct.miles.models")
model_config_from_hf = core_models.model_config_from_hf
iter_export_state = core_models.iter_export_state
prepare_checkpoint_path = import_module("open_instruct.miles.checkpoint").prepare_checkpoint_path


class ObjectiveModule:
    def __init__(self, model):
        self.model = model
        self.contexts = []

    @contextlib.contextmanager
    def _train_microbatch_context(self, index, count):
        self.contexts.append((index, count))
        yield


def test_custom_objective_accumulation_matches_full_batch():
    torch.manual_seed(17)
    model = nn.Linear(4, 3)
    reference = nn.Linear(4, 3)
    reference.load_state_dict(model.state_dict())
    x = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 0, 2])
    module = ObjectiveModule(model)

    def objective(module, batch):
        loss = F.cross_entropy(module.model(batch["x"]), batch["y"], reduction="sum") / 5
        return loss, {"loss": loss}

    metrics = train_batch_with_loss(module, [{"x": x[:2], "y": labels[:2]}, {"x": x[2:], "y": labels[2:]}], objective)
    F.cross_entropy(reference(x), labels).backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)
    assert module.contexts == [(0, 2), (1, 2)]
    assert all(not metric["loss"].requires_grad for metric in metrics)


@pytest.mark.parametrize("recompute", [False, True])
def test_router_replay_keeps_experts_and_router_gradients(recompute):
    router = MoERouterConfigV2(d_model=8, num_experts=4, top_k=2).build()
    router.reset_parameters() if hasattr(router, "reset_parameters") else nn.init.normal_(router.weight, std=0.1)
    model = nn.Module()
    model.add_module("routed_experts_router", router)
    indices = torch.tensor([[[0, 2], [1, 3], [2, 3]]])
    x = torch.randn(1, 3, 8, requires_grad=True)
    seen = []

    def forward(x):
        weights, actual, _, _ = router(x, scores_only=False)
        seen.append(actual.detach().clone())
        return weights

    with replay_routes(model, {"routed_experts_router": indices}):
        weights = checkpoint(forward, x, use_reentrant=True) if recompute else forward(x)
        (weights * torch.tensor([1.0, 2.0])).sum().backward()
    assert seen and all(torch.equal(actual, indices) for actual in seen)
    assert torch.isfinite(router.weight.grad).all() and router.weight.grad.abs().sum() > 0
    assert router.replay_expert_indices is None


@pytest.mark.parametrize("hf_cls", [LlamaConfig, Qwen2Config, Qwen3Config, Olmo2Config])
def test_hf_core_logits_and_roundtrip(hf_cls):
    torch.manual_seed(13)
    hf = hf_cls(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    hf._attn_implementation = "eager"
    reference = AutoModelForCausalLM.from_config(hf).to(torch.bfloat16).eval()
    config = model_config_from_hf(hf, CoreConfig(attention_backend="torch", activation_checkpointing=False))
    native = config.build(init_device="cpu").eval()
    state = convert.convert_state_from_hf(hf, reference.state_dict(), model_type=hf.model_type)
    native.load_state_dict(state, strict=True)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        actual = native(ids)
        expected = reference(ids).logits
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.015, rtol=0.025)
    exported = convert.convert_state_to_hf(hf, native.state_dict())
    streamed = dict(iter_export_state(ObjectiveModule(native), hf))
    assert set(streamed) == set(exported)
    for key in exported:
        torch.testing.assert_close(streamed[key], exported[key], rtol=0, atol=0)
    assert set(exported) == set(reference.state_dict())
    for key, value in exported.items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)


def test_interrupted_checkpoint_can_be_retried_without_replacing_committed_data(tmp_path):
    path = tmp_path / "rollout_0000001"
    path.mkdir()
    (path / "partial-model").write_bytes(b"partial")
    prepare_checkpoint_path(path)
    preserved = list(tmp_path.glob("rollout_0000001.incomplete-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "partial-model").read_bytes() == b"partial"
    assert list(path.iterdir()) == []
    (path / "complete.json").write_text("{}")
    with pytest.raises(FileExistsError, match="committed checkpoint"):
        prepare_checkpoint_path(path)
    assert (path / "complete.json").read_text() == "{}"
