"""Frozen image conditioning, publication scope and immutable checkpoint identity."""

from types import SimpleNamespace

import pytest
import torch
from miles.backends.core_utils import models, vision_models
from olmo_sglang.vision import model as vision_model
from safetensors import torch as safetensors_torch

from open_instruct.miles.configuration.config import CoreConfig


@pytest.fixture
def module():
    config = {
        "model_type": "olmo3moe_vision",
        "text_config": {"model_type": "olmo3moe", "vocab_size": 16, "hidden_size": 8},
        "image_patch_token_id": 6,
        "vit_layers": [1, 0],
        "vision_config": {
            "name": "siglip",
            "image_num_layers": 2,
            "image_emb_dim": 8,
            "image_num_heads": 2,
            "image_num_key_value_heads": 2,
            "image_head_dim": 4,
            "image_mlp_dim": 16,
            "image_patch_size": 2,
            "image_default_input_size": [4, 4],
            "image_num_pos": 4,
            "patch_embedding_bias": True,
            "use_cls_token": False,
            "use_pre_ln": False,
        },
        "connector_config": {
            "image_emb_dim": 8,
            "image_num_heads": 2,
            "image_num_key_value_heads": 2,
            "image_head_dim": 4,
            "output_dim": 8,
            "num_input_layers": 2,
            "pooling_type": "attention_meanq",
            "pooling_attention_mask": True,
            "projector_type": "mlp",
            "mlp_hidden_size": 16,
        },
    }
    stack = vision_model.OlmoVisionStack(config).requires_grad_(False).eval()
    # Mimic a loaded checkpoint; the connector's initial residual projection is zero.
    for parameter in stack.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    embedding = torch.nn.Embedding(16, 8)
    norm = torch.nn.LayerNorm(8)
    language = torch.nn.Sequential(embedding, norm)
    language.forward_embed = language.forward
    return SimpleNamespace(
        model=language,
        _miles_vision=stack,
        _miles_image_patch_token_id=6,
        _miles_model_backend="vision",
        _miles_vision_identity=vision_models.tensor_identity(stack.state_dict()),
    )


def batch():
    return {
        "tokens": torch.tensor([[0, 6, 6, 1]]),
        "response_lengths": [1],
        "multimodal_train_inputs": [
            {
                "images": torch.randn(1, 1, 4, 12),
                "pooled_patches_idx": torch.tensor([[[0, 1], [2, 3]]]),
                "image_positions": torch.tensor([1, 2]),
            }
        ],
    }


def test_actual_vision_changes_conditioning_and_only_language_receives_gradients(module):
    data = batch()
    identity = vision_models.tensor_identity(module._miles_vision.state_dict())
    conditioned = models.conditioning_inputs(module, data)["input_embeddings"]
    original = module.model.forward_embed(data["tokens"])
    torch.testing.assert_close(conditioned[:, [0, 3]], original[:, [0, 3]])
    assert not torch.allclose(conditioned[:, 1:3], original[:, 1:3])
    conditioned.square().sum().backward()
    assert module.model[0].weight.grad[6].abs().sum() > 0
    assert all(p.grad is None for p in module._miles_vision.parameters())
    optimizer = torch.optim.SGD(module.model.parameters(), lr=0.01)
    optimizer.step()
    next_conditioned = models.conditioning_inputs(module, data)["input_embeddings"]
    assert not torch.allclose(next_conditioned, conditioned)
    assert vision_models.tensor_identity(module._miles_vision.state_dict()) == identity
    data["multimodal_train_inputs"][0]["images"] *= -1
    counterfactual = models.conditioning_inputs(module, data)["input_embeddings"]
    assert not torch.allclose(next_conditioned[:, 1:3], counterfactual[:, 1:3])


@pytest.mark.parametrize("damage", ["missing", "positions", "train_mode", "unfreeze"])
def test_missing_or_misaligned_conditioning_is_rejected(module, damage):
    data = batch()
    if damage == "missing":
        data.pop("multimodal_train_inputs")
    elif damage == "positions":
        data["multimodal_train_inputs"][0]["image_positions"] = torch.tensor([0, 1])
    elif damage == "train_mode":
        module._miles_vision.train()
    else:
        module._miles_vision.connector.requires_grad_(True)
    with pytest.raises(ValueError):
        models.conditioning_inputs(module, data)


def test_export_and_publication_have_different_visual_scope(module, monkeypatch):
    monkeypatch.setattr(
        vision_models.moe_models, "iter_export_state", lambda *a, **kw: iter([("lm.weight", torch.ones(2))])
    )
    hf = SimpleNamespace(text_config=object(), model_type="olmo3moe_vision")
    publication = dict(models.iter_export_state(module, hf, for_publication=True))
    export = dict(models.iter_export_state(module, hf))
    assert set(publication) == {"language_model.lm.weight"}
    assert set(export) == set(publication) | set(module._miles_vision.state_dict())


def test_frozen_checkpoint_roundtrip_rejects_changed_weights(module, tmp_path, monkeypatch):
    monkeypatch.setattr(vision_models.moe_models, "save_native", lambda *a: None)
    monkeypatch.setattr(vision_models.moe_models, "load_native", lambda *a, **kw: None)
    monkeypatch.setattr(vision_models.dist, "get_rank", lambda: 0)
    vision_models.save_native(module, tmp_path)
    vision_models.load_native(module, tmp_path)
    path = tmp_path / "frozen-vision.safetensors"
    state = safetensors_torch.load_file(str(path))
    next(iter(state.values())).add_(1)
    safetensors_torch.save_file(state, str(path))
    with pytest.raises(ValueError, match="tensors changed"):
        vision_models.load_native(module, tmp_path)
    next(module._miles_vision.parameters()).add_(1)
    with pytest.raises(ValueError, match="changed during training"):
        vision_models.save_native(module, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"freeze_connector": False},
        {"freeze_image_encoder": False},
        {"sequence_packing": True},
        {"publication_mode": "refresh"},
    ],
)
def test_unqualified_config_fails_before_allocation(change):
    args = SimpleNamespace(
        olmo_core=CoreConfig(**change), use_rollout_routing_replay=False, use_rollout_indexer_replay=False
    )
    with pytest.raises(ValueError):
        vision_models.validate_training_options(args)
