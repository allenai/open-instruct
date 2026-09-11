"""Dense Olmo 3 uses standard Core modules with independently checked HF math."""

from types import SimpleNamespace

import pytest
import torch
from olmo_core.nn.hf import convert
from transformers import AutoModelForCausalLM, Olmo3Config

from open_instruct.miles import models
from open_instruct.miles.config import CoreConfig


@pytest.mark.parametrize("sliding", [False, True])
def test_olmo3_standard_logits_and_streamed_weights(sliding, monkeypatch):
    torch.manual_seed(13)
    hf = Olmo3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=4,
        layer_types=["sliding_attention" if sliding else "full_attention", "full_attention"],
        tie_word_embeddings=False,
    )
    hf._attn_implementation = "eager"
    real_backend = models._backend
    used = []

    def standard_only(kind):
        assert kind == "standard", "Dense Olmo 3 must not enter the MoE backend"
        used.append(kind)
        return real_backend(kind)

    monkeypatch.setattr(models, "_backend", standard_only)
    reference = AutoModelForCausalLM.from_config(hf).to(torch.bfloat16).eval()
    config = models.model_config_from_hf(hf, CoreConfig(attention_backend="torch", activation_checkpointing=False))
    native = config.build(init_device="cpu").eval()
    native.load_state_dict(
        convert.convert_state_from_hf(hf, reference.state_dict(), model_type=hf.model_type), strict=True
    )
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    with torch.no_grad():
        actual, expected = native(ids), reference(ids).logits
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.015, rtol=0.025)
    exported = dict(models.iter_export_state(SimpleNamespace(model=native), hf))
    assert exported.keys() == reference.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(exported[name], value, rtol=0, atol=0)
    assert used == ["standard", "standard"]
