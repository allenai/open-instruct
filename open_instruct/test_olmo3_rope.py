"""Olmo 3 applies YaRN only on full-attention layers; transformers before 5.13 also applied it to sliding-window ones."""

import unittest

import torch
from transformers import Olmo3Config, Olmo3ForCausalLM

YARN = {
    "rope_type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 64,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "attention_factor": 1.2,
}


def sliding_window_logits(**rope) -> torch.Tensor:
    torch.manual_seed(0)
    config = Olmo3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        sliding_window=256,
        layer_types=["sliding_attention"] * 2,
        rope_theta=500000,
        pad_token_id=0,
        eos_token_id=0,
        **rope,
    )
    model = Olmo3ForCausalLM(config).eval()
    with torch.no_grad():
        return model(torch.arange(200)[None] % 128).logits


class TestOlmo3SlidingWindowRope(unittest.TestCase):
    def test_yarn_does_not_reach_sliding_window_layers(self):
        torch.testing.assert_close(sliding_window_logits(rope_scaling=YARN), sliding_window_logits(), rtol=0, atol=0)
