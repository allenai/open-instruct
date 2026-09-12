"""Tests for Olmo Hybrid (GDN + attention) config and HF state conversion."""

import types
import unittest
from unittest import mock

import torch

from open_instruct import olmo_core_hybrid, olmo_core_utils

# A 4-layer stand-in for Olmo-Hybrid-7B: same block pattern and head geometry,
# small enough to build in a test.
SMALL_HYBRID = dict(
    d_model=128,
    vocab_size=256,
    n_layers=4,
    n_heads=4,
    n_kv_heads=4,
    head_dim=32,
    intermediate_size=256,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=32,
)

OLMO_CORE_TO_HF = {**{v: k for k, v in olmo_core_hybrid.HF_TO_OLMO_CORE_SHARED_KEYS.items()}}


def _hf_key_for(olmo_key: str, layer_types: list[str]) -> str:
    """Invert the maps in olmo_core_hybrid to get the HF name for an olmo-core key."""
    if olmo_key in OLMO_CORE_TO_HF:
        return OLMO_CORE_TO_HF[olmo_key]
    _, layer_str, suffix = olmo_key.split(".", 2)
    key_map = (
        olmo_core_hybrid.HF_TO_OLMO_CORE_GDN_KEYS
        if layer_types[int(layer_str)] == "linear_attention"
        else olmo_core_hybrid.HF_TO_OLMO_CORE_ATTN_KEYS
    )
    inverted = {v: k for k, v in key_map.items()}
    return f"model.layers.{layer_str}.{inverted[suffix]}"


class TestOlmoHybridConfig(unittest.TestCase):
    def test_7b_config_shape(self) -> None:
        """The preset must match allenai/Olmo-Hybrid-7B: 32 layers, 3 GDN per attention block."""
        config = olmo_core_hybrid.olmo3_hybrid_7B(vocab_size=100352)
        self.assertEqual(config.n_layers, 32)
        self.assertEqual(config.d_model, 3840)
        self.assertEqual(config.block_pattern, ["gdn", "gdn", "gdn", "attn"])
        # 7.43B total / 7.05B non-embedding, per the released checkpoint.
        self.assertEqual(config.num_params, 7_430_870_688)
        self.assertEqual(config.num_non_embedding_params, 7_045_519_008)

    def test_attention_blocks_have_no_rope(self) -> None:
        """Olmo Hybrid is NoPE; fabricating a RoPE would not match the trained weights."""
        config = olmo_core_hybrid.olmo3_hybrid_7B(vocab_size=100352)
        self.assertIsNone(config.block["attn"].sequence_mixer.rope)

    def test_resolvable_by_config_name(self) -> None:
        """--config_name olmo3_hybrid_7B must resolve even though olmo-core has no such preset."""
        config = olmo_core_utils.get_transformer_config("olmo3_hybrid_7B", 100352, "flash_2")
        self.assertEqual(config.n_layers, 32)

    def test_model_name_maps_to_config(self) -> None:
        config = olmo_core_utils.get_transformer_config("allenai/Olmo-Hybrid-7B", 100352, "flash_2")
        self.assertEqual(config.d_model, 3840)


class TestHybridStateConversion(unittest.TestCase):
    def _build_small(self):
        config = olmo_core_hybrid.olmo_hybrid_like(**SMALL_HYBRID)
        model = config.build(init_device="meta")
        layer_types = ["linear_attention"] * 4
        for i in (3,):
            layer_types[i] = "full_attention"
        return model, layer_types

    def test_conversion_covers_every_model_parameter(self) -> None:
        """Every olmo-core parameter must come back from the HF state, and nothing else."""
        model, layer_types = self._build_small()
        olmo_keys = set(model.state_dict())
        hf_state = {_hf_key_for(k, layer_types): torch.empty(0) for k in olmo_keys}
        converted = olmo_core_hybrid.convert_hybrid_state_from_hf(hf_state, layer_types)
        self.assertEqual(set(converted), olmo_keys)

    def test_gdn_and_attention_layers_use_different_maps(self) -> None:
        """The same HF norm name means different things in the two block types."""
        layer_types = ["linear_attention", "full_attention"]
        hf_state = {
            "model.layers.0.post_attention_layernorm.weight": torch.tensor([0.0]),
            "model.layers.1.post_attention_layernorm.weight": torch.tensor([1.0]),
        }
        converted = olmo_core_hybrid.convert_hybrid_state_from_hf(hf_state, layer_types)
        self.assertEqual(converted["blocks.0.feed_forward_norm.weight"].item(), 0.0)
        self.assertEqual(converted["blocks.1.attention_norm.weight"].item(), 1.0)

    def test_export_dispatches_on_model_type(self) -> None:
        """Export must not reach olmo-core's generic converter, which raises on every
        GDN parameter -- aborting DPO and GRPO at their pre-training export check."""
        hf_config = types.SimpleNamespace(
            model_type=olmo_core_hybrid.OLMO_HYBRID_MODEL_TYPE, layer_types=["linear_attention", "full_attention"]
        )
        state = {"blocks.0.attention.A_log": torch.empty(0), "blocks.1.attention.w_q.weight": torch.empty(0)}
        converted = olmo_core_utils.convert_olmo_core_state_to_hf(hf_config, state)
        self.assertEqual(
            set(converted), {"model.layers.0.linear_attn.A_log", "model.layers.1.self_attn.q_proj.weight"}
        )

    def test_unmapped_key_raises(self) -> None:
        """A silently dropped weight would leave randomly initialised parameters behind."""
        with self.assertRaises(KeyError):
            olmo_core_hybrid.convert_hybrid_state_from_hf(
                {"model.layers.0.mystery.weight": torch.empty(0)}, ["linear_attention"]
            )

    def test_hybrid_weight_loading_uses_model_type_dispatch(self) -> None:
        """The dispatcher must route Olmo Hybrid around olmo-core's generic converter."""
        state = {"blocks.0.attention.A_log": torch.zeros(1)}
        hf_config = types.SimpleNamespace(model_type=olmo_core_hybrid.OLMO_HYBRID_MODEL_TYPE)

        with (
            mock.patch.object(olmo_core_utils.transformers.AutoConfig, "from_pretrained", return_value=hf_config),
            mock.patch.object(olmo_core_hybrid, "load_hf_hybrid_model") as load_hybrid,
            mock.patch.object(olmo_core_utils, "load_hf_model") as load_generic,
        ):
            olmo_core_utils.load_hf_weights_into_olmo_core(state, "allenai/Olmo-Hybrid-7B", work_dir="out")

        load_hybrid.assert_called_once_with("allenai/Olmo-Hybrid-7B", state)
        load_generic.assert_not_called()

    def test_model_loader_applies_replaced_state_dict_values(self) -> None:
        """Hybrid conversion replaces state-dict entries, so the model must load them back."""
        model = torch.nn.Linear(2, 1, bias=False)
        replacement = torch.full_like(model.weight, 3.0)

        def replace_state(model_state_dict, model_name_or_path, work_dir=None):
            self.assertEqual(model_name_or_path, "allenai/Olmo-Hybrid-7B")
            self.assertEqual(work_dir, "out")
            model_state_dict["weight"] = replacement

        with mock.patch.object(olmo_core_utils, "load_hf_weights_into_olmo_core", side_effect=replace_state):
            olmo_core_utils.load_hf_weights_into_model(model, "allenai/Olmo-Hybrid-7B", work_dir="out")

        torch.testing.assert_close(model.weight, replacement)


if __name__ == "__main__":
    unittest.main()
