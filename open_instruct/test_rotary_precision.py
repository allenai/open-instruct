# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import deepspeed
import deepspeed.comm.torch
import torch
import torch.nn as nn
from transformers import Olmo2Config, Olmo2ForCausalLM, Olmo3Config, Olmo3ForCausalLM

from open_instruct.rotary_utils import parse_rotary_buffer_name, recompute_rotary_inv_freq, restore_fp32_rotary


class TestRotaryPrecision(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.head_dim = 64
        self.theta = 500000.0
        self.default_config = Olmo3Config(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            rope_theta=self.theta,
        )
        self.yarn_config = Olmo3Config(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            rope_theta=self.theta,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 8.0,
                "original_max_position_embeddings": 256,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.2,
            },
        )

    def _independent_math_ref_f64(self, head_dim: int, theta: float) -> torch.Tensor:
        arange = torch.arange(0, head_dim, 2, dtype=torch.float64)
        return 1.0 / (theta ** (arange / head_dim))

    def test_01_bf16_conversion_reproduces_original_precision_loss(self):
        """Acceptance test 1: BF16 conversion reproduces the original precision loss."""
        model = Olmo3ForCausalLM(self.default_config)
        orig_buf = model.model.rotary_emb.inv_freq.clone()

        # Simulate DeepSpeed engine module.bfloat16()
        model.bfloat16()
        bf16_buf = model.model.rotary_emb.inv_freq

        self.assertEqual(bf16_buf.dtype, torch.bfloat16)

        # Truncated mantissa loses precision; casting back does not restore it
        re_upcast = bf16_buf.float()
        rel_error = (torch.abs(re_upcast - orig_buf) / orig_buf).max().item()
        self.assertGreater(rel_error, 1e-4, f"Expected relative error from bf16 downcast > 1e-4, but got {rel_error}")

    def test_02_fixed_preparation_keeps_expected_fp32_frequency_values(self):
        """Acceptance test 2: The fixed training preparation keeps expected fp32 frequency values."""
        model = Olmo3ForCausalLM(self.default_config)
        ref = self._independent_math_ref_f64(self.head_dim, self.theta).float()

        model.bfloat16()
        restored = restore_fp32_rotary(model)
        self.assertGreater(restored, 0)

        buf = model.model.rotary_emb.inv_freq
        self.assertEqual(buf.dtype, torch.float32)

        max_rel_error = ((buf - ref).abs() / ref).max().item()
        self.assertLess(max_rel_error, 1e-6)

    def test_03_fixed_rope_values_match_independently_calculated_reference(self):
        """Acceptance test 3: Fixed RoPE values match independently calculated references."""
        model = Olmo3ForCausalLM(self.default_config)
        model.bfloat16()
        restore_fp32_rotary(model)

        ref_f64 = self._independent_math_ref_f64(self.head_dim, self.theta)
        inv_freq_f32 = model.model.rotary_emb.inv_freq

        torch.testing.assert_close(inv_freq_f32.double(), ref_f64, rtol=1e-5, atol=1e-5)

    def test_04_model_parameters_still_use_intended_training_dtype(self):
        """Acceptance test 4: Model parameters still use their intended training dtype."""
        model = Olmo3ForCausalLM(self.default_config)
        model.bfloat16()
        restore_fp32_rotary(model)

        for name, param in model.named_parameters():
            self.assertEqual(
                param.dtype, torch.bfloat16, f"Parameter {name} was unexpectedly altered to {param.dtype}"
            )

    def test_05_default_and_yarn_configurations_behave_correctly(self):
        """Acceptance test 5: Default and YaRN configurations behave correctly."""
        for cfg, name in [(self.default_config, "default"), (self.yarn_config, "yarn")]:
            model = Olmo3ForCausalLM(cfg)
            orig_inv_freq = model.model.rotary_emb.inv_freq.clone()
            orig_orig_inv_freq = model.model.rotary_emb.original_inv_freq.clone()

            model.bfloat16()
            restore_fp32_rotary(model)

            self.assertEqual(model.model.rotary_emb.inv_freq.dtype, torch.float32)
            self.assertEqual(model.model.rotary_emb.original_inv_freq.dtype, torch.float32)

            torch.testing.assert_close(
                model.model.rotary_emb.inv_freq, orig_inv_freq, rtol=0, atol=0, msg=f"{name} inv_freq mismatch"
            )
            torch.testing.assert_close(
                model.model.rotary_emb.original_inv_freq,
                orig_orig_inv_freq,
                rtol=0,
                atol=0,
                msg=f"{name} original_inv_freq mismatch",
            )

    def test_06_buffer_name_parsing_and_unrelated_buffer_protection(self):
        """Acceptance test 6: Validate buffer parsing rejects flat keys and non-rotary modules."""

        # 1. Non-rotary module must be completely rejected
        class UnrelatedLinearModule(nn.Module):
            pass

        unrelated = UnrelatedLinearModule()
        self.assertEqual(parse_rotary_buffer_name(unrelated, "inv_freq"), (False, None))

        # 2. Rotary module validation
        class MockRotaryEmbedding(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
                arange = torch.arange(0, dim, 2, dtype=torch.float32)
                self.register_buffer("inv_freq", 1.0 / (10000.0 ** (arange / dim)), persistent=False)
                # Unrelated buffer containing 'inv_freq' in name
                self.register_buffer("inv_freq_cache", torch.tensor([123.0], dtype=torch.bfloat16), persistent=False)
                # Arbitrary name that should NOT be accepted
                self.register_buffer("random_inv_freq", torch.tensor([256.0], dtype=torch.bfloat16), persistent=False)
                # Negative test: rope_theta_inv_freq against flat dictionary must NOT be recognized
                self.register_buffer("rope_theta_inv_freq", torch.tensor([500.0], dtype=torch.bfloat16), persistent=False)

        # Flat rope_parameters config
        flat_cfg = Olmo3Config(
            hidden_size=128,
            num_attention_heads=2,
            rope_parameters={"rope_theta": 500000.0, "factor": 1.0, "rope_type": "default"},
        )
        rot_mod = MockRotaryEmbedding(flat_cfg)

        # Standard names are recognized
        self.assertEqual(parse_rotary_buffer_name(rot_mod, "inv_freq", config=flat_cfg), (True, None))
        self.assertEqual(parse_rotary_buffer_name(rot_mod, "original_inv_freq", config=flat_cfg), (True, None))

        # Negative test: rope_theta is a flat key, NOT a nested layer type -> must be REJECTED
        self.assertEqual(parse_rotary_buffer_name(rot_mod, "rope_theta_inv_freq", config=flat_cfg), (False, None))

        # Arbitrary names ending in _inv_freq or containing inv_freq must be REJECTED
        self.assertEqual(parse_rotary_buffer_name(rot_mod, "random_inv_freq", config=flat_cfg), (False, None))
        self.assertEqual(parse_rotary_buffer_name(rot_mod, "inv_freq_cache", config=flat_cfg), (False, None))

        # Genuinely nested configuration: layer keys are recognized
        nested_cfg = Olmo3Config(
            hidden_size=128,
            num_attention_heads=4,
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
        )
        self.assertEqual(
            parse_rotary_buffer_name(rot_mod, "full_attention_inv_freq", config=nested_cfg),
            (True, "full_attention"),
        )
        self.assertEqual(
            parse_rotary_buffer_name(rot_mod, "sliding_attention_original_inv_freq", config=nested_cfg),
            (True, "sliding_attention"),
        )

        # 3. Execution protection: arbitrary and unrelated buffers remain untouched
        rot_mod.bfloat16()
        self.assertEqual(rot_mod.inv_freq.dtype, torch.bfloat16)
        self.assertEqual(rot_mod.inv_freq_cache.dtype, torch.bfloat16)
        self.assertEqual(rot_mod.random_inv_freq.dtype, torch.bfloat16)
        self.assertEqual(rot_mod.rope_theta_inv_freq.dtype, torch.bfloat16)

        restore_fp32_rotary(rot_mod)

        # Only validated inv_freq is restored to float32
        self.assertEqual(rot_mod.inv_freq.dtype, torch.float32)
        # Unrelated buffers remain completely untouched
        self.assertEqual(rot_mod.inv_freq_cache.dtype, torch.bfloat16)
        self.assertEqual(rot_mod.inv_freq_cache.item(), 123.0)
        self.assertEqual(rot_mod.random_inv_freq.dtype, torch.bfloat16)
        self.assertEqual(rot_mod.random_inv_freq.item(), 256.0)
        self.assertEqual(rot_mod.rope_theta_inv_freq.dtype, torch.bfloat16)
        self.assertEqual(rot_mod.rope_theta_inv_freq.item(), 500.0)

    def test_07_persistence_semantics_are_preserved(self):
        """Acceptance test 7: Original persistence semantics are preserved (not forced to False)."""

        class MockRotaryPersistence(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
                arange = torch.arange(0, dim, 2, dtype=torch.float32)
                # Register one as persistent, one as non-persistent
                self.register_buffer("inv_freq", 1.0 / (10000.0 ** (arange / dim)), persistent=True)
                self.register_buffer("original_inv_freq", 1.0 / (10000.0 ** (arange / dim)), persistent=False)

        mod = MockRotaryPersistence(self.default_config)
        self.assertIn("inv_freq", mod.state_dict())
        self.assertNotIn("original_inv_freq", mod.state_dict())

        mod.bfloat16()
        self.assertEqual(mod.inv_freq.dtype, torch.bfloat16)
        self.assertEqual(mod.original_inv_freq.dtype, torch.bfloat16)

        restored = restore_fp32_rotary(mod)
        self.assertEqual(restored, 2)

        # Both buffers must be restored to float32
        self.assertEqual(mod.inv_freq.dtype, torch.float32)
        self.assertEqual(mod.original_inv_freq.dtype, torch.float32)

        # Persistence semantics must be exactly preserved
        self.assertIn("inv_freq", mod.state_dict(), "Persistent buffer was converted to non-persistent!")
        self.assertNotIn("original_inv_freq", mod.state_dict(), "Non-persistent buffer was converted to persistent!")

    def test_08_unsupported_scaling_variants_fail_safely_without_breaking_restoration(self):
        """Acceptance test 8: Unsupported variants fail safely; full restoration continues without crashing."""

        class MockRotaryEmbedding(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
                arange = torch.arange(0, dim, 2, dtype=torch.float32)
                self.register_buffer("inv_freq", 1.0 / (10000.0 ** (arange / dim)), persistent=False)

        # 1. Direct recomputation explicitly raises NotImplementedError
        cfg_dynamic = Olmo3Config(
            hidden_size=128, num_attention_heads=2, rope_scaling={"rope_type": "dynamic", "factor": 2.0}
        )
        mod_dynamic = MockRotaryEmbedding(cfg_dynamic)
        with self.assertRaises(NotImplementedError):
            recompute_rotary_inv_freq(mod_dynamic, cfg_dynamic)

        # 2. Full model restoration logs warning and does not crash training
        mod_dynamic.bfloat16()
        restored = restore_fp32_rotary(mod_dynamic)
        self.assertEqual(restored, 0)
        # Buffer remains safely at current precision without crashing
        self.assertEqual(mod_dynamic.inv_freq.dtype, torch.bfloat16)

    def test_09_older_and_newer_rotary_layouts_are_accounted_for(self):
        """Acceptance test 9: Realistic native computation with config.rope_parameters[layer_type]."""
        # Olmo2 architecture
        olmo2_cfg = Olmo2Config(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=2048,
        )

        model2 = Olmo2ForCausalLM(olmo2_cfg)
        model2.bfloat16()
        restore_fp32_rotary(model2)
        self.assertEqual(model2.model.rotary_emb.inv_freq.dtype, torch.float32)

        # Realistic native computation method that accesses config.rope_parameters[layer_type]
        class RealisticPerLayerRotary(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                dim = 32
                arange = torch.arange(0, dim, 2, dtype=torch.float32)
                self.register_buffer("full_attention_inv_freq", 1.0 / (500000.0 ** (arange / dim)), persistent=False)
                self.register_buffer("sliding_attention_inv_freq", 1.0 / (10000.0 ** (arange / dim)), persistent=False)

            def compute_default_rope_parameters(self, config, device=None, seq_len=None, layer_type=None):
                # Must NOT be passed flattened config; must access config.rope_parameters[layer_type]
                if layer_type is None:
                    raise ValueError("RealisticPerLayerRotary requires layer_type parameter!")
                layer_conf = config.rope_parameters[layer_type]
                theta = layer_conf["rope_theta"]
                dim = 32
                arange = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32)
                return 1.0 / (theta ** (arange / dim)), 1.0

        cfg_nested = Olmo3Config(
            hidden_size=128,
            num_attention_heads=4,
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
        )
        mock_mod = RealisticPerLayerRotary(cfg_nested)
        mock_mod.bfloat16()
        self.assertEqual(mock_mod.full_attention_inv_freq.dtype, torch.bfloat16)

        restored = restore_fp32_rotary(mock_mod)
        self.assertEqual(restored, 2)
        self.assertEqual(mock_mod.full_attention_inv_freq.dtype, torch.float32)
        self.assertEqual(mock_mod.sliding_attention_inv_freq.dtype, torch.float32)
        self.assertNotEqual(mock_mod.full_attention_inv_freq[1], mock_mod.sliding_attention_inv_freq[1])

    def test_10_partial_rotary_factor_and_validation(self):
        """Acceptance test 10: partial_rotary_factor=0.5 effective dim calculation and shape validation."""
        # head_dim = 64, partial_rotary_factor = 0.5 -> effective dim = 32 -> buffer length = 16
        cfg_partial = Olmo3Config(
            hidden_size=128,
            num_attention_heads=2,
            partial_rotary_factor=0.5,
            rope_theta=10000.0,
        )

        class MockPartialRotary(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                dim = 32  # 64 * 0.5
                arange = torch.arange(0, dim, 2, dtype=torch.float32)
                self.register_buffer("inv_freq", 1.0 / (10000.0 ** (arange / dim)), persistent=False)

        mod = MockPartialRotary(cfg_partial)
        self.assertEqual(mod.inv_freq.shape, (16,))
        mod.bfloat16()

        restored = restore_fp32_rotary(mod)
        self.assertEqual(restored, 1)
        self.assertEqual(mod.inv_freq.dtype, torch.float32)
        self.assertEqual(mod.inv_freq.shape, (16,))

        # Negative check: if recomputed shape does not match existing buffer shape, do not overwrite
        class MockMismatchedShapeRotary(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                # Register shape (8,) when config implies (16,)
                self.register_buffer("inv_freq", torch.zeros(8, dtype=torch.bfloat16), persistent=False)

        bad_mod = MockMismatchedShapeRotary(cfg_partial)
        restored = restore_fp32_rotary(bad_mod)
        # Validation fails, buffer is NOT overwritten
        self.assertEqual(restored, 0)
        self.assertEqual(bad_mod.inv_freq.dtype, torch.bfloat16)

    def test_11_deepspeed_cpu_engine_integration_isolated(self):
        """Integration test with actual DeepSpeed ZeRO-3 engine initialization on CPU with scoped cleanup."""
        env_keys = ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]
        orig_env = {k: os.environ.get(k) for k in env_keys}
        orig_build_shm_op = getattr(deepspeed.comm.torch, "build_shm_op", None)
        was_dist_initialized = torch.distributed.is_initialized()

        try:
            os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29519", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
            deepspeed.comm.torch.build_shm_op = lambda: None

            model = Olmo3ForCausalLM(self.default_config)
            expected_f32 = model.model.rotary_emb.inv_freq.clone()

            ds_cfg = {
                "bf16": {"enabled": True},
                "zero_optimization": {"stage": 3},
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "optimizer": {"type": "AdamW", "params": {"lr": 1e-5, "torch_adam": True}},
            }

            engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_cfg)

            # DeepSpeed bf16 engine casts buffers to bfloat16
            self.assertEqual(engine.module.model.rotary_emb.inv_freq.dtype, torch.bfloat16)

            # Apply restoration
            restored = restore_fp32_rotary(engine)
            self.assertGreater(restored, 0)

            # Buffer restored to fp32
            self.assertEqual(engine.module.model.rotary_emb.inv_freq.dtype, torch.float32)
            torch.testing.assert_close(engine.module.model.rotary_emb.inv_freq, expected_f32, rtol=0, atol=0)

            # Parameters remain bfloat16
            for _name, param in engine.module.named_parameters():
                self.assertEqual(param.dtype, torch.bfloat16)

        finally:
            # Restore environment variables
            for k, v in orig_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

            # Restore patched function
            if orig_build_shm_op is not None:
                deepspeed.comm.torch.build_shm_op = orig_build_shm_op

            # Clean up distributed state only if this test created it
            if not was_dist_initialized and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
