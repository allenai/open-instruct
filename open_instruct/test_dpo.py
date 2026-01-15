"""Tests for DPO training module."""

import unittest

from parameterized import parameterized

from open_instruct import dpo
from open_instruct.dpo_utils import DPOLossType, config_to_json_serializable


class TestComputeReferenceLogprobsCacheHash(unittest.TestCase):
    """Tests for compute_reference_logprobs_cache_hash function."""

    def test_deterministic_hash(self):
        hash1 = dpo.compute_reference_logprobs_cache_hash(
            model_name_or_path="allenai/OLMo-2-1124-7B",
            model_revision="main",
            dpo_loss_type="dpo",
            concatenated_forward=True,
            packing=False,
            use_lora=False,
            use_qlora=False,
            max_train_samples=None,
            dataset_config_hash="abc123",
        )
        hash2 = dpo.compute_reference_logprobs_cache_hash(
            model_name_or_path="allenai/OLMo-2-1124-7B",
            model_revision="main",
            dpo_loss_type="dpo",
            concatenated_forward=True,
            packing=False,
            use_lora=False,
            use_qlora=False,
            max_train_samples=None,
            dataset_config_hash="abc123",
        )
        self.assertEqual(hash1, hash2)

    def test_different_inputs_different_hash(self):
        hash1 = dpo.compute_reference_logprobs_cache_hash(
            model_name_or_path="allenai/OLMo-2-1124-7B",
            model_revision="main",
            dpo_loss_type="dpo",
            concatenated_forward=True,
            packing=False,
            use_lora=False,
            use_qlora=False,
            max_train_samples=None,
            dataset_config_hash="abc123",
        )
        hash2 = dpo.compute_reference_logprobs_cache_hash(
            model_name_or_path="allenai/OLMo-2-1124-7B",
            model_revision="main",
            dpo_loss_type="simpo",
            concatenated_forward=True,
            packing=False,
            use_lora=False,
            use_qlora=False,
            max_train_samples=None,
            dataset_config_hash="abc123",
        )
        self.assertNotEqual(hash1, hash2)


class TestConfigToJsonSerializable(unittest.TestCase):
    """Tests for config_to_json_serializable function."""

    @parameterized.expand(
        [
            ("enum_conversion", {"loss_type": DPOLossType.dpo}, {"loss_type": "dpo"}),
            ("nested_dict", {"outer": {"inner": DPOLossType.simpo}}, {"outer": {"inner": "simpo"}}),
            ("list_conversion", [DPOLossType.dpo, DPOLossType.wpo], ["dpo", "wpo"]),
            (
                "primitive_passthrough",
                {"str": "hello", "int": 42, "float": 3.14},
                {"str": "hello", "int": 42, "float": 3.14},
            ),
        ]
    )
    def test_config_to_json_serializable(self, name, input_val, expected):
        self.assertEqual(config_to_json_serializable(input_val), expected)


if __name__ == "__main__":
    unittest.main()
