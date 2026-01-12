"""Tests for DPO training module."""

import logging
import unittest

from open_instruct import dpo

logging.basicConfig(level=logging.INFO)


class TestDPOLossType(unittest.TestCase):
    """Tests for DPOLossType enum."""

    def test_loss_types_exist(self):
        self.assertEqual(dpo.DPOLossType.dpo.value, "dpo")
        self.assertEqual(dpo.DPOLossType.dpo_norm.value, "dpo_norm")
        self.assertEqual(dpo.DPOLossType.simpo.value, "simpo")
        self.assertEqual(dpo.DPOLossType.wpo.value, "wpo")


class TestDPOConfig(unittest.TestCase):
    """Tests for DPOConfig."""

    def test_default_values(self):
        config = dpo.DPOConfig()
        self.assertEqual(config.dpo_beta, 0.1)
        self.assertEqual(config.dpo_loss_type, dpo.DPOLossType.dpo)
        self.assertEqual(config.dpo_gamma_beta_ratio, 0.3)
        self.assertEqual(config.dpo_label_smoothing, 0.0)
        self.assertFalse(config.load_balancing_loss)
        self.assertEqual(config.load_balancing_weight, 1e-3)
        self.assertTrue(config.concatenated_forward)
        self.assertFalse(config.packing)

    def test_custom_values(self):
        config = dpo.DPOConfig(
            dpo_beta=0.5, dpo_loss_type=dpo.DPOLossType.simpo, dpo_gamma_beta_ratio=0.5, load_balancing_loss=True
        )
        self.assertEqual(config.dpo_beta, 0.5)
        self.assertEqual(config.dpo_loss_type, dpo.DPOLossType.simpo)
        self.assertEqual(config.dpo_gamma_beta_ratio, 0.5)
        self.assertTrue(config.load_balancing_loss)


class TestDPOExperimentConfig(unittest.TestCase):
    """Tests for DPOExperimentConfig."""

    def test_default_values(self):
        config = dpo.DPOExperimentConfig()
        self.assertEqual(config.exp_name, "dpo_experiment")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.num_epochs, 2)
        self.assertEqual(config.per_device_train_batch_size, 8)
        self.assertEqual(config.learning_rate, 2e-5)
        self.assertEqual(config.max_seq_length, 2048)

    def test_nested_dpo_config(self):
        config = dpo.DPOExperimentConfig()
        self.assertIsInstance(config.dpo_config, dpo.DPOConfig)
        self.assertEqual(config.dpo_config.dpo_beta, 0.1)


class TestDPOTrainModuleInit(unittest.TestCase):
    """Tests for DPOTrainModule initialization."""

    def test_average_log_prob_for_simpo(self):
        config = dpo.DPOConfig(dpo_loss_type=dpo.DPOLossType.simpo)
        average_log_prob = config.dpo_loss_type in (dpo.DPOLossType.simpo, dpo.DPOLossType.dpo_norm)
        self.assertTrue(average_log_prob)

    def test_average_log_prob_for_dpo_norm(self):
        config = dpo.DPOConfig(dpo_loss_type=dpo.DPOLossType.dpo_norm)
        average_log_prob = config.dpo_loss_type in (dpo.DPOLossType.simpo, dpo.DPOLossType.dpo_norm)
        self.assertTrue(average_log_prob)

    def test_average_log_prob_for_dpo(self):
        config = dpo.DPOConfig(dpo_loss_type=dpo.DPOLossType.dpo)
        average_log_prob = config.dpo_loss_type in (dpo.DPOLossType.simpo, dpo.DPOLossType.dpo_norm)
        self.assertFalse(average_log_prob)


class TestComputeReferenceLogprobsCacheHash(unittest.TestCase):
    """Tests for compute_reference_logprobs_cache_hash function."""

    def test_deterministic_hash(self):
        hash1 = dpo.compute_reference_logprobs_cache_hash(
            model_name_or_path="allenai/OLMo-2-1124-7B",
            model_revision="main",
            dpo_loss_type=dpo.DPOLossType.dpo,
            concatenated_forward=True,
            packing=False,
            use_lora=False,
            dataset_config_hash="abc123",
        )
        hash2 = dpo.compute_reference_logprobs_cache_hash(
            model_name_or_path="allenai/OLMo-2-1124-7B",
            model_revision="main",
            dpo_loss_type=dpo.DPOLossType.dpo,
            concatenated_forward=True,
            packing=False,
            use_lora=False,
            dataset_config_hash="abc123",
        )
        self.assertEqual(hash1, hash2)

    def test_different_inputs_different_hash(self):
        hash1 = dpo.compute_reference_logprobs_cache_hash(
            model_name_or_path="allenai/OLMo-2-1124-7B",
            model_revision="main",
            dpo_loss_type=dpo.DPOLossType.dpo,
            concatenated_forward=True,
            packing=False,
            use_lora=False,
            dataset_config_hash="abc123",
        )
        hash2 = dpo.compute_reference_logprobs_cache_hash(
            model_name_or_path="allenai/OLMo-2-1124-7B",
            model_revision="main",
            dpo_loss_type=dpo.DPOLossType.simpo,
            concatenated_forward=True,
            packing=False,
            use_lora=False,
            dataset_config_hash="abc123",
        )
        self.assertNotEqual(hash1, hash2)

    def test_hash_length(self):
        hash_value = dpo.compute_reference_logprobs_cache_hash(
            model_name_or_path="allenai/OLMo-2-1124-7B",
            model_revision=None,
            dpo_loss_type=dpo.DPOLossType.dpo,
            concatenated_forward=True,
            packing=False,
            use_lora=False,
            dataset_config_hash="test",
        )
        self.assertEqual(len(hash_value), 16)


class TestConfigToJsonSerializable(unittest.TestCase):
    """Tests for config_to_json_serializable function."""

    def test_enum_conversion(self):
        result = dpo.config_to_json_serializable({"loss_type": dpo.DPOLossType.dpo})
        self.assertEqual(result, {"loss_type": "dpo"})

    def test_nested_dict(self):
        result = dpo.config_to_json_serializable({"outer": {"inner": dpo.DPOLossType.simpo}})
        self.assertEqual(result, {"outer": {"inner": "simpo"}})

    def test_list_conversion(self):
        result = dpo.config_to_json_serializable([dpo.DPOLossType.dpo, dpo.DPOLossType.wpo])
        self.assertEqual(result, ["dpo", "wpo"])

    def test_primitive_passthrough(self):
        result = dpo.config_to_json_serializable({"str": "hello", "int": 42, "float": 3.14})
        self.assertEqual(result, {"str": "hello", "int": 42, "float": 3.14})


if __name__ == "__main__":
    unittest.main()
