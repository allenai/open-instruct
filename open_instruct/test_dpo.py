"""Tests for DPO training module."""

import logging
import unittest

import torch

from open_instruct import dpo, model_utils

logging.basicConfig(level=logging.INFO)


class TestDPOLossType(unittest.TestCase):
    """Tests for DPOLossType enum."""

    def test_loss_types_exist(self):
        self.assertEqual(dpo.DPOLossType.dpo.value, "dpo")
        self.assertEqual(dpo.DPOLossType.dpo_norm.value, "dpo_norm")
        self.assertEqual(dpo.DPOLossType.simpo.value, "simpo")
        self.assertEqual(dpo.DPOLossType.wpo.value, "wpo")


class TestTensorCache(unittest.TestCase):
    """Tests for TensorCache."""

    def test_getitem_returns_correct_tensors(self):
        chosen_logps = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        rejected_logps = torch.tensor([[0.5, 1.5], [2.5, 3.5]])

        cache = model_utils.TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        result = cache[torch.tensor([0])]
        self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([[1.0, 2.0]])))
        self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([[0.5, 1.5]])))

        result = cache[torch.tensor([1])]
        self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([[3.0, 4.0]])))
        self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([[2.5, 3.5]])))

    def test_getitem_with_multiple_indices(self):
        chosen_logps = torch.tensor([[1.0], [2.0], [3.0]])
        rejected_logps = torch.tensor([[0.5], [1.5], [2.5]])

        cache = model_utils.TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        result = cache[torch.tensor([0, 2])]
        self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([[1.0], [3.0]])))
        self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([[0.5], [2.5]])))


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


if __name__ == "__main__":
    unittest.main()
