"""Tests for DPO training module."""

import logging
import unittest
from unittest import mock

import torch
from datasets import Dataset

from open_instruct import dpo

logging.basicConfig(level=logging.INFO)


class TestDPOLossType(unittest.TestCase):
    """Tests for DPOLossType enum."""

    def test_loss_types_exist(self):
        self.assertEqual(dpo.DPOLossType.dpo.value, "dpo")
        self.assertEqual(dpo.DPOLossType.dpo_norm.value, "dpo_norm")
        self.assertEqual(dpo.DPOLossType.simpo.value, "simpo")
        self.assertEqual(dpo.DPOLossType.wpo.value, "wpo")


class TestReferenceLogprobsCache(unittest.TestCase):
    """Tests for ReferenceLogprobsCache."""

    def test_get_returns_correct_tensors(self):
        chosen_logps = [[torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]]
        rejected_logps = [[torch.tensor([0.5, 1.5]), torch.tensor([2.5, 3.5])]]

        cache = dpo.ReferenceLogprobsCache(chosen_logps=chosen_logps, rejected_logps=rejected_logps)
        device = torch.device("cpu")

        chosen, rejected = cache.get(epoch=0, batch_idx=0, device=device)
        self.assertTrue(torch.allclose(chosen, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(rejected, torch.tensor([0.5, 1.5])))

        chosen, rejected = cache.get(epoch=0, batch_idx=1, device=device)
        self.assertTrue(torch.allclose(chosen, torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.allclose(rejected, torch.tensor([2.5, 3.5])))

    def test_get_multiple_epochs(self):
        chosen_logps = [[torch.tensor([1.0])], [torch.tensor([2.0])]]
        rejected_logps = [[torch.tensor([0.5])], [torch.tensor([1.5])]]

        cache = dpo.ReferenceLogprobsCache(chosen_logps=chosen_logps, rejected_logps=rejected_logps)
        device = torch.device("cpu")

        chosen_e0, _ = cache.get(epoch=0, batch_idx=0, device=device)
        chosen_e1, _ = cache.get(epoch=1, batch_idx=0, device=device)

        self.assertTrue(torch.allclose(chosen_e0, torch.tensor([1.0])))
        self.assertTrue(torch.allclose(chosen_e1, torch.tensor([2.0])))


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


class TestDPODataLoader(unittest.TestCase):
    """Tests for DPODataLoader."""

    def setUp(self):
        self.mock_tokenizer = mock.MagicMock()
        self.mock_tokenizer.pad_token_id = 0

        self.mock_dataset = Dataset.from_dict(
            {
                "chosen_input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                "chosen_labels": [[-100, 2, 3], [-100, 5, 6], [-100, 8, 9], [-100, 11, 12]],
                "chosen_attention_mask": [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                "rejected_input_ids": [[1, 2, 4], [4, 5, 7], [7, 8, 10], [10, 11, 13]],
                "rejected_labels": [[-100, 2, 4], [-100, 5, 7], [-100, 8, 10], [-100, 11, 13]],
                "rejected_attention_mask": [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
            }
        )
        self.mock_dataset.set_format(type="pt")

    def test_initialization(self):
        loader = dpo.DPODataLoader(
            dataset=self.mock_dataset,
            batch_size=2,
            seed=42,
            rank=0,
            world_size=1,
            work_dir="/tmp",
            tokenizer=self.mock_tokenizer,
        )
        self.assertEqual(loader._batch_size, 2)
        self.assertEqual(loader._epoch, 0)
        self.assertFalse(loader._packing)

    def test_total_batches(self):
        loader = dpo.DPODataLoader(
            dataset=self.mock_dataset,
            batch_size=2,
            seed=42,
            rank=0,
            world_size=1,
            work_dir="/tmp",
            tokenizer=self.mock_tokenizer,
        )
        self.assertEqual(loader.total_batches, 2)

    def test_state_dict(self):
        loader = dpo.DPODataLoader(
            dataset=self.mock_dataset,
            batch_size=2,
            seed=42,
            rank=0,
            world_size=1,
            work_dir="/tmp",
            tokenizer=self.mock_tokenizer,
        )
        loader._epoch = 3
        loader.batches_processed = 5

        state = loader.state_dict()
        self.assertEqual(state["epoch"], 3)
        self.assertEqual(state["batches_processed"], 5)
        self.assertIn("base_loader_state", state)


class TestDPOTrainModuleInit(unittest.TestCase):
    """Tests for DPOTrainModule initialization."""

    def test_average_log_prob_for_simpo(self):
        config = dpo.DPOConfig(dpo_loss_type=dpo.DPOLossType.simpo)

        with mock.patch.object(dpo.TransformerTrainModule, "__init__", return_value=None):
            module = dpo.DPOTrainModule.__new__(dpo.DPOTrainModule)
            module.dpo_config = config
            module.average_log_prob = config.dpo_loss_type in (dpo.DPOLossType.simpo, dpo.DPOLossType.dpo_norm)

        self.assertTrue(module.average_log_prob)

    def test_average_log_prob_for_dpo_norm(self):
        config = dpo.DPOConfig(dpo_loss_type=dpo.DPOLossType.dpo_norm)

        with mock.patch.object(dpo.TransformerTrainModule, "__init__", return_value=None):
            module = dpo.DPOTrainModule.__new__(dpo.DPOTrainModule)
            module.dpo_config = config
            module.average_log_prob = config.dpo_loss_type in (dpo.DPOLossType.simpo, dpo.DPOLossType.dpo_norm)

        self.assertTrue(module.average_log_prob)

    def test_average_log_prob_for_dpo(self):
        config = dpo.DPOConfig(dpo_loss_type=dpo.DPOLossType.dpo)

        with mock.patch.object(dpo.TransformerTrainModule, "__init__", return_value=None):
            module = dpo.DPOTrainModule.__new__(dpo.DPOTrainModule)
            module.dpo_config = config
            module.average_log_prob = config.dpo_loss_type in (dpo.DPOLossType.simpo, dpo.DPOLossType.dpo_norm)

        self.assertFalse(module.average_log_prob)


if __name__ == "__main__":
    unittest.main()
