"""Tests for DPO utility functions."""

import logging
import unittest

import torch

from open_instruct import dpo_utils
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.dpo_utils import DPOLossType, ExperimentConfig

logging.basicConfig(level=logging.INFO)


def make_test_args(**overrides) -> ExperimentConfig:
    """Create an ExperimentConfig with test defaults."""
    defaults = {
        "model_name_or_path": "allenai/OLMo-2-1124-7B",
        "mixer_list": ["allenai/tulu-3-wildchat-reused-on-policy-8b", "1.0"],
        "config_hash": "test_dataset_config_hash",
    }
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class TestDPOLoss(unittest.TestCase):
    """Tests for dpo_loss function."""

    def test_basic_loss_computation(self):
        policy_chosen = torch.tensor([0.0, -1.0, -2.0])
        policy_rejected = torch.tensor([-1.0, -2.0, -3.0])
        ref_chosen = torch.tensor([0.0, -1.0, -2.0])
        ref_rejected = torch.tensor([-1.0, -2.0, -3.0])

        losses, chosen_rewards, rejected_rewards = dpo_utils.dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )

        self.assertEqual(losses.shape, (3,))
        self.assertEqual(chosen_rewards.shape, (3,))
        self.assertEqual(rejected_rewards.shape, (3,))

    def test_chosen_preferred_gives_lower_loss(self):
        policy_chosen = torch.tensor([0.0])
        policy_rejected = torch.tensor([-5.0])
        ref_chosen = torch.tensor([-1.0])
        ref_rejected = torch.tensor([-1.0])

        losses_good, _, _ = dpo_utils.dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)

        policy_chosen_bad = torch.tensor([-5.0])
        policy_rejected_bad = torch.tensor([0.0])

        losses_bad, _, _ = dpo_utils.dpo_loss(
            policy_chosen_bad, policy_rejected_bad, ref_chosen, ref_rejected, beta=0.1
        )

        self.assertLess(losses_good.item(), losses_bad.item())

    def test_reference_free(self):
        policy_chosen = torch.tensor([0.0])
        policy_rejected = torch.tensor([-1.0])
        ref_chosen = torch.tensor([-5.0])
        ref_rejected = torch.tensor([-10.0])

        losses_ref, _, _ = dpo_utils.dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1, reference_free=False
        )

        losses_ref_free, _, _ = dpo_utils.dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1, reference_free=True
        )

        self.assertNotEqual(losses_ref.item(), losses_ref_free.item())

    def test_rewards_are_detached(self):
        policy_chosen = torch.tensor([0.0], requires_grad=True)
        policy_rejected = torch.tensor([-1.0], requires_grad=True)
        ref_chosen = torch.tensor([-1.0])
        ref_rejected = torch.tensor([-1.0])

        _, chosen_rewards, rejected_rewards = dpo_utils.dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )

        self.assertFalse(chosen_rewards.requires_grad)
        self.assertFalse(rejected_rewards.requires_grad)


class TestSimPOLoss(unittest.TestCase):
    """Tests for simpo_loss function."""

    def test_basic_loss_computation(self):
        policy_chosen = torch.tensor([0.0, -1.0, -2.0])
        policy_rejected = torch.tensor([-1.0, -2.0, -3.0])

        losses, chosen_rewards, rejected_rewards = dpo_utils.simpo_loss(
            policy_chosen, policy_rejected, beta=0.1, gamma_beta_ratio=0.3
        )

        self.assertEqual(losses.shape, (3,))
        self.assertEqual(chosen_rewards.shape, (3,))
        self.assertEqual(rejected_rewards.shape, (3,))

    def test_gamma_affects_loss(self):
        policy_chosen = torch.tensor([0.0])
        policy_rejected = torch.tensor([-1.0])

        losses_low_gamma, _, _ = dpo_utils.simpo_loss(policy_chosen, policy_rejected, beta=0.1, gamma_beta_ratio=0.1)

        losses_high_gamma, _, _ = dpo_utils.simpo_loss(policy_chosen, policy_rejected, beta=0.1, gamma_beta_ratio=1.0)

        self.assertNotEqual(losses_low_gamma.item(), losses_high_gamma.item())


class TestWPOLoss(unittest.TestCase):
    """Tests for wpo_loss function."""

    def test_basic_loss_computation(self):
        policy_chosen = torch.tensor([[0.0, -1.0, -2.0]])
        policy_rejected = torch.tensor([[-1.0, -2.0, -3.0]])
        ref_chosen = torch.tensor([[0.0, -1.0, -2.0]])
        ref_rejected = torch.tensor([[-1.0, -2.0, -3.0]])
        chosen_mask = torch.tensor([[True, True, True]])
        rejected_mask = torch.tensor([[True, True, True]])

        losses, chosen_rewards, rejected_rewards = dpo_utils.wpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            beta=0.1,
            chosen_loss_mask=chosen_mask,
            rejected_loss_mask=rejected_mask,
        )

        self.assertEqual(losses.shape, (1, 3))
        self.assertEqual(chosen_rewards.shape, (1, 3))
        self.assertEqual(rejected_rewards.shape, (1, 3))


class TestComputeReferenceCacheHash(unittest.TestCase):
    """Tests for compute_reference_cache_hash function."""

    def test_deterministic_hash(self):
        args = make_test_args()
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")

        hash1 = dpo_utils.compute_reference_cache_hash(args, tc)
        hash2 = dpo_utils.compute_reference_cache_hash(args, tc)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)

    def test_different_model_different_hash(self):
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")

        args1 = make_test_args(model_name_or_path="allenai/OLMo-2-1124-7B")
        args2 = make_test_args(model_name_or_path="allenai/OLMo-2-1124-13B")

        hash1 = dpo_utils.compute_reference_cache_hash(args1, tc)
        hash2 = dpo_utils.compute_reference_cache_hash(args2, tc)

        self.assertNotEqual(hash1, hash2)

    def test_different_loss_type_different_hash(self):
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")

        args1 = make_test_args(loss_type=DPOLossType.dpo)
        args2 = make_test_args(loss_type=DPOLossType.simpo)

        hash1 = dpo_utils.compute_reference_cache_hash(args1, tc)
        hash2 = dpo_utils.compute_reference_cache_hash(args2, tc)

        self.assertNotEqual(hash1, hash2)

    def test_different_packing_different_hash(self):
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")

        args1 = make_test_args(packing=False)
        args2 = make_test_args(packing=True)

        hash1 = dpo_utils.compute_reference_cache_hash(args1, tc)
        hash2 = dpo_utils.compute_reference_cache_hash(args2, tc)

        self.assertNotEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main()
