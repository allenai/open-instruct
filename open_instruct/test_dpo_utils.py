"""Tests for DPO utility functions."""

import logging
import unittest

import torch
from parameterized import parameterized

from open_instruct import dpo_utils, utils
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


class TestTokenCountReduction(unittest.TestCase):
    """Tests that the metric reduction in the DPO training loop correctly
    computes total token count and average loss.

    Verifies the fix for the token_count undercount bug (commit 98120776)
    where the old code only multiplied by num_processes but not by
    gradient_accumulation_steps * logging_steps.
    """

    @parameterized.expand(
        [
            ("2proc_4accum_1log", 2, 4, 1),
            ("8proc_4accum_1log", 8, 4, 1),
            ("32proc_2accum_5log", 32, 2, 5),
            ("1proc_1accum_1log", 1, 1, 1),
        ]
    )
    def test_token_count_equals_true_total(self, _name, num_processes, grad_accum_steps, logging_steps):
        num_microbatches = grad_accum_steps * logging_steps
        torch.manual_seed(42)
        per_process_tokens = torch.randint(100, 1000, (num_processes, num_microbatches)).float()
        true_total_tokens = per_process_tokens.sum().item()

        # Simulate what each process accumulates.
        per_process_accumulated = per_process_tokens.sum(dim=1)

        # Simulate accelerator.reduce(mean).
        reduced_mean = per_process_accumulated.mean()

        # Apply the division (same as training loop).
        reduced_mean /= grad_accum_steps * logging_steps

        # Apply the token_count correction (the fix).
        corrected = reduced_mean * (num_processes * grad_accum_steps * logging_steps)

        self.assertAlmostEqual(corrected.item(), true_total_tokens, places=0)

    @parameterized.expand(
        [
            ("2proc_4accum_1log", 2, 4, 1),
            ("8proc_4accum_1log", 8, 4, 1),
            ("32proc_2accum_5log", 32, 2, 5),
            ("1proc_1accum_1log", 1, 1, 1),
        ]
    )
    def test_old_formula_undercounts(self, _name, num_processes, grad_accum_steps, logging_steps):
        num_microbatches = grad_accum_steps * logging_steps
        if num_microbatches == 1:
            self.skipTest("No undercount when grad_accum * logging == 1")

        torch.manual_seed(42)
        per_process_tokens = torch.randint(100, 1000, (num_processes, num_microbatches)).float()
        true_total_tokens = per_process_tokens.sum().item()

        per_process_accumulated = per_process_tokens.sum(dim=1)
        reduced_mean = per_process_accumulated.mean()
        reduced_mean /= grad_accum_steps * logging_steps

        # Old (buggy) formula: only multiply by num_processes.
        old_result = reduced_mean * num_processes

        expected_undercount_factor = grad_accum_steps * logging_steps
        self.assertAlmostEqual(old_result.item() * expected_undercount_factor, true_total_tokens, places=0)
        self.assertLess(old_result.item(), true_total_tokens)

    @parameterized.expand(
        [
            ("2proc_4accum_1log", 2, 4, 1),
            ("8proc_4accum_1log", 8, 4, 1),
            ("32proc_2accum_5log", 32, 2, 5),
            ("1proc_1accum_1log", 1, 1, 1),
        ]
    )
    def test_loss_average_is_correct(self, _name, num_processes, grad_accum_steps, logging_steps):
        num_microbatches = grad_accum_steps * logging_steps
        torch.manual_seed(42)
        per_process_losses = torch.rand(num_processes, num_microbatches)
        true_average_loss = per_process_losses.mean().item()

        per_process_accumulated = per_process_losses.sum(dim=1)
        reduced_mean = per_process_accumulated.mean()
        average_loss = (reduced_mean / (grad_accum_steps * logging_steps)).item()

        self.assertAlmostEqual(average_loss, true_average_loss, places=5)

    def test_with_metrics_tracker(self):
        """End-to-end test using MetricsTracker, matching the training loop."""
        num_processes = 4
        grad_accum_steps = 4
        logging_steps = 1
        num_microbatches = grad_accum_steps * logging_steps

        torch.manual_seed(42)
        per_process_tokens = torch.randint(100, 1000, (num_processes, num_microbatches)).float()
        per_process_losses = torch.rand(num_processes, num_microbatches)

        true_total_tokens = per_process_tokens.sum().item()
        true_average_loss = per_process_losses.mean().item()

        # Simulate each process accumulating into a MetricsTracker.
        trackers = []
        for proc in range(num_processes):
            tracker = utils.MetricsTracker(device="cpu")
            for mb in range(num_microbatches):
                tracker["train_loss"] += per_process_losses[proc, mb]
                tracker["token_count"] += per_process_tokens[proc, mb]
            trackers.append(tracker)

        # Simulate accelerator.reduce(mean).
        stacked = torch.stack([t.metrics for t in trackers])
        global_metrics_tensor = stacked.mean(dim=0)

        # Apply the division.
        global_metrics_tensor /= grad_accum_steps * logging_steps

        # Apply the token_count correction.
        token_idx = trackers[0].names2idx["token_count"]
        global_metrics_tensor[token_idx] *= num_processes * grad_accum_steps * logging_steps

        result_token_count = global_metrics_tensor[token_idx].item()
        loss_idx = trackers[0].names2idx["train_loss"]
        result_loss = global_metrics_tensor[loss_idx].item()

        self.assertAlmostEqual(result_token_count, true_total_tokens, places=0)
        self.assertAlmostEqual(result_loss, true_average_loss, places=5)


if __name__ == "__main__":
    unittest.main()
