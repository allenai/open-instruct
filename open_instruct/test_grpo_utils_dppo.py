"""Unit tests for the DPPO loss helpers in grpo_utils."""

import unittest
from unittest.mock import MagicMock

import torch

from open_instruct import grpo_utils
from open_instruct.utils import INVALID_LOGPROB


def _make_grpo_config(**kwargs) -> grpo_utils.GRPOExperimentConfig:
    defaults = {
        "clip_lower": 0.2,
        "clip_higher": 0.2,
        "beta": 0.0,
        "kl_estimator": 2,
        "loss_fn": grpo_utils.GRPOLossType.dapo,
        "load_ref_policy": False,
        "dppo_divergence_type": grpo_utils.DPPODivergenceType.tv,
        "dppo_divergence_threshold": 0.1,
    }
    defaults.update(kwargs)
    config = MagicMock(spec=grpo_utils.GRPOExperimentConfig)
    for key, value in defaults.items():
        setattr(config, key, value)
    return config


class TestComputeBinaryDivergence(unittest.TestCase):
    def test_tv_matches_definition(self):
        # Eq. 13 in arXiv:2602.04879: D_TV^Bin = |μ - π|.
        behavior_logprobs = torch.log(torch.tensor([[0.1, 0.5, 0.9]]))
        policy_logprobs = torch.log(torch.tensor([[0.2, 0.5, 0.3]]))
        response_mask = torch.ones_like(behavior_logprobs, dtype=torch.bool)

        divergence = grpo_utils.compute_binary_divergence(
            behavior_logprobs=behavior_logprobs,
            policy_logprobs=policy_logprobs,
            response_mask=response_mask,
            divergence_type=grpo_utils.DPPODivergenceType.tv,
        )

        expected = torch.tensor([[0.1, 0.0, 0.6]])
        torch.testing.assert_close(divergence, expected, atol=1e-5, rtol=1e-5)

    def test_kl_zero_when_distributions_match(self):
        logprobs = torch.log(torch.tensor([[0.3, 0.7]]))
        response_mask = torch.ones_like(logprobs, dtype=torch.bool)

        divergence = grpo_utils.compute_binary_divergence(
            behavior_logprobs=logprobs,
            policy_logprobs=logprobs,
            response_mask=response_mask,
            divergence_type=grpo_utils.DPPODivergenceType.kl,
        )

        torch.testing.assert_close(divergence, torch.zeros_like(divergence), atol=1e-5, rtol=1e-5)

    def test_response_mask_zeroes_invalid_positions(self):
        behavior_logprobs = torch.tensor([[INVALID_LOGPROB, -0.1]])
        policy_logprobs = torch.tensor([[INVALID_LOGPROB, -2.0]])
        response_mask = torch.tensor([[False, True]])

        divergence = grpo_utils.compute_binary_divergence(
            behavior_logprobs=behavior_logprobs,
            policy_logprobs=policy_logprobs,
            response_mask=response_mask,
            divergence_type=grpo_utils.DPPODivergenceType.tv,
        )

        self.assertEqual(float(divergence[0, 0]), 0.0)
        self.assertGreater(float(divergence[0, 1]), 0.0)

    def test_unknown_divergence_type_raises(self):
        with self.assertRaises(ValueError):
            grpo_utils.compute_binary_divergence(
                behavior_logprobs=torch.zeros(1, 1),
                policy_logprobs=torch.zeros(1, 1),
                response_mask=torch.ones(1, 1, dtype=torch.bool),
                divergence_type="not_a_divergence",
            )


class TestComputeDPPOMask(unittest.TestCase):
    def _make_inputs(self):
        # Behavior probs: [0.1, 0.1, 0.1]; policy probs: [0.5, 0.5, 0.5].
        # Binary TV per token is 0.4 -> well above any small δ.
        behavior_logprobs = torch.log(torch.tensor([[0.1, 0.1, 0.1]]))
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
        # ratio = exp(new - behavior) = 5 for all tokens.
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        return new_logprobs, behavior_logprobs, ratio, response_mask

    def test_blocks_only_unsafe_directions(self):
        new_logprobs, behavior_logprobs, ratio, response_mask = self._make_inputs()
        # Per Eq. 12: A>0 and r>1 with D>δ -> mask. A<0 and r>1 -> safe (ratio
        # heading back towards 1 under negative advantage), so don't mask.
        advantages = torch.tensor([[1.0, -1.0, 0.0]])

        mask, divergence = grpo_utils.compute_dppo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_type=grpo_utils.DPPODivergenceType.tv,
            divergence_threshold=0.05,
        )

        self.assertTrue(torch.all(divergence > 0.05))
        torch.testing.assert_close(mask, torch.tensor([[0.0, 1.0, 1.0]]))

    def test_below_threshold_keeps_all_tokens(self):
        # Same μ/π so divergence is 0 everywhere.
        logprobs = torch.log(torch.tensor([[0.4, 0.6]]))
        ratio = torch.ones_like(logprobs)
        response_mask = torch.ones_like(logprobs, dtype=torch.bool)
        advantages = torch.tensor([[1.0, -1.0]])

        mask, _ = grpo_utils.compute_dppo_mask(
            new_logprobs=logprobs,
            behavior_logprobs=logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_type=grpo_utils.DPPODivergenceType.tv,
            divergence_threshold=0.01,
        )

        torch.testing.assert_close(mask, torch.ones_like(mask))

    def test_response_mask_zeroes_padding(self):
        new_logprobs, behavior_logprobs, ratio, _ = self._make_inputs()
        advantages = torch.tensor([[-1.0, -1.0, -1.0]])
        # Only middle token is a real response position. A<0 with r>1 is the
        # "safe" direction (moving back towards 1), so it is never masked; the
        # padding positions are always zeroed via response_mask.
        response_mask = torch.tensor([[False, True, False]])

        mask, _ = grpo_utils.compute_dppo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_type=grpo_utils.DPPODivergenceType.tv,
            divergence_threshold=0.05,
        )

        torch.testing.assert_close(mask, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_mask_does_not_propagate_gradients(self):
        new_logprobs = torch.log(torch.tensor([[0.5]])).requires_grad_(True)
        behavior_logprobs = torch.log(torch.tensor([[0.1]]))
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        advantages = torch.tensor([[1.0]])

        mask, _ = grpo_utils.compute_dppo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_type=grpo_utils.DPPODivergenceType.tv,
            divergence_threshold=0.05,
        )

        self.assertFalse(mask.requires_grad)


class TestDPPOLoss(unittest.TestCase):
    def test_dppo_loss_masks_and_has_no_symmetric_clip(self):
        config = _make_grpo_config(loss_fn=grpo_utils.GRPOLossType.dppo)
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        behavior_logprobs = torch.log(torch.tensor([[0.1, 0.1]]))
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        advantages = torch.tensor([[1.0, -1.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        rho_weights = torch.ones_like(new_logprobs)

        dppo_mask, _ = grpo_utils.compute_dppo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_type=config.dppo_divergence_type,
            divergence_threshold=config.dppo_divergence_threshold,
        )

        pg_loss, clipfrac, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=rho_weights,
            dppo_mask=dppo_mask,
        )

        expected = -advantages * ratio * dppo_mask
        torch.testing.assert_close(pg_loss, expected)
        torch.testing.assert_close(clipfrac, torch.zeros_like(clipfrac))
        torch.testing.assert_close(kl, torch.zeros_like(kl))


class TestDPPOConfigValidation(unittest.TestCase):
    def test_dppo_requires_positive_threshold(self):
        with self.assertRaisesRegex(ValueError, "dppo_divergence_threshold"):
            grpo_utils.GRPOExperimentConfig(
                loss_fn=grpo_utils.GRPOLossType.dppo,
                dppo_divergence_threshold=0.0,
                use_vllm_logprobs=True,
                use_rho_correction=False,
            )

    def test_dppo_requires_use_vllm_logprobs(self):
        with self.assertRaisesRegex(ValueError, "use_vllm_logprobs"):
            grpo_utils.GRPOExperimentConfig(loss_fn=grpo_utils.GRPOLossType.dppo, use_vllm_logprobs=False)


if __name__ == "__main__":
    unittest.main()
