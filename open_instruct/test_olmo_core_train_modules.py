import unittest
from unittest.mock import MagicMock

import torch
from parameterized import parameterized

from open_instruct import data_types, grpo_utils
from open_instruct.utils import INVALID_LOGPROB


def _make_mock_model(vocab_size: int = 10, seq_len: int = 5, batch_size: int = 2) -> MagicMock:
    model = MagicMock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    logits = torch.randn(batch_size, seq_len, vocab_size)
    model.return_value = logits
    return model


def _make_batch_data(
    batch_size: int = 2, seq_len: int = 5, vocab_size: int = 10, num_samples: int = 2
) -> data_types.CollatedBatchData:
    query_responses = []
    attention_masks = []
    position_ids = []
    advantages = []
    response_masks = []
    vllm_logprobs = []

    for _ in range(num_samples):
        query_responses.append(torch.randint(0, vocab_size, (batch_size, seq_len)))
        attention_masks.append(torch.ones(batch_size, seq_len, dtype=torch.long))
        position_ids.append(torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1))
        advantages.append(torch.randn(batch_size, seq_len))
        response_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        response_mask[:, :2] = False
        response_masks.append(response_mask)
        vllm_logprobs.append(torch.randn(batch_size, seq_len - 1))

    return data_types.CollatedBatchData(
        query_responses=query_responses,
        attention_masks=attention_masks,
        position_ids=position_ids,
        advantages=advantages,
        response_masks=response_masks,
        vllm_logprobs=vllm_logprobs,
    )


class TestComputeLogprobs(unittest.TestCase):
    def test_basic(self):
        batch_size, seq_len, vocab_size = 2, 5, 10
        model = _make_mock_model(vocab_size, seq_len, batch_size)
        data_BT = _make_batch_data(batch_size, seq_len, vocab_size, num_samples=2)

        result = grpo_utils.compute_logprobs(model, data_BT, pad_token_id=0, temperature=1.0, use_grad=False)

        self.assertEqual(len(result), 2)
        for logprob in result:
            self.assertEqual(logprob.shape, (batch_size, seq_len - 1))
            self.assertTrue(torch.all(logprob <= INVALID_LOGPROB))

    def test_with_response_mask(self):
        batch_size, seq_len, vocab_size = 2, 5, 10
        model = _make_mock_model(vocab_size, seq_len, batch_size)
        data_BT = _make_batch_data(batch_size, seq_len, vocab_size, num_samples=1)
        data_BT.response_masks[0][:, :] = 0

        result = grpo_utils.compute_logprobs(model, data_BT, pad_token_id=0, temperature=1.0, use_grad=False)

        self.assertEqual(len(result), 1)
        self.assertTrue(torch.all(result[0] == INVALID_LOGPROB))

    def test_use_grad(self):
        batch_size, seq_len, vocab_size = 2, 5, 10
        model = _make_mock_model(vocab_size, seq_len, batch_size)
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        model.return_value = logits
        data_BT = _make_batch_data(batch_size, seq_len, vocab_size, num_samples=1)

        result = grpo_utils.compute_logprobs(model, data_BT, pad_token_id=0, temperature=1.0, use_grad=True)

        self.assertTrue(result[0].requires_grad)


class TestForwardForLogprobs(unittest.TestCase):
    def test_log_probabilities(self):
        batch_size, seq_len, vocab_size = 2, 5, 10
        model = _make_mock_model(vocab_size, seq_len, batch_size)
        query_responses = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        logprob, entropy = grpo_utils.forward_for_logprobs(
            model, query_responses, attention_mask, position_ids, pad_token_id=0, temperature=1.0, return_entropy=False
        )

        self.assertEqual(logprob.shape, (batch_size, seq_len - 1))
        self.assertTrue(torch.all(logprob <= 0))
        self.assertIsNone(entropy)

    def test_with_entropy(self):
        batch_size, seq_len, vocab_size = 2, 5, 10
        model = _make_mock_model(vocab_size, seq_len, batch_size)
        query_responses = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        logprob, entropy = grpo_utils.forward_for_logprobs(
            model, query_responses, attention_mask, position_ids, pad_token_id=0, temperature=1.0, return_entropy=True
        )

        self.assertEqual(logprob.shape, (batch_size, seq_len - 1))
        self.assertIsNotNone(entropy)
        self.assertEqual(entropy.shape, (batch_size, seq_len - 1))
        self.assertTrue(torch.all(entropy >= 0))

    def test_temperature_scaling(self):
        batch_size, seq_len, vocab_size = 2, 5, 10
        model = _make_mock_model(vocab_size, seq_len, batch_size)
        query_responses = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        logprob_t1, _ = grpo_utils.forward_for_logprobs(
            model, query_responses, attention_mask, position_ids, pad_token_id=0, temperature=1.0, return_entropy=False
        )
        logprob_t2, _ = grpo_utils.forward_for_logprobs(
            model, query_responses, attention_mask, position_ids, pad_token_id=0, temperature=2.0, return_entropy=False
        )

        self.assertFalse(torch.allclose(logprob_t1, logprob_t2))


class TestDAPOLoss(unittest.TestCase):
    def test_negative_advantages_clipping(self):
        batch_size, seq_len = 2, 5
        clip_lower = 0.2
        clip_higher = 0.28

        advantages = -torch.ones(batch_size, seq_len)
        ratio = torch.tensor([[1.5, 0.5, 1.0, 1.3], [0.7, 1.4, 0.9, 1.1]])

        pg_losses = -advantages[:, 1:] * ratio
        pg_losses2 = -advantages[:, 1:] * torch.clamp(ratio, 1.0 - clip_lower, 1.0 + clip_higher)
        pg_loss = torch.max(pg_losses, pg_losses2)

        self.assertTrue(torch.all(pg_loss >= pg_losses))
        self.assertTrue(torch.all(pg_loss >= pg_losses2))

        high_ratio_mask = ratio > 1.0 + clip_higher
        if high_ratio_mask.any():
            self.assertTrue(torch.all(pg_losses[high_ratio_mask] > pg_losses2[high_ratio_mask]))


def _make_grpo_config(**kwargs) -> grpo_utils.GRPOExperimentConfig:
    defaults = {
        "clip_lower": 0.2,
        "clip_higher": 0.2,
        "beta": 0.05,
        "kl_estimator": 2,
        "loss_fn": grpo_utils.GRPOLossType.dapo,
        "load_ref_policy": False,
        "rho_mask_sequence_level": False,
        "rho_clamp_lower_bound": 0.0,
        "rho_clamp_upper_bound": 0.0,
        "rho_mask_lower_bound": 0.0,
        "rho_mask_upper_bound": 0.0,
        "rho_divergence_type": grpo_utils.RhoDivergenceType.rho,
        "use_rho_correction": False,
    }
    defaults.update(kwargs)
    config = MagicMock(spec=grpo_utils.GRPOExperimentConfig)
    for key, value in defaults.items():
        setattr(config, key, value)
    return config


def _old_logprobs_for_ratio(new_logprobs: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    """Old logprobs such that exp(new - old) equals ``ratio``."""
    return new_logprobs.detach() - torch.log(ratio)


class TestComputeGRPOLoss(unittest.TestCase):
    @parameterized.expand(
        [
            ("dapo", grpo_utils.GRPOLossType.dapo),
            ("cispo", grpo_utils.GRPOLossType.cispo),
            ("dppo", grpo_utils.GRPOLossType.dppo),
        ]
    )
    def test_output_shapes(self, _name, loss_type):
        batch_size, seq_len = 2, 4
        config = _make_grpo_config(loss_fn=loss_type)
        new_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            vllm_logprobs=old_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            config=config,
        )

        self.assertEqual(output.pg_loss.shape, (batch_size, seq_len))
        self.assertEqual(output.clipfrac.shape, (batch_size, seq_len))
        self.assertEqual(output.kl.shape, (batch_size, seq_len))
        self.assertEqual(output.ratio.shape, (batch_size, seq_len))

    def test_dapo_clipping(self):
        config = _make_grpo_config(clip_lower=0.2, clip_higher=0.2)
        ratio = torch.tensor([[1.5, 0.5, 1.0]])
        new_logprobs = torch.randn(1, 3)
        old_logprobs = _old_logprobs_for_ratio(new_logprobs, ratio)
        advantages = torch.ones(1, 3)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            vllm_logprobs=old_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=torch.ones(1, 3, dtype=torch.bool),
            config=config,
        )

        expected_clamped = torch.clamp(ratio, 0.8, 1.2)
        expected_unclipped = -advantages * ratio
        expected_clipped = -advantages * expected_clamped
        torch.testing.assert_close(output.pg_loss, torch.max(expected_unclipped, expected_clipped))
        torch.testing.assert_close(output.clipfrac, (expected_clipped > expected_unclipped).float())

    def test_cispo_uses_detached_ratio(self):
        config = _make_grpo_config(loss_fn=grpo_utils.GRPOLossType.cispo, clip_higher=0.2)
        ratio = torch.tensor([[1.5, 0.5, 1.0]])
        new_logprobs = torch.randn(1, 3, requires_grad=True)
        old_logprobs = _old_logprobs_for_ratio(new_logprobs, ratio)
        advantages = torch.ones(1, 3)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            vllm_logprobs=old_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=torch.ones(1, 3, dtype=torch.bool),
            config=config,
        )

        output.pg_loss.sum().backward()
        # The clipped ratio is detached, so the only gradient path is the REINFORCE term.
        torch.testing.assert_close(new_logprobs.grad, -advantages * torch.clamp(ratio, max=1.2))
        torch.testing.assert_close(output.clipfrac, torch.tensor([[1.0, 0.0, 0.0]]))

    def test_with_ref_logprobs(self):
        config = _make_grpo_config(beta=0.05, kl_estimator=2)
        batch_size, seq_len = 2, 4
        new_logprobs = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            vllm_logprobs=old_logprobs,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            response_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            config=config,
        )

        self.assertFalse(torch.all(output.kl == 0))

    def test_without_ref_logprobs(self):
        config = _make_grpo_config()
        new_logprobs = torch.randn(2, 4)
        old_logprobs = torch.randn(2, 4)
        advantages = torch.randn(2, 4)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            vllm_logprobs=old_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=torch.ones(2, 4, dtype=torch.bool),
            config=config,
        )

        torch.testing.assert_close(output.kl, torch.zeros_like(output.kl))

    def test_rho_weights_scale_loss(self):
        # ρ = π_old / μ = 2 everywhere; with clamps disabled the kept tokens are reweighted by 2.
        config = _make_grpo_config(use_rho_correction=True)
        new_logprobs = torch.randn(2, 4)
        old_logprobs = torch.randn(2, 4)
        vllm_logprobs = old_logprobs - torch.log(torch.tensor(2.0))
        advantages = torch.randn(2, 4)
        response_mask = torch.ones(2, 4, dtype=torch.bool)

        common = dict(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=response_mask,
            config=config,
        )
        output_no_rho = grpo_utils.compute_grpo_loss(vllm_logprobs=old_logprobs, **common)
        output_rho = grpo_utils.compute_grpo_loss(vllm_logprobs=vllm_logprobs, **common)

        torch.testing.assert_close(output_rho.pg_loss, output_no_rho.pg_loss * 2.0)
        torch.testing.assert_close(output_rho.clipfrac, output_no_rho.clipfrac)

    def test_rho_mask(self):
        config = _make_grpo_config(use_rho_correction=True, rho_mask_lower_bound=0.5, rho_mask_upper_bound=2.0)
        response_mask = torch.tensor([[True, True, True, True, True]])
        # ρ values: 0.25 (drop, < lower=0.5), 0.5 (keep), 1.0 (keep), 2.0 (keep), 4.0 (drop, > upper=2.0).
        # In-range tokens are reweighted by ρ, not gated to 1.
        old_logprob = torch.log(torch.tensor([[0.25, 0.5, 1.0, 2.0, 4.0]]))
        vllm_logprobs = torch.zeros_like(old_logprob)
        advantages = torch.ones_like(old_logprob)
        correction = grpo_utils.compute_rho_correction(
            old_logprob, vllm_logprobs, old_logprob, response_mask, advantages, config
        )
        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 0.5, 1.0, 2.0, 0.0]]))
        torch.testing.assert_close(
            correction.metrics["val/rho_drop_low_frac"], torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(
            correction.metrics["val/rho_drop_high_frac"], torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        )

        # Padding tokens (response_mask=False) should always be 0 / not counted as dropped.
        response_mask_with_pad = torch.tensor([[False, True, True, True, False]])
        correction_pad = grpo_utils.compute_rho_correction(
            old_logprob, vllm_logprobs, old_logprob, response_mask_with_pad, advantages, config
        )
        torch.testing.assert_close(correction_pad.weights, torch.tensor([[0.0, 0.5, 1.0, 2.0, 0.0]]))
        torch.testing.assert_close(correction_pad.metrics["val/rho_drop_low_frac"], torch.zeros((1, 5)))
        torch.testing.assert_close(correction_pad.metrics["val/rho_drop_high_frac"], torch.zeros((1, 5)))

    def test_rho_mask_sequence_level(self):
        config = _make_grpo_config(
            use_rho_correction=True, rho_mask_lower_bound=0.5, rho_mask_upper_bound=2.0, rho_mask_sequence_level=True
        )
        # Row 0: per-token ρ = [0.25, 1.0, 4.0]; mean log ρ = 0 → ρ_seq = 1 (kept).
        # Row 1: per-token ρ = [4.0, 4.0, 4.0]; mean log ρ = log 4 → ρ_seq = 4 (drop high).
        # Row 2: per-token ρ = [0.25, 0.25, 0.25]; ρ_seq = 0.25 (drop low).
        old_logprob = torch.log(torch.tensor([[0.25, 1.0, 4.0], [4.0, 4.0, 4.0], [0.25, 0.25, 0.25]]))
        vllm_logprobs = torch.zeros_like(old_logprob)
        response_mask = torch.ones_like(old_logprob, dtype=torch.bool)
        advantages = torch.ones_like(old_logprob)
        correction = grpo_utils.compute_rho_correction(
            old_logprob, vllm_logprobs, old_logprob, response_mask, advantages, config
        )
        torch.testing.assert_close(
            correction.weights, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(
            correction.metrics["val/rho_drop_low_frac"],
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        )
        torch.testing.assert_close(
            correction.metrics["val/rho_drop_high_frac"],
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        )

    def test_rho_mask_tv_divergence(self):
        config = _make_grpo_config(
            use_rho_correction=True,
            rho_mask_lower_bound=0.0,
            rho_mask_upper_bound=2.0,
            rho_divergence_type=grpo_utils.RhoDivergenceType.tv,
        )
        # Row 0: mean |ρ - 1| = 1.25, below upper=2.0, so it is kept.
        # Row 1: mean |ρ - 1| = 3.0, above upper=2.0, so TV-increasing tokens are dropped.
        old_logprob = torch.log(torch.tensor([[0.25, 1.0, 4.0], [4.0, 4.0, 4.0]]))
        vllm_logprobs = torch.zeros_like(old_logprob)
        response_mask = torch.ones_like(old_logprob, dtype=torch.bool)
        advantages = torch.ones_like(old_logprob)

        correction = grpo_utils.compute_rho_correction(
            old_logprob, vllm_logprobs, old_logprob, response_mask, advantages, config
        )

        torch.testing.assert_close(correction.weights, torch.tensor([[0.25, 1.0, 4.0], [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(
            correction.metrics["val/rho_drop_high_frac"], torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        )
        torch.testing.assert_close(
            correction.metrics["val/rho_divergence"], torch.tensor([[1.25, 1.25, 1.25], [3.0, 3.0, 3.0]])
        )

    def test_rho_mask_zeroes_loss(self):
        # ρ values [1.0, 4.0, 1.0] with mask bounds (0.5, 2.0): the middle token is dropped.
        config = _make_grpo_config(use_rho_correction=True, rho_mask_lower_bound=0.5, rho_mask_upper_bound=2.0)
        new_logprobs = torch.randn(1, 3)
        old_logprobs = torch.randn(1, 3)
        vllm_logprobs = old_logprobs - torch.log(torch.tensor([[1.0, 4.0, 1.0]]))
        advantages = torch.randn(1, 3)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=torch.ones(1, 3, dtype=torch.bool),
            config=config,
        )
        self.assertEqual(output.pg_loss[0, 1].item(), 0.0)
        self.assertEqual(output.clipfrac[0, 1].item(), 0.0)
        self.assertNotEqual(output.pg_loss[0, 0].item(), 0.0)

    def test_invalid_loss_fn(self):
        config = _make_grpo_config(loss_fn="invalid")
        with self.assertRaises(ValueError):
            grpo_utils.compute_grpo_loss(
                new_logprobs=torch.randn(2, 4),
                old_logprobs=torch.randn(2, 4),
                vllm_logprobs=torch.randn(2, 4),
                advantages=torch.randn(2, 4),
                ref_logprobs=None,
                response_mask=torch.ones(2, 4, dtype=torch.bool),
                config=config,
            )


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
            divergence_type=grpo_utils.RhoDivergenceType.dppo_tv,
        )

        torch.testing.assert_close(divergence, torch.tensor([[0.1, 0.0, 0.6]]), atol=1e-5, rtol=1e-5)

    def test_kl_zero_when_distributions_match(self):
        logprobs = torch.log(torch.tensor([[0.3, 0.7]]))
        response_mask = torch.ones_like(logprobs, dtype=torch.bool)

        divergence = grpo_utils.compute_binary_divergence(
            behavior_logprobs=logprobs,
            policy_logprobs=logprobs,
            response_mask=response_mask,
            divergence_type=grpo_utils.RhoDivergenceType.dppo_kl,
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
            divergence_type=grpo_utils.RhoDivergenceType.dppo_tv,
        )

        self.assertEqual(float(divergence[0, 0]), 0.0)
        self.assertGreater(float(divergence[0, 1]), 0.0)

    def test_non_dppo_divergence_type_raises(self):
        with self.assertRaises(ValueError):
            grpo_utils.compute_binary_divergence(
                behavior_logprobs=torch.zeros(1, 1),
                policy_logprobs=torch.zeros(1, 1),
                response_mask=torch.ones(1, 1, dtype=torch.bool),
                divergence_type=grpo_utils.RhoDivergenceType.tv,
            )


def _make_dppo_config(**kwargs) -> grpo_utils.GRPOExperimentConfig:
    defaults = {
        "use_rho_correction": False,
        "rho_divergence_type": grpo_utils.RhoDivergenceType.dppo_tv,
        "rho_mask_upper_bound": 0.05,
        "rho_mask_lower_bound": 0.0,
    }
    defaults.update(kwargs)
    return _make_grpo_config(**defaults)


class TestDPPODivergenceMask(unittest.TestCase):
    def test_blocks_only_unsafe_directions(self):
        config = _make_dppo_config()
        # μ = 0.1, π_θ = 0.5 → binary TV = 0.4 > δ = 0.05 everywhere; ratio > 1.
        # Per Eq. 12: A>0 with π_θ>μ is masked; A<0 with π_θ>μ moves back towards μ
        # (safe), and A=0 contributes no update — neither is masked.
        vllm_logprobs = torch.log(torch.tensor([[0.1, 0.1, 0.1]]))
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        advantages = torch.tensor([[1.0, -1.0, 0.0]])

        correction = grpo_utils.compute_rho_correction(
            vllm_logprobs, vllm_logprobs, new_logprobs, response_mask, advantages, config
        )

        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 1.0, 1.0]]))
        torch.testing.assert_close(correction.metrics["val/rho_drop_high_frac"], torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(correction.metrics["val/rho_drop_low_frac"], torch.zeros(1, 3))
        torch.testing.assert_close(
            correction.metrics["val/rho_divergence"], torch.full((1, 3), 0.4), atol=1e-5, rtol=1e-5
        )

    def test_below_threshold_keeps_all_tokens(self):
        config = _make_dppo_config(rho_mask_upper_bound=0.5)
        vllm_logprobs = torch.log(torch.tensor([[0.4, 0.6]]))
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        advantages = torch.tensor([[1.0, -1.0]])

        correction = grpo_utils.compute_rho_correction(
            vllm_logprobs, vllm_logprobs, new_logprobs, response_mask, advantages, config
        )

        torch.testing.assert_close(correction.weights, torch.ones(1, 2))

    def test_negative_advantage_masked_when_ratio_below_one(self):
        config = _make_dppo_config()
        # μ = 0.5, π_θ = 0.1 → TV = 0.4 > δ; ratio < 1, so A<0 is the unsafe direction.
        vllm_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        new_logprobs = torch.log(torch.tensor([[0.1, 0.1]]))
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        advantages = torch.tensor([[-1.0, 1.0]])

        correction = grpo_utils.compute_rho_correction(
            vllm_logprobs, vllm_logprobs, new_logprobs, response_mask, advantages, config
        )

        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 1.0]]))
        torch.testing.assert_close(correction.metrics["val/rho_drop_low_frac"], torch.tensor([[1.0, 0.0]]))

    def test_padding_positions_zeroed(self):
        config = _make_dppo_config()
        vllm_logprobs = torch.log(torch.tensor([[0.1, 0.1, 0.1]]))
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
        response_mask = torch.tensor([[False, True, False]])
        advantages = torch.full((1, 3), -1.0)

        correction = grpo_utils.compute_rho_correction(
            vllm_logprobs, vllm_logprobs, new_logprobs, response_mask, advantages, config
        )

        # A<0 with π_θ>μ is the safe direction, so the middle token is kept; padding is 0.
        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 1.0, 0.0]]))

    def test_mask_does_not_propagate_gradients(self):
        config = _make_dppo_config()
        vllm_logprobs = torch.log(torch.tensor([[0.1]]))
        new_logprobs = torch.log(torch.tensor([[0.5]])).requires_grad_(True)
        response_mask = torch.ones(1, 1, dtype=torch.bool)
        advantages = torch.ones(1, 1)

        correction = grpo_utils.compute_rho_correction(
            vllm_logprobs, vllm_logprobs, new_logprobs, response_mask, advantages, config
        )

        self.assertFalse(correction.weights.requires_grad)

    def test_composes_with_rho_correction(self):
        # With use_rho_correction, kept tokens are still reweighted by ρ = π_old / μ
        # while the DPPO mask drops unsafe out-of-region tokens.
        config = _make_dppo_config(use_rho_correction=True)
        vllm_logprobs = torch.log(torch.tensor([[0.1, 0.1]]))
        old_logprobs = vllm_logprobs + torch.log(torch.tensor(2.0))
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        advantages = torch.tensor([[1.0, -1.0]])

        correction = grpo_utils.compute_rho_correction(
            old_logprobs, vllm_logprobs, new_logprobs, response_mask, advantages, config
        )

        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 2.0]]))


class TestDPPOLoss(unittest.TestCase):
    def test_dppo_loss_masks_and_has_no_symmetric_clip(self):
        config = _make_dppo_config(loss_fn=grpo_utils.GRPOLossType.dppo, rho_mask_upper_bound=0.1)
        vllm_logprobs = torch.log(torch.tensor([[0.1, 0.1]]))
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        advantages = torch.tensor([[1.0, -1.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=vllm_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=response_mask,
            config=config,
        )

        # ratio = π_θ / μ = 5 for both tokens; only the A>0 token is masked (Eq. 12),
        # and the kept A<0 token is unclipped despite ratio 5 (Eq. 11).
        expected_mask = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(output.pg_loss, -advantages * output.ratio * expected_mask)
        torch.testing.assert_close(output.clipfrac, torch.zeros_like(output.clipfrac))
        torch.testing.assert_close(output.kl, torch.zeros_like(output.kl))


class TestDPPOConfigValidation(unittest.TestCase):
    def test_dppo_loss_requires_dppo_divergence_type(self):
        with self.assertRaisesRegex(ValueError, "rho_divergence_type"):
            grpo_utils.GRPOExperimentConfig(loss_fn=grpo_utils.GRPOLossType.dppo)

    def test_dppo_divergence_requires_threshold(self):
        with self.assertRaisesRegex(ValueError, "rho_mask_upper_bound"):
            grpo_utils.GRPOExperimentConfig(rho_divergence_type=grpo_utils.RhoDivergenceType.dppo_tv)

    def test_dppo_divergence_requires_rollout_anchoring(self):
        with self.assertRaisesRegex(ValueError, "rollout policy"):
            grpo_utils.GRPOExperimentConfig(
                rho_divergence_type=grpo_utils.RhoDivergenceType.dppo_tv,
                rho_mask_upper_bound=0.1,
                use_rho_correction=False,
            )

    def test_dppo_with_rho_correction_is_valid(self):
        grpo_utils.GRPOExperimentConfig(
            loss_fn=grpo_utils.GRPOLossType.dppo,
            rho_divergence_type=grpo_utils.RhoDivergenceType.dppo_kl,
            rho_mask_upper_bound=0.1,
        )

    def test_dppo_with_vllm_logprobs_is_valid(self):
        grpo_utils.GRPOExperimentConfig(
            loss_fn=grpo_utils.GRPOLossType.dppo,
            rho_divergence_type=grpo_utils.RhoDivergenceType.dppo_tv,
            rho_mask_upper_bound=0.1,
            use_rho_correction=False,
            use_vllm_logprobs=True,
        )

    def test_tv_divergence_allows_sub_one_threshold(self):
        grpo_utils.GRPOExperimentConfig(rho_divergence_type=grpo_utils.RhoDivergenceType.tv, rho_mask_upper_bound=0.5)


if __name__ == "__main__":
    unittest.main()
