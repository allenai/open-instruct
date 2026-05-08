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
        response_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        response_mask[:, :2] = 0
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
        "use_rho_correction": False,
    }
    defaults.update(kwargs)
    config = MagicMock(spec=grpo_utils.GRPOExperimentConfig)
    for key, value in defaults.items():
        setattr(config, key, value)
    return config


class TestComputeGRPOLoss(unittest.TestCase):
    @parameterized.expand([("dapo", grpo_utils.GRPOLossType.dapo), ("cispo", grpo_utils.GRPOLossType.cispo)])
    def test_output_shapes(self, _name, loss_type):
        batch_size, seq_len = 2, 4
        config = _make_grpo_config(loss_fn=loss_type)
        new_logprobs = torch.randn(batch_size, seq_len)
        ratio = torch.exp(torch.randn(batch_size, seq_len))
        advantages = torch.randn(batch_size, seq_len)

        pg_losses, pg_losses2, pg_loss_max, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=torch.ones_like(ratio),
        )

        self.assertEqual(pg_losses.shape, (batch_size, seq_len))
        self.assertEqual(pg_losses2.shape, (batch_size, seq_len))
        self.assertEqual(pg_loss_max.shape, (batch_size, seq_len))
        self.assertEqual(kl.shape, (batch_size, seq_len))

    def test_dapo_clipping(self):
        config = _make_grpo_config(clip_lower=0.2, clip_higher=0.2)
        ratio = torch.tensor([[1.5, 0.5, 1.0]])
        new_logprobs = torch.randn(1, 3)
        advantages = torch.ones(1, 3)

        pg_losses, pg_losses2, pg_loss_max, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=torch.ones_like(ratio),
        )

        expected_clamped = torch.clamp(ratio, 0.8, 1.2)
        torch.testing.assert_close(pg_losses2, -advantages * expected_clamped)

    def test_cispo_uses_detached_ratio(self):
        config = _make_grpo_config(loss_fn=grpo_utils.GRPOLossType.cispo, clip_higher=0.2)
        ratio = torch.tensor([[1.5, 0.5, 1.0]], requires_grad=True)
        new_logprobs = torch.randn(1, 3, requires_grad=True)
        advantages = torch.ones(1, 3)

        pg_losses, pg_losses2, pg_loss_max, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=torch.ones_like(ratio),
        )

        pg_loss_max.sum().backward()
        self.assertIsNone(ratio.grad)
        self.assertIsNotNone(new_logprobs.grad)

    def test_with_ref_logprobs(self):
        config = _make_grpo_config(beta=0.05, kl_estimator=2)
        batch_size, seq_len = 2, 4
        new_logprobs = torch.randn(batch_size, seq_len)
        ratio = torch.exp(torch.randn(batch_size, seq_len))
        advantages = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)

        _, _, _, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            config=config,
            rho_weights=torch.ones_like(ratio),
        )

        self.assertFalse(torch.all(kl == 0))

    def test_without_ref_logprobs(self):
        config = _make_grpo_config()
        new_logprobs = torch.randn(2, 4)
        ratio = torch.exp(torch.randn(2, 4))
        advantages = torch.randn(2, 4)

        _, _, _, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=torch.ones_like(ratio),
        )

        torch.testing.assert_close(kl, torch.zeros_like(kl))

    def test_rho_weights(self):
        config = _make_grpo_config()
        new_logprobs = torch.randn(2, 4)
        ratio = torch.exp(torch.randn(2, 4))
        advantages = torch.randn(2, 4)
        rho_weights = torch.full((2, 4), 2.0)

        pg_no_rho, pg2_no_rho, _, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=torch.ones_like(new_logprobs),
        )

        pg_rho, pg2_rho, _, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=rho_weights,
        )

        torch.testing.assert_close(pg_rho, pg_no_rho * 2.0)
        torch.testing.assert_close(pg2_rho, pg2_no_rho * 2.0)

    def test_rho_mask(self):
        config = _make_grpo_config(use_rho_correction=True, rho_mask_lower_bound=0.5, rho_mask_upper_bound=2.0)
        response_mask = torch.tensor([[True, True, True, True, True]])
        # ρ values: 0.25 (drop, < lower=0.5), 0.5 (keep), 1.0 (keep), 2.0 (keep), 4.0 (drop, > upper=2.0).
        # In-range tokens are reweighted by ρ, not gated to 1.
        old_logprob = torch.log(torch.tensor([[0.25, 0.5, 1.0, 2.0, 4.0]]))
        vllm_logprobs = torch.zeros_like(old_logprob)
        correction = grpo_utils.compute_rho_correction(old_logprob, vllm_logprobs, response_mask, config)
        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 0.5, 1.0, 2.0, 0.0]]))
        torch.testing.assert_close(
            correction.metrics["val/rho_drop_low_frac"], torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        )
        torch.testing.assert_close(
            correction.metrics["val/rho_drop_high_frac"], torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        )

        # Padding tokens (response_mask=False) should always be 0 / not counted as dropped.
        response_mask_with_pad = torch.tensor([[False, True, True, True, False]])
        correction_pad = grpo_utils.compute_rho_correction(old_logprob, vllm_logprobs, response_mask_with_pad, config)
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
        correction = grpo_utils.compute_rho_correction(old_logprob, vllm_logprobs, response_mask, config)
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

    def test_rho_mask_zeroes_loss(self):
        config = _make_grpo_config()
        new_logprobs = torch.randn(1, 3)
        ratio = torch.exp(torch.randn(1, 3))
        advantages = torch.randn(1, 3)
        rho_mask = torch.tensor([[1.0, 0.0, 1.0]])

        pg_losses, pg_losses2, _, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=rho_mask,
        )
        self.assertEqual(pg_losses[0, 1].item(), 0.0)
        self.assertEqual(pg_losses2[0, 1].item(), 0.0)
        self.assertNotEqual(pg_losses[0, 0].item(), 0.0)

    def test_invalid_loss_fn(self):
        config = _make_grpo_config(loss_fn="invalid")
        with self.assertRaises(ValueError):
            grpo_utils.compute_grpo_loss(
                new_logprobs=torch.randn(2, 4),
                ratio=torch.exp(torch.randn(2, 4)),
                advantages=torch.randn(2, 4),
                ref_logprobs=None,
                config=config,
                rho_weights=torch.ones(2, 4),
            )


if __name__ == "__main__":
    unittest.main()
