import unittest
from types import SimpleNamespace
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


class _TinyBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 4):
        super().__init__()
        self.model = _TinyBackbone(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden_states))


class TestForwardForLigerGRPOLoss(unittest.TestCase):
    def test_returns_hidden_states_and_shifted_labels(self):
        model = _TinyCausalLM()
        query_responses = torch.tensor([[1, 2, 0, 3]])
        attention_mask = torch.ones_like(query_responses)
        position_ids = torch.arange(query_responses.shape[1]).unsqueeze(0)

        output = grpo_utils.forward_for_liger_grpo_loss(
            model, query_responses, attention_mask, position_ids, pad_token_id=0
        )

        self.assertEqual(output.hidden_states.shape, (1, 3, 4))
        self.assertIs(output.lm_head_weight, model.lm_head.weight)
        self.assertIsNone(output.lm_head_bias)
        torch.testing.assert_close(output.selected_token_ids, torch.tensor([[2, 0, 3]]))

    def test_liger_fp32_casts_hidden_states(self):
        model = _TinyCausalLM().to(torch.bfloat16)
        query_responses = torch.tensor([[1, 2, 0, 3]])
        attention_mask = torch.ones_like(query_responses)
        position_ids = torch.arange(query_responses.shape[1]).unsqueeze(0)

        output = grpo_utils.forward_for_liger_grpo_loss(
            model, query_responses, attention_mask, position_ids, pad_token_id=0, lm_head_fp32=True
        )

        self.assertEqual(output.hidden_states.dtype, torch.float32)
        self.assertEqual(output.lm_head_weight.dtype, torch.bfloat16)

    def test_chunked_lm_head_logprobs_match_full_logits(self):
        model = _TinyCausalLM(vocab_size=11, hidden_size=5)
        query_responses = torch.tensor([[1, 2, 0, 3], [4, 5, 6, 7]])
        attention_mask = torch.ones_like(query_responses)
        position_ids = torch.arange(query_responses.shape[1]).unsqueeze(0).expand_as(query_responses)

        full_logprobs, _ = grpo_utils.forward_for_logprobs(
            model, query_responses, attention_mask, position_ids, pad_token_id=0, temperature=1.3
        )
        chunked_logprobs = grpo_utils.forward_for_chunked_lm_head_logprobs(
            model,
            query_responses,
            attention_mask,
            position_ids,
            pad_token_id=0,
            temperature=1.3,
            lm_head_chunk_size=2,
        )

        torch.testing.assert_close(chunked_logprobs, full_logprobs)


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
        "dppo_divergence_type": grpo_utils.DPPODivergenceType.tv,
        "dppo_divergence_threshold": 0.02,
        "tvpo_divergence_threshold": 0.02,
        "tvpo_truncation_cap": 20.0,
    }
    defaults.update(kwargs)
    config = MagicMock(spec=grpo_utils.GRPOExperimentConfig)
    for key, value in defaults.items():
        setattr(config, key, value)
    return config


class TestComputeTISMask(unittest.TestCase):
    def test_upper_bound_is_absolute_ratio(self):
        ratios = torch.tensor([[0.49, 0.5, 1.0, 1.99, 2.0, 3.0]])
        new_logprobs = torch.log(ratios)
        vllm_logprobs = torch.zeros_like(new_logprobs)
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        mask = grpo_utils.compute_tis_mask(
            new_logprobs, vllm_logprobs, response_mask, lower_bound=0.5, upper_bound=2.0
        )

        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
        torch.testing.assert_close(mask, expected)


class TestComputeGRPOLoss(unittest.TestCase):
    @parameterized.expand([("dapo", grpo_utils.GRPOLossType.dapo), ("cispo", grpo_utils.GRPOLossType.cispo)])
    def test_output_shapes(self, _name, loss_type):
        batch_size, seq_len = 2, 4
        config = _make_grpo_config(loss_fn=loss_type)
        new_logprobs = torch.randn(batch_size, seq_len)
        ratio = torch.exp(torch.randn(batch_size, seq_len))
        advantages = torch.randn(batch_size, seq_len)

        pg_losses, pg_losses2, pg_loss_max, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs, ratio=ratio, advantages=advantages, ref_logprobs=None, config=config
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
            new_logprobs=new_logprobs, ratio=ratio, advantages=advantages, ref_logprobs=None, config=config
        )

        expected_clamped = torch.clamp(ratio, 0.8, 1.2)
        torch.testing.assert_close(pg_losses2, -advantages * expected_clamped)

    def test_cispo_uses_detached_ratio(self):
        config = _make_grpo_config(loss_fn=grpo_utils.GRPOLossType.cispo, clip_higher=0.2)
        ratio = torch.tensor([[1.5, 0.5, 1.0]], requires_grad=True)
        new_logprobs = torch.randn(1, 3, requires_grad=True)
        advantages = torch.ones(1, 3)

        pg_losses, pg_losses2, pg_loss_max, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs, ratio=ratio, advantages=advantages, ref_logprobs=None, config=config
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
            new_logprobs=new_logprobs, ratio=ratio, advantages=advantages, ref_logprobs=ref_logprobs, config=config
        )

        self.assertFalse(torch.all(kl == 0))

    def test_without_ref_logprobs(self):
        config = _make_grpo_config()
        new_logprobs = torch.randn(2, 4)
        ratio = torch.exp(torch.randn(2, 4))
        advantages = torch.randn(2, 4)

        _, _, _, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs, ratio=ratio, advantages=advantages, ref_logprobs=None, config=config
        )

        torch.testing.assert_close(kl, torch.zeros_like(kl))

    def test_tis_weights(self):
        config = _make_grpo_config()
        new_logprobs = torch.randn(2, 4)
        ratio = torch.exp(torch.randn(2, 4))
        advantages = torch.randn(2, 4)
        tis_weights = torch.full((2, 4), 2.0)

        pg_no_tis, pg2_no_tis, _, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            tis_weights=None,
        )

        pg_tis, pg2_tis, _, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            tis_weights=tis_weights,
        )

        torch.testing.assert_close(pg_tis, pg_no_tis * 2.0)
        torch.testing.assert_close(pg2_tis, pg2_no_tis * 2.0)

    def test_invalid_loss_fn(self):
        config = _make_grpo_config(loss_fn="invalid")
        with self.assertRaises(ValueError):
            grpo_utils.compute_grpo_loss(
                new_logprobs=torch.randn(2, 4),
                ratio=torch.exp(torch.randn(2, 4)),
                advantages=torch.randn(2, 4),
                ref_logprobs=None,
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
        # Three tokens with varying behavior/policy probabilities.
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
        # Only middle token is a real response position.
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

        # All response positions are unsafe (A<0 and r>1 -> safe direction
        # actually); but masking is by trust region only. r>1 with A<0 means
        # we're moving back towards 1, which is the "safe" direction, so no
        # mask. Padding positions are always 0 via response_mask.
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
    def test_liger_grpo_loss_rejects_dppo(self):
        with self.assertRaisesRegex(ValueError, "only supports"):
            grpo_utils.GRPOExperimentConfig(
                use_liger_grpo_loss=True,
                loss_fn=grpo_utils.GRPOLossType.dppo,
                use_vllm_logprobs=True,
                truncated_importance_sampling_ratio_cap=0.0,
            )

    def test_liger_grpo_loss_rejects_non_default_kl_estimator(self):
        with self.assertRaisesRegex(ValueError, "kl_estimator=2"):
            grpo_utils.GRPOExperimentConfig(use_liger_grpo_loss=True, load_ref_policy=True, beta=0.1, kl_estimator=1)

    def test_liger_grpo_loss_allows_lm_head_fp32(self):
        config = grpo_utils.GRPOExperimentConfig(use_liger_grpo_loss=True, lm_head_fp32=True)

        self.assertTrue(config.use_liger_grpo_loss)
        self.assertTrue(config.lm_head_fp32)

    def test_dppo_loss_matches_masked_reinforce(self):
        config = _make_grpo_config(loss_fn=grpo_utils.GRPOLossType.dppo)
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        ratio = torch.tensor([[2.0, 0.5]])
        advantages = torch.tensor([[1.0, -1.0]])
        # Mask token 0 only.
        tis_weights = torch.tensor([[0.0, 1.0]])

        pg_losses, pg_losses2, pg_loss_max, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            tis_weights=tis_weights,
        )

        # DPPO has no symmetric clipping, so pg_losses == pg_losses2.
        torch.testing.assert_close(pg_losses, pg_losses2)
        # Eq. 11: -A * r * M.
        expected = torch.tensor([[-0.0, 0.5]])
        torch.testing.assert_close(pg_loss_max, expected)

    def test_dppo_threshold_validation(self):
        with self.assertRaises(ValueError):
            grpo_utils.GRPOExperimentConfig(
                loss_fn=grpo_utils.GRPOLossType.dppo,
                dppo_divergence_threshold=0.0,
                use_vllm_logprobs=True,
                truncated_importance_sampling_ratio_cap=0.0,
            )

    def test_dppo_requires_use_vllm_logprobs(self):
        with self.assertRaisesRegex(ValueError, "use_vllm_logprobs"):
            grpo_utils.GRPOExperimentConfig(
                loss_fn=grpo_utils.GRPOLossType.dppo,
                use_vllm_logprobs=False,
                truncated_importance_sampling_ratio_cap=0.0,
            )


class TestComputeTVPOMask(unittest.TestCase):
    def test_in_budget_returns_all_ones(self):
        # ratio = 1 everywhere → per-token TV contribution is 0 → all prompts
        # in budget regardless of advantages, so mask is the response_mask.
        new_logprobs = torch.log(torch.tensor([[0.4, 0.4, 0.4, 0.4]]))
        ratio = torch.ones_like(new_logprobs)
        advantages = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        mask, prompt_tv = grpo_utils.compute_tvpo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=new_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_threshold=0.01,
            rollout_ids=torch.tensor([[0, 0, 1, 1]]),
            num_samples_per_prompt=2,
        )

        torch.testing.assert_close(mask, torch.ones_like(mask))
        torch.testing.assert_close(prompt_tv, torch.zeros_like(prompt_tv))

    def test_short_circuit_skips_directional_filter(self):
        # Even with adversarial advantage/sign-of-shift combos, when every
        # prompt is in budget the mask is all-ones (no directional gating).
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        behavior_logprobs = torch.log(torch.tensor([[0.49, 0.51]]))
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        advantages = torch.tensor([[1.0, -1.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        mask, _ = grpo_utils.compute_tvpo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_threshold=0.5,
            rollout_ids=torch.tensor([[0, 1]]),
            num_samples_per_prompt=2,
        )

        torch.testing.assert_close(mask, torch.ones_like(mask))

    def test_over_budget_directional_release_valve(self):
        # Drive a single prompt well over budget (ratio = 5), then check that
        # only tokens whose gradient *reduces* TV survive: A·sign(p−p_old) ≤ 0.
        # token 0: A=+1, p>p_old → product>0 → BLOCK
        # token 1: A=-1, p>p_old → product<0 → KEEP
        # token 2: A=+1, p<p_old → product<0 → KEEP
        # token 3: A=-1, p<p_old → product>0 → BLOCK
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5, 0.1, 0.1]]))
        behavior_logprobs = torch.log(torch.tensor([[0.1, 0.1, 0.5, 0.5]]))
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        advantages = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        mask, prompt_tv = grpo_utils.compute_tvpo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_threshold=0.01,
            rollout_ids=torch.tensor([[0, 0, 0, 0]]),
            num_samples_per_prompt=1,
        )

        torch.testing.assert_close(mask, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))
        # All tokens belong to the same prompt, so prompt_tv is constant and >> 0.
        self.assertTrue(torch.all(prompt_tv[response_mask] > 0.01))

    def test_in_budget_prompt_keeps_all_its_tokens_when_other_prompt_over(self):
        # Two prompts. Prompt 0 (rollouts 0,1) is in budget — its per-token TV
        # contribution |1.005 - 1|/2 = 0.0025 averaged across rollouts is below
        # δ=0.01 — and so its tokens all keep regardless of direction. Prompt 1
        # (rollouts 2,3) has ratio 5.0 (TV=2.0) and is well over budget, so the
        # directional gate fires only there.
        in_budget = torch.full((1, 4), 1.005)
        over_budget = torch.full((1, 4), 5.0)
        ratio = torch.cat([in_budget, over_budget], dim=1)
        # Sign of p − p_old is positive everywhere (π = 0.4 > μ = 0.1).
        new_logprobs = torch.log(torch.full_like(ratio, 0.4))
        behavior_logprobs = torch.log(torch.full_like(ratio, 0.1))
        advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        rollout_ids = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])

        mask, _ = grpo_utils.compute_tvpo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_threshold=0.01,
            rollout_ids=rollout_ids,
            num_samples_per_prompt=2,
        )

        # Prompt 0 (tokens 0..3) is in budget → fully kept.
        # Prompt 1 (tokens 4..7) over budget: A>0,sign>0 → block; A<0,sign>0 → keep.
        expected = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]])
        torch.testing.assert_close(mask, expected)

    def test_response_mask_zeroes_padding(self):
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
        behavior_logprobs = torch.log(torch.tensor([[0.1, 0.1, 0.1]]))
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        advantages = torch.tensor([[1.0, 1.0, 1.0]])
        response_mask = torch.tensor([[False, True, False]])

        mask, _ = grpo_utils.compute_tvpo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_threshold=0.01,
            rollout_ids=torch.tensor([[0, 0, 0]]),
            num_samples_per_prompt=1,
        )

        # Padding positions are always 0; the lone valid position has A>0 and
        # p>p_old so it's blocked under the directional gate.
        torch.testing.assert_close(mask, torch.tensor([[0.0, 0.0, 0.0]]))

    def test_no_rollout_ids_falls_back_to_per_microbatch_tv(self):
        # Without rollout_ids, every token is treated as one prompt with one
        # rollout: per_sample_tv = mean(|r-1|)/2 averaged over the whole pack.
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
        behavior_logprobs = torch.log(torch.tensor([[0.1, 0.1, 0.1]]))
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        advantages = torch.tensor([[1.0, -1.0, 0.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        mask, prompt_tv = grpo_utils.compute_tvpo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_threshold=0.01,
        )

        # All response tokens share the same per-microbatch prompt_tv.
        expected_tv = 0.5 * float((ratio - 1.0).abs().mean())
        torch.testing.assert_close(prompt_tv, torch.full_like(prompt_tv, expected_tv))
        # Over budget; A>0 with p>p_old → block; A<0 → keep; A=0 → keep.
        torch.testing.assert_close(mask, torch.tensor([[0.0, 1.0, 1.0]]))

    def test_mask_does_not_propagate_gradients(self):
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]])).requires_grad_(True)
        behavior_logprobs = torch.log(torch.tensor([[0.1, 0.1]]))
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        advantages = torch.tensor([[1.0, 1.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        mask, _ = grpo_utils.compute_tvpo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_threshold=0.01,
            rollout_ids=torch.tensor([[0, 0]]),
            num_samples_per_prompt=1,
        )

        self.assertFalse(mask.requires_grad)


class TestTVPOLoss(unittest.TestCase):
    def test_tvpo_uses_truncated_detached_ratio(self):
        # Surrogate: -A · clamp(r, max=cap).detach() · log π. With r ≤ cap and
        # no mask, gradient flows only through new_logprobs (log π), not ratio.
        config = _make_grpo_config(loss_fn=grpo_utils.GRPOLossType.tvpo, tvpo_truncation_cap=20.0)
        ratio = torch.tensor([[1.5, 0.5, 1.0]], requires_grad=True)
        new_logprobs = torch.log(torch.tensor([[0.4, 0.5, 0.6]])).requires_grad_(True)
        advantages = torch.ones(1, 3)

        pg_losses, pg_losses2, pg_loss_max, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs, ratio=ratio, advantages=advantages, ref_logprobs=None, config=config
        )

        # No symmetric clipping for TVPO.
        torch.testing.assert_close(pg_losses, pg_losses2)
        # Forward value: -A · trunc(r) · log π.
        expected = -advantages * torch.clamp(ratio.detach(), max=20.0) * new_logprobs
        torch.testing.assert_close(pg_loss_max, expected)

        # Backward only flows through log π, not r (ratio is detached in surrogate).
        pg_loss_max.sum().backward()
        self.assertIsNone(ratio.grad)
        self.assertIsNotNone(new_logprobs.grad)

    def test_tvpo_truncation_caps_extreme_ratios(self):
        config = _make_grpo_config(loss_fn=grpo_utils.GRPOLossType.tvpo, tvpo_truncation_cap=2.0)
        ratio = torch.tensor([[10.0, 1.0]])
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        advantages = torch.ones(1, 2)

        _, _, pg_loss_max, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs, ratio=ratio, advantages=advantages, ref_logprobs=None, config=config
        )

        expected = -advantages * torch.clamp(ratio, max=2.0) * new_logprobs
        torch.testing.assert_close(pg_loss_max, expected)

    def test_tvpo_freeze_mask_zeroes_gradient_but_keeps_value(self):
        config = _make_grpo_config(loss_fn=grpo_utils.GRPOLossType.tvpo)
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]])).requires_grad_(True)
        ratio = torch.tensor([[2.0, 0.5]])
        advantages = torch.tensor([[1.0, -1.0]])
        # Freeze token 0's gradient, keep token 1.
        freeze_mask = torch.tensor([[0.0, 1.0]])

        _, _, pg_loss_max, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            policy_freeze_mask=freeze_mask,
        )

        # Loss VALUE preserved everywhere (surrogate would otherwise be 0 at
        # mask=0 if we used multiplicative masking, but TVPO uses where()).
        expected = (
            -advantages
            * torch.clamp(ratio, max=config.tvpo_truncation_cap).detach()
            * torch.log(torch.tensor([[0.5, 0.5]]))
        )
        torch.testing.assert_close(pg_loss_max, expected)

        pg_loss_max.sum().backward()
        # Gradient should be zero for the masked-out token.
        assert new_logprobs.grad is not None
        self.assertEqual(float(new_logprobs.grad[0, 0]), 0.0)
        self.assertNotEqual(float(new_logprobs.grad[0, 1]), 0.0)

    def test_tvpo_threshold_validation(self):
        with self.assertRaises(ValueError):
            grpo_utils.GRPOExperimentConfig(
                loss_fn=grpo_utils.GRPOLossType.tvpo,
                tvpo_divergence_threshold=0.0,
                use_vllm_logprobs=True,
                truncated_importance_sampling_ratio_cap=0.0,
            )

    def test_tvpo_truncation_cap_validation(self):
        with self.assertRaises(ValueError):
            grpo_utils.GRPOExperimentConfig(
                loss_fn=grpo_utils.GRPOLossType.tvpo,
                tvpo_truncation_cap=0.0,
                use_vllm_logprobs=True,
                truncated_importance_sampling_ratio_cap=0.0,
            )

    def test_tvpo_requires_use_vllm_logprobs(self):
        with self.assertRaisesRegex(ValueError, "use_vllm_logprobs"):
            grpo_utils.GRPOExperimentConfig(
                loss_fn=grpo_utils.GRPOLossType.tvpo,
                use_vllm_logprobs=False,
                truncated_importance_sampling_ratio_cap=0.0,
            )


if __name__ == "__main__":
    unittest.main()
