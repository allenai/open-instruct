import unittest
from unittest.mock import MagicMock

import torch
import transformers
from parameterized import parameterized

from open_instruct import data_types, grpo_utils
from open_instruct.distillkit.losses import forward_kl_topk_from_logprobs
from open_instruct.rl_utils import masked_mean
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

    def test_forward_for_logprobs_and_topk(self):
        batch_size, seq_len, vocab_size, topk = 2, 5, 10, 3
        model = _make_mock_model(vocab_size, seq_len, batch_size)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        model.return_value = logits
        query_responses = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        topk_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len - 1, topk))

        sampled_logprobs, entropy, topk_logprobs = grpo_utils.forward_for_logprobs_and_topk(
            model,
            query_responses,
            attention_mask,
            position_ids,
            pad_token_id=0,
            temperature=1.0,
            return_entropy=False,
            topk_token_ids=topk_token_ids,
        )

        expected_log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        expected_sampled = torch.gather(
            expected_log_probs, dim=-1, index=query_responses[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        expected_topk = torch.gather(expected_log_probs, dim=-1, index=topk_token_ids)

        torch.testing.assert_close(sampled_logprobs, expected_sampled)
        self.assertIsNone(entropy)
        assert topk_logprobs is not None
        torch.testing.assert_close(topk_logprobs, expected_topk)

    def test_forward_for_logprobs_and_topk_uses_raw_logits_for_topk(self):
        batch_size, seq_len, vocab_size = 1, 3, 4
        model = _make_mock_model(vocab_size, seq_len, batch_size)
        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 0.0, -1.0], [0.5, 0.0, -0.5, -1.0]]])
        model.return_value = logits
        query_responses = torch.tensor([[0, 3, 1]])
        attention_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        topk_token_ids = torch.tensor([[[0, 3], [1, 2]]])

        sampled_logprobs, _, topk_logprobs = grpo_utils.forward_for_logprobs_and_topk(
            model,
            query_responses,
            attention_mask,
            position_ids,
            pad_token_id=99,
            temperature=0.5,
            return_entropy=False,
            topk_token_ids=topk_token_ids,
        )

        raw_log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        policy_log_probs = torch.log_softmax((logits[:, :-1] / 0.5).float(), dim=-1)
        expected_sampled = torch.gather(policy_log_probs, dim=-1, index=query_responses[:, 1:].unsqueeze(-1)).squeeze(
            -1
        )
        expected_topk_raw = torch.gather(raw_log_probs, dim=-1, index=topk_token_ids)
        expected_topk_policy = torch.gather(policy_log_probs, dim=-1, index=topk_token_ids)

        torch.testing.assert_close(sampled_logprobs, expected_sampled)
        assert topk_logprobs is not None
        torch.testing.assert_close(topk_logprobs, expected_topk_raw)
        self.assertFalse(torch.allclose(topk_logprobs, expected_topk_policy))


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
        "rho_mask_tv_divergence": False,
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

        pg_loss, clipfrac, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=torch.ones_like(ratio),
        )

        self.assertEqual(pg_loss.shape, (batch_size, seq_len))
        self.assertEqual(clipfrac.shape, (batch_size, seq_len))
        self.assertEqual(kl.shape, (batch_size, seq_len))

    def test_dapo_clipping(self):
        config = _make_grpo_config(clip_lower=0.2, clip_higher=0.2)
        ratio = torch.tensor([[1.5, 0.5, 1.0]])
        new_logprobs = torch.randn(1, 3)
        advantages = torch.ones(1, 3)

        pg_loss, clipfrac, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=torch.ones_like(ratio),
        )

        expected_clamped = torch.clamp(ratio, 0.8, 1.2)
        expected_unclipped = -advantages * ratio
        expected_clipped = -advantages * expected_clamped
        torch.testing.assert_close(pg_loss, torch.max(expected_unclipped, expected_clipped))
        torch.testing.assert_close(clipfrac, (expected_clipped > expected_unclipped).float())

    def test_cispo_uses_detached_ratio(self):
        config = _make_grpo_config(loss_fn=grpo_utils.GRPOLossType.cispo, clip_higher=0.2)
        ratio = torch.tensor([[1.5, 0.5, 1.0]], requires_grad=True)
        new_logprobs = torch.randn(1, 3, requires_grad=True)
        advantages = torch.ones(1, 3)

        pg_loss, clipfrac, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=torch.ones_like(ratio),
        )

        pg_loss.sum().backward()
        self.assertIsNone(ratio.grad)
        self.assertIsNotNone(new_logprobs.grad)
        torch.testing.assert_close(clipfrac, torch.tensor([[1.0, 0.0, 0.0]]))

    def test_with_ref_logprobs(self):
        config = _make_grpo_config(beta=0.05, kl_estimator=2)
        batch_size, seq_len = 2, 4
        new_logprobs = torch.randn(batch_size, seq_len)
        ratio = torch.exp(torch.randn(batch_size, seq_len))
        advantages = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)

        _, _, kl = grpo_utils.compute_grpo_loss(
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

        _, _, kl = grpo_utils.compute_grpo_loss(
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

        pg_no_rho, clipfrac_no_rho, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=torch.ones_like(new_logprobs),
        )

        pg_rho, clipfrac_rho, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=rho_weights,
        )

        torch.testing.assert_close(pg_rho, pg_no_rho * 2.0)
        torch.testing.assert_close(clipfrac_rho, clipfrac_no_rho)

    def test_rho_mask(self):
        config = _make_grpo_config(use_rho_correction=True, rho_mask_lower_bound=0.5, rho_mask_upper_bound=2.0)
        response_mask = torch.tensor([[True, True, True, True, True]])
        # ρ values: 0.25 (drop, < lower=0.5), 0.5 (keep), 1.0 (keep), 2.0 (keep), 4.0 (drop, > upper=2.0).
        # In-range tokens are reweighted by ρ, not gated to 1.
        old_logprob = torch.log(torch.tensor([[0.25, 0.5, 1.0, 2.0, 4.0]]))
        vllm_logprobs = torch.zeros_like(old_logprob)
        advantages = torch.ones_like(old_logprob)
        correction = grpo_utils.compute_rho_correction(old_logprob, vllm_logprobs, response_mask, advantages, config)
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
            old_logprob, vllm_logprobs, response_mask_with_pad, advantages, config
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
        correction = grpo_utils.compute_rho_correction(old_logprob, vllm_logprobs, response_mask, advantages, config)
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
            use_rho_correction=True, rho_mask_lower_bound=0.0, rho_mask_upper_bound=2.0, rho_mask_tv_divergence=True
        )
        # Row 0: mean |ρ - 1| = 1.25, below upper=2.0, so it is kept.
        # Row 1: mean |ρ - 1| = 3.0, above upper=2.0, so TV-increasing tokens are dropped.
        old_logprob = torch.log(torch.tensor([[0.25, 1.0, 4.0], [4.0, 4.0, 4.0]]))
        vllm_logprobs = torch.zeros_like(old_logprob)
        response_mask = torch.ones_like(old_logprob, dtype=torch.bool)
        advantages = torch.ones_like(old_logprob)

        correction = grpo_utils.compute_rho_correction(old_logprob, vllm_logprobs, response_mask, advantages, config)

        torch.testing.assert_close(correction.weights, torch.tensor([[0.25, 1.0, 4.0], [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(
            correction.metrics["val/rho_drop_high_frac"], torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        )

    def test_rho_mask_zeroes_loss(self):
        config = _make_grpo_config()
        new_logprobs = torch.randn(1, 3)
        ratio = torch.exp(torch.randn(1, 3))
        advantages = torch.randn(1, 3)
        rho_mask = torch.tensor([[1.0, 0.0, 1.0]])

        pg_loss, clipfrac, _ = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=None,
            config=config,
            rho_weights=rho_mask,
        )
        self.assertEqual(pg_loss[0, 1].item(), 0.0)
        self.assertEqual(clipfrac[0, 1].item(), 0.0)
        self.assertNotEqual(pg_loss[0, 0].item(), 0.0)

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


_OPD_MODEL_NAME = "EleutherAI/pythia-14m"


class TestOPDLearnerLossEndToEnd(unittest.TestCase):
    """End-to-end OPD learner path on a real model.

    The unit tests above use a mocked model. This exercises the actual learner
    OPD computation that ``GRPOTrainModule.train_batch`` performs, against a real
    transformer forward, with real backprop:

      forward -> raw (T=1) top-k gather -> forward_kl_topk -> masked_mean -> backward

    It mirrors ``train_batch``'s OPD block (the ``[:, 1:, :]`` teacher slice, the
    sampled-token-in-top-k metric, and the masked-mean reduction) so a regression
    in that assembly or in the temperature handling is caught without standing up
    the full OLMo-core Trainer. Runs on GPU when available, otherwise CPU.
    """

    K = 4
    # Teacher top-k probabilities for response positions (mass < 1 on purpose, so
    # teacher_topk_mass is a meaningful diagnostic rather than trivially 1.0).
    TEACHER_PROBS = [0.5, 0.3, 0.1, 0.05]

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(_OPD_MODEL_NAME)
        cls.model = transformers.AutoModelForCausalLM.from_pretrained(_OPD_MODEL_NAME).to(cls.device)
        cls.pad_token_id = cls.tokenizer.eos_token_id

    def _build_batch(self):
        query = self.tokenizer.encode("The capital of France is")
        response = self.tokenizer.encode(" Paris")
        self.assertGreater(len(query), 0)
        self.assertGreater(len(response), 0)

        qr = query + response
        seq_len = len(qr)
        vocab_size = self.model.config.vocab_size
        query_responses = torch.tensor([qr], dtype=torch.long, device=self.device)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.device)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        # Response positions are the slots holding response tokens.
        response_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=self.device)
        response_mask[0, len(query) :] = True

        # Teacher tensors are [B, T, K]: at full-tensor position j the teacher row is
        # the distribution that *predicts* token j (the same convention pack_sequences
        # uses). Query/sentinel positions get id 0 and -inf logprobs (contribute zero).
        teacher_ids = torch.zeros(1, seq_len, self.K, dtype=torch.long, device=self.device)
        teacher_logprobs = torch.full((1, seq_len, self.K), float("-inf"), device=self.device)
        resp_lp = torch.log(torch.tensor(self.TEACHER_PROBS, device=self.device))
        for pos in range(len(query), seq_len):
            predicted = qr[pos]
            # Put the actually-realized token first so sampled_token_in_topk is exercised.
            ids = [predicted] + [(predicted + offset) % vocab_size for offset in (1, 2, 3)]
            teacher_ids[0, pos] = torch.tensor(ids[: self.K], dtype=torch.long, device=self.device)
            teacher_logprobs[0, pos] = resp_lp

        return query_responses, attention_mask, position_ids, response_mask, teacher_ids, teacher_logprobs

    def test_opd_loss_backprops_through_real_model(self):
        qr, attn, pos, response_mask, teacher_ids_full, teacher_lp_full = self._build_batch()

        # Mirror train_batch: teacher tensors and labels are shifted by one.
        teacher_ids = teacher_ids_full[:, 1:, :]
        teacher_lp = teacher_lp_full[:, 1:, :]
        response_mask_shifted = response_mask[:, 1:]

        self.model.zero_grad(set_to_none=True)
        new_logprobs, _, student_topk_logprobs = grpo_utils.forward_for_logprobs_and_topk(
            self.model,
            qr,
            attn,
            pos,
            self.pad_token_id,
            temperature=0.7,
            return_entropy=False,
            pass_olmo_core_doc_lens=False,
            topk_token_ids=teacher_ids,
        )
        self.assertEqual(student_topk_logprobs.shape, teacher_ids.shape)

        opd_output = forward_kl_topk_from_logprobs(student_topk_logprobs, teacher_lp)
        opd_loss = masked_mean(opd_output.loss, response_mask_shifted)

        # Loss is a finite scalar that still carries gradient to the model.
        self.assertEqual(opd_loss.shape, ())
        self.assertTrue(torch.isfinite(opd_loss))
        self.assertTrue(opd_loss.requires_grad)

        opd_loss.backward()
        embed_grad = self.model.get_input_embeddings().weight.grad
        self.assertIsNotNone(embed_grad, "OPD loss did not backprop into the model embeddings")
        self.assertTrue(torch.isfinite(embed_grad).all())
        self.assertGreater(embed_grad.abs().sum().item(), 0.0)

        # teacher_topk_mass is a real (0, 1] diagnostic on response positions.
        mass = masked_mean(opd_output.teacher_topk_mass, response_mask_shifted)
        self.assertGreater(mass.item(), 0.0)
        self.assertLessEqual(mass.item(), 1.0 + 1e-5)

        # sampled_token_in_topk: we seeded the realized token into the teacher top-k,
        # so it must be 1.0 averaged over response positions.
        sampled_tokens = qr[:, 1:].unsqueeze(-1)
        sampled_in_topk = (teacher_ids == sampled_tokens).any(dim=-1).float()
        self.assertAlmostEqual(masked_mean(sampled_in_topk, response_mask_shifted).item(), 1.0, places=5)

    def test_opd_topk_logprobs_are_temperature_invariant(self):
        """The student OPD view uses raw T=1 logits, so it must not move with rollout temperature,
        while the GRPO sampled-token logprobs must."""
        qr, attn, pos, _, teacher_ids_full, _ = self._build_batch()
        teacher_ids = teacher_ids_full[:, 1:, :]

        with torch.no_grad():
            sampled_a, _, topk_a = grpo_utils.forward_for_logprobs_and_topk(
                self.model, qr, attn, pos, self.pad_token_id, temperature=1.0, topk_token_ids=teacher_ids
            )
            sampled_b, _, topk_b = grpo_utils.forward_for_logprobs_and_topk(
                self.model, qr, attn, pos, self.pad_token_id, temperature=0.5, topk_token_ids=teacher_ids
            )

        # OPD top-k logprobs come from raw logits -> identical regardless of temperature.
        torch.testing.assert_close(topk_a, topk_b)
        # GRPO sampled-token logprobs are temperature-scaled -> must differ.
        self.assertFalse(torch.allclose(sampled_a, sampled_b))


if __name__ == "__main__":
    unittest.main()
