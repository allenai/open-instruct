import unittest
from unittest.mock import MagicMock, patch

import torch
from parameterized import parameterized

from open_instruct import data_types, grpo_utils, model_utils
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


class TestForwardExtraKwargs(unittest.TestCase):
    def test_no_packing_preserves_attention_mask(self):
        attention_mask = torch.ones(1, 4)
        position_ids = torch.arange(4).unsqueeze(0)

        returned_mask, extra_kwargs = grpo_utils._compute_forward_extra_kwargs(position_ids, attention_mask)

        self.assertIs(returned_mask, attention_mask)
        self.assertEqual(extra_kwargs, {})

    def test_cp_context_forces_packed_kwargs_without_local_reset(self):
        attention_mask = torch.ones(1, 4)
        position_ids = torch.tensor([[8, 9, 10, 11]])
        cp_context = object()

        returned_mask, extra_kwargs = grpo_utils._compute_forward_extra_kwargs(
            position_ids, attention_mask, cp_context=cp_context
        )

        self.assertIsNone(returned_mask)
        self.assertIs(extra_kwargs["cp_context"], cp_context)
        torch.testing.assert_close(extra_kwargs["seq_idx"], torch.tensor([[0, 0, 0, 0]], dtype=torch.int32))
        torch.testing.assert_close(extra_kwargs["cu_seqlens"], torch.tensor([0, 4], dtype=torch.int32))

    def test_packed_kwargs_include_leading_local_continuation(self):
        position_ids = torch.tensor([[8, 9, 0, 1]])

        extra_kwargs = grpo_utils._compute_packing_kwargs(position_ids)

        torch.testing.assert_close(extra_kwargs["seq_idx"], torch.tensor([[0, 0, 1, 1]], dtype=torch.int32))
        torch.testing.assert_close(extra_kwargs["cu_seqlens"], torch.tensor([0, 2, 4], dtype=torch.int32))


class TestTiledGRPOLMHeadLoss(unittest.TestCase):
    def test_matches_dense_dapo_loss_and_grads(self):
        torch.manual_seed(0)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 4, 11
        lm_head_dense = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled.load_state_dict(lm_head_dense.state_dict())

        hidden_dense = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        hidden_tiled = hidden_dense.detach().clone().requires_grad_(True)
        selected_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        response_mask = torch.tensor([[True, True, False, True, False], [False, True, True, False, True]])
        advantages = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)
        beta = 0.05
        clip_lower = 0.2
        clip_higher = 0.28
        loss_scale = torch.tensor(0.75)

        logits = lm_head_dense(hidden_dense)
        new_logprobs = model_utils.log_softmax_and_gather(logits, selected_token_ids)
        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_losses, pg_losses2, pg_loss, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            config=_make_grpo_config(beta=beta, clip_lower=clip_lower, clip_higher=clip_higher),
        )
        dense_loss = ((pg_loss + beta * kl) * response_mask).sum() / response_mask.sum() * loss_scale
        dense_loss.backward()

        tiled_loss, tiled_kl, tiled_clipfrac, tiled_ratio = grpo_utils.tiled_grpo_lm_head_loss(
            lm_head=lm_head_tiled,
            hidden_states=hidden_tiled,
            selected_token_ids=selected_token_ids,
            response_mask=response_mask,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            temperature=1.0,
            beta=beta,
            clip_lower=clip_lower,
            clip_higher=clip_higher,
            shards=3,
            loss_scale=loss_scale,
        )
        tiled_loss.backward()

        torch.testing.assert_close(tiled_loss, dense_loss.detach())
        expected_kl = model_utils.estimate_kl((new_logprobs - ref_logprobs).clamp(-40.0, 40.0), ratio)
        expected_kl = (expected_kl * response_mask).sum(dim=(-2, -1)) / response_mask.sum()
        torch.testing.assert_close(tiled_kl, expected_kl)
        expected_clipfrac = ((pg_losses2 > pg_losses).float() * response_mask).sum() / response_mask.sum()
        torch.testing.assert_close(tiled_clipfrac, expected_clipfrac)
        torch.testing.assert_close(tiled_ratio, (ratio * response_mask).sum() / response_mask.sum())
        torch.testing.assert_close(hidden_tiled.grad, hidden_dense.grad)
        torch.testing.assert_close(lm_head_tiled.weight.grad, lm_head_dense.weight.grad)
        torch.testing.assert_close(lm_head_tiled.bias.grad, lm_head_dense.bias.grad)

    def test_matches_dense_cispo_loss_and_grads_with_policy_mask(self):
        torch.manual_seed(7)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 4, 11
        lm_head_dense = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled.load_state_dict(lm_head_dense.state_dict())

        hidden_dense = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        hidden_tiled = hidden_dense.detach().clone().requires_grad_(True)
        selected_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        response_mask = torch.tensor([[True, True, False, True, False], [False, True, True, False, True]])
        policy_mask = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0]])
        advantages = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)
        beta = 0.05
        clip_higher = 0.28
        loss_scale = torch.tensor(0.75)

        logits = lm_head_dense(hidden_dense)
        new_logprobs = model_utils.log_softmax_and_gather(logits, selected_token_ids)
        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_losses, pg_losses2, pg_loss, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            config=_make_grpo_config(beta=beta, clip_higher=clip_higher, loss_fn=grpo_utils.GRPOLossType.cispo),
            tis_weights=policy_mask,
        )
        dense_loss = ((pg_loss + beta * kl) * response_mask).sum() / response_mask.sum() * loss_scale
        dense_loss.backward()

        tiled_loss, tiled_kl, tiled_clipfrac, tiled_ratio = grpo_utils.tiled_grpo_lm_head_loss(
            lm_head=lm_head_tiled,
            hidden_states=hidden_tiled,
            selected_token_ids=selected_token_ids,
            response_mask=response_mask,
            policy_mask=policy_mask,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            temperature=1.0,
            beta=beta,
            clip_lower=0.2,
            clip_higher=clip_higher,
            shards=3,
            loss_scale=loss_scale,
            loss_fn=grpo_utils.GRPOLossType.cispo,
        )
        tiled_loss.backward()

        torch.testing.assert_close(tiled_loss, dense_loss.detach())
        expected_kl = model_utils.estimate_kl((new_logprobs - ref_logprobs).clamp(-40.0, 40.0), ratio)
        expected_kl = (expected_kl * response_mask).sum(dim=(-2, -1)) / response_mask.sum()
        torch.testing.assert_close(tiled_kl, expected_kl)
        expected_clipfrac = ((pg_losses2 > pg_losses).float() * response_mask).sum() / response_mask.sum()
        torch.testing.assert_close(tiled_clipfrac, expected_clipfrac)
        torch.testing.assert_close(tiled_ratio, (ratio * response_mask).sum() / response_mask.sum())
        torch.testing.assert_close(hidden_tiled.grad, hidden_dense.grad)
        torch.testing.assert_close(lm_head_tiled.weight.grad, lm_head_dense.weight.grad)
        torch.testing.assert_close(lm_head_tiled.bias.grad, lm_head_dense.bias.grad)

    def test_matches_dense_dppo_loss_and_grads(self):
        torch.manual_seed(5)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 4, 11
        lm_head_dense = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled.load_state_dict(lm_head_dense.state_dict())

        hidden_dense = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        hidden_tiled = hidden_dense.detach().clone().requires_grad_(True)
        selected_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        response_mask = torch.tensor([[True, True, False, True, False], [False, True, True, False, True]])
        advantages = torch.randn(batch_size, seq_len)
        old_logprobs = -torch.rand(batch_size, seq_len) * 3.0
        ref_logprobs = -torch.rand(batch_size, seq_len) * 3.0
        beta = 0.05
        divergence_threshold = 0.02
        loss_scale = torch.tensor(0.75)

        logits = lm_head_dense(hidden_dense)
        new_logprobs = model_utils.log_softmax_and_gather(logits, selected_token_ids)
        ratio = torch.exp(new_logprobs - old_logprobs)
        dppo_mask, _ = grpo_utils.compute_dppo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=old_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_type=grpo_utils.DPPODivergenceType.tv,
            divergence_threshold=divergence_threshold,
        )
        pg_losses, pg_losses2, pg_loss, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            config=_make_grpo_config(
                beta=beta, loss_fn=grpo_utils.GRPOLossType.dppo, dppo_divergence_threshold=divergence_threshold
            ),
            tis_weights=dppo_mask,
        )
        dense_loss = ((pg_loss + beta * kl) * response_mask).sum() / response_mask.sum() * loss_scale
        dense_loss.backward()

        tiled_loss, tiled_kl, tiled_clipfrac, tiled_ratio = grpo_utils.tiled_grpo_lm_head_loss(
            lm_head=lm_head_tiled,
            hidden_states=hidden_tiled,
            selected_token_ids=selected_token_ids,
            response_mask=response_mask,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            temperature=1.0,
            beta=beta,
            clip_lower=0.2,
            clip_higher=0.28,
            shards=3,
            loss_scale=loss_scale,
            loss_fn=grpo_utils.GRPOLossType.dppo,
            dppo_divergence_type=grpo_utils.DPPODivergenceType.tv,
            dppo_divergence_threshold=divergence_threshold,
        )
        tiled_loss.backward()

        torch.testing.assert_close(tiled_loss, dense_loss.detach())
        expected_kl = model_utils.estimate_kl((new_logprobs - ref_logprobs).clamp(-40.0, 40.0), ratio)
        expected_kl = (expected_kl * response_mask).sum(dim=(-2, -1)) / response_mask.sum()
        torch.testing.assert_close(tiled_kl, expected_kl)
        expected_clipfrac = ((pg_losses2 > pg_losses).float() * response_mask).sum() / response_mask.sum()
        torch.testing.assert_close(tiled_clipfrac, expected_clipfrac)
        torch.testing.assert_close(tiled_ratio, (ratio * response_mask).sum() / response_mask.sum())
        torch.testing.assert_close(hidden_tiled.grad, hidden_dense.grad)
        torch.testing.assert_close(lm_head_tiled.weight.grad, lm_head_dense.weight.grad)
        torch.testing.assert_close(lm_head_tiled.bias.grad, lm_head_dense.bias.grad)

    def test_matches_dense_tvpo_loss_and_grads(self):
        torch.manual_seed(6)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 4, 11
        lm_head_dense = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled.load_state_dict(lm_head_dense.state_dict())

        hidden_dense = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        hidden_tiled = hidden_dense.detach().clone().requires_grad_(True)
        selected_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        response_mask = torch.tensor([[True, True, False, True, False], [False, True, True, False, True]])
        advantages = torch.randn(batch_size, seq_len)
        old_logprobs = -torch.rand(batch_size, seq_len) * 3.0
        ref_logprobs = -torch.rand(batch_size, seq_len) * 3.0
        beta = 0.05
        tvpo_truncation_cap = 2.0
        loss_scale = torch.tensor(0.75)

        logits = lm_head_dense(hidden_dense)
        new_logprobs = model_utils.log_softmax_and_gather(logits, selected_token_ids)
        ratio = torch.exp(new_logprobs - old_logprobs)
        tvpo_mask, _ = grpo_utils.compute_tvpo_mask(
            new_logprobs=new_logprobs,
            behavior_logprobs=old_logprobs,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_threshold=0.01,
            rollout_ids=torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]),
            num_samples_per_prompt=2,
        )
        pg_losses, pg_losses2, pg_loss, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            config=_make_grpo_config(
                beta=beta, loss_fn=grpo_utils.GRPOLossType.tvpo, tvpo_truncation_cap=tvpo_truncation_cap
            ),
            policy_freeze_mask=tvpo_mask,
        )
        dense_loss = ((pg_loss + beta * kl) * response_mask).sum() / response_mask.sum() * loss_scale
        dense_loss.backward()

        tiled_loss, tiled_kl, tiled_clipfrac, tiled_ratio = grpo_utils.tiled_grpo_lm_head_loss(
            lm_head=lm_head_tiled,
            hidden_states=hidden_tiled,
            selected_token_ids=selected_token_ids,
            response_mask=response_mask,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            temperature=1.0,
            beta=beta,
            clip_lower=0.2,
            clip_higher=0.28,
            shards=3,
            loss_scale=loss_scale,
            loss_fn=grpo_utils.GRPOLossType.tvpo,
            tvpo_truncation_cap=tvpo_truncation_cap,
            policy_freeze_mask=tvpo_mask,
        )
        tiled_loss.backward()

        torch.testing.assert_close(tiled_loss, dense_loss.detach())
        expected_kl = model_utils.estimate_kl((new_logprobs - ref_logprobs).clamp(-40.0, 40.0), ratio)
        expected_kl = (expected_kl * response_mask).sum(dim=(-2, -1)) / response_mask.sum()
        torch.testing.assert_close(tiled_kl, expected_kl)
        expected_clipfrac = ((pg_losses2 > pg_losses).float() * response_mask).sum() / response_mask.sum()
        torch.testing.assert_close(tiled_clipfrac, expected_clipfrac)
        torch.testing.assert_close(tiled_ratio, (ratio * response_mask).sum() / response_mask.sum())
        torch.testing.assert_close(hidden_tiled.grad, hidden_dense.grad)
        torch.testing.assert_close(lm_head_tiled.weight.grad, lm_head_dense.weight.grad)
        torch.testing.assert_close(lm_head_tiled.bias.grad, lm_head_dense.bias.grad)

    def test_returns_kl_metrics_when_beta_is_zero(self):
        torch.manual_seed(1)
        batch_size, seq_len, hidden_size, vocab_size = 1, 4, 3, 7
        lm_head = torch.nn.Linear(hidden_size, vocab_size)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        selected_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        response_mask = torch.tensor([[True, True, False, True]])
        advantages = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)

        logits = lm_head(hidden_states)
        new_logprobs = model_utils.log_softmax_and_gather(logits, selected_token_ids)
        ratio = torch.exp(new_logprobs - old_logprobs)
        expected_kl = model_utils.estimate_kl((new_logprobs - ref_logprobs).clamp(-40.0, 40.0), ratio)
        expected_kl = (expected_kl * response_mask).sum(dim=(-2, -1)) / response_mask.sum()

        _, tiled_kl, _, _ = grpo_utils.tiled_grpo_lm_head_loss(
            lm_head=lm_head,
            hidden_states=hidden_states.detach().clone().requires_grad_(True),
            selected_token_ids=selected_token_ids,
            response_mask=response_mask,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            temperature=1.0,
            beta=0.0,
            clip_lower=0.2,
            clip_higher=0.28,
            shards=2,
            loss_scale=torch.tensor(1.0),
        )

        torch.testing.assert_close(tiled_kl, expected_kl)

    def test_matches_dense_sequence_weighted_loss_and_grads(self):
        torch.manual_seed(4)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 4, 11
        lm_head_dense = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled = torch.nn.Linear(hidden_size, vocab_size)
        lm_head_tiled.load_state_dict(lm_head_dense.state_dict())

        hidden_dense = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        hidden_tiled = hidden_dense.detach().clone().requires_grad_(True)
        selected_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        response_mask = torch.tensor([[True, True, False, False, False], [False, True, True, True, True]])
        advantages = torch.randn(batch_size, seq_len)
        old_logprobs = torch.randn(batch_size, seq_len)
        ref_logprobs = torch.randn(batch_size, seq_len)
        beta = 0.05
        clip_lower = 0.2
        clip_higher = 0.28
        loss_scale = torch.tensor(0.75)

        logits = lm_head_dense(hidden_dense)
        new_logprobs = model_utils.log_softmax_and_gather(logits, selected_token_ids)
        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_losses, pg_losses2, pg_loss, kl = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            ratio=ratio,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
            config=_make_grpo_config(beta=beta, clip_lower=clip_lower, clip_higher=clip_higher),
        )
        dense_loss = (
            grpo_utils.sequence_weighted_mean(pg_loss + beta * kl, response_mask, denominator=2.0) * loss_scale
        )
        dense_loss.backward()

        tiled_loss, tiled_kl, tiled_clipfrac, tiled_ratio = grpo_utils.tiled_grpo_lm_head_loss(
            lm_head=lm_head_tiled,
            hidden_states=hidden_tiled,
            selected_token_ids=selected_token_ids,
            response_mask=response_mask,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            temperature=1.0,
            beta=beta,
            clip_lower=clip_lower,
            clip_higher=clip_higher,
            shards=3,
            loss_scale=loss_scale,
            loss_denominator="sequence",
        )
        tiled_loss.backward()

        torch.testing.assert_close(tiled_loss, dense_loss.detach())
        expected_kl = model_utils.estimate_kl((new_logprobs - ref_logprobs).clamp(-40.0, 40.0), ratio)
        expected_kl = (expected_kl * response_mask).sum(dim=(-2, -1)) / response_mask.sum()
        torch.testing.assert_close(tiled_kl, expected_kl)
        expected_clipfrac = ((pg_losses2 > pg_losses).float() * response_mask).sum() / response_mask.sum()
        torch.testing.assert_close(tiled_clipfrac, expected_clipfrac)
        torch.testing.assert_close(tiled_ratio, (ratio * response_mask).sum() / response_mask.sum())
        torch.testing.assert_close(hidden_tiled.grad, hidden_dense.grad)
        torch.testing.assert_close(lm_head_tiled.weight.grad, lm_head_dense.weight.grad)
        torch.testing.assert_close(lm_head_tiled.bias.grad, lm_head_dense.bias.grad)


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

    def test_sequence_weighted_mean_averages_rows_equally(self):
        per_token_loss = torch.tensor([[1.0, 2.0, 100.0], [4.0, 6.0, 8.0]])
        response_mask = torch.tensor([[True, True, False], [True, True, True]])

        loss = grpo_utils.sequence_weighted_mean(per_token_loss, response_mask, denominator=2.0)

        torch.testing.assert_close(loss, torch.tensor(3.75))

    def test_sequence_weighted_mean_uses_rollout_sample_ids_for_packed_rows(self):
        per_token_loss = torch.tensor([[1.0, 2.0, 10.0, 20.0, 100.0]])
        response_mask = torch.tensor([[True, True, True, True, False]])
        rollout_sample_ids = torch.tensor([[3, 3, 7, 7, -1]])

        loss = grpo_utils.sequence_weighted_mean(
            per_token_loss, response_mask, denominator=2.0, rollout_sample_ids=rollout_sample_ids
        )

        torch.testing.assert_close(loss, torch.tensor(8.25))

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


class TestLigerGRPOLossConfig(unittest.TestCase):
    def test_liger_grpo_loss_rejects_unsupported_loss_fn(self):
        with self.assertRaisesRegex(ValueError, "loss_fn=dapo.*loss_fn=cispo.*loss_fn=dppo.*loss_fn=tvpo"):
            grpo_utils.GRPOExperimentConfig(
                use_liger_grpo_loss=True,
                loss_fn="not_a_loss",
                use_vllm_logprobs=True,
                truncated_importance_sampling_ratio_cap=0.0,
            )

    def test_liger_grpo_loss_accepts_trainer_logprobs(self):
        config = grpo_utils.GRPOExperimentConfig(use_liger_grpo_loss=True, use_vllm_logprobs=False)

        self.assertTrue(config.use_liger_grpo_loss)
        self.assertFalse(config.use_vllm_logprobs)

    def test_liger_grpo_loss_accepts_dapo_with_vllm_logprobs(self):
        config = grpo_utils.GRPOExperimentConfig(
            use_liger_grpo_loss=True,
            loss_fn=grpo_utils.GRPOLossType.dapo,
            use_vllm_logprobs=True,
            truncated_importance_sampling_ratio_cap=0.0,
        )

        self.assertTrue(config.use_liger_grpo_loss)

    def test_liger_grpo_loss_accepts_cispo_with_vllm_logprobs(self):
        config = grpo_utils.GRPOExperimentConfig(
            use_liger_grpo_loss=True,
            loss_fn=grpo_utils.GRPOLossType.cispo,
            use_vllm_logprobs=True,
            truncated_importance_sampling_ratio_cap=0.0,
        )

        self.assertTrue(config.use_liger_grpo_loss)

    def test_liger_grpo_loss_accepts_dppo_with_vllm_logprobs(self):
        config = grpo_utils.GRPOExperimentConfig(
            use_liger_grpo_loss=True,
            loss_fn=grpo_utils.GRPOLossType.dppo,
            use_vllm_logprobs=True,
            truncated_importance_sampling_ratio_cap=0.0,
        )

        self.assertTrue(config.use_liger_grpo_loss)

    def test_liger_grpo_loss_accepts_tvpo_with_vllm_logprobs(self):
        config = grpo_utils.GRPOExperimentConfig(
            use_liger_grpo_loss=True,
            loss_fn=grpo_utils.GRPOLossType.tvpo,
            use_vllm_logprobs=True,
            truncated_importance_sampling_ratio_cap=0.0,
        )

        self.assertTrue(config.use_liger_grpo_loss)

    def test_liger_grpo_loss_accepts_sequence_loss_denominator(self):
        config = grpo_utils.GRPOExperimentConfig(
            use_liger_grpo_loss=True,
            loss_denominator="sequence",
            use_vllm_logprobs=True,
            truncated_importance_sampling_ratio_cap=0.0,
        )

        self.assertEqual(config.loss_denominator, "sequence")

    def test_liger_grpo_loss_accepts_token_tis_mask(self):
        config = grpo_utils.GRPOExperimentConfig(
            use_liger_grpo_loss=True,
            loss_fn=grpo_utils.GRPOLossType.cispo,
            use_vllm_logprobs=True,
            truncated_importance_sampling_ratio_cap=0.0,
            tis_mask_lower=0.5,
            tis_mask_upper=2.0,
        )

        self.assertEqual(config.tis_mask_lower, 0.5)
        self.assertEqual(config.tis_mask_upper, 2.0)


class TestGRPOLossDenominatorConfig(unittest.TestCase):
    def test_accepts_sequence_loss_denominator(self):
        config = grpo_utils.GRPOExperimentConfig(loss_denominator="sequence")

        self.assertEqual(config.loss_denominator, "sequence")

    def test_sequence_loss_denominator_accepts_sequence_parallelism(self):
        config = grpo_utils.GRPOExperimentConfig(
            loss_denominator="sequence", sequence_parallel_size=2, deepspeed_stage=3
        )

        self.assertEqual(config.loss_denominator, "sequence")
        self.assertEqual(config.sequence_parallel_size, 2)


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

    def test_sequence_process_group_aggregates_rollout_tv(self):
        new_logprobs = torch.log(torch.tensor([[0.11]]))
        behavior_logprobs = torch.log(torch.tensor([[0.1]]))
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        advantages = torch.tensor([[1.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        process_group = object()
        float_calls = 0

        def fake_all_reduce(tensor, op=None, group=None):
            nonlocal float_calls
            self.assertIs(group, process_group)
            if tensor.dtype == torch.long:
                tensor.fill_(0)
                return
            float_calls += 1
            if float_calls == 1:
                tensor.fill_(2.05)
            else:
                tensor.fill_(2.0)

        with patch.object(grpo_utils.dist, "all_reduce", side_effect=fake_all_reduce):
            mask, prompt_tv = grpo_utils.compute_tvpo_mask(
                new_logprobs=new_logprobs,
                behavior_logprobs=behavior_logprobs,
                advantages=advantages,
                ratio=ratio,
                response_mask=response_mask,
                divergence_threshold=0.5,
                rollout_ids=torch.tensor([[0]]),
                num_samples_per_prompt=1,
                sequence_process_group=process_group,
            )

        torch.testing.assert_close(prompt_tv, torch.tensor([[1.025]]))
        torch.testing.assert_close(mask, torch.zeros_like(mask))


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
