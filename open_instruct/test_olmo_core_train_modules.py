import dataclasses
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


def _make_grpo_config(**kwargs) -> grpo_utils.GRPOExperimentConfig:
    defaults = {
        "clip_lower": 0.2,
        "clip_higher": 0.2,
        "beta": 0.0,
        "kl_estimator": 2,
        "loss_fn": grpo_utils.GRPOLossType.dapo,
        "load_ref_policy": False,
        "mask_reference_kl_with_policy": False,
        "policy_ratio_denominator": "rollout_policy",
        "rollout_importance_correction": "none",
        "rho_clamp_lower_bound": 0.0,
        "rho_clamp_upper_bound": 0.0,
        "rho_mask_lower_bound": 0.0,
        "rho_mask_upper_bound": 0.0,
        "rho_mask_metric": "ratio",
        "rho_mask_source": "current_policy",
        "rho_mask_level": "token",
        "rho_mask_direction": "symmetric",
    }
    defaults.update(kwargs)
    return grpo_utils.GRPOExperimentConfig(**defaults)


def _make_dppo_config(**kwargs) -> grpo_utils.GRPOExperimentConfig:
    defaults = {
        "loss_fn": grpo_utils.GRPOLossType.dppo,
        "policy_ratio_denominator": "rollout_policy",
        "rollout_importance_correction": "none",
        "rho_mask_metric": "tv",
        "rho_mask_source": "current_policy",
        "rho_mask_level": "token",
        "rho_mask_direction": "increase_only",
        "rho_mask_upper_bound": 0.05,
        "rho_mask_lower_bound": 0.0,
    }
    defaults.update(kwargs)
    return _make_grpo_config(**defaults)


def _logprobs_for_rho(rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vllm_logprobs = torch.full_like(rho, -30.0)
    new_logprobs = vllm_logprobs + torch.log(rho)
    return vllm_logprobs, new_logprobs


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
        config = (
            _make_dppo_config() if loss_type == grpo_utils.GRPOLossType.dppo else _make_grpo_config(loss_fn=loss_type)
        )
        new_logprobs = -torch.rand(batch_size, seq_len)
        vllm_logprobs = -torch.rand(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            config=config,
        )

        self.assertEqual(output.pg_loss.shape, (batch_size, seq_len))
        self.assertEqual(output.clipfrac.shape, (batch_size, seq_len))
        self.assertEqual(output.kl.shape, (batch_size, seq_len))
        self.assertEqual(output.ratio.shape, (batch_size, seq_len))
        self.assertEqual(output.rho.mask.shape, (batch_size, seq_len))
        self.assertEqual(output.kl_mask.shape, (batch_size, seq_len))

    def test_default_preserves_ppo_ratio_with_clipped_rollout_correction(self):
        config = grpo_utils.GRPOExperimentConfig(load_ref_policy=False, beta=0.0)
        vllm_logprobs = torch.log(torch.tensor([[0.1]]))
        old_logprobs = torch.log(torch.tensor([[0.4]]))
        new_logprobs = old_logprobs.clone().requires_grad_(True)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=torch.ones_like(new_logprobs),
            ref_logprobs=None,
            response_mask=torch.ones_like(new_logprobs, dtype=torch.bool),
            config=config,
        )

        torch.testing.assert_close(output.ratio, torch.ones_like(output.ratio))
        torch.testing.assert_close(output.rho.rho, torch.full_like(output.rho.rho, 4.0))
        torch.testing.assert_close(output.rho.weights, torch.full_like(output.rho.weights, 2.0))
        output.pg_loss.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, torch.full_like(new_logprobs, -2.0))

    def test_disabling_rollout_correction_keeps_the_ppo_ratio(self):
        config = grpo_utils.GRPOExperimentConfig(
            load_ref_policy=False, beta=0.0, rollout_importance_correction="none", rho_mask_metric="none"
        )
        vllm_logprobs = torch.log(torch.tensor([[0.01]]))
        old_logprobs = torch.log(torch.tensor([[0.2]]))
        new_logprobs = torch.log(torch.tensor([[0.3]])).requires_grad_(True)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=-torch.ones_like(new_logprobs),
            ref_logprobs=None,
            response_mask=torch.ones_like(new_logprobs, dtype=torch.bool),
            config=config,
        )

        torch.testing.assert_close(output.ratio, torch.full_like(output.ratio, 1.5))
        torch.testing.assert_close(output.rho.rho, torch.full_like(output.rho.rho, 20.0))
        torch.testing.assert_close(output.rho.weights, torch.full_like(output.rho.weights, 1.5))
        output.pg_loss.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, torch.full_like(new_logprobs, 1.5))

    def test_old_logprobs_match_the_selected_denominator(self):
        shape = (1, 1)
        common = {
            "new_logprobs": torch.full(shape, -1.0),
            "vllm_logprobs": torch.full(shape, -2.0),
            "advantages": torch.ones(shape),
            "ref_logprobs": None,
            "response_mask": torch.ones(shape, dtype=torch.bool),
        }
        with self.assertRaisesRegex(ValueError, "old_logprobs is required"):
            grpo_utils.compute_grpo_loss(**common, config=_make_grpo_config(policy_ratio_denominator="old_policy"))
        with self.assertRaisesRegex(ValueError, "old_logprobs must be omitted"):
            grpo_utils.compute_grpo_loss(**common, old_logprobs=torch.full(shape, -1.5), config=_make_grpo_config())

    def test_old_logprobs_are_lazily_fixed_for_single_minibatch(self):
        cache: list[torch.Tensor | None] = [None]
        first = torch.tensor([[-1.0]], requires_grad=True)
        resolved = grpo_utils.resolve_old_logprobs(cache, 0, 0, 1, first)
        later = grpo_utils.resolve_old_logprobs(cache, 0, 1, 1, torch.tensor([[-2.0]]))

        self.assertIs(resolved, later)
        self.assertFalse(resolved.requires_grad)
        torch.testing.assert_close(resolved, first.detach())

    def test_objective_is_detached_rho_times_advantage_times_training_logprob(self):
        rho = torch.tensor([[2.0, 0.5]])
        vllm_logprobs, new_values = _logprobs_for_rho(rho)
        new_logprobs = new_values.requires_grad_(True)
        vllm_logprobs.requires_grad_(True)
        advantages = torch.tensor([[3.0, -4.0]], requires_grad=True)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=torch.ones_like(new_logprobs, dtype=torch.bool),
            config=_make_dppo_config(),
        )

        torch.testing.assert_close(output.rho.rho, rho)
        torch.testing.assert_close(output.pg_loss, -rho * advantages.detach() * new_logprobs.detach())
        output.pg_loss.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, -rho * advantages.detach())
        self.assertIsNone(vllm_logprobs.grad)
        self.assertIsNone(advantages.grad)

    def test_rho_is_always_present_when_optional_correction_is_disabled(self):
        rho = torch.tensor([[3.0]])
        vllm_logprobs, new_values = _logprobs_for_rho(rho)
        new_logprobs = new_values.requires_grad_(True)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=torch.ones_like(new_logprobs),
            ref_logprobs=None,
            response_mask=torch.ones_like(new_logprobs, dtype=torch.bool),
            config=_make_dppo_config(),
        )

        torch.testing.assert_close(output.rho.weights, rho)
        output.pg_loss.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, -rho)

    def test_rho_is_not_arbitrarily_clamped_in_log_space(self):
        rho = torch.tensor([[torch.exp(torch.tensor(20.0))]])
        vllm_logprobs, new_logprobs = _logprobs_for_rho(rho)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=torch.ones_like(rho),
            ref_logprobs=None,
            response_mask=torch.ones_like(rho, dtype=torch.bool),
            config=_make_dppo_config(),
        )

        torch.testing.assert_close(output.rho.rho, rho, rtol=1e-5, atol=0)

    def test_ratio_mask_drops_overflow_before_ratio_and_kl_autograd(self):
        new_logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=torch.tensor([[-101.0, -2.0]]),
            advantages=torch.tensor([[-1.0, 1.0]]),
            ref_logprobs=torch.tensor([[-2.0, -3.0]]),
            response_mask=torch.ones(1, 2, dtype=torch.bool),
            config=_make_grpo_config(loss_fn=grpo_utils.GRPOLossType.dapo, kl_estimator=3, rho_mask_upper_bound=10.0),
        )

        torch.testing.assert_close(output.rho.mask, torch.tensor([[False, True]]))
        torch.testing.assert_close(output.ratio, torch.tensor([[0.0, 1.0]]))
        torch.testing.assert_close(output.rho.metrics["val/rho_overflow_frac"], torch.tensor([[1.0, 0.0]]))
        histograms: dict[str, list[torch.Tensor]] = {}
        grpo_utils.accumulate_rho_histograms(histograms, output.rho)
        rho_hist = grpo_utils.finalize_rho_histograms(histograms)["val/rho_hist"]
        torch.testing.assert_close(torch.from_numpy(rho_hist), torch.ones(1))
        self.assertTrue(torch.isfinite(output.pg_loss).all())
        self.assertTrue(torch.isfinite(output.kl).all())

        (output.pg_loss + output.kl * 0.05).sum().backward()
        self.assertEqual(new_logprobs.grad[0, 0].item(), 0.0)
        self.assertNotEqual(new_logprobs.grad[0, 1].item(), 0.0)

    def test_retained_overflow_still_fails(self):
        with self.assertRaisesRegex(FloatingPointError, "retained response token"):
            grpo_utils.compute_grpo_loss(
                new_logprobs=torch.tensor([[-1.0]], requires_grad=True),
                vllm_logprobs=torch.tensor([[-101.0]]),
                advantages=torch.tensor([[-1.0]]),
                ref_logprobs=None,
                response_mask=torch.ones(1, 1, dtype=torch.bool),
                config=_make_grpo_config(loss_fn=grpo_utils.GRPOLossType.dapo),
            )

    def test_clipped_rollout_correction_contains_raw_rho_overflow(self):
        config = grpo_utils.GRPOExperimentConfig(load_ref_policy=False, beta=0.0, rho_mask_metric="none")
        new_logprobs = torch.tensor([[-1.0]], requires_grad=True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=torch.tensor([[-1.0]]),
            vllm_logprobs=torch.tensor([[-101.0]]),
            advantages=torch.ones(1, 1),
            ref_logprobs=None,
            response_mask=torch.ones(1, 1, dtype=torch.bool),
            config=config,
        )

        torch.testing.assert_close(output.rho.weights, torch.tensor([[2.0]]))
        self.assertEqual(output.rho.metrics["val/rho_overflow_frac"].item(), 1.0)
        output.pg_loss.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, torch.tensor([[-2.0]]))

    def test_dapo_clipping_is_a_structural_directional_mask(self):
        rho = torch.tensor([[1.5, 0.5, 1.5, 0.5]])
        vllm_logprobs, new_values = _logprobs_for_rho(rho)
        new_logprobs = new_values.requires_grad_(True)
        advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0]])

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=torch.ones_like(new_logprobs, dtype=torch.bool),
            config=_make_grpo_config(loss_fn=grpo_utils.GRPOLossType.dapo),
        )

        torch.testing.assert_close(output.rho.mask, torch.tensor([[False, True, True, False]]))
        output.pg_loss.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, torch.tensor([[0.0, -0.5, 1.5, 0.0]]))

    def test_cispo_caps_the_detached_rho_coefficient(self):
        rho = torch.tensor([[2.0, 1.2, 0.5]])
        vllm_logprobs, new_values = _logprobs_for_rho(rho)
        new_logprobs = new_values.requires_grad_(True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=torch.ones_like(new_logprobs),
            ref_logprobs=None,
            response_mask=torch.ones_like(new_logprobs, dtype=torch.bool),
            config=_make_grpo_config(loss_fn=grpo_utils.GRPOLossType.cispo, clip_higher=0.2),
        )

        torch.testing.assert_close(output.rho.weights, torch.tensor([[1.2, 1.2, 0.5]]))
        output.pg_loss.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, -output.rho.weights)

    def test_rho_mask_sequence_level(self):
        config = _make_grpo_config(
            loss_fn=grpo_utils.GRPOLossType.cispo,
            clip_higher=10.0,
            rho_mask_lower_bound=0.5,
            rho_mask_upper_bound=2.0,
            rho_mask_level="sequence",
        )
        rho = torch.tensor([[0.25, 1.0, 4.0], [4.0, 4.0, 4.0], [0.25, 0.25, 0.25]])
        vllm_logprobs, new_logprobs = _logprobs_for_rho(rho)
        response_mask = torch.ones_like(rho, dtype=torch.bool)

        correction = grpo_utils.compute_rho_correction(
            vllm_logprobs, new_logprobs, response_mask, torch.ones_like(rho), config
        )

        torch.testing.assert_close(
            correction.weights, torch.tensor([[0.25, 1.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )

    def test_sequence_tv_divergence_uses_old_policy(self):
        config = _make_grpo_config(
            policy_ratio_denominator="old_policy",
            rho_mask_source="old_policy",
            rho_mask_upper_bound=0.3,
            rho_mask_metric="tv",
            rho_mask_level="sequence",
            rho_mask_direction="increase_only",
        )
        vllm_logprobs = torch.log(torch.tensor([[0.5] * 3, [0.1] * 3, [0.1] * 3]))
        old_logprobs = torch.log(torch.tensor([[0.4] * 3, [0.6] * 3, [0.6] * 3]))
        advantages = torch.tensor([[1.0] * 3, [1.0] * 3, [-1.0] * 3])

        correction = grpo_utils.compute_rho_correction(
            vllm_logprobs,
            old_logprobs,
            torch.ones_like(old_logprobs, dtype=torch.bool),
            advantages,
            config,
            old_logprobs=old_logprobs,
        )

        torch.testing.assert_close(correction.weights, torch.tensor([[1.0] * 3, [0.0] * 3, [1.0] * 3]))
        torch.testing.assert_close(
            correction.metrics["val/rho_divergence"], torch.tensor([[0.1] * 3, [0.5] * 3, [0.5] * 3])
        )

    def test_sequence_kl_divergence_uses_old_policy(self):
        config = _make_grpo_config(
            policy_ratio_denominator="old_policy",
            rho_mask_source="old_policy",
            rho_mask_upper_bound=1.0,
            rho_mask_metric="kl",
            rho_mask_level="sequence",
            rho_mask_direction="increase_only",
        )
        vllm_logprobs = torch.log(torch.tensor([[0.5, 0.5], [0.1, 0.1]]))
        old_logprobs = torch.log(torch.tensor([[0.5, 0.5], [0.9, 0.9]]))

        correction = grpo_utils.compute_rho_correction(
            vllm_logprobs,
            old_logprobs,
            torch.ones_like(old_logprobs, dtype=torch.bool),
            torch.ones_like(old_logprobs),
            config,
            old_logprobs=old_logprobs,
        )

        torch.testing.assert_close(correction.weights, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
        expected_divergence = 0.8 * torch.log(torch.tensor(9.0))
        torch.testing.assert_close(
            correction.metrics["val/rho_divergence"],
            torch.stack([torch.zeros(2), expected_divergence.expand(2)]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_configured_rho_mask_zeroes_policy_and_reference_kl_gradients(self):
        rho = torch.tensor([[1.0, 4.0, 1.0]])
        vllm_logprobs, new_values = _logprobs_for_rho(rho)
        new_logprobs = new_values.requires_grad_(True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=torch.ones_like(new_logprobs),
            ref_logprobs=torch.full_like(new_logprobs, -31.0),
            response_mask=torch.ones(1, 3, dtype=torch.bool),
            config=_make_grpo_config(
                loss_fn=grpo_utils.GRPOLossType.cispo, rho_mask_upper_bound=2.0, mask_reference_kl_with_policy=True
            ),
        )

        (output.pg_loss + output.kl).sum().backward()
        torch.testing.assert_close(output.rho.mask, torch.tensor([[True, False, True]]))
        self.assertEqual(new_logprobs.grad[0, 1].item(), 0.0)
        self.assertEqual(output.pg_loss[0, 1].item(), 0.0)
        self.assertEqual(output.kl[0, 1].item(), 0.0)

    def test_reference_kl_is_independent_from_policy_mask_by_default(self):
        rho = torch.tensor([[1.0, 4.0]])
        vllm_logprobs, new_values = _logprobs_for_rho(rho)
        new_logprobs = new_values.requires_grad_(True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=torch.ones_like(new_logprobs),
            ref_logprobs=torch.full_like(new_logprobs, -31.0),
            response_mask=torch.ones_like(new_logprobs, dtype=torch.bool),
            config=_make_grpo_config(loss_fn=grpo_utils.GRPOLossType.cispo, rho_mask_upper_bound=2.0, kl_estimator=0),
        )

        torch.testing.assert_close(output.rho.mask, torch.tensor([[True, False]]))
        torch.testing.assert_close(output.kl_mask, torch.tensor([[True, True]]))
        output.pg_loss.sum().backward(retain_graph=True)
        self.assertEqual(new_logprobs.grad[0, 1].item(), 0.0)
        new_logprobs.grad.zero_()
        output.kl.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, torch.ones_like(new_logprobs))

    def test_masked_selected_token_has_zero_gradient_for_every_logit(self):
        logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]], requires_grad=True)
        selected_tokens = torch.tensor([[[0], [1]]])
        new_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, selected_tokens).squeeze(-1)
        vllm_logprobs = new_logprobs.detach() - torch.log(torch.tensor([[1.0, 4.0]]))

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=torch.ones_like(new_logprobs),
            ref_logprobs=torch.full_like(new_logprobs, -3.0),
            response_mask=torch.ones_like(new_logprobs, dtype=torch.bool),
            config=_make_grpo_config(
                loss_fn=grpo_utils.GRPOLossType.cispo, rho_mask_upper_bound=2.0, mask_reference_kl_with_policy=True
            ),
        )

        (output.pg_loss + output.kl).sum().backward()
        self.assertGreater(logits.grad[0, 0].abs().sum().item(), 0.0)
        torch.testing.assert_close(logits.grad[0, 1], torch.zeros(3))

    def test_nonresponse_nonfinite_training_logprob_has_zero_gradient(self):
        new_logprobs = torch.tensor([[-2.0, float("nan")]], requires_grad=True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=torch.full((1, 2), -2.0),
            advantages=torch.ones(1, 2),
            ref_logprobs=torch.full((1, 2), -3.0),
            response_mask=torch.tensor([[True, False]]),
            config=_make_dppo_config(),
        )

        (output.pg_loss + output.kl).sum().backward()
        self.assertEqual(new_logprobs.grad[0, 1].item(), 0.0)
        self.assertTrue(torch.isfinite(output.pg_loss).all())
        self.assertTrue(torch.isfinite(output.kl).all())

    def test_invalid_response_training_logprob_fails_fast(self):
        with self.assertRaisesRegex(FloatingPointError, "new_logprobs"):
            grpo_utils.compute_grpo_loss(
                new_logprobs=torch.tensor([[-1.0, float("nan")]], requires_grad=True),
                vllm_logprobs=torch.full((1, 2), -1.0),
                advantages=torch.ones(1, 2),
                ref_logprobs=None,
                response_mask=torch.ones(1, 2, dtype=torch.bool),
                config=_make_dppo_config(),
            )

    def test_invalid_reference_logprob_only_removes_its_kl_gradient(self):
        new_logprobs = torch.full((1, 2), -2.0, requires_grad=True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=torch.full((1, 2), -2.0),
            advantages=torch.ones(1, 2),
            ref_logprobs=torch.tensor([[-3.0, float("nan")]]),
            response_mask=torch.ones(1, 2, dtype=torch.bool),
            config=_make_dppo_config(kl_estimator=0),
        )

        output.kl.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, torch.tensor([[1.0, 0.0]]))
        self.assertNotEqual(output.pg_loss[0, 1].item(), 0.0)

    def test_invalid_behavior_metadata_structurally_removes_all_gradient(self):
        new_logprobs = torch.full((1, 3), -2.0, requires_grad=True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=torch.tensor([[-2.0, INVALID_LOGPROB, float("nan")]]),
            advantages=torch.ones(1, 3),
            ref_logprobs=torch.full((1, 3), -3.0),
            response_mask=torch.ones(1, 3, dtype=torch.bool),
            config=_make_dppo_config(mask_reference_kl_with_policy=True),
        )

        (output.pg_loss + output.kl).sum().backward()
        torch.testing.assert_close(output.rho.mask, torch.tensor([[True, False, False]]))
        torch.testing.assert_close(new_logprobs.grad[0, 1:], torch.zeros(2))
        self.assertTrue(torch.isfinite(output.pg_loss).all())
        self.assertTrue(torch.isfinite(output.kl).all())

    def test_loss_metrics_use_the_effective_kl_mask(self):
        rho = torch.tensor([[1.0, 4.0, 1.0]])
        vllm_logprobs, new_logprobs = _logprobs_for_rho(rho)
        response_mask = torch.ones_like(rho, dtype=torch.bool)
        config = _make_grpo_config(
            loss_fn=grpo_utils.GRPOLossType.cispo,
            load_ref_policy=True,
            rho_mask_upper_bound=2.0,
            kl_estimator=0,
            beta=0.5,
            mask_reference_kl_with_policy=True,
        )
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=torch.ones_like(rho),
            ref_logprobs=torch.full_like(rho, -31.0),
            response_mask=response_mask,
            config=config,
        )
        stats = grpo_utils.create_loss_stats(1, torch.device("cpu"))

        grpo_utils.populate_sample_loss_stats(
            stats,
            0,
            output,
            (output.pg_loss + config.beta * output.kl).mean(),
            response_mask,
            new_logprobs,
            torch.full_like(rho, -31.0),
            None,
            config,
        )

        torch.testing.assert_close(
            stats["loss/kl_avg"][0], grpo_utils.masked_mean(output.kl, response_mask) * config.beta
        )

    def test_ratio_metric_excludes_sanitized_overflow(self):
        response_mask = torch.ones(1, 2, dtype=torch.bool)
        config = _make_dppo_config()
        new_logprobs = torch.tensor([[-1.0, -1.0]], requires_grad=True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=torch.tensor([[-101.0, -1.0]]),
            advantages=torch.ones(1, 2),
            ref_logprobs=None,
            response_mask=response_mask,
            config=config,
        )
        stats = grpo_utils.create_loss_stats(1, torch.device("cpu"))

        grpo_utils.populate_sample_loss_stats(
            stats, 0, output, output.pg_loss.mean(), response_mask, new_logprobs, None, None, config
        )

        torch.testing.assert_close(output.ratio, torch.tensor([[0.0, 1.0]]))
        self.assertTrue(torch.isinf(output.rho.ratio[0, 0]))
        torch.testing.assert_close(stats["val/ratio"][0], torch.tensor(1.0))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "vllm_logprobs shape"):
            grpo_utils.compute_grpo_loss(
                new_logprobs=torch.full((1, 2), -1.0),
                vllm_logprobs=torch.full((1, 3), -1.0),
                advantages=torch.ones(1, 2),
                ref_logprobs=None,
                response_mask=torch.ones(1, 2, dtype=torch.bool),
                config=_make_grpo_config(),
            )

    def test_removed_use_vllm_logprobs_config_field(self):
        field_names = {field.name for field in dataclasses.fields(grpo_utils.GRPOExperimentConfig)}
        self.assertNotIn("use_vllm_logprobs", field_names)

    def test_reference_kl_policy_masking_is_disabled_by_default(self):
        self.assertFalse(grpo_utils.GRPOExperimentConfig().mask_reference_kl_with_policy)


class TestVLLMDebugMetrics(unittest.TestCase):
    def test_reverse_kl_does_not_double_weight_behavior_samples(self):
        metrics = grpo_utils.compute_vllm_local_debug_metrics(
            local_logprobs=torch.tensor([[-3.0, -4.0]]),
            vllm_logprobs=torch.tensor([[-2.0, -1.0]]),
            response_mask=torch.ones(1, 2, dtype=torch.bool),
        )

        self.assertAlmostEqual(metrics["debug/vllm_local_reverse_kl"], 2.0)


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
            divergence_type="tv",
        )

        torch.testing.assert_close(divergence, torch.tensor([[0.1, 0.0, 0.6]]), atol=1e-5, rtol=1e-5)

    def test_kl_zero_when_distributions_match(self):
        logprobs = torch.log(torch.tensor([[0.3, 0.7]]))
        response_mask = torch.ones_like(logprobs, dtype=torch.bool)

        divergence = grpo_utils.compute_binary_divergence(
            behavior_logprobs=logprobs, policy_logprobs=logprobs, response_mask=response_mask, divergence_type="kl"
        )

        torch.testing.assert_close(divergence, torch.zeros_like(divergence), atol=1e-5, rtol=1e-5)

    def test_kl_remains_finite_for_probabilities_near_one(self):
        behavior_logprobs = torch.log(torch.tensor([[1.0 - 1e-7]], dtype=torch.float64))
        policy_logprobs = torch.log(torch.tensor([[1.0 - 2e-7]], dtype=torch.float64))

        divergence = grpo_utils.compute_binary_divergence(
            behavior_logprobs=behavior_logprobs,
            policy_logprobs=policy_logprobs,
            response_mask=torch.ones_like(behavior_logprobs, dtype=torch.bool),
            divergence_type="kl",
        )

        self.assertTrue(torch.isfinite(divergence).all())
        self.assertGreaterEqual(divergence.item(), 0.0)

    def test_response_mask_zeroes_invalid_positions(self):
        behavior_logprobs = torch.tensor([[INVALID_LOGPROB, -0.1]])
        policy_logprobs = torch.tensor([[INVALID_LOGPROB, -2.0]])
        response_mask = torch.tensor([[False, True]])

        divergence = grpo_utils.compute_binary_divergence(
            behavior_logprobs=behavior_logprobs,
            policy_logprobs=policy_logprobs,
            response_mask=response_mask,
            divergence_type="tv",
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

        correction = grpo_utils.compute_rho_correction(vllm_logprobs, new_logprobs, response_mask, advantages, config)

        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 5.0, 5.0]]))
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

        correction = grpo_utils.compute_rho_correction(vllm_logprobs, new_logprobs, response_mask, advantages, config)

        torch.testing.assert_close(correction.weights, torch.tensor([[1.25, 5.0 / 6.0]]))

    def test_negative_advantage_masked_when_ratio_below_one(self):
        config = _make_dppo_config()
        # μ = 0.5, π_θ = 0.1 → TV = 0.4 > δ; ratio < 1, so A<0 is the unsafe direction.
        vllm_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        new_logprobs = torch.log(torch.tensor([[0.1, 0.1]]))
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        advantages = torch.tensor([[-1.0, 1.0]])

        correction = grpo_utils.compute_rho_correction(vllm_logprobs, new_logprobs, response_mask, advantages, config)

        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 0.2]]))
        torch.testing.assert_close(correction.metrics["val/rho_drop_high_frac"], torch.tensor([[1.0, 0.0]]))

    def test_padding_positions_zeroed(self):
        config = _make_dppo_config()
        vllm_logprobs = torch.log(torch.tensor([[0.1, 0.1, 0.1]]))
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
        response_mask = torch.tensor([[False, True, False]])
        advantages = torch.full((1, 3), -1.0)

        correction = grpo_utils.compute_rho_correction(vllm_logprobs, new_logprobs, response_mask, advantages, config)

        # A<0 with π_θ>μ is the safe direction, so the middle token is kept; padding is 0.
        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 5.0, 0.0]]))

    def test_mask_does_not_propagate_gradients(self):
        config = _make_dppo_config()
        vllm_logprobs = torch.log(torch.tensor([[0.1]]))
        new_logprobs = torch.log(torch.tensor([[0.5]])).requires_grad_(True)
        response_mask = torch.ones(1, 1, dtype=torch.bool)
        advantages = torch.ones(1, 1)

        correction = grpo_utils.compute_rho_correction(vllm_logprobs, new_logprobs, response_mask, advantages, config)

        self.assertFalse(correction.weights.requires_grad)

    def test_dppo_uses_raw_rho_even_when_correction_bounds_are_configured(self):
        # DPPO's trust region is its directional mask, so retained tokens keep
        # the raw Eq. 11 ratio rather than receiving a second clamp.
        config = _make_dppo_config(rho_clamp_upper_bound=2.0)
        vllm_logprobs = torch.log(torch.tensor([[0.1, 0.1]]))
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
        advantages = torch.tensor([[1.0, -1.0]])

        correction = grpo_utils.compute_rho_correction(vllm_logprobs, new_logprobs, response_mask, advantages, config)

        torch.testing.assert_close(correction.weights, torch.tensor([[0.0, 5.0]]))

    def test_dppo_drops_overflow_before_retained_token_check(self):
        new_logprobs = torch.tensor([[-1.0]], requires_grad=True)
        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=torch.tensor([[-101.0]]),
            advantages=torch.ones(1, 1),
            ref_logprobs=torch.tensor([[-2.0]]),
            response_mask=torch.ones(1, 1, dtype=torch.bool),
            config=_make_dppo_config(mask_reference_kl_with_policy=True),
        )

        self.assertFalse(output.rho.mask.item())
        self.assertEqual(output.ratio.item(), 0.0)
        self.assertEqual(output.rho.metrics["val/rho_overflow_frac"].item(), 1.0)
        histograms: dict[str, list[torch.Tensor]] = {}
        grpo_utils.accumulate_rho_histograms(histograms, output.rho)
        self.assertEqual(grpo_utils.finalize_rho_histograms(histograms), {})
        (output.pg_loss + output.kl).sum().backward()
        self.assertEqual(new_logprobs.grad.item(), 0.0)


class TestDPPOLoss(unittest.TestCase):
    def test_dppo_loss_masks_and_has_no_symmetric_clip(self):
        config = _make_dppo_config(loss_fn=grpo_utils.GRPOLossType.dppo, rho_mask_upper_bound=0.1)
        vllm_logprobs = torch.log(torch.tensor([[0.1, 0.1]]))
        new_logprobs = torch.log(torch.tensor([[0.5, 0.5]])).requires_grad_(True)
        advantages = torch.tensor([[1.0, -1.0]])
        response_mask = torch.ones_like(new_logprobs, dtype=torch.bool)

        output = grpo_utils.compute_grpo_loss(
            new_logprobs=new_logprobs,
            vllm_logprobs=vllm_logprobs,
            advantages=advantages,
            ref_logprobs=None,
            response_mask=response_mask,
            config=config,
        )

        # ratio = π_θ / μ = 5 for both tokens; only the A>0 token is masked (Eq. 12),
        # and the kept A<0 token is unclipped despite ratio 5 (Eq. 11).
        expected_weights = torch.tensor([[0.0, 5.0]])
        torch.testing.assert_close(output.pg_loss, -expected_weights * advantages * new_logprobs.detach())
        torch.testing.assert_close(output.clipfrac, torch.zeros_like(output.clipfrac))
        torch.testing.assert_close(output.kl, torch.zeros_like(output.kl))
        output.pg_loss.sum().backward()
        torch.testing.assert_close(new_logprobs.grad, -expected_weights * advantages)


class TestDPPOConfigValidation(unittest.TestCase):
    def test_dppo_requires_rollout_policy_properties(self):
        with self.assertRaisesRegex(ValueError, "policy_ratio_denominator"):
            grpo_utils.GRPOExperimentConfig(loss_fn=grpo_utils.GRPOLossType.dppo)

    def test_dppo_requires_threshold(self):
        with self.assertRaisesRegex(ValueError, "rho_mask_upper_bound"):
            grpo_utils.GRPOExperimentConfig(
                loss_fn=grpo_utils.GRPOLossType.dppo,
                policy_ratio_denominator="rollout_policy",
                rollout_importance_correction="none",
                rho_mask_metric="tv",
                rho_mask_source="current_policy",
                rho_mask_direction="increase_only",
            )

    @parameterized.expand([("nan", float("nan")), ("infinity", float("inf"))])
    def test_dppo_requires_finite_threshold(self, _name, threshold):
        with self.assertRaisesRegex(ValueError, "finite"):
            grpo_utils.GRPOExperimentConfig(
                loss_fn=grpo_utils.GRPOLossType.dppo,
                policy_ratio_denominator="rollout_policy",
                rollout_importance_correction="none",
                rho_mask_metric="tv",
                rho_mask_source="current_policy",
                rho_mask_direction="increase_only",
                rho_mask_upper_bound=threshold,
            )

    def test_dppo_rejects_sequence_level_masking(self):
        with self.assertRaisesRegex(ValueError, "rho_mask_level"):
            grpo_utils.GRPOExperimentConfig(
                loss_fn=grpo_utils.GRPOLossType.dppo,
                policy_ratio_denominator="rollout_policy",
                rollout_importance_correction="none",
                rho_mask_metric="tv",
                rho_mask_source="current_policy",
                rho_mask_level="sequence",
                rho_mask_direction="increase_only",
                rho_mask_upper_bound=0.1,
            )

    def test_dppo_kl_is_valid(self):
        grpo_utils.GRPOExperimentConfig(
            loss_fn=grpo_utils.GRPOLossType.dppo,
            policy_ratio_denominator="rollout_policy",
            rollout_importance_correction="none",
            rho_mask_metric="kl",
            rho_mask_source="current_policy",
            rho_mask_direction="increase_only",
            rho_mask_upper_bound=0.1,
        )

    def test_rollout_policy_rejects_additional_correction(self):
        with self.assertRaisesRegex(ValueError, "rollout_importance_correction"):
            grpo_utils.GRPOExperimentConfig(
                policy_ratio_denominator="rollout_policy",
                rollout_importance_correction="clipped",
                rho_mask_source="current_policy",
            )

    def test_rollout_policy_rejects_old_policy_mask_source(self):
        with self.assertRaisesRegex(ValueError, "rho_mask_source"):
            grpo_utils.GRPOExperimentConfig(
                policy_ratio_denominator="rollout_policy", rollout_importance_correction="none"
            )

    def test_property_based_sequence_divergence_mask_is_valid(self):
        grpo_utils.GRPOExperimentConfig(
            rho_mask_metric="tv",
            rho_mask_source="old_policy",
            rho_mask_level="sequence",
            rho_mask_direction="increase_only",
            rho_mask_upper_bound=0.5,
        )

    def test_removed_configuration_fields_are_absent(self):
        field_names = {field.name for field in dataclasses.fields(grpo_utils.GRPOExperimentConfig)}
        self.assertNotIn("use_vllm_logprobs", field_names)
        self.assertNotIn("use_rho_correction", field_names)
        self.assertNotIn("rho_divergence_algo", field_names)
        self.assertNotIn("rho_divergence_type", field_names)


if __name__ == "__main__":
    unittest.main()
