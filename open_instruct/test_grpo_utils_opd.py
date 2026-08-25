"""Tests for the on-policy distillation (OPD) advantage transform and config validation."""

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from open_instruct import data_types, grpo_utils


def _make_inputs():
    # [B=2, T=5] advantages; logprobs/masks are [B, T-1=4].
    advantages = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [-1.0, -1.0, -1.0, -1.0, -1.0]])
    behavior_logprobs = torch.tensor([[-1.0, -2.0, -0.5, -3.0], [-0.1, -0.2, -0.3, -0.4]])
    teacher_logprobs = torch.tensor([[-1.5, -1.0, -0.5, -2.0], [-0.1, -0.7, -0.1, -0.9]])
    response_mask = torch.tensor([[True, True, True, False], [False, True, True, True]])
    return advantages, behavior_logprobs, teacher_logprobs, response_mask


class TestComputeOPDAdvantages(unittest.TestCase):
    def test_additive_mode(self):
        advantages, behavior, teacher, mask = _make_inputs()
        new_adv, reverse_kl = grpo_utils.compute_opd_advantages(
            advantages, behavior, teacher, mask, kl_coef=2.0, pure=False
        )
        expected_kl = torch.where(mask, behavior - teacher, torch.zeros_like(behavior))
        torch.testing.assert_close(reverse_kl, expected_kl)
        # Masked positions: A ← A − coef · kl. Unmasked positions untouched.
        expected = advantages.clone()
        expected[:, 1:] = torch.where(mask, advantages[:, 1:] - 2.0 * expected_kl, advantages[:, 1:])
        torch.testing.assert_close(new_adv, expected)
        # Column 0 is never modified.
        torch.testing.assert_close(new_adv[:, 0], advantages[:, 0])

    def test_pure_mode_replaces_env_advantage(self):
        advantages, behavior, teacher, mask = _make_inputs()
        new_adv, reverse_kl = grpo_utils.compute_opd_advantages(
            advantages, behavior, teacher, mask, kl_coef=1.0, pure=True
        )
        # Masked positions carry only the distillation term.
        torch.testing.assert_close(new_adv[:, 1:][mask], -reverse_kl[mask])
        # Unmasked positions keep the original (irrelevant) values.
        torch.testing.assert_close(new_adv[:, 1:][~mask], advantages[:, 1:][~mask])

    def test_teacher_equals_student_is_noop(self):
        advantages, behavior, _, mask = _make_inputs()
        new_adv, reverse_kl = grpo_utils.compute_opd_advantages(
            advantages, behavior, behavior.clone(), mask, kl_coef=1.0, pure=False
        )
        torch.testing.assert_close(reverse_kl, torch.zeros_like(reverse_kl))
        torch.testing.assert_close(new_adv, advantages)

    def test_sign_pushes_towards_teacher(self):
        # Token where the teacher likes the sampled token more than the student
        # (teacher logprob > behavior logprob) must get a positive advantage boost.
        advantages = torch.zeros(1, 3)
        behavior = torch.tensor([[-2.0, -0.5]])
        teacher = torch.tensor([[-1.0, -1.5]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        new_adv, _ = grpo_utils.compute_opd_advantages(advantages, behavior, teacher, mask, kl_coef=1.0, pure=False)
        self.assertGreater(new_adv[0, 1].item(), 0.0)  # teacher prefers it → push up
        self.assertLess(new_adv[0, 2].item(), 0.0)  # student overconfident → push down

    def test_does_not_mutate_input(self):
        advantages, behavior, teacher, mask = _make_inputs()
        original = advantages.clone()
        grpo_utils.compute_opd_advantages(advantages, behavior, teacher, mask, kl_coef=1.0, pure=True)
        torch.testing.assert_close(advantages, original)

    def test_dtype_preserved(self):
        advantages, behavior, teacher, mask = _make_inputs()
        new_adv, _ = grpo_utils.compute_opd_advantages(
            advantages.to(torch.bfloat16), behavior, teacher, mask, kl_coef=1.0, pure=False
        )
        self.assertEqual(new_adv.dtype, torch.bfloat16)


class TestComputeLogprobsTiled(unittest.TestCase):
    def test_matches_untiled_compute_logprobs(self):
        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = Qwen2ForCausalLM(config).eval()
        batch, seq_len, pad_token_id = 2, 16, 0
        query_responses = torch.randint(1, 128, (batch, 1, seq_len)).unbind(0)
        position_ids = [torch.arange(seq_len).unsqueeze(0) for _ in range(batch)]
        response_masks = [
            torch.cat([torch.zeros(1, 4, dtype=torch.bool), torch.ones(1, seq_len - 4, dtype=torch.bool)], dim=1)
            for _ in range(batch)
        ]
        data_BT = data_types.CollatedBatchData(
            query_responses=list(query_responses),
            attention_masks=[torch.ones(1, seq_len, dtype=torch.long) for _ in range(batch)],
            position_ids=position_ids,
            advantages=[torch.zeros(1, seq_len) for _ in range(batch)],
            response_masks=response_masks,
            vllm_logprobs=[torch.zeros(1, seq_len) for _ in range(batch)],
        )
        reference = grpo_utils.compute_logprobs(model, data_BT, pad_token_id, temperature=0.7)
        for shards in (1, 3, 64):
            tiled = grpo_utils.compute_logprobs_tiled(model, data_BT, pad_token_id, temperature=0.7, shards=shards)
            for ref, til in zip(reference, tiled, strict=True):
                torch.testing.assert_close(til, ref, atol=1e-4, rtol=1e-4)


class TestOPDConfigValidation(unittest.TestCase):
    def _base_kwargs(self, **overrides):
        kwargs = {"opd_teacher_model_name_or_path": "some/teacher"}
        kwargs.update(overrides)
        return kwargs

    def test_valid_opd_config(self):
        config = grpo_utils.GRPOExperimentConfig(**self._base_kwargs())
        self.assertEqual(config.opd_kl_coef, 1.0)
        self.assertFalse(config.opd_pure)

    def test_opd_kl_coef_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "opd_kl_coef"):
            grpo_utils.GRPOExperimentConfig(**self._base_kwargs(opd_kl_coef=0.0))

    def test_opd_pure_requires_teacher(self):
        with self.assertRaisesRegex(ValueError, "opd_pure"):
            grpo_utils.GRPOExperimentConfig(opd_pure=True)

    def test_opd_incompatible_with_value_model(self):
        with self.assertRaisesRegex(ValueError, "use_value_model"):
            grpo_utils.GRPOExperimentConfig(**self._base_kwargs(use_value_model=True))


if __name__ == "__main__":
    unittest.main()
