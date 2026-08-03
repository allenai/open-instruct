"""Tests for the on-policy distillation (OPD) advantage transform and config validation."""

import unittest

import torch

from open_instruct import grpo_utils


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
