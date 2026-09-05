"""Tests for the on-policy distillation (OPD) advantage transform and config validation."""

import unittest

import numpy as np
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


class TestComputePerDatasetEvalMetrics(unittest.TestCase):
    def test_separates_tasks_and_preserves_pass_at_k_semantics(self):
        metrics = grpo_utils.compute_per_dataset_eval_metrics(
            scores=np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
            response_lengths=np.array([1, 2, 1, 3, 1, 1, 2, 2]),
            finish_reasons=["stop", "length", "stop", "stop", "length", "length", "stop", "length"],
            dataset_names=[["math_aime_2025"]] * 4 + [["math_brumo_2025"]] * 4,
            eval_k=2,
            max_possible_score=1.0,
        )

        self.assertEqual(metrics["eval/math_aime_2025/scores"], 0.75)
        self.assertEqual(metrics["eval/math_aime_2025/pass_at_1"], 0.75)
        self.assertEqual(metrics["eval/math_aime_2025/pass_at_2"], 1.0)
        self.assertEqual(metrics["eval/math_aime_2025/sequence_lengths"], 1.75)
        self.assertEqual(metrics["eval/math_aime_2025/stop_rate"], 0.75)
        self.assertEqual(metrics["eval/math_brumo_2025/scores"], 0.25)
        self.assertEqual(metrics["eval/math_brumo_2025/pass_at_1"], 0.25)
        self.assertEqual(metrics["eval/math_brumo_2025/pass_at_2"], 0.5)
        self.assertEqual(metrics["eval/math_brumo_2025/sequence_lengths"], 1.5)
        self.assertEqual(metrics["eval/math_brumo_2025/stop_rate"], 0.25)

    def test_response_can_belong_to_multiple_datasets(self):
        metrics = grpo_utils.compute_per_dataset_eval_metrics(
            scores=np.array([1.0, 0.0]),
            response_lengths=np.array([1, 2]),
            finish_reasons=["stop", "length"],
            dataset_names=[["math_aime_2025", "math_brumo_2025"], ["math_brumo_2025"]],
            eval_k=1,
            max_possible_score=1.0,
        )

        self.assertEqual(metrics["eval/math_aime_2025/scores"], 1.0)
        self.assertEqual(metrics["eval/math_brumo_2025/scores"], 0.5)

    def test_returns_no_metrics_when_result_lengths_differ(self):
        metrics = grpo_utils.compute_per_dataset_eval_metrics(
            scores=np.array([1.0]),
            response_lengths=np.array([1]),
            finish_reasons=["stop"],
            dataset_names=["math_aime_2025", "math_brumo_2025"],
            eval_k=1,
            max_possible_score=1.0,
        )

        self.assertEqual(metrics, {})


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

    def test_adv_clip_bounds_folded_term_but_not_logged_kl(self):
        advantages = torch.zeros(1, 3)
        behavior = torch.tensor([[-0.1, -0.1]])
        teacher = torch.tensor([[-10.0, -0.2]])  # first token: huge disagreement (kl = 9.9)
        mask = torch.ones(1, 2, dtype=torch.bool)
        new_adv, reverse_kl = grpo_utils.compute_opd_advantages(
            advantages, behavior, teacher, mask, kl_coef=1.0, pure=True, adv_clip=2.0
        )
        # The folded advantage is clipped to ±2, the logged KL is not.
        torch.testing.assert_close(new_adv[0, 1:], torch.tensor([-2.0, -(-0.1 - -0.2)]))
        torch.testing.assert_close(reverse_kl[0, 0], torch.tensor(9.9))


class TestCombineOPDTeacherLogprobs(unittest.TestCase):
    def _teachers(self):
        # [K=2, B=1, T-1=4]
        t0 = torch.tensor([[-1.0, -2.0, -0.5, -3.0]])
        t1 = torch.tensor([[-2.0, -1.0, -1.5, -0.5]])
        mask = torch.tensor([[True, True, True, False]])
        return torch.stack([t0, t1]), mask

    def test_single_teacher_all_strategies_identity(self):
        lp = torch.tensor([[[-1.0, -2.0, -0.5, -3.0]]])
        mask = torch.tensor([[True, True, True, False]])
        expected = torch.where(mask, lp[0], torch.zeros_like(lp[0]))
        for strategy in ("mixture", "max", "min"):
            combined = grpo_utils.combine_opd_teacher_logprobs(lp, strategy, mask)
            torch.testing.assert_close(combined, expected, msg=strategy)
        route = grpo_utils.combine_opd_teacher_logprobs(
            lp, "route", mask, teacher_ids=torch.zeros(1, 4, dtype=torch.long)
        )
        torch.testing.assert_close(route, expected)

    def test_mixture_uniform_is_logsumexp_minus_log_k(self):
        stacked, mask = self._teachers()
        combined = grpo_utils.combine_opd_teacher_logprobs(stacked, "mixture", mask)
        expected = torch.logsumexp(stacked, dim=0) - torch.log(torch.tensor(2.0))
        torch.testing.assert_close(combined[mask], expected[mask])
        torch.testing.assert_close(combined[~mask], torch.zeros_like(combined[~mask]))

    def test_mixture_weighted(self):
        stacked, mask = self._teachers()
        weights = torch.tensor([0.75, 0.25])
        combined = grpo_utils.combine_opd_teacher_logprobs(stacked, "mixture", mask, log_weights=torch.log(weights))
        expected = torch.logsumexp(stacked + torch.log(weights).view(-1, 1, 1), dim=0)
        torch.testing.assert_close(combined[mask], expected[mask])
        # Mixture probability is bounded by the best teacher and above the weighted worst.
        self.assertTrue((combined[mask] <= stacked.amax(dim=0)[mask] + 1e-6).all())

    def test_max_and_min_envelopes(self):
        stacked, mask = self._teachers()
        combined_max = grpo_utils.combine_opd_teacher_logprobs(stacked, "max", mask)
        combined_min = grpo_utils.combine_opd_teacher_logprobs(stacked, "min", mask)
        torch.testing.assert_close(combined_max[mask], stacked.amax(dim=0)[mask])
        torch.testing.assert_close(combined_min[mask], stacked.amin(dim=0)[mask])

    def test_route_gathers_per_token_teacher(self):
        stacked, mask = self._teachers()
        teacher_ids = torch.tensor([[0, 1, 1, -1]])  # -1 on the masked position
        combined = grpo_utils.combine_opd_teacher_logprobs(stacked, "route", mask, teacher_ids=teacher_ids)
        torch.testing.assert_close(combined[0, :3], torch.tensor([-1.0, -1.0, -1.5]))
        torch.testing.assert_close(combined[0, 3], torch.tensor(0.0))

    def test_route_rejects_invalid_ids_on_response_tokens(self):
        stacked, mask = self._teachers()
        with self.assertRaisesRegex(ValueError, "teacher ids"):
            grpo_utils.combine_opd_teacher_logprobs(stacked, "route", mask, teacher_ids=torch.tensor([[0, -1, 1, 0]]))
        with self.assertRaisesRegex(ValueError, "teacher_ids"):
            grpo_utils.combine_opd_teacher_logprobs(stacked, "route", mask, teacher_ids=None)


class TestParseOPDTeacherDomains(unittest.TestCase):
    def test_parse_and_resolve(self):
        mapping, catch_all = grpo_utils.parse_opd_teacher_domains(["gsm8k,math", "swerl", "*"])
        self.assertEqual(mapping, {"gsm8k": 0, "math": 0, "swerl": 1})
        self.assertEqual(catch_all, 2)
        self.assertEqual(grpo_utils.resolve_opd_teacher_for_dataset("math", mapping, catch_all), 0)
        self.assertEqual(grpo_utils.resolve_opd_teacher_for_dataset("unknown", mapping, catch_all), 2)

    def test_case_insensitive_and_list_valued_dataset_fields(self):
        mapping, catch_all = grpo_utils.parse_opd_teacher_domains(["GSM8K,math", "swerl"])
        self.assertEqual(mapping, {"gsm8k": 0, "math": 0, "swerl": 1})
        # The RLVR `dataset` field may be a list of verifier names; the first claimed one wins.
        self.assertEqual(grpo_utils.resolve_opd_teacher_for_dataset(["GSM8K"], mapping, catch_all), 0)
        self.assertEqual(grpo_utils.resolve_opd_teacher_for_dataset(["nope", "swerl"], mapping, catch_all), 1)

    def test_unmatched_dataset_without_catch_all_raises(self):
        mapping, catch_all = grpo_utils.parse_opd_teacher_domains(["gsm8k", "swerl"])
        self.assertIsNone(catch_all)
        with self.assertRaisesRegex(ValueError, "not claimed"):
            grpo_utils.resolve_opd_teacher_for_dataset("unknown", mapping, catch_all)

    def test_duplicate_claim_raises(self):
        with self.assertRaisesRegex(ValueError, "both"):
            grpo_utils.parse_opd_teacher_domains(["gsm8k,math", "math"])

    def test_empty_entry_raises(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            grpo_utils.parse_opd_teacher_domains(["gsm8k", " , "])

    def test_multiple_catch_all_raises(self):
        with self.assertRaisesRegex(ValueError, "catch-all"):
            grpo_utils.parse_opd_teacher_domains(["*", "*"])


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

    def test_bare_string_teacher_coerced_to_list(self):
        config = grpo_utils.GRPOExperimentConfig(**self._base_kwargs())
        self.assertEqual(config.opd_teacher_model_name_or_path, ["some/teacher"])

    def test_multi_teacher_defaults_valid(self):
        config = grpo_utils.GRPOExperimentConfig(opd_teacher_model_name_or_path=["t/a", "t/b"])
        self.assertEqual(config.opd_teacher_combine, "mixture")

    def test_unknown_combine_strategy_raises(self):
        with self.assertRaisesRegex(ValueError, "opd_teacher_combine"):
            grpo_utils.GRPOExperimentConfig(opd_teacher_model_name_or_path=["t/a", "t/b"], opd_teacher_combine="vote")

    def test_revision_count_must_match(self):
        with self.assertRaisesRegex(ValueError, "opd_teacher_model_revision"):
            grpo_utils.GRPOExperimentConfig(
                opd_teacher_model_name_or_path=["t/a", "t/b"], opd_teacher_model_revision=["main"]
            )

    def test_weights_require_mixture(self):
        with self.assertRaisesRegex(ValueError, "opd_teacher_weights"):
            grpo_utils.GRPOExperimentConfig(
                opd_teacher_model_name_or_path=["t/a", "t/b"],
                opd_teacher_combine="max",
                opd_teacher_weights=[0.5, 0.5],
            )

    def test_weights_count_and_positivity(self):
        with self.assertRaisesRegex(ValueError, "one entry per teacher"):
            grpo_utils.GRPOExperimentConfig(opd_teacher_model_name_or_path=["t/a", "t/b"], opd_teacher_weights=[1.0])
        with self.assertRaisesRegex(ValueError, "> 0"):
            grpo_utils.GRPOExperimentConfig(
                opd_teacher_model_name_or_path=["t/a", "t/b"], opd_teacher_weights=[1.0, 0.0]
            )

    def test_route_requires_domains(self):
        with self.assertRaisesRegex(ValueError, "opd_teacher_domains"):
            grpo_utils.GRPOExperimentConfig(opd_teacher_model_name_or_path=["t/a", "t/b"], opd_teacher_combine="route")

    def test_route_domains_count_must_match(self):
        with self.assertRaisesRegex(ValueError, "one entry per teacher"):
            grpo_utils.GRPOExperimentConfig(
                opd_teacher_model_name_or_path=["t/a", "t/b"],
                opd_teacher_combine="route",
                opd_teacher_domains=["gsm8k"],
            )

    def test_domains_without_route_raises(self):
        with self.assertRaisesRegex(ValueError, "route"):
            grpo_utils.GRPOExperimentConfig(
                opd_teacher_model_name_or_path=["t/a", "t/b"], opd_teacher_domains=["gsm8k", "*"]
            )

    def test_valid_route_config(self):
        config = grpo_utils.GRPOExperimentConfig(
            opd_teacher_model_name_or_path=["t/a", "t/b"],
            opd_teacher_combine="route",
            opd_teacher_domains=["gsm8k,math", "*"],
        )
        self.assertEqual(config.opd_teacher_domains, ["gsm8k,math", "*"])

    def test_adv_clip_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "opd_adv_clip"):
            grpo_utils.GRPOExperimentConfig(**self._base_kwargs(opd_adv_clip=0.0))


if __name__ == "__main__":
    unittest.main()
