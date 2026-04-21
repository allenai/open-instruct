# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Smoke tests for the value-model code paths added on hamish/vip.

These tests deliberately avoid anything that requires vLLM, DeepSpeed, or a GPU so they run on a
laptop and in CI. They focus on:

(a) GAE variants (standard, SAE, VAPO, SAE+VAPO) on a tiny packed sequence;
(b) sibling-rollout assembly helpers in data_loader.py;
(c) the conditioning text builders in value_model_utils.py;
(d) score parsing for the generative value model.
"""
from __future__ import annotations

import unittest

import numpy as np


class TestGAEVariants(unittest.TestCase):
    def _inputs(self):
        # Single packed sequence with one sub-sequence. Prompt tokens 0..2, response 3..7.
        # Reward of 1.0 at t=7 (terminal). One low-probability token at t=4 for SAE.
        B, T = 1, 8
        values = np.array([[0.1] * T])
        rewards = np.zeros((B, T))
        rewards[0, 7] = 1.0
        dones = np.zeros((B, T))
        dones[0, 7] = 1
        response_masks = np.zeros((B, T))
        response_masks[0, 3:8] = 1
        logprobs = np.array([[-0.1, -0.1, -0.1, -0.1, -2.5, -0.1, -0.1, -0.1]])
        return values, rewards, dones, response_masks, logprobs

    def test_standard_gae_runs(self):
        from open_instruct.rl_utils import calculate_advantages_packed

        v, r, d, m, _ = self._inputs()
        adv, returns = calculate_advantages_packed(v, r, gamma=1.0, lam=0.95, dones=d, response_masks=m)
        self.assertEqual(adv.shape, v.shape)
        self.assertEqual(returns.shape, v.shape)
        # Terminal step should have a positive advantage (reward minus baseline value).
        self.assertGreater(adv[0, 7], 0)

    def test_vapo_has_two_outputs(self):
        from open_instruct.rl_utils import calculate_advantages_packed_vapo

        v, r, d, m, _ = self._inputs()
        pa, cr, avg_lam = calculate_advantages_packed_vapo(v, r, gamma=1.0, dones=d, response_masks=m)
        self.assertEqual(pa.shape, v.shape)
        self.assertEqual(cr.shape, v.shape)
        self.assertEqual(avg_lam, 0.95)

    def test_sae_marks_boundary(self):
        from open_instruct.rl_utils import calculate_advantages_packed_sae

        v, r, d, m, logp = self._inputs()
        adv, returns, bf = calculate_advantages_packed_sae(
            v, r, gamma=1.0, lam=0.2, dones=d, response_masks=m, logprobs=logp, sae_threshold=0.2
        )
        # t=4 logprob < log(0.2) ≈ -1.609, so exactly one boundary among the response tokens.
        expected_frac = 1 / 5
        self.assertAlmostEqual(bf, expected_frac, places=6)
        self.assertEqual(adv.shape, v.shape)

    def test_sae_vapo_combines_variants(self):
        from open_instruct.rl_utils import calculate_advantages_packed_sae_vapo

        v, r, d, m, logp = self._inputs()
        pa, cr, bf = calculate_advantages_packed_sae_vapo(
            v, r, gamma=1.0, dones=d, response_masks=m, logprobs=logp, sae_threshold=0.2, lam_policy=0.5
        )
        self.assertEqual(pa.shape, v.shape)
        self.assertEqual(cr.shape, v.shape)
        self.assertGreater(bf, 0)

    def test_length_adaptive_lambda(self):
        from open_instruct.rl_utils import calculate_length_adaptive_lambda

        # alpha*length = 1 -> lambda = 0
        self.assertEqual(calculate_length_adaptive_lambda(1, alpha=1.0), 0.0)
        # alpha*length = 100 -> lambda close to 1
        self.assertGreater(calculate_length_adaptive_lambda(100, alpha=1.0), 0.98)


def _data_loader_available() -> bool:
    try:
        import vllm  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_data_loader_available(), "data_loader requires vllm")
class TestSiblingAssembly(unittest.TestCase):
    def test_extract_subseq_indices_per_pack(self):
        import torch

        from open_instruct.data_loader import _extract_subseq_indices_per_pack

        # pack 1: [1,1,1,0,2,2,2], pack 2: [3,3,0,0]
        rm = [torch.tensor([[1, 1, 1, 0, 2, 2, 2]]), torch.tensor([[3, 3, 0, 0]])]
        got = _extract_subseq_indices_per_pack(rm)
        self.assertEqual(got, [[1, 2], [3]])

    def test_populate_value_model_fields_minimal(self):
        import torch

        from open_instruct.data_loader import populate_value_model_fields
        from open_instruct.rl_utils import PackedSequences

        # Fake packed sequences with one sub-seq per pack.
        ps = PackedSequences(
            query_responses=[torch.tensor([[0, 1, 2, 3]])],
            attention_masks=[torch.tensor([[1, 1, 1, 1]])],
            response_masks=[torch.tensor([[0, 1, 1, 1]])],
            original_responses=[[1, 2, 3]],
            advantages=None,
            num_actions=[torch.tensor([[3]])],
            position_ids=[torch.tensor([[0, 1, 2, 3]])],
            packed_seq_lens=[torch.tensor([[4]])],
            vllm_logprobs=[torch.tensor([[0.0, -0.1, -3.0, -0.1]])],
            dones=[torch.tensor([[0, 0, 0, 1]])],
        )
        populate_value_model_fields(
            packed_sequences=ps,
            scores=np.array([0.5]),
            batch_ground_truths=["42"],
            decoded_responses=["hello"],
            num_samples_per_prompt=1,
            max_possible_score=1.0,
            use_sae=True,
            sae_threshold=0.2,
            need_ground_truths=True,
            need_siblings=False,
            num_siblings_to_sample=0,
            rng=np.random.default_rng(0),
        )
        self.assertIsNotNone(ps.rewards)
        self.assertEqual(ps.rewards[0].tolist(), [[0.0, 0.0, 0.0, 0.5]])
        self.assertIsNotNone(ps.segment_boundaries)
        # t=2 has logprob -3.0 < log(0.2) -> boundary.
        self.assertTrue(bool(ps.segment_boundaries[0][0, 2].item()))
        self.assertEqual(ps.ground_truths, [["42"]])


class TestConditioningBuilders(unittest.TestCase):
    def test_every_template_builds(self):
        from open_instruct.value_model_utils import build_conditioning_text

        siblings = [
            {"text": "abc", "is_correct": True},
            {"text": "def", "is_correct": False},
        ]
        for t in [
            "answer_prefix",
            "boxed_answer",
            "cot_spoiler",
            "expected_accuracy",
            "rollout_context",
            "correct_demo",
        ]:
            txt = build_conditioning_text(t, ground_truth="42", siblings=siblings)
            self.assertIsInstance(txt, str)
            self.assertGreater(len(txt), 0)

    def test_unknown_template_raises(self):
        from open_instruct.value_model_utils import build_conditioning_text

        with self.assertRaises(ValueError):
            build_conditioning_text("bogus", "42", [])

    def test_is_postfix_template(self):
        from open_instruct.value_model_utils import is_postfix_template

        self.assertFalse(is_postfix_template("answer_prefix"))
        self.assertTrue(is_postfix_template("expected_accuracy"))


class TestScoreParsing(unittest.TestCase):
    def test_direct_parsing(self):
        from open_instruct.value_model_utils import parse_generative_value_score

        self.assertEqual(parse_generative_value_score("7"), 7.0)
        self.assertEqual(parse_generative_value_score(" 10 foo"), 10.0)
        self.assertEqual(parse_generative_value_score("5.5"), 5.5)
        self.assertIsNone(parse_generative_value_score("no digits here"))

    def test_cot_parsing(self):
        from open_instruct.value_model_utils import parse_generative_value_score

        self.assertEqual(
            parse_generative_value_score("The approach is good... {score: 7.5}", allow_cot=True), 7.5
        )
        self.assertIsNone(parse_generative_value_score("no json", allow_cot=True))

    def test_clamping(self):
        from open_instruct.value_model_utils import parse_generative_value_score

        self.assertEqual(parse_generative_value_score("42", score_min=0, score_max=10), 10.0)
        self.assertEqual(parse_generative_value_score("-5", score_min=0, score_max=10), 0.0)

    def test_prompt_has_conditioning(self):
        from open_instruct.value_model_utils import build_generative_value_prompt

        p = build_generative_value_prompt(
            "partial", conditioning="gt", ground_truth="42", allow_cot=False
        )
        self.assertIn("The correct answer is 42", p)
        self.assertIn("<rollout>", p)
        self.assertIn("Thus, the score is", p)


class TestValueLoss(unittest.TestCase):
    def test_mse_loss_no_clip(self):
        import torch

        from open_instruct.value_model_utils import value_clipped_mse_loss

        new_v = torch.tensor([[1.0, 2.0, 3.0]])
        ret = torch.tensor([[1.0, 1.0, 1.0]])
        mask = torch.tensor([[True, True, True]])
        per_tok, clipfrac = value_clipped_mse_loss(new_v, ret, None, mask, clip_range=0.0)
        self.assertEqual(per_tok.shape, new_v.shape)
        self.assertEqual(float(clipfrac), 0.0)

    def test_mse_loss_with_clip(self):
        import torch

        from open_instruct.value_model_utils import value_clipped_mse_loss

        new_v = torch.tensor([[10.0, 2.0]])
        old_v = torch.tensor([[0.0, 0.0]])
        ret = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[True, True]])
        per_tok, clipfrac = value_clipped_mse_loss(new_v, ret, old_v, mask, clip_range=0.1)
        # PPO2 clipping is pessimistic: it uses the MAX of clipped and unclipped losses, so the
        # final per-token loss is dominated by (new - ret)^2 here (= 100), giving 50 after the
        # 0.5 factor. clipfrac stays 0 because the clipped loss never exceeds the unclipped one.
        self.assertAlmostEqual(float(per_tok[0, 0]), 50.0, places=5)
        self.assertEqual(per_tok.shape, new_v.shape)
        self.assertGreaterEqual(float(clipfrac), 0.0)


class TestGenValueSegmentation(unittest.TestCase):
    def test_fixed_segmentation(self):
        from open_instruct.grpo_fast_genvalue import segment_rollout

        boundaries = segment_rollout(list(range(1500)), None, mode="fixed", fixed_chunk_size=500)
        # Boundaries at 500 and 1000, plus a final boundary at L-1.
        self.assertEqual(boundaries, [500, 1000, 1499])

    def test_sae_segmentation(self):
        from open_instruct.grpo_fast_genvalue import segment_rollout

        logps = [-0.1] * 10 + [-3.0] + [-0.1] * 10  # one boundary at t=10
        boundaries = segment_rollout([0] * 21, logps, mode="sae", sae_threshold=0.2)
        self.assertIn(10, boundaries)
        self.assertEqual(boundaries[-1], 20)


if __name__ == "__main__":
    unittest.main()
