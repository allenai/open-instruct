import unittest

import numpy as np

from open_instruct import length_reward_shaping as lrs


class TestApplyLengthRewardShaping(unittest.TestCase):
    def test_none_returns_copy(self):
        scores = np.array([[10.0, 10.0, 0.0, 10.0]])
        lengths = np.array([[100, 200, 50, 400]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "none", 1.0)
        np.testing.assert_array_equal(out, scores)
        self.assertIsNot(out, scores)

    def test_warmup_zero_returns_unshaped(self):
        scores = np.array([[10.0, 10.0]])
        lengths = np.array([[100, 200]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "linear", 1.0, warmup_weight=0.0)
        np.testing.assert_array_equal(out, scores)

    def test_incorrect_responses_unchanged(self):
        # incorrect responses (score == 0) must keep their score regardless of length.
        scores = np.array([[10.0, 0.0, 10.0]])
        lengths = np.array([[100, 50, 300]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "linear", 1.0)
        self.assertEqual(out[0, 1], 0.0)

    def test_linear_shortest_keeps_full_reward(self):
        scores = np.array([[10.0, 10.0, 10.0]])
        lengths = np.array([[100, 200, 300]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "linear", 1.0)
        self.assertAlmostEqual(out[0, 0], 10.0)
        # alpha=1.0, L=200, L_min=100 => factor = 1 - 1*(100/100) = 0
        self.assertAlmostEqual(out[0, 1], 0.0)
        # alpha=1.0, L=300 => factor = max(0, 1 - 2) = 0
        self.assertAlmostEqual(out[0, 2], 0.0)

    def test_linear_partial_decay(self):
        scores = np.array([[10.0, 10.0]])
        lengths = np.array([[100, 150]])
        # alpha=0.5, L=150 => factor = 1 - 0.5*(50/100) = 0.75
        out = lrs.apply_length_reward_shaping(scores, lengths, "linear", 0.5)
        self.assertAlmostEqual(out[0, 0], 10.0)
        self.assertAlmostEqual(out[0, 1], 7.5)

    def test_exponential_decay(self):
        scores = np.array([[10.0, 10.0]])
        lengths = np.array([[100, 200]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "exponential", 1.0)
        self.assertAlmostEqual(out[0, 0], 10.0)
        # lambda=1.0, rel=1.0 => exp(-1) * 10
        self.assertAlmostEqual(out[0, 1], 10.0 * np.exp(-1.0))

    def test_rank_based(self):
        scores = np.array([[10.0, 10.0, 10.0, 10.0]])
        lengths = np.array([[100, 200, 300, 400]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "rank", 0.0)
        # ranks are 0,1,2,3; factors are 1, 1/2, 1/3, 1/4
        np.testing.assert_allclose(out[0], [10.0, 5.0, 10.0 / 3.0, 2.5])

    def test_rank_based_with_ties(self):
        scores = np.array([[10.0, 10.0, 10.0]])
        lengths = np.array([[100, 100, 200]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "rank", 0.0)
        # tied at rank 0 => both get factor 1; longer gets rank 1 => factor 0.5
        self.assertAlmostEqual(out[0, 0], 10.0)
        self.assertAlmostEqual(out[0, 1], 10.0)
        self.assertAlmostEqual(out[0, 2], 5.0)

    def test_binary_shortest(self):
        scores = np.array([[10.0, 10.0, 0.0, 10.0]])
        lengths = np.array([[200, 100, 50, 300]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "binary_shortest", 0.0)
        # incorrect (50) stays 0; shortest correct is index 1 (length 100); others zeroed.
        np.testing.assert_array_equal(out[0], [0.0, 10.0, 0.0, 0.0])

    def test_soft_threshold(self):
        # Two correct: median = 150; threshold = 1.0 * 150.
        # L=100 (below threshold): factor 1; L=200 (above): factor 1 - (200-150)/150 = 2/3.
        scores = np.array([[10.0, 10.0]])
        lengths = np.array([[100, 200]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "soft_threshold", 1.0)
        self.assertAlmostEqual(out[0, 0], 10.0)
        self.assertAlmostEqual(out[0, 1], 10.0 * (1.0 - 50.0 / 150.0))

    def test_warmup_blend(self):
        scores = np.array([[10.0, 10.0]])
        lengths = np.array([[100, 200]])
        # Full shaping zeroes the longer correct response (alpha=1 linear).
        # warmup_weight=0.5 should average shaped (0) and unshaped (10) => 5.
        out = lrs.apply_length_reward_shaping(scores, lengths, "linear", 1.0, warmup_weight=0.5)
        self.assertAlmostEqual(out[0, 0], 10.0)
        self.assertAlmostEqual(out[0, 1], 5.0)

    def test_independent_groups(self):
        # Per-group shaping: each row computes its own L_min.
        scores = np.array([[10.0, 10.0], [10.0, 10.0]])
        lengths = np.array([[100, 200], [400, 500]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "linear", 1.0)
        # Group 0: L_min=100, factors [1, 0]; Group 1: L_min=400, factors [1, 1-100/400=0.75]
        np.testing.assert_allclose(out[0], [10.0, 0.0])
        np.testing.assert_allclose(out[1], [10.0, 7.5])

    def test_all_incorrect_group_unchanged(self):
        scores = np.array([[0.0, 0.0]])
        lengths = np.array([[100, 200]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "linear", 1.0)
        np.testing.assert_array_equal(out, scores)

    def test_unknown_method_raises(self):
        scores = np.array([[10.0]])
        lengths = np.array([[100]])
        with self.assertRaises(ValueError):
            lrs.apply_length_reward_shaping(scores, lengths, "bogus", 1.0)

    def test_shape_mismatch_raises(self):
        scores = np.zeros((2, 3))
        lengths = np.zeros((2, 4))
        with self.assertRaises(ValueError):
            lrs.apply_length_reward_shaping(scores, lengths, "linear", 1.0)

    def test_correctness_threshold_excludes_format_only(self):
        # Additive format reward gives a well-formatted-but-wrong response score
        # equal to format_reward (1.0 by default). With threshold = 1.0, that
        # response should not set L_min.
        scores = np.array([[1.0, 11.0, 11.0]])  # format-only, fully-correct, fully-correct
        lengths = np.array([[10, 200, 250]])
        out = lrs.apply_length_reward_shaping(scores, lengths, "linear", 1.0, correctness_threshold=1.0)
        # The format-only response (score 1.0) is below threshold so it stays unchanged
        # AND does not contribute to L_min. L_min = 200 (the shortest fully-correct).
        # The response at length 250 gets factor 1 - 50/200 = 0.75 -> 11 * 0.75 = 8.25.
        self.assertAlmostEqual(out[0, 0], 1.0)
        self.assertAlmostEqual(out[0, 1], 11.0)
        self.assertAlmostEqual(out[0, 2], 8.25)


class TestComputeWarmupWeight(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(lrs.compute_warmup_weight(0, 1000, "constant", 0.25, 0.3, None), 1.0)
        self.assertEqual(lrs.compute_warmup_weight(500, 1000, "constant", 0.25, 0.3, 0.5), 1.0)

    def test_linear_warmup(self):
        # warmup_fraction=0.5, num_steps=100 => end_step=50.
        self.assertAlmostEqual(lrs.compute_warmup_weight(0, 100, "linear", 0.5, 0.3, None), 0.0)
        self.assertAlmostEqual(lrs.compute_warmup_weight(25, 100, "linear", 0.5, 0.3, None), 0.5)
        self.assertAlmostEqual(lrs.compute_warmup_weight(50, 100, "linear", 0.5, 0.3, None), 1.0)
        self.assertAlmostEqual(lrs.compute_warmup_weight(75, 100, "linear", 0.5, 0.3, None), 1.0)

    def test_solve_rate_gating(self):
        # below threshold => 0; at/above => 1.
        self.assertEqual(lrs.compute_warmup_weight(0, 100, "solve_rate", 0.0, 0.3, 0.1), 0.0)
        self.assertEqual(lrs.compute_warmup_weight(0, 100, "solve_rate", 0.0, 0.3, 0.3), 1.0)
        self.assertEqual(lrs.compute_warmup_weight(0, 100, "solve_rate", 0.0, 0.3, 0.5), 1.0)

    def test_solve_rate_missing_value(self):
        self.assertEqual(lrs.compute_warmup_weight(0, 100, "solve_rate", 0.0, 0.3, group_solve_rate=None), 0.0)

    def test_unknown_warmup_raises(self):
        with self.assertRaises(ValueError):
            lrs.compute_warmup_weight(0, 100, "bogus", 0.5, 0.3, None)


if __name__ == "__main__":
    unittest.main()
