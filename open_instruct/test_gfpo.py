import unittest

import numpy as np

from open_instruct import gfpo


class TestApplyGFPOFilter(unittest.TestCase):
    def test_shortest_keeps_k_shortest(self):
        scores = np.array([[1.0, 1.0, 0.0, 1.0]])
        lengths = np.array([[300, 100, 50, 200]])
        mask = gfpo.apply_gfpo_filter(scores, lengths, retain_k=2, metric="shortest")
        # Two shortest are lengths 50 (idx 2) and 100 (idx 1).
        np.testing.assert_array_equal(mask, [[False, True, True, False]])

    def test_shortest_ignores_correctness(self):
        # The filter is length-only: a short *wrong* response is kept over a long correct one.
        scores = np.array([[1.0, 0.0]])  # idx0 correct/long, idx1 wrong/short
        lengths = np.array([[500, 10]])
        mask = gfpo.apply_gfpo_filter(scores, lengths, retain_k=1, metric="shortest")
        np.testing.assert_array_equal(mask, [[False, True]])

    def test_token_efficiency_keeps_highest_reward_per_token(self):
        scores = np.array([[1.0, 1.0, 0.0]])
        lengths = np.array([[100, 50, 10]])
        # reward/length = [0.01, 0.02, 0.0]; keep the 2 highest -> idx 0 and 1.
        mask = gfpo.apply_gfpo_filter(scores, lengths, retain_k=2, metric="token_efficiency")
        np.testing.assert_array_equal(mask, [[True, True, False]])

    def test_token_efficiency_guards_zero_length(self):
        scores = np.array([[1.0, 0.0]])
        lengths = np.array([[0, 5]])  # zero-length guarded to 1.0 -> efficiency 1.0 vs 0.0
        mask = gfpo.apply_gfpo_filter(scores, lengths, retain_k=1, metric="token_efficiency")
        np.testing.assert_array_equal(mask, [[True, False]])

    def test_each_row_keeps_exactly_k(self):
        scores = np.ones((3, 6))
        lengths = np.tile(np.arange(6), (3, 1))
        mask = gfpo.apply_gfpo_filter(scores, lengths, retain_k=4, metric="shortest")
        np.testing.assert_array_equal(mask.sum(axis=1), [4, 4, 4])

    def test_k_clamped_to_group_size(self):
        scores = np.ones((1, 3))
        lengths = np.array([[1, 2, 3]])
        mask = gfpo.apply_gfpo_filter(scores, lengths, retain_k=10, metric="shortest")
        np.testing.assert_array_equal(mask, [[True, True, True]])

    def test_independent_rows(self):
        scores = np.ones((2, 3))
        lengths = np.array([[10, 20, 30], [30, 10, 20]])
        mask = gfpo.apply_gfpo_filter(scores, lengths, retain_k=1, metric="shortest")
        np.testing.assert_array_equal(mask, [[True, False, False], [False, True, False]])

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            gfpo.apply_gfpo_filter(np.ones((1, 2)), np.ones((1, 2)), 1, "bogus")

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gfpo.apply_gfpo_filter(np.ones((1, 2)), np.ones((1, 3)), 1, "shortest")


class TestComputeGFPOAdvantages(unittest.TestCase):
    def test_non_retained_get_zero_advantage(self):
        scores = np.array([[1.0, 1.0, 0.0, 0.0]])
        mask = np.array([[True, True, False, False]])
        adv = gfpo.compute_gfpo_advantages(scores, mask, "standard")
        # Dropped responses (idx 2, 3) must be exactly 0.
        self.assertEqual(adv[0, 2], 0.0)
        self.assertEqual(adv[0, 3], 0.0)

    def test_subset_normalization_standard(self):
        # Retained subset = {1.0, 0.0}: mu=0.5, sigma=0.5. Advantages over subset:
        # (1-0.5)/0.5 = 1, (0-0.5)/0.5 = -1.
        scores = np.array([[1.0, 0.0, 1.0]])
        mask = np.array([[True, True, False]])
        adv = gfpo.compute_gfpo_advantages(scores, mask, "standard")
        self.assertAlmostEqual(adv[0, 0], 1.0, places=5)
        self.assertAlmostEqual(adv[0, 1], -1.0, places=5)
        self.assertEqual(adv[0, 2], 0.0)

    def test_subset_normalization_centered(self):
        scores = np.array([[1.0, 0.0, 1.0]])
        mask = np.array([[True, True, False]])
        adv = gfpo.compute_gfpo_advantages(scores, mask, "centered")
        # mu over subset = 0.5; centered advantages without std division.
        self.assertAlmostEqual(adv[0, 0], 0.5)
        self.assertAlmostEqual(adv[0, 1], -0.5)
        self.assertEqual(adv[0, 2], 0.0)

    def test_baseline_excludes_dropped_responses(self):
        # A dropped high reward must not raise the baseline of the retained subset.
        scores = np.array([[1.0, 1.0, 9.0]])  # idx2 dropped
        mask = np.array([[True, True, False]])
        adv = gfpo.compute_gfpo_advantages(scores, mask, "standard")
        # Retained subset is {1.0, 1.0}: mu=1, sigma=0 -> advantages 0 (not negative
        # as they would be if the dropped 9.0 inflated the mean).
        self.assertAlmostEqual(adv[0, 0], 0.0, places=5)
        self.assertAlmostEqual(adv[0, 1], 0.0, places=5)

    def test_degenerate_subset_zero_std(self):
        # All retained equal -> sigma_S = 0 -> advantages 0 (no blowup from 1e-8).
        scores = np.array([[2.0, 2.0, 5.0]])
        mask = np.array([[True, True, False]])
        adv = gfpo.compute_gfpo_advantages(scores, mask, "standard")
        np.testing.assert_allclose(adv[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gfpo.compute_gfpo_advantages(np.ones((1, 3)), np.ones((1, 4), dtype=bool), "standard")

    def test_unknown_normalization_raises(self):
        with self.assertRaises(ValueError):
            gfpo.compute_gfpo_advantages(np.ones((1, 2)), np.ones((1, 2), dtype=bool), "bogus")


if __name__ == "__main__":
    unittest.main()
