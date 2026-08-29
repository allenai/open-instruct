import unittest

import numpy as np

from open_instruct.data_loader_utils import (
    NeverGiveUpAccumulationState,
    compute_grouped_advantages,
    get_never_give_up_chain_id,
    get_never_give_up_retry_suffix,
    pop_pending_never_give_up_state,
    should_accept_never_give_up_batch,
    store_pending_never_give_up_state,
)


class TestNeverGiveUpChainIds(unittest.TestCase):
    def test_retry_suffix_increments_existing_suffix(self):
        self.assertEqual(get_never_give_up_retry_suffix("7_0", epoch_number=7, index=0), "_1")
        self.assertEqual(get_never_give_up_retry_suffix("7_0_1", epoch_number=7, index=0), "_2")
        self.assertEqual(get_never_give_up_retry_suffix("7_0_4", epoch_number=7, index=0), "_5")

    def test_chain_id_strips_retry_suffix(self):
        self.assertEqual(get_never_give_up_chain_id("7_0"), "7_0")
        self.assertEqual(get_never_give_up_chain_id("7_0_1"), "7_0")
        self.assertEqual(get_never_give_up_chain_id("12_345_9"), "12_345")

    def test_chain_id_rejects_unexpected_format(self):
        with self.assertRaises(ValueError):
            get_never_give_up_chain_id("7_0_1_2")


class TestShouldAcceptNeverGiveUpBatch(unittest.TestCase):
    def test_no_pending_falls_back_to_zero_std_filter(self):
        self.assertTrue(should_accept_never_give_up_batch(np.array([0.0, 1.0]), None, filter_zero_std_samples=True))
        self.assertFalse(should_accept_never_give_up_batch(np.array([1.0, 1.0]), None, filter_zero_std_samples=True))

    def test_pending_accepts_only_when_strictly_better(self):
        self.assertTrue(should_accept_never_give_up_batch(np.array([0.0, 5.0]), 1.0, filter_zero_std_samples=True))
        self.assertFalse(should_accept_never_give_up_batch(np.array([0.0, 1.0]), 1.0, filter_zero_std_samples=True))
        self.assertFalse(should_accept_never_give_up_batch(np.array([1.0, 1.0]), 1.0, filter_zero_std_samples=True))

    def test_disabled_filter_always_accepts(self):
        self.assertTrue(should_accept_never_give_up_batch(np.array([1.0, 1.0]), 5.0, filter_zero_std_samples=False))


class TestPendingNeverGiveUpState(unittest.TestCase):
    def test_store_then_pop_round_trips_counts(self):
        state = NeverGiveUpAccumulationState()
        popped = pop_pending_never_give_up_state(state, "3_1", current_model_step=10, maintain_pending_ngu_age=4)
        self.assertEqual(popped.attempt_count, 0)
        self.assertIsNone(popped.best_reward)

        popped.response_count += 4
        popped.reward_sum += 2.0
        store_pending_never_give_up_state(state, "3_1", popped, best_reward=1.0, attempt_count=1)

        again = pop_pending_never_give_up_state(state, "3_1", current_model_step=11, maintain_pending_ngu_age=4)
        self.assertEqual(again.response_count, 4)
        self.assertEqual(again.reward_sum, 2.0)
        self.assertEqual(again.attempt_count, 1)
        self.assertEqual(again.best_reward, 1.0)
        # Popping consumes the entry.
        self.assertNotIn("3_1", state.pending_best_reward)

    def test_negative_age_keeps_all_pending_completions(self):
        state = NeverGiveUpAccumulationState()
        state.pending_results["3_1"] = [_FakeResult(model_step=0)]
        state.pending_metrics["3_1"] = [None]
        popped = pop_pending_never_give_up_state(state, "3_1", current_model_step=100, maintain_pending_ngu_age=-1)
        self.assertEqual(len(popped.results), 1)


class _FakeResult:
    def __init__(self, model_step: int):
        self.model_step = model_step


class TestComputeGroupedAdvantages(unittest.TestCase):
    def _inline_reference(self, scores: np.ndarray, n: int, norm: str) -> np.ndarray:
        """The pre-NGU inline math from DataPreparationActor, for regression comparison."""
        per_prompt = scores.reshape(-1, n)
        mean = np.repeat(per_prompt.mean(axis=-1), n, axis=0)
        std = np.repeat(per_prompt.std(axis=-1), n, axis=0)
        if norm == "standard":
            return (scores - mean) / (std + 1e-8)
        return scores - mean

    def test_matches_inline_reference_without_baseline(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=12).astype(np.float64)
        for norm in ("centered", "standard"):
            expected = self._inline_reference(scores, 4, norm)
            got = compute_grouped_advantages(scores, [4, 4, 4], advantage_normalization_type=norm)
            np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_anchor_pos_is_noop_when_baseline_equals_batch(self):
        scores = np.array([0.0, 0.0, 10.0, 0.0], dtype=np.float64)
        baseline = compute_grouped_advantages(
            scores, [4], prompt_baseline_sample_counts=[4], prompt_baseline_reward_sums=[float(scores.sum())]
        )
        plain = compute_grouped_advantages(scores, [4])
        np.testing.assert_allclose(baseline, plain, rtol=1e-9, atol=1e-9)

    def test_anchor_pos_rescales_and_recenters_when_chain_baseline_differs(self):
        # Batch has one positive (10) and three zeros; the chain saw 8 samples summing to 10,
        # so the chain mean (10/8 = 1.25) is lower than the batch mean (2.5).
        scores = np.array([0.0, 0.0, 10.0, 0.0], dtype=np.float64)
        adv = compute_grouped_advantages(
            scores, [4], prompt_baseline_sample_counts=[8], prompt_baseline_reward_sums=[10.0]
        )
        # Positive sample keeps its advantage against the chain mean.
        self.assertAlmostEqual(adv[2], 10.0 - 1.25, places=6)
        # Group is re-centered to sum to zero.
        self.assertAlmostEqual(float(adv.sum()), 0.0, places=6)
        # Negative samples stay negative.
        self.assertTrue((adv[[0, 1, 3]] < 0).all())

    def test_variable_group_sizes(self):
        scores = np.array([0.0, 2.0, 4.0, 1.0, 1.0], dtype=np.float64)
        adv = compute_grouped_advantages(scores, [3, 2])
        np.testing.assert_allclose(adv[:3], scores[:3] - 2.0, atol=1e-9)
        np.testing.assert_allclose(adv[3:], scores[3:] - 1.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
