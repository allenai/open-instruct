import tempfile
import unittest

import numpy as np
import parameterized
import torch
from datasets import Dataset

from open_instruct import data_loader
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO


def _make_dpo_dataset(num_samples: int, max_seq_length: int) -> Dataset:
    rng = torch.Generator().manual_seed(42)
    data = {
        "chosen_input_ids": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_labels": [],
        "index": list(range(num_samples)),
    }
    for _ in range(num_samples):
        chosen_len = torch.randint(1, max_seq_length + 1, (1,), generator=rng).item()
        rejected_len = torch.randint(1, max_seq_length + 1, (1,), generator=rng).item()
        data["chosen_input_ids"].append(torch.randint(0, 1000, (chosen_len,), generator=rng))
        data["chosen_labels"].append(torch.randint(0, 1000, (chosen_len,), generator=rng))
        data["rejected_input_ids"].append(torch.randint(0, 1000, (rejected_len,), generator=rng))
        data["rejected_labels"].append(torch.randint(0, 1000, (rejected_len,), generator=rng))
    ds = Dataset.from_dict(data)
    ds.set_format(type="pt")
    return ds


class TestWorldAwarePacking(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("olmo3_7b_dp2", 16384, 8, 2, True, 200),
            ("olmo3_7b_dp4", 16384, 16, 4, True, 200),
            ("olmo3_32b_dp4", 8192, 8, 4, True, 200),
            ("olmo3_32b_dp8", 8192, 16, 8, True, 200),
            ("debug_multi_node", 16384, 32, 2, True, 200),
            ("olmo3_7b_dp2_no_drop", 16384, 8, 2, False, 200),
            ("olmo3_32b_dp4_no_drop", 8192, 8, 4, False, 200),
        ]
    )
    def test_packing_equal_batches_across_ranks(
        self, _name, max_seq_length, global_batch_size, dp_world_size, drop_last, num_samples
    ):
        dataset = _make_dpo_dataset(num_samples, max_seq_length)
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)

        with tempfile.TemporaryDirectory() as work_dir:
            loaders = [
                data_loader.HFDataLoader(
                    dataset=dataset,
                    batch_size=global_batch_size,
                    seed=42,
                    dp_rank=rank,
                    dp_world_size=dp_world_size,
                    work_dir=work_dir,
                    collator=collator,
                    drop_last=drop_last,
                )
                for rank in range(dp_world_size)
            ]

            batch_counts = [loader.total_batches for loader in loaders]
            self.assertTrue(
                all(c == batch_counts[0] for c in batch_counts), f"Batch counts differ across ranks: {batch_counts}"
            )

            all_indices = set()
            for loader in loaders:
                for batch in loader:
                    if "index" in batch:
                        all_indices.update(batch["index"].tolist())

            if not drop_last:
                expected_indices = set(range(num_samples))
                self.assertEqual(all_indices, expected_indices, f"Missing indices: {expected_indices - all_indices}")


class TestToggleBudgetTracker(unittest.TestCase):
    def test_disabled_when_m_zero(self):
        tracker = data_loader.ToggleBudgetTracker(m=0, lambda_=0.5, percentile=90.0, warmup_steps=0)
        scores = np.array([1.0, 0.0, 1.0, 0.0])
        out_scores, metrics = tracker.maybe_apply(
            step=10,
            scores=scores,
            sequence_lengths=np.array([100, 200, 300, 400]),
            datasets=["d", "d", "d", "d"],
            num_samples_per_prompt=4,
            max_possible_score=1.0,
        )
        np.testing.assert_array_equal(out_scores, scores)
        self.assertEqual(metrics, {})

    def test_budget_none_until_correct_seen(self):
        tracker = data_loader.ToggleBudgetTracker(m=2, lambda_=0.5, percentile=50.0, warmup_steps=0)
        self.assertIsNone(tracker.budget("d"))
        tracker.update(["d", "d"], np.array([10, 20]), np.array([0.0, 0.0]), 1.0)
        self.assertIsNone(tracker.budget("d"))
        tracker.update(["d"], np.array([30]), np.array([1.0]), 1.0)
        self.assertEqual(tracker.budget("d"), 30.0)

    def test_percentile_matches_numpy(self):
        tracker = data_loader.ToggleBudgetTracker(m=2, lambda_=0.5, percentile=80.0, warmup_steps=0)
        lengths = np.array([10, 20, 30, 40, 50])
        tracker.update(["d"] * 5, lengths, np.array([1.0] * 5), 1.0)
        self.assertAlmostEqual(tracker.budget("d"), float(np.percentile(lengths, 80.0)))

    def test_list_valued_dataset_field(self):
        # Per-sample dataset can be list[str] (multiple verifier sources). Tracker must
        # not crash on `unhashable type: list`.
        tracker = data_loader.ToggleBudgetTracker(m=2, lambda_=0.5, percentile=50.0, warmup_steps=0)
        datasets = [["math", "code"], ["math", "code"], ["math", "code"], ["math", "code"]]
        tracker.update(datasets, np.array([10, 20, 30, 40]), np.array([1.0, 1.0, 1.0, 1.0]), 1.0)
        self.assertEqual(tracker.budget(["math", "code"]), 25.0)
        scores = np.array([1.0, 1.0, 1.0, 0.0])
        out_scores, metrics = tracker.maybe_apply(
            step=0,
            scores=scores,
            sequence_lengths=np.array([5, 50, 15, 100]),
            datasets=datasets,
            num_samples_per_prompt=4,
            max_possible_score=1.0,
        )
        self.assertEqual(metrics["toggle/phase"], 0)
        self.assertIn("toggle/budget/math|code", metrics)

    def test_per_dataset_isolation(self):
        tracker = data_loader.ToggleBudgetTracker(m=2, lambda_=0.5, percentile=50.0, warmup_steps=0)
        tracker.update(["a", "b"], np.array([100, 200]), np.array([1.0, 1.0]), 1.0)
        self.assertEqual(tracker.budget("a"), 100.0)
        self.assertEqual(tracker.budget("b"), 200.0)

    def test_phase1_passthrough(self):
        tracker = data_loader.ToggleBudgetTracker(m=2, lambda_=0.5, percentile=50.0, warmup_steps=0)
        # step // m == 0 -> Phase0; step // m == 1 -> Phase1
        scores = np.array([1.0, 1.0, 0.0, 0.0])
        out_scores, metrics = tracker.maybe_apply(
            step=2,
            scores=scores,
            sequence_lengths=np.array([5, 1000, 5, 5]),
            datasets=["d"] * 4,
            num_samples_per_prompt=4,
            max_possible_score=1.0,
        )
        np.testing.assert_array_equal(out_scores, scores)
        self.assertEqual(metrics["toggle/phase"], 1)

    def test_phase0_zeros_overlong_correct(self):
        tracker = data_loader.ToggleBudgetTracker(m=2, lambda_=0.5, percentile=50.0, warmup_steps=0)
        # Seed budget for dataset "d" so percentile-50 == 20 (median of [10, 20, 30]).
        tracker.update(["d"] * 3, np.array([10, 20, 30]), np.array([1.0, 1.0, 1.0]), 1.0)
        # K=4 single prompt, mean_acc = 0.75 > lambda=0.5 -> mask enforced.
        scores = np.array([1.0, 1.0, 1.0, 0.0])
        sequence_lengths = np.array([5, 50, 15, 100])
        out_scores, metrics = tracker.maybe_apply(
            step=0,
            scores=scores,
            sequence_lengths=sequence_lengths,
            datasets=["d"] * 4,
            num_samples_per_prompt=4,
            max_possible_score=1.0,
        )
        # idx 0,2: len <= budget(20) -> kept. idx 1: 50 > 20 and correct -> zeroed.
        # idx 3: 100 > 20 but already zero score -> still zero.
        np.testing.assert_array_equal(out_scores, np.array([1.0, 0.0, 1.0, 0.0]))
        self.assertEqual(metrics["toggle/phase"], 0)

    def test_phase0_lambda_protects_hard_prompts(self):
        tracker = data_loader.ToggleBudgetTracker(m=2, lambda_=0.5, percentile=50.0, warmup_steps=0)
        tracker.update(["d"] * 3, np.array([10, 20, 30]), np.array([1.0, 1.0, 1.0]), 1.0)
        # mean_acc = 0.25 < lambda=0.5 -> all kept regardless of length.
        scores = np.array([1.0, 0.0, 0.0, 0.0])
        out_scores, _ = tracker.maybe_apply(
            step=0,
            scores=scores,
            sequence_lengths=np.array([5, 5, 5, 1000]),
            datasets=["d"] * 4,
            num_samples_per_prompt=4,
            max_possible_score=1.0,
        )
        np.testing.assert_array_equal(out_scores, scores)

    def test_warmup_blocks_phase0(self):
        tracker = data_loader.ToggleBudgetTracker(m=2, lambda_=0.5, percentile=50.0, warmup_steps=10)
        tracker.update(["d"] * 3, np.array([10, 20, 30]), np.array([1.0, 1.0, 1.0]), 1.0)
        scores = np.array([1.0, 1.0, 1.0, 1.0])
        out_scores, metrics = tracker.maybe_apply(
            step=4,
            scores=scores,
            sequence_lengths=np.array([1000, 1000, 1000, 1000]),
            datasets=["d"] * 4,
            num_samples_per_prompt=4,
            max_possible_score=1.0,
        )
        np.testing.assert_array_equal(out_scores, scores)
        self.assertEqual(metrics["toggle/phase"], 1)


if __name__ == "__main__":
    unittest.main()
