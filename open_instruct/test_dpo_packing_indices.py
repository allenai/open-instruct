"""Tests for DPO packing collator index handling.

This test verifies that the TensorDataCollatorWithFlatteningDPO correctly
handles indices when packing sequences, especially when truncation occurs.
"""

import unittest

import torch
from parameterized import parameterized

from open_instruct.padding_free_collator import (
    TensorDataCollatorWithFlatteningDPO,
    concatenated_inputs,
    get_batch_logps,
)


def make_dpo_features(
    num_samples: int, chosen_lengths: list[int], rejected_lengths: list[int], start_index: int = 0
) -> list[dict]:
    """Create mock DPO features with known indices and sequence lengths."""
    features = []
    for i in range(num_samples):
        chosen_len = chosen_lengths[i % len(chosen_lengths)]
        rejected_len = rejected_lengths[i % len(rejected_lengths)]
        features.append(
            {
                "chosen_input_ids": torch.ones(chosen_len, dtype=torch.long),
                "chosen_labels": torch.ones(chosen_len, dtype=torch.long),
                "rejected_input_ids": torch.ones(rejected_len, dtype=torch.long),
                "rejected_labels": torch.ones(rejected_len, dtype=torch.long),
                "index": start_index + i,
            }
        )
    return features


class TestDPOPackingIndices(unittest.TestCase):
    """Tests for DPO packing collator index handling."""

    def test_indices_preserved_no_truncation(self):
        """Test that indices are preserved when no truncation occurs."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=1000)
        features = make_dpo_features(num_samples=4, chosen_lengths=[50], rejected_lengths=[50], start_index=10)

        batch = collator(features)

        self.assertIn("index", batch)
        expected_indices = torch.tensor([10, 11, 12, 13])
        torch.testing.assert_close(batch["index"], expected_indices)

    def test_indices_preserved_with_padding(self):
        """Test that indices are preserved when padding is needed."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=500)
        features = make_dpo_features(num_samples=4, chosen_lengths=[50], rejected_lengths=[50], start_index=0)

        batch = collator(features)

        self.assertIn("index", batch)
        self.assertEqual(len(batch["index"]), 4)
        expected_indices = torch.tensor([0, 1, 2, 3])
        torch.testing.assert_close(batch["index"], expected_indices)

    def test_indices_truncated_when_exceeding_max_length(self):
        """Test that indices are correctly truncated when packed length exceeds max."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=150)
        features = make_dpo_features(num_samples=4, chosen_lengths=[100], rejected_lengths=[100], start_index=0)

        batch = collator(features)

        self.assertIn("index", batch)
        self.assertLess(len(batch["index"]), 4)

    def test_cu_seq_lens_matches_index_count(self):
        """Test that cu_seq_lens has correct length relative to indices."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=1000)
        features = make_dpo_features(num_samples=4, chosen_lengths=[50], rejected_lengths=[50], start_index=0)

        batch = collator(features)

        num_indices = len(batch["index"])
        self.assertEqual(len(batch["chosen_cu_seq_lens_k"]), num_indices + 1)
        self.assertEqual(len(batch["rejected_cu_seq_lens_k"]), num_indices + 1)

    def test_cu_seq_lens_matches_after_truncation(self):
        """Test that cu_seq_lens is correctly synced after truncation."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=200)
        features = make_dpo_features(num_samples=4, chosen_lengths=[100], rejected_lengths=[100], start_index=0)

        batch = collator(features)

        num_indices = len(batch["index"])
        self.assertEqual(
            len(batch["chosen_cu_seq_lens_k"]),
            num_indices + 1,
            f"chosen_cu_seq_lens has {len(batch['chosen_cu_seq_lens_k'])} entries "
            f"but should have {num_indices + 1} for {num_indices} indices",
        )
        self.assertEqual(
            len(batch["rejected_cu_seq_lens_k"]),
            num_indices + 1,
            f"rejected_cu_seq_lens has {len(batch['rejected_cu_seq_lens_k'])} entries "
            f"but should have {num_indices + 1} for {num_indices} indices",
        )

    def test_asymmetric_truncation_chosen_rejected(self):
        """Test handling when chosen and rejected have different truncation amounts."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=250)
        features = make_dpo_features(num_samples=4, chosen_lengths=[100], rejected_lengths=[50], start_index=0)

        batch = collator(features)

        num_indices = len(batch["index"])
        self.assertEqual(len(batch["chosen_cu_seq_lens_k"]), num_indices + 1)
        self.assertEqual(len(batch["rejected_cu_seq_lens_k"]), num_indices + 1)

    def test_concatenated_inputs_returns_correct_bs(self):
        """Test that concatenated_inputs returns correct batch size."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=1000)
        features = make_dpo_features(num_samples=4, chosen_lengths=[50], rejected_lengths=[50], start_index=0)

        batch = collator(features)
        concat_batch, bs = concatenated_inputs(batch)

        self.assertEqual(bs, len(batch["index"]))
        self.assertIn("concatenated_cu_seq_lens_k", concat_batch)
        self.assertEqual(len(concat_batch["concatenated_cu_seq_lens_k"]), 2 * bs + 1)

    @parameterized.expand([("no_truncation", 1000, 4), ("slight_truncation", 300, 3), ("heavy_truncation", 150, 1)])
    def test_logps_count_matches_indices(self, name, max_seq_length, expected_min_indices):
        """Test that get_batch_logps returns same count as indices."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)
        features = make_dpo_features(num_samples=4, chosen_lengths=[100], rejected_lengths=[100], start_index=0)

        batch = collator(features)
        concat_batch, bs = concatenated_inputs(batch)

        num_indices = len(batch["index"])
        self.assertGreaterEqual(num_indices, expected_min_indices)

        logits = torch.randn(1, concat_batch["concatenated_input_ids"].shape[1], 100)
        labels = concat_batch["concatenated_labels"]
        cu_seq_lens = concat_batch["concatenated_cu_seq_lens_k"]

        logps = get_batch_logps(logits, labels, cu_seq_lens)
        expected_logps_count = 2 * bs

        self.assertEqual(
            len(logps),
            expected_logps_count,
            f"get_batch_logps returned {len(logps)} logps but expected {expected_logps_count} "
            f"(2 * {bs} for chosen + rejected). batch has {num_indices} indices.",
        )

        chosen_logps = logps[:bs]
        rejected_logps = logps[bs:]
        self.assertEqual(len(chosen_logps), num_indices)
        self.assertEqual(len(rejected_logps), num_indices)

    def test_simulate_reference_cache_no_truncation(self):
        """Test cache assignment with no truncation (production-like scenario)."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=1000)
        num_total_samples = 16
        all_features = make_dpo_features(
            num_samples=num_total_samples, chosen_lengths=[100], rejected_lengths=[100], start_index=0
        )

        chosen_tensor = torch.full((num_total_samples,), float("-inf"))
        rejected_tensor = torch.full((num_total_samples,), float("-inf"))

        batch_size = 4
        for batch_start in range(0, num_total_samples, batch_size):
            batch_features = all_features[batch_start : batch_start + batch_size]
            batch = collator(batch_features)

            concat_batch, bs = concatenated_inputs(batch)
            logits = torch.randn(1, concat_batch["concatenated_input_ids"].shape[1], 100)
            logps = get_batch_logps(
                logits, concat_batch["concatenated_labels"], concat_batch["concatenated_cu_seq_lens_k"]
            )
            chosen_logps = logps[:bs]
            rejected_logps = logps[bs:]

            self.assertEqual(
                len(chosen_logps),
                len(batch["index"]),
                f"Mismatch: batch['index'] has {len(batch['index'])} elements, "
                f"but chosen_logps has {len(chosen_logps)} elements",
            )

            chosen_tensor[batch["index"]] = chosen_logps
            rejected_tensor[batch["index"]] = rejected_logps

        missing_chosen = torch.where(chosen_tensor == float("-inf"))[0]
        missing_rejected = torch.where(rejected_tensor == float("-inf"))[0]

        self.assertEqual(
            len(missing_chosen), 0, f"Missing {len(missing_chosen)} chosen indices: {missing_chosen[:10].tolist()}"
        )
        self.assertEqual(
            len(missing_rejected),
            0,
            f"Missing {len(missing_rejected)} rejected indices: {missing_rejected[:10].tolist()}",
        )

    def test_simulate_reference_cache_with_truncation(self):
        """Test cache assignment with truncation - indices should be missing."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=300)
        num_total_samples = 16
        all_features = make_dpo_features(
            num_samples=num_total_samples, chosen_lengths=[100], rejected_lengths=[100], start_index=0
        )

        chosen_tensor = torch.full((num_total_samples,), float("-inf"))
        rejected_tensor = torch.full((num_total_samples,), float("-inf"))

        batch_size = 4
        for batch_start in range(0, num_total_samples, batch_size):
            batch_features = all_features[batch_start : batch_start + batch_size]
            batch = collator(batch_features)

            concat_batch, bs = concatenated_inputs(batch)
            logits = torch.randn(1, concat_batch["concatenated_input_ids"].shape[1], 100)
            logps = get_batch_logps(
                logits, concat_batch["concatenated_labels"], concat_batch["concatenated_cu_seq_lens_k"]
            )
            chosen_logps = logps[:bs]
            rejected_logps = logps[bs:]

            self.assertEqual(
                len(chosen_logps),
                len(batch["index"]),
                f"Mismatch: batch['index'] has {len(batch['index'])} elements, "
                f"but chosen_logps has {len(chosen_logps)} elements",
            )

            chosen_tensor[batch["index"]] = chosen_logps
            rejected_tensor[batch["index"]] = rejected_logps

        missing_chosen = torch.where(chosen_tensor == float("-inf"))[0]

        self.assertGreater(len(missing_chosen), 0, "Expected some missing indices due to truncation")
        self.assertEqual(
            missing_chosen.tolist(), [3, 7, 11, 15], "Expected the last index in each batch to be truncated"
        )

    def test_concatenated_cu_seq_lens_with_padding(self):
        """Test that cu_seq_lens correctly accounts for padding in concatenation."""
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=500)
        features = make_dpo_features(num_samples=2, chosen_lengths=[50], rejected_lengths=[50], start_index=0)

        batch = collator(features)
        concat_batch, bs = concatenated_inputs(batch)

        self.assertEqual(bs, 2)
        self.assertEqual(batch["chosen_input_ids"].shape[-1], 500)
        self.assertEqual(batch["rejected_input_ids"].shape[-1], 500)
        self.assertEqual(concat_batch["concatenated_input_ids"].shape[-1], 1000)

        cu_seq_lens = concat_batch["concatenated_cu_seq_lens_k"]
        self.assertEqual(len(cu_seq_lens), 5)
        self.assertEqual(cu_seq_lens[0].item(), 0)
        self.assertEqual(cu_seq_lens[2].item(), 100)
        self.assertEqual(cu_seq_lens[3].item(), 550)
        self.assertEqual(cu_seq_lens[4].item(), 600)


if __name__ == "__main__":
    unittest.main()
