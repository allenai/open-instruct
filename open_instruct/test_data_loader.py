import os
import tempfile
import unittest

import numpy as np
import parameterized
import torch
from datasets import Dataset
from olmo_core import data as oc_data
from olmo_core.data import utils as oc_data_utils

from open_instruct import data_loader
from open_instruct.padding_free_collator import PackedSFTCollator, TensorDataCollatorWithFlatteningDPO


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


_NUMPY_SFT_DIR = os.path.join(os.path.dirname(__file__), "test_data", "numpy_sft")


class TestNumpyDataLoading(unittest.TestCase):
    def test_numpy_packed_fsl_dataset_loads_correctly(self):
        max_seq_length = 64

        raw_token_ids = np.memmap(os.path.join(_NUMPY_SFT_DIR, "token_ids_part_0000.npy"), dtype=np.uint32, mode="r")
        raw_label_mask = np.memmap(os.path.join(_NUMPY_SFT_DIR, "labels_mask_part_0000.npy"), dtype=np.bool_, mode="r")
        self.assertEqual(len(raw_token_ids), len(raw_label_mask))
        total_raw_tokens = len(raw_token_ids)

        with tempfile.TemporaryDirectory() as work_dir:
            tokenizer_config = oc_data.TokenizerConfig.dolma2()
            dataset_config = oc_data.NumpyPackedFSLDatasetConfig(
                tokenizer=tokenizer_config,
                work_dir=work_dir,
                paths=[os.path.join(_NUMPY_SFT_DIR, "token_ids_part_*.npy")],
                expand_glob=True,
                label_mask_paths=[os.path.join(_NUMPY_SFT_DIR, "labels_mask_*.npy")],
                generate_doc_lengths=True,
                long_doc_strategy=oc_data.LongDocStrategy.truncate,
                sequence_length=max_seq_length,
            )
            np_dataset = dataset_config.build()
            np_dataset.prepare()

            self.assertGreater(len(np_dataset), 0)

            total_non_pad = 0
            raw_token_set = set(raw_token_ids.tolist())
            for i in range(len(np_dataset)):
                item = np_dataset[i]
                self.assertIn("input_ids", item)
                self.assertIn("label_mask", item)
                self.assertEqual(len(item["input_ids"]), max_seq_length)
                self.assertEqual(len(item["label_mask"]), max_seq_length)
                for token_id in item["input_ids"].tolist():
                    if token_id != tokenizer_config.pad_token_id:
                        total_non_pad += 1
                        self.assertIn(token_id, raw_token_set)

            self.assertGreater(total_non_pad, 0)
            self.assertLessEqual(total_non_pad, total_raw_tokens)


class TestPackedSFTCollatorEquivalence(unittest.TestCase):
    def test_labels_match_get_labels(self):
        input_ids = torch.tensor([10, 20, 30, 40, 50])
        labels = torch.tensor([-100, -100, 30, 40, 50])

        collator = PackedSFTCollator(sequence_length=5, pad_token_id=0)
        batch = collator([{"input_ids": input_ids, "labels": labels}])

        self.assertEqual(batch["input_ids"].shape, (1, 5))
        self.assertEqual(batch["label_mask"].shape, (1, 5))
        self.assertNotIn("labels", batch)

        generated_labels = oc_data_utils.get_labels(batch)
        expected = torch.tensor([[-100, 30, 40, 50, -100]])
        self.assertTrue(torch.equal(generated_labels, expected))

    def test_multi_sequence_packing(self):
        examples = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3])},
            {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5])},
        ]
        collator = PackedSFTCollator(sequence_length=6, pad_token_id=0)
        batch = collator(examples)

        self.assertEqual(batch["input_ids"].shape[1], 6)
        self.assertEqual(batch["label_mask"].shape[1], 6)
        self.assertEqual(batch["input_ids"].shape[0], 1)

        expected_ids = torch.tensor([[1, 2, 3, 4, 5, 0]])
        expected_mask = torch.tensor([[False, True, True, True, True, False]])
        self.assertTrue(torch.equal(batch["input_ids"], expected_ids))
        self.assertTrue(torch.equal(batch["label_mask"], expected_mask))

        generated_labels = oc_data_utils.get_labels(batch)
        expected_labels = torch.tensor([[2, 3, 4, 5, -100, -100]])
        self.assertTrue(torch.equal(generated_labels, expected_labels))

    def test_sequences_overflow_to_new_row(self):
        examples = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([4, 5, 6])},
        ]
        collator = PackedSFTCollator(sequence_length=4, pad_token_id=0)
        batch = collator(examples)

        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["label_mask"].shape, (2, 4))

    def test_all_masked_sequence(self):
        input_ids = torch.tensor([10, 20, 30])
        labels = torch.tensor([-100, -100, -100])

        collator = PackedSFTCollator(sequence_length=4, pad_token_id=0)
        batch = collator([{"input_ids": input_ids, "labels": labels}])

        expected_mask = torch.tensor([[False, False, False, False]])
        self.assertTrue(torch.equal(batch["label_mask"], expected_mask))

    def test_no_labels_field(self):
        examples = [{"input_ids": torch.tensor([10, 20, 30])}]
        collator = PackedSFTCollator(sequence_length=4, pad_token_id=0)
        batch = collator(examples)

        expected_mask = torch.tensor([[True, True, True, False]])
        self.assertTrue(torch.equal(batch["label_mask"], expected_mask))

    def test_dpo_collator_unchanged(self):
        dpo_collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=20)
        features = [
            {
                "chosen_input_ids": torch.tensor([1, 2, 3]),
                "chosen_labels": torch.tensor([-100, 2, 3]),
                "rejected_input_ids": torch.tensor([4, 5]),
                "rejected_labels": torch.tensor([-100, 5]),
                "index": 0,
            }
        ]
        batch = dpo_collator(features)
        self.assertIn("chosen_labels", batch)
        self.assertIn("rejected_labels", batch)
        self.assertNotIn("label_mask", batch)


if __name__ == "__main__":
    unittest.main()
