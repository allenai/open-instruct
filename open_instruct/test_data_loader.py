import gzip
import os
import tempfile
import unittest

import numpy as np
import parameterized
import torch
import torch.nn.functional as F
from datasets import Dataset
from olmo_core import data as oc_data
from olmo_core.data import utils as oc_data_utils
from olmo_core.data.utils import InstancePacker

from open_instruct import data_loader, dataset_transformation
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


def _convert_to_numpy(dataset: Dataset, output_dir: str) -> None:
    token_ids: list[int] = []
    labels_mask: list[int] = []
    document_boundaries: list[tuple[int, int]] = []
    current_position = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        sample_ids = sample["input_ids"].tolist()
        sample_labels = sample["labels"].tolist()
        sample_length = len(sample_ids)

        token_ids.extend(sample_ids)
        labels_mask.extend([1 if label != -100 else 0 for label in sample_labels])
        document_boundaries.append((current_position, current_position + sample_length))
        current_position += sample_length

    mmap_ids = np.memmap(
        os.path.join(output_dir, "token_ids_part_0000.npy"), mode="w+", dtype=np.uint32, shape=(len(token_ids),)
    )
    mmap_ids[:] = token_ids
    mmap_ids.flush()

    mmap_mask = np.memmap(
        os.path.join(output_dir, "labels_mask_part_0000.npy"), mode="w+", dtype=np.bool_, shape=(len(labels_mask),)
    )
    mmap_mask[:] = labels_mask
    mmap_mask.flush()

    with gzip.open(os.path.join(output_dir, "token_ids_part_0000.csv.gz"), "wt") as f:
        for start, end in document_boundaries:
            f.write(f"{start},{end}\n")


class OBFDCollator:
    """Collator that uses olmo-core's OBFD packing algorithm.

    This replicates the packing logic of NumpyPackedFSLDataset so that
    HFDataLoader produces batches identical to the numpy path.
    """

    def __init__(self, sequence_length: int, pad_token_id: int):
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        lengths = np.array([len(f["input_ids"]) for f in features])
        doc_indices = np.zeros((len(features), 2), dtype=np.uint64)
        pos = 0
        for i, length in enumerate(lengths):
            doc_indices[i] = [pos, pos + length]
            pos += length

        packer = InstancePacker(self.sequence_length)
        instances, sorted_doc_indices, _ = packer.pack_documents(doc_indices)

        sorted_lengths = sorted_doc_indices[:, 1] - sorted_doc_indices[:, 0]
        original_sort_order = np.argsort(-1 * lengths.astype(np.int64))

        packed_input_ids = []
        packed_label_mask = []
        for instance in instances:
            row_ids = []
            row_mask = []
            for sorted_doc_id in instance:
                original_idx = original_sort_order[sorted_doc_id]
                f = features[original_idx]
                ids = f["input_ids"]
                doc_len = int(sorted_lengths[sorted_doc_id])
                ids = ids[:doc_len]
                if "labels" in f:
                    labels = f["labels"][:doc_len]
                    mask = labels != -100
                else:
                    mask = torch.ones(doc_len, dtype=torch.bool)
                row_ids.append(ids)
                row_mask.append(mask)

            cat_ids = torch.cat(row_ids)
            cat_mask = torch.cat(row_mask)
            cat_ids = F.pad(cat_ids, (0, self.sequence_length - len(cat_ids)), value=self.pad_token_id)
            cat_mask = F.pad(cat_mask, (0, self.sequence_length - len(cat_mask)), value=False)
            packed_input_ids.append(cat_ids)
            packed_label_mask.append(cat_mask)

        return {
            "input_ids": torch.stack(packed_input_ids),
            "label_mask": torch.stack(packed_label_mask),
            "index": torch.tensor([f["index"] for f in features]),
        }


class TestHFNumpyEquivalence(unittest.TestCase):
    def test_batches_match(self):
        sequence_length = 4096
        num_examples = 100

        tc = dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path="allenai/OLMo-2-1124-7B", chat_template_name="olmo"
        )
        oc_tokenizer_config = oc_data.TokenizerConfig.dolma2()
        pad_token_id = tc.tokenizer.pad_token_id

        dataset = dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list=["allenai/tulu-3-sft-olmo-2-mixture-0225", str(num_examples)],
            dataset_mixer_list_splits=["train"],
            tc=tc,
            dataset_transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
            transform_fn_args=[{"max_seq_length": sequence_length}, {}],
            target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
            dataset_cache_mode="local",
            dataset_skip_cache=True,
        )
        if "index" not in dataset.column_names:
            dataset = dataset.add_column("index", list(range(len(dataset))))
        dataset.set_format(type="pt")

        with tempfile.TemporaryDirectory() as tmpdir:
            hf_work_dir = os.path.join(tmpdir, "hf_work")
            os.makedirs(hf_work_dir)
            numpy_dir = os.path.join(tmpdir, "numpy")
            os.makedirs(numpy_dir)
            np_work_dir = os.path.join(tmpdir, "np_work")
            os.makedirs(np_work_dir)

            collator = OBFDCollator(sequence_length=sequence_length, pad_token_id=pad_token_id)
            hf_loader = data_loader.HFDataLoader(
                dataset=dataset,
                batch_size=len(dataset),
                seed=42,
                dp_rank=0,
                dp_world_size=1,
                work_dir=hf_work_dir,
                collator=collator,
                drop_last=False,
                max_seq_length=sequence_length,
            )

            generator = torch.Generator().manual_seed(42)
            shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()
            shuffled_dataset = dataset.select(shuffled_indices)
            shuffled_dataset.set_format(type="pt")
            _convert_to_numpy(shuffled_dataset, numpy_dir)

            np_dataset_config = oc_data.NumpyPackedFSLDatasetConfig(
                tokenizer=oc_tokenizer_config,
                work_dir=np_work_dir,
                paths=[os.path.join(numpy_dir, "token_ids_part_*.npy")],
                expand_glob=True,
                label_mask_paths=[os.path.join(numpy_dir, "labels_mask_*.npy")],
                generate_doc_lengths=True,
                long_doc_strategy=oc_data.LongDocStrategy.truncate,
                sequence_length=sequence_length,
            )
            np_dataset = np_dataset_config.build()
            np_dataset.prepare()

            hf_batch = next(iter(hf_loader))
            hf_ids = hf_batch["input_ids"]
            hf_mask = hf_batch["label_mask"]

            np_ids_list = []
            np_mask_list = []
            for i in range(len(np_dataset)):
                item = np_dataset[i]
                np_ids_list.append(item["input_ids"])
                np_mask_list.append(item["label_mask"])
            np_ids = torch.stack(np_ids_list)
            np_mask = torch.stack(np_mask_list)

            self.assertEqual(hf_ids.shape, np_ids.shape, f"Shape mismatch: HF {hf_ids.shape} vs numpy {np_ids.shape}")
            self.assertEqual(hf_mask.shape, np_mask.shape)
            self.assertTrue(torch.equal(hf_ids, np_ids), "input_ids mismatch between HF and numpy paths")
            self.assertTrue(torch.equal(hf_mask, np_mask), "label_mask mismatch between HF and numpy paths")


if __name__ == "__main__":
    unittest.main()
