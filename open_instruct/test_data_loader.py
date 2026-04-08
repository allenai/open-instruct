import os
import tempfile
import unittest

import numpy as np
import parameterized
import torch
from datasets import Dataset
from olmo_core import data as oc_data

from open_instruct import data_loader
from open_instruct.padding_free_collator import TensorDataCollatorWithFlattening, TensorDataCollatorWithFlatteningDPO


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


def _make_sft_dataset(num_samples: int, max_seq_length: int, vocab_size: int = 1000) -> Dataset:
    rng = torch.Generator().manual_seed(123)
    data: dict[str, list] = {"input_ids": [], "labels": [], "attention_mask": [], "index": []}
    for i in range(num_samples):
        seq_len = torch.randint(4, max_seq_length + 1, (1,), generator=rng).item()
        input_ids = torch.randint(1, vocab_size, (seq_len,), generator=rng)
        labels = input_ids.clone()
        num_masked = torch.randint(0, seq_len // 2 + 1, (1,), generator=rng).item()
        labels[:num_masked] = -100
        data["input_ids"].append(input_ids)
        data["labels"].append(labels)
        data["attention_mask"].append(torch.ones(seq_len, dtype=torch.long))
        data["index"].append(i)
    ds = Dataset.from_dict(data)
    ds.set_format(type="pt")
    return ds


def _convert_hf_to_numpy(dataset: Dataset, output_dir: str) -> None:
    all_token_ids = []
    all_labels_mask = []
    for sample in dataset:
        all_token_ids.extend(sample["input_ids"].tolist())
        all_labels_mask.extend([1 if label != -100 else 0 for label in sample["labels"].tolist()])
    token_ids_arr = np.array(all_token_ids, dtype=np.uint16)
    labels_mask_arr = np.array(all_labels_mask, dtype=np.bool_)
    np.save(os.path.join(output_dir, "token_ids_part_000.npy"), token_ids_arr)
    np.save(os.path.join(output_dir, "labels_mask_000.npy"), labels_mask_arr)


class TestHFAndNumpyEquivalence(unittest.TestCase):
    def test_same_tokens_and_labels(self):
        num_samples = 20
        max_seq_length = 64
        batch_size_seqs = 4
        dataset = _make_sft_dataset(num_samples, max_seq_length)

        with tempfile.TemporaryDirectory() as tmpdir:
            collator = TensorDataCollatorWithFlattening(
                return_position_ids=True,
                return_flash_attn_kwargs=True,
                max_seq_length=batch_size_seqs * max_seq_length,
            )
            hf_loader = data_loader.HFDataLoader(
                dataset=dataset,
                batch_size=batch_size_seqs,
                seed=42,
                dp_rank=0,
                dp_world_size=1,
                work_dir=tmpdir,
                collator=collator,
                drop_last=False,
                max_seq_length=max_seq_length,
            )

            hf_tokens = []
            hf_label_mask = []
            for batch in hf_loader:
                ids = batch["input_ids"].squeeze(0)
                labs = batch["labels"].squeeze(0)
                for token_id, label in zip(ids.tolist(), labs.tolist()):
                    if token_id == 0 and label == -100:
                        continue
                    hf_tokens.append(token_id)
                    hf_label_mask.append(label != -100)

            numpy_dir = os.path.join(tmpdir, "numpy_data")
            os.makedirs(numpy_dir)
            _convert_hf_to_numpy(dataset, numpy_dir)

            tokenizer_config = oc_data.TokenizerConfig.dolma2()
            dataset_config = oc_data.NumpyPackedFSLDatasetConfig(
                tokenizer=tokenizer_config,
                work_dir=os.path.join(numpy_dir, "_work"),
                paths=[os.path.join(numpy_dir, "token_ids_part_*.npy")],
                expand_glob=True,
                label_mask_paths=[os.path.join(numpy_dir, "labels_mask_*.npy")],
                generate_doc_lengths=True,
                long_doc_strategy=oc_data.LongDocStrategy.truncate,
                sequence_length=max_seq_length,
            )
            np_dataset = dataset_config.build()
            np_dataset.prepare()

            numpy_tokens = []
            numpy_label_mask = []
            for i in range(len(np_dataset)):
                item = np_dataset[i]
                ids = item["input_ids"]
                mask = item["label_mask"]
                for token_id, is_train in zip(ids.tolist(), mask.tolist()):
                    if token_id == tokenizer_config.pad_token_id:
                        continue
                    numpy_tokens.append(token_id)
                    numpy_label_mask.append(bool(is_train))

            self.assertEqual(hf_tokens, numpy_tokens, "Token sequences differ between HF and numpy loaders")
            self.assertEqual(hf_label_mask, numpy_label_mask, "Label masks differ between HF and numpy loaders")


if __name__ == "__main__":
    unittest.main()
