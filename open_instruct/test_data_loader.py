import tempfile
import unittest

import parameterized
import torch
from datasets import Dataset

from open_instruct import data_loader, padding_free_collator
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
                        all_indices.update(batch["index"][batch["index"] >= 0].tolist())

            if not drop_last:
                expected_indices = set(range(num_samples))
                self.assertEqual(all_indices, expected_indices, f"Missing indices: {expected_indices - all_indices}")


def _make_fixed_length_dpo_dataset(num_samples: int, seq_len: int) -> Dataset:
    rng = torch.Generator().manual_seed(42)
    data = {
        "chosen_input_ids": [torch.randint(0, 1000, (seq_len,), generator=rng) for _ in range(num_samples)],
        "chosen_labels": [torch.randint(0, 1000, (seq_len,), generator=rng) for _ in range(num_samples)],
        "rejected_input_ids": [torch.randint(0, 1000, (seq_len,), generator=rng) for _ in range(num_samples)],
        "rejected_labels": [torch.randint(0, 1000, (seq_len,), generator=rng) for _ in range(num_samples)],
        "index": list(range(num_samples)),
    }
    ds = Dataset.from_dict(data)
    ds.set_format(type="pt")
    return ds


class TestTokenBudgetPacking(unittest.TestCase):
    def test_packs_to_token_budget_not_sample_cap(self):
        max_seq_length = 16384
        seq_len = 100
        num_samples = 200
        global_batch_size = 4
        dataset = _make_fixed_length_dpo_dataset(num_samples, seq_len)
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)

        with tempfile.TemporaryDirectory() as work_dir:
            loader = data_loader.HFDataLoader(
                dataset=dataset,
                batch_size=global_batch_size,
                seed=42,
                dp_rank=0,
                dp_world_size=1,
                work_dir=work_dir,
                collator=collator,
                drop_last=False,
            )

            row_sizes = []
            seen_indices = set()
            for batch in loader:
                for row in padding_free_collator.unstack_packed_rows(batch):
                    row_sizes.append(len(row["index"]))
                    seen_indices.update(row["index"].tolist())
                    self.assertLessEqual(row["chosen_cu_seq_lens_k"][-1].item(), max_seq_length)
                    self.assertLessEqual(row["rejected_cu_seq_lens_k"][-1].item(), max_seq_length)

            self.assertGreater(max(row_sizes), global_batch_size)
            self.assertEqual(seen_indices, set(range(num_samples)))

    def test_microbatch_sample_cap_binds(self):
        max_seq_length = 16384
        seq_len = 100
        num_samples = 200
        cap = 3
        dataset = _make_fixed_length_dpo_dataset(num_samples, seq_len)
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)

        with tempfile.TemporaryDirectory() as work_dir:
            loader = data_loader.HFDataLoader(
                dataset=dataset,
                batch_size=4,
                seed=42,
                dp_rank=0,
                dp_world_size=1,
                work_dir=work_dir,
                collator=collator,
                drop_last=False,
                microbatch_sample_cap=cap,
            )

            for batch in loader:
                for row in padding_free_collator.unstack_packed_rows(batch):
                    self.assertLessEqual(len(row["index"]), cap)


class TestStackedPackedBatches(unittest.TestCase):
    @parameterized.parameterized.expand([("rows2_dp1", 2, 1), ("rows4_dp1", 4, 1), ("rows2_dp2", 2, 2)])
    def test_yields_per_rank_rows_per_batch(self, _name, rows_per_rank, dp_world_size):
        max_seq_length = 16384
        seq_len = 100
        num_samples = 200
        cap = 2
        dataset = _make_fixed_length_dpo_dataset(num_samples, seq_len)
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)

        with tempfile.TemporaryDirectory() as work_dir:
            loaders = [
                data_loader.HFDataLoader(
                    dataset=dataset,
                    batch_size=rows_per_rank * dp_world_size,
                    seed=42,
                    dp_rank=rank,
                    dp_world_size=dp_world_size,
                    work_dir=work_dir,
                    collator=collator,
                    drop_last=True,
                    microbatch_sample_cap=cap,
                )
                for rank in range(dp_world_size)
            ]

            batch_counts = [loader.total_batches for loader in loaders]
            self.assertTrue(all(c == batch_counts[0] for c in batch_counts), f"Step counts differ: {batch_counts}")

            for loader in loaders:
                num_batches = 0
                for batch in loader:
                    self.assertIsInstance(batch, dict)
                    self.assertEqual(batch["chosen_input_ids"].shape, (rows_per_rank, max_seq_length))
                    rows = padding_free_collator.unstack_packed_rows(batch)
                    self.assertEqual(len(rows), rows_per_rank)
                    for row in rows:
                        self.assertLessEqual(len(row["index"]), cap)
                    num_batches += 1
                self.assertEqual(num_batches, loader.total_batches)

    def test_stack_unstack_round_trip(self):
        max_seq_length = 512
        dataset = _make_dpo_dataset(num_samples=7, max_seq_length=max_seq_length)
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)
        rows = [
            collator([dataset[0], dataset[1]]) | {"is_padding": False},
            collator([dataset[2]]) | {"is_padding": False},
            collator([dataset[3], dataset[4], dataset[5]]) | {"is_padding": True},
        ]

        stacked = padding_free_collator.stack_packed_rows(rows)
        unstacked = padding_free_collator.unstack_packed_rows(stacked)

        self.assertEqual(len(unstacked), len(rows))
        for original, restored in zip(rows, unstacked):
            self.assertEqual(set(original.keys()), set(restored.keys()))
            for k, v in original.items():
                if k.endswith(("max_length_q", "max_length_k")):
                    # Stacking reduces max_length to a batch-level max (a safe upper bound).
                    self.assertEqual(restored[k], max(r[k] for r in rows))
                elif isinstance(v, torch.Tensor):
                    torch.testing.assert_close(restored[k], v)
                else:
                    self.assertEqual(restored[k], v)


if __name__ == "__main__":
    unittest.main()
