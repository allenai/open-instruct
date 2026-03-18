import tempfile
import unittest

import datasets
import numpy as np
import parameterized
import torch

import open_instruct.data_loader
from open_instruct import data_loader
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO


def single_example_collator(examples: list[dict]) -> dict:
    assert len(examples) == 1
    return examples[0]


def batch_collator(examples: list[dict]) -> dict:
    keys = examples[0].keys()
    return {key: [ex[key] for ex in examples] for key in keys}


def make_test_dataset(num_examples: int) -> datasets.Dataset:
    data = {"text": [f"example_{i}" for i in range(num_examples)], "label": list(range(num_examples))}
    dataset = datasets.Dataset.from_dict(data)
    return dataset.add_column("index", range(num_examples))


def _make_dpo_dataset(num_samples: int, max_seq_length: int) -> datasets.Dataset:
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
    ds = datasets.Dataset.from_dict(data)
    ds.set_format(type="pt")
    return ds


class TestHFDataLoader(unittest.TestCase):
    def test_smoke(self):
        dataset = make_test_dataset(20)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )

        batches = list(loader)
        self.assertEqual(len(batches), 20)
        for batch in batches:
            self.assertIn("index", batch)
            self.assertIn("text", batch)
            self.assertIn("label", batch)

        self.assertEqual(loader.total_batches, 20)

        mock_batch = loader.get_mock_batch()
        self.assertIn("index", mock_batch)

    @parameterized.parameterized.expand([("dp_world_size_2", 2), ("dp_world_size_4", 4), ("dp_world_size_8", 8)])
    def test_multi_rank_sampling(self, name, dp_world_size):
        num_examples = 100
        batch_size = dp_world_size
        dataset = make_test_dataset(num_examples)

        loaders = [
            open_instruct.data_loader.HFDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                seed=42,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                work_dir=tempfile.gettempdir(),
                collator=single_example_collator,
            )
            for dp_rank in range(dp_world_size)
        ]

        all_indices = []
        for _dp_rank, loader in enumerate(loaders):
            rank_indices = []
            for batch in loader:
                rank_indices.append(batch["index"])
            all_indices.append(set(rank_indices))

        for i in range(dp_world_size):
            for j in range(i + 1, dp_world_size):
                overlap = all_indices[i] & all_indices[j]
                self.assertEqual(len(overlap), 0, f"Rank {i} and {j} have overlapping indices: {overlap}")

        union = set()
        for indices in all_indices:
            union |= indices
        total_batches = num_examples // batch_size
        usable_size = total_batches * batch_size
        rng = np.random.default_rng(42)
        shuffled = np.arange(num_examples)
        rng.shuffle(shuffled)
        expected_indices = set(shuffled[:usable_size].tolist())
        self.assertEqual(union, expected_indices)

    def test_reshuffle(self):
        dataset = make_test_dataset(20)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )

        first_pass = [batch["index"] for batch in loader]

        loader.reshuffle()
        second_pass = [batch["index"] for batch in loader]

        self.assertNotEqual(first_pass, second_pass)
        self.assertEqual(set(first_pass), set(second_pass))

    def test_state_dict_load_state_dict(self):
        dataset = make_test_dataset(20)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )

        for _ in range(5):
            next(iter(loader))

        loader.reshuffle(epoch=1)
        for _ in range(3):
            next(iter(loader))

        state = loader.state_dict()

        new_loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )
        new_loader.load_state_dict(state)

        self.assertEqual(new_loader._epoch, loader._epoch)
        self.assertEqual(new_loader.batches_processed, loader.batches_processed)

    def test_reproducibility_across_runs(self):
        dataset = make_test_dataset(50)

        loader1 = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )
        loader2 = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )

        indices1 = [batch["index"] for batch in loader1]
        indices2 = [batch["index"] for batch in loader2]
        self.assertEqual(indices1, indices2)

        loader1.reshuffle(epoch=1)
        loader2.reshuffle(epoch=1)
        indices1_epoch1 = [batch["index"] for batch in loader1]
        indices2_epoch1 = [batch["index"] for batch in loader2]
        self.assertEqual(indices1_epoch1, indices2_epoch1)

        self.assertNotEqual(indices1, indices1_epoch1)

    def test_checkpoint_resumption_exact_position(self):
        dataset = make_test_dataset(50)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )

        loader.reshuffle(epoch=1)
        loader_iter = iter(loader)
        first_10 = []
        for _ in range(10):
            first_10.append(next(loader_iter)["index"])

        state = loader.state_dict()

        remaining_original = []
        for _ in range(40):
            remaining_original.append(next(loader_iter)["index"])

        new_loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )
        new_loader.load_state_dict(state)

        new_loader_iter = iter(new_loader)
        remaining_restored = []
        for _ in range(40):
            remaining_restored.append(next(new_loader_iter)["index"])

        self.assertEqual(remaining_original, remaining_restored)

    def test_batches_processed_increments_during_iteration(self):
        dataset = make_test_dataset(20)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )

        self.assertEqual(loader.batches_processed, 0)

        for i, _ in enumerate(loader, start=1):
            self.assertEqual(loader.batches_processed, i)

        self.assertEqual(loader.batches_processed, 20)

        state = loader.state_dict()
        self.assertEqual(state["batches_processed"], 20)

    def test_checkpoint_mid_epoch_restores_position(self):
        dataset = make_test_dataset(20)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )

        loader.reshuffle(epoch=1)
        first_half = []
        for _ in range(10):
            first_half.append(next(loader)["index"])

        state = loader.state_dict()
        self.assertEqual(state["batches_processed"], 10)

        new_loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=single_example_collator,
        )
        new_loader.load_state_dict(state)

        remaining = [batch["index"] for batch in new_loader]
        self.assertEqual(len(remaining), 10)

    def test_infinite_loop_all_excluded(self):
        dataset = make_test_dataset(10)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            automatic_reshuffle=True,
            collator=single_example_collator,
        )

        for batch in loader:
            loader.exclude_index(batch["index"])

        with self.assertRaises(RuntimeError) as context:
            next(loader)

        self.assertIn("All dataset examples have been excluded", str(context.exception))

    def test_unique_prompt_ids_across_iterations(self):
        dataset = make_test_dataset(10)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            automatic_reshuffle=False,
            collator=single_example_collator,
        )

        all_prompt_ids = []

        first_pass = list(loader)
        all_prompt_ids.extend([batch["prompt_id"] for batch in first_pass])

        with self.assertRaises(StopIteration):
            next(loader)

        second_pass = list(loader)
        all_prompt_ids.extend([batch["prompt_id"] for batch in second_pass])

        self.assertEqual(len(all_prompt_ids), len(set(all_prompt_ids)))

    @parameterized.parameterized.expand(
        [
            ("size_17_batch_4", 17, 4),
            ("size_23_batch_8", 23, 8),
            ("size_10_batch_3", 10, 3),
            ("size_33_batch_16", 33, 16),
        ]
    )
    def test_drop_last_true_drops_remainder(self, name, num_examples, batch_size):
        dataset = make_test_dataset(num_examples)
        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=batch_collator,
            drop_last=True,
        )
        indices = [idx for batch in loader for idx in batch["index"]]
        expected_count = (num_examples // batch_size) * batch_size
        self.assertEqual(len(indices), expected_count)

    @parameterized.parameterized.expand(
        [
            ("size_17_batch_4", 17, 4),
            ("size_23_batch_8", 23, 8),
            ("size_10_batch_3", 10, 3),
            ("size_33_batch_16", 33, 16),
        ]
    )
    def test_drop_last_false_covers_all_indices(self, name, num_examples, batch_size):
        dataset = make_test_dataset(num_examples)
        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            seed=42,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            collator=batch_collator,
            drop_last=False,
        )
        indices = [idx for batch in loader for idx in batch["index"]]
        self.assertEqual(set(indices), set(range(num_examples)))

    @parameterized.parameterized.expand(
        [("dp2_size_17_batch_4", 17, 4, 2), ("dp4_size_23_batch_8", 23, 8, 4), ("dp2_size_33_batch_16", 33, 16, 2)]
    )
    def test_drop_last_false_multi_rank_covers_all_indices(self, name, num_examples, batch_size, dp_world_size):
        dataset = make_test_dataset(num_examples)
        all_indices = []
        for dp_rank in range(dp_world_size):
            loader = open_instruct.data_loader.HFDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                seed=42,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                work_dir=tempfile.gettempdir(),
                collator=batch_collator,
                drop_last=False,
            )
            all_indices.extend(idx for batch in loader for idx in batch["index"])
        self.assertEqual(set(all_indices), set(range(num_examples)))


class TestStreamingDataLoaderConfigSaveTraces(unittest.TestCase):
    def test_save_traces_requires_rollouts_save_path(self):
        with self.assertRaises(ValueError) as context:
            open_instruct.data_loader.StreamingDataLoaderConfig(save_traces=True, rollouts_save_path="")
        self.assertIn("rollouts_save_path", str(context.exception))
        self.assertIn("save_traces", str(context.exception))

    def test_save_traces_with_valid_path_succeeds(self):
        config = open_instruct.data_loader.StreamingDataLoaderConfig(
            save_traces=True, rollouts_save_path="/tmp/rollouts"
        )
        self.assertTrue(config.save_traces)
        self.assertEqual(config.rollouts_save_path, "/tmp/rollouts")


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


if __name__ == "__main__":
    unittest.main()
