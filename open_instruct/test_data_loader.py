import tempfile
import unittest

import datasets
import parameterized

import open_instruct.data_loader


class TestHFDataLoader(unittest.TestCase):
    def test_smoke(self):
        data = {"text": [f"example_{i}" for i in range(20)], "label": list(range(20))}
        dataset = datasets.Dataset.from_dict(data)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        batches = list(loader)
        self.assertEqual(len(batches), 20)
        for batch in batches:
            self.assertIn("dataset_index", batch)
            self.assertIn("text", batch)
            self.assertIn("label", batch)

        self.assertEqual(loader.total_batches, 20)

        mock_batch = loader.get_mock_batch()
        self.assertIn("dataset_index", mock_batch)

    @parameterized.parameterized.expand([("world_size_2", 2), ("world_size_4", 4), ("world_size_8", 8)])
    def test_multi_rank_sampling(self, name, world_size):
        num_examples = 100
        data = {"text": [f"example_{i}" for i in range(num_examples)], "label": list(range(num_examples))}
        dataset = datasets.Dataset.from_dict(data)

        loaders = [
            open_instruct.data_loader.HFDataLoader(
                dataset=dataset,
                batch_size=1,
                seed=42,
                rank=rank,
                world_size=world_size,
                work_dir=tempfile.gettempdir(),
            )
            for rank in range(world_size)
        ]

        all_indices = []
        for _rank, loader in enumerate(loaders):
            rank_indices = []
            for batch in loader:
                rank_indices.append(batch["dataset_index"])
            all_indices.append(set(rank_indices))

        for i in range(world_size):
            for j in range(i + 1, world_size):
                overlap = all_indices[i] & all_indices[j]
                self.assertEqual(len(overlap), 0, f"Rank {i} and {j} have overlapping indices: {overlap}")

        union = set()
        for indices in all_indices:
            union |= indices
        expected_indices = set(range(num_examples))
        self.assertEqual(union, expected_indices)

    def test_reshuffle(self):
        data = {"text": [f"example_{i}" for i in range(20)], "label": list(range(20))}
        dataset = datasets.Dataset.from_dict(data)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        first_pass = [batch["dataset_index"] for batch in loader]

        loader.reshuffle()
        second_pass = [batch["dataset_index"] for batch in loader]

        self.assertNotEqual(first_pass, second_pass)
        self.assertEqual(set(first_pass), set(second_pass))

    def test_state_dict_load_state_dict(self):
        data = {"text": [f"example_{i}" for i in range(20)], "label": list(range(20))}
        dataset = datasets.Dataset.from_dict(data)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        for _ in range(5):
            next(iter(loader))

        loader.reshuffle(epoch=1)
        for _ in range(3):
            next(iter(loader))

        state = loader.state_dict()

        new_loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )
        new_loader.load_state_dict(state)

        self.assertEqual(new_loader._epoch, loader._epoch)
        self.assertEqual(new_loader.batches_processed, loader.batches_processed)

    def test_reproducibility_across_runs(self):
        data = {"text": [f"example_{i}" for i in range(50)], "label": list(range(50))}
        dataset = datasets.Dataset.from_dict(data)

        loader1 = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )
        loader2 = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        indices1 = [batch["dataset_index"] for batch in loader1]
        indices2 = [batch["dataset_index"] for batch in loader2]
        self.assertEqual(indices1, indices2)

        loader1.reshuffle(epoch=1)
        loader2.reshuffle(epoch=1)
        indices1_epoch1 = [batch["dataset_index"] for batch in loader1]
        indices2_epoch1 = [batch["dataset_index"] for batch in loader2]
        self.assertEqual(indices1_epoch1, indices2_epoch1)

        self.assertNotEqual(indices1, indices1_epoch1)

    def test_checkpoint_resumption_exact_position(self):
        data = {"text": [f"example_{i}" for i in range(50)], "label": list(range(50))}
        dataset = datasets.Dataset.from_dict(data)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        loader.reshuffle(epoch=1)
        loader_iter = iter(loader)
        first_10 = []
        for _ in range(10):
            first_10.append(next(loader_iter)["dataset_index"])

        state = loader.state_dict()

        remaining_original = []
        for _ in range(40):
            remaining_original.append(next(loader_iter)["dataset_index"])

        new_loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )
        new_loader.load_state_dict(state)

        new_loader_iter = iter(new_loader)
        remaining_restored = []
        for _ in range(40):
            remaining_restored.append(next(new_loader_iter)["dataset_index"])

        self.assertEqual(remaining_original, remaining_restored)

    def test_batches_processed_increments_during_iteration(self):
        data = {"text": [f"example_{i}" for i in range(20)], "label": list(range(20))}
        dataset = datasets.Dataset.from_dict(data)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        self.assertEqual(loader.batches_processed, 0)

        for i, _ in enumerate(loader, start=1):
            self.assertEqual(loader.batches_processed, i)

        self.assertEqual(loader.batches_processed, 20)

        state = loader.state_dict()
        self.assertEqual(state["batches_processed"], 20)

    def test_checkpoint_mid_epoch_restores_position(self):
        data = {"text": [f"example_{i}" for i in range(20)], "label": list(range(20))}
        dataset = datasets.Dataset.from_dict(data)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        loader.reshuffle(epoch=1)
        first_half = []
        for _ in range(10):
            first_half.append(next(loader)["dataset_index"])

        state = loader.state_dict()
        self.assertEqual(state["batches_processed"], 10)

        new_loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )
        new_loader.load_state_dict(state)

        remaining = [batch["dataset_index"] for batch in new_loader]
        self.assertEqual(len(remaining), 10)

    def test_infinite_loop_all_excluded(self):
        data = {"text": [f"example_{i}" for i in range(10)], "label": list(range(10))}
        dataset = datasets.Dataset.from_dict(data)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            rank=0,
            world_size=1,
            work_dir=tempfile.gettempdir(),
            automatic_reshuffle=True,
        )

        for batch in loader:
            loader.exclude_index(batch["dataset_index"])

        with self.assertRaises(RuntimeError) as context:
            next(loader)

        self.assertIn("All dataset examples have been excluded", str(context.exception))

    def test_unique_prompt_ids_across_iterations(self):
        data = {"text": [f"example_{i}" for i in range(10)], "label": list(range(10))}
        dataset = datasets.Dataset.from_dict(data)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=42,
            rank=0,
            world_size=1,
            work_dir=tempfile.gettempdir(),
            automatic_reshuffle=False,
        )

        all_prompt_ids = []

        first_pass = list(loader)
        all_prompt_ids.extend([batch["prompt_id"] for batch in first_pass])

        with self.assertRaises(StopIteration):
            next(loader)

        second_pass = list(loader)
        all_prompt_ids.extend([batch["prompt_id"] for batch in second_pass])

        self.assertEqual(len(all_prompt_ids), len(set(all_prompt_ids)))


if __name__ == "__main__":
    unittest.main()
