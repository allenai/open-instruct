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

        self.assertEqual(new_loader.epoch_number, loader.epoch_number)
        self.assertEqual(new_loader.batches_processed, loader.batches_processed)


if __name__ == "__main__":
    unittest.main()
