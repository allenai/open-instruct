import os
import tempfile
import unittest

import datasets
import parameterized

import open_instruct.data_loader
import open_instruct.dataset_transformation

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data", "rlvr_test.jsonl")


class TestHFDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_test_dataset(self, num_examples: int) -> datasets.Dataset:
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path="Qwen/Qwen3-1.7B",
            tokenizer_revision="main",
            chat_template_name="r1_simple_chat_postpend_think",
        )
        dataset = open_instruct.dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list=[TEST_DATA_PATH, str(num_examples)],
            dataset_mixer_list_splits=["train"],
            tc=tc,
            dataset_transform_fn=["rlvr_tokenize_v1", "rlvr_max_length_filter_v1"],
            transform_fn_args=[{}, {"max_prompt_token_length": 512}],
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
        )
        return dataset

    def test_smoke(self):
        dataset = self._create_test_dataset(20)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        batches = list(loader)
        self.assertEqual(len(batches), 20)
        for batch in batches:
            self.assertIn("index", batch)

        self.assertEqual(loader.total_batches, 20)

        mock_batch = loader.get_mock_batch()
        self.assertIn("index", mock_batch)

    @parameterized.parameterized.expand([("world_size_2", 2), ("world_size_4", 4), ("world_size_8", 8)])
    def test_multi_rank_sampling(self, name, world_size):
        num_examples = 100
        dataset = self._create_test_dataset(num_examples)

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
                rank_indices.append(batch["index"])
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
        dataset = self._create_test_dataset(20)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        first_pass = [batch["index"] for batch in loader]

        loader.reshuffle()
        second_pass = [batch["index"] for batch in loader]

        self.assertNotEqual(first_pass, second_pass)
        self.assertEqual(set(first_pass), set(second_pass))

    def test_state_dict_load_state_dict(self):
        dataset = self._create_test_dataset(20)

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
        dataset = self._create_test_dataset(50)

        loader1 = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )
        loader2 = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
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
        dataset = self._create_test_dataset(50)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
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
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )
        new_loader.load_state_dict(state)

        new_loader_iter = iter(new_loader)
        remaining_restored = []
        for _ in range(40):
            remaining_restored.append(next(new_loader_iter)["index"])

        self.assertEqual(remaining_original, remaining_restored)

    def test_batches_processed_increments_during_iteration(self):
        dataset = self._create_test_dataset(20)

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
        dataset = self._create_test_dataset(20)

        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        loader.reshuffle(epoch=1)
        first_half = []
        for _ in range(10):
            first_half.append(next(loader)["index"])

        state = loader.state_dict()
        self.assertEqual(state["batches_processed"], 10)

        new_loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )
        new_loader.load_state_dict(state)

        remaining = [batch["index"] for batch in new_loader]
        self.assertEqual(len(remaining), 10)

    def test_infinite_loop_all_excluded(self):
        dataset = self._create_test_dataset(10)

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
            loader.exclude_index(batch["index"])

        with self.assertRaises(RuntimeError) as context:
            next(loader)

        self.assertIn("All dataset examples have been excluded", str(context.exception))

    def test_unique_prompt_ids_across_iterations(self):
        dataset = self._create_test_dataset(10)

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

    def test_getitem_by_index(self):
        dataset = self._create_test_dataset(20)
        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        for batch in loader:
            original_index = batch["index"]
            retrieved = loader[original_index]
            self.assertEqual(retrieved["index"], original_index)

    def test_getitem_after_reshuffle(self):
        dataset = self._create_test_dataset(20)
        loader = open_instruct.data_loader.HFDataLoader(
            dataset=dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir=tempfile.gettempdir()
        )

        indices_before = [batch["index"] for batch in loader]
        loader.reshuffle()
        indices_after = [batch["index"] for batch in loader]

        self.assertNotEqual(indices_before, indices_after)
        for idx in indices_after:
            retrieved = loader[idx]
            self.assertEqual(retrieved["index"], idx)


if __name__ == "__main__":
    unittest.main()
