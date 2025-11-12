import os
import random
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

import open_instruct.dpo_tune_cache as dpo_cache


@patch("open_instruct.dpo_tune_cache.logger")
class TestDPOCacheCompatibility(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _create_mock_dataloader(self, all_indices, world_size, rank, seed):
        rng = random.Random(seed)
        shuffled_indices = list(all_indices)
        rng.shuffle(shuffled_indices)

        shard_indices = shuffled_indices[rank::world_size]

        mock_dataloader = Mock()
        batches = [{"dataset_index": [idx]} for idx in shard_indices]
        mock_dataloader.__iter__ = Mock(return_value=iter(batches))
        return mock_dataloader, shard_indices

    def test_save_and_load_different_world_sizes(self, mock_logger):
        merged_cache_path = os.path.join(self.tmpdir, "model_hash.npz")

        all_indices = list(range(100))
        all_chosen = torch.randn(100)
        all_rejected = torch.randn(100)

        seed = 42
        for rank in range(4):
            mock_dataloader, shard_indices = self._create_mock_dataloader(
                all_indices, world_size=4, rank=rank, seed=seed
            )

            process_chosen = all_chosen[shard_indices]
            process_rejected = all_rejected[shard_indices]

            dpo_cache.save_per_process_ref_logprobs(
                cache_dir=self.tmpdir,
                process_rank=rank,
                world_size=4,
                indices=shard_indices,
                chosen_logps_tensors=[process_chosen],
                rejected_logps_tensors=[process_rejected],
                merged_cache_path=merged_cache_path,
            )

        dpo_cache.merge_ref_logprobs_from_processes(
            cache_dir=self.tmpdir, world_size=4, merged_cache_path=merged_cache_path
        )

        self.assertTrue(os.path.exists(merged_cache_path))

        load_seed = 123
        load_rank = 0
        load_world_size = 2
        mock_dataloader, load_shard_indices = self._create_mock_dataloader(
            all_indices, world_size=load_world_size, rank=load_rank, seed=load_seed
        )
        mock_accelerator = Mock()

        logprobs_dict = dpo_cache.load_ref_logprobs_from_disk(merged_cache_path, mock_dataloader, mock_accelerator)

        self.assertIsNotNone(logprobs_dict)
        self.assertEqual(len(logprobs_dict), len(load_shard_indices))

        for idx in load_shard_indices:
            self.assertIn(idx, logprobs_dict)
            chosen_logp, rejected_logp = logprobs_dict[idx]
            self.assertAlmostEqual(chosen_logp.item(), all_chosen[idx].item(), places=5)
            self.assertAlmostEqual(rejected_logp.item(), all_rejected[idx].item(), places=5)

    def test_cache_load_with_different_seeds_matches(self, mock_logger):
        merged_cache_path = os.path.join(self.tmpdir, "model_hash.npz")

        all_indices = list(range(30))
        all_chosen = torch.randn(30)
        all_rejected = torch.randn(30)

        save_seed = 42
        for rank in range(2):
            mock_dataloader, shard_indices = self._create_mock_dataloader(
                all_indices, world_size=2, rank=rank, seed=save_seed
            )

            process_chosen = all_chosen[shard_indices]
            process_rejected = all_rejected[shard_indices]

            dpo_cache.save_per_process_ref_logprobs(
                cache_dir=self.tmpdir,
                process_rank=rank,
                world_size=2,
                indices=shard_indices,
                chosen_logps_tensors=[process_chosen],
                rejected_logps_tensors=[process_rejected],
                merged_cache_path=merged_cache_path,
            )

        dpo_cache.merge_ref_logprobs_from_processes(
            cache_dir=self.tmpdir, world_size=2, merged_cache_path=merged_cache_path
        )

        load_seed = 999
        mock_dataloader, load_shard_indices = self._create_mock_dataloader(
            all_indices, world_size=2, rank=0, seed=load_seed
        )
        mock_accelerator = Mock()

        logprobs_dict = dpo_cache.load_ref_logprobs_from_disk(merged_cache_path, mock_dataloader, mock_accelerator)

        for idx in load_shard_indices:
            self.assertIn(idx, logprobs_dict)
            actual_chosen, actual_rejected = logprobs_dict[idx]
            self.assertAlmostEqual(actual_chosen.item(), all_chosen[idx].item(), places=5)
            self.assertAlmostEqual(actual_rejected.item(), all_rejected[idx].item(), places=5)

    def test_save_with_world_size_2_load_with_world_size_4(self, mock_logger):
        merged_cache_path = os.path.join(self.tmpdir, "model_hash.npz")

        all_indices = list(range(20))
        all_chosen = torch.arange(20, dtype=torch.float32) * 0.1
        all_rejected = torch.arange(20, dtype=torch.float32) * 0.2

        save_seed = 42
        for rank in range(2):
            mock_dataloader, shard_indices = self._create_mock_dataloader(
                all_indices, world_size=2, rank=rank, seed=save_seed
            )

            process_chosen = all_chosen[shard_indices]
            process_rejected = all_rejected[shard_indices]

            dpo_cache.save_per_process_ref_logprobs(
                cache_dir=self.tmpdir,
                process_rank=rank,
                world_size=2,
                indices=shard_indices,
                chosen_logps_tensors=[process_chosen],
                rejected_logps_tensors=[process_rejected],
                merged_cache_path=merged_cache_path,
            )

        dpo_cache.merge_ref_logprobs_from_processes(
            cache_dir=self.tmpdir, world_size=2, merged_cache_path=merged_cache_path
        )

        load_seed = 123
        mock_dataloader, load_shard_indices = self._create_mock_dataloader(
            all_indices, world_size=4, rank=1, seed=load_seed
        )
        mock_accelerator = Mock()

        logprobs_dict = dpo_cache.load_ref_logprobs_from_disk(merged_cache_path, mock_dataloader, mock_accelerator)

        self.assertEqual(len(logprobs_dict), len(load_shard_indices))

        for idx in load_shard_indices:
            self.assertIn(idx, logprobs_dict)
            actual_chosen, actual_rejected = logprobs_dict[idx]
            self.assertAlmostEqual(actual_chosen.item(), all_chosen[idx].item(), places=5)
            self.assertAlmostEqual(actual_rejected.item(), all_rejected[idx].item(), places=5)


if __name__ == "__main__":
    unittest.main()
