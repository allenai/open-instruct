"""Unit tests for cache-validation and checkpoint-detection helpers."""

import json
import os
import tempfile
import unittest
from unittest import mock

from parameterized import parameterized

from open_instruct import olmo_core_finetune, olmo_core_utils


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w"):
        pass


class NumpyDirIsPopulatedTest(unittest.TestCase):
    def test_empty_dir_is_not_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(olmo_core_finetune._numpy_dir_is_populated(tmp))

    def test_token_ids_only_is_not_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "token_ids_part_0000.npy"))
            self.assertFalse(olmo_core_finetune._numpy_dir_is_populated(tmp))

    def test_missing_metadata_is_not_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "token_ids_part_0000.npy"))
            _touch(os.path.join(tmp, "labels_mask_part_0000.npy"))
            self.assertFalse(olmo_core_finetune._numpy_dir_is_populated(tmp))

    def test_complete_single_chunk_is_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "token_ids_part_0000.npy"))
            _touch(os.path.join(tmp, "labels_mask_part_0000.npy"))
            _touch(os.path.join(tmp, "token_ids_part_0000.csv.gz"))
            self.assertTrue(olmo_core_finetune._numpy_dir_is_populated(tmp))

    def test_partial_second_chunk_is_not_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for i in (0, 1):
                _touch(os.path.join(tmp, f"token_ids_part_{i:04d}.npy"))
            _touch(os.path.join(tmp, "labels_mask_part_0000.npy"))
            _touch(os.path.join(tmp, "token_ids_part_0000.csv.gz"))
            self.assertFalse(olmo_core_finetune._numpy_dir_is_populated(tmp))


class NumpyCacheProvenanceTest(unittest.TestCase):
    def test_fresh_cache_records_source_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GIT_COMMIT": "abc123"}):
            olmo_core_finetune._record_numpy_cache_provenance(tmp)
            with open(
                os.path.join(tmp, olmo_core_finetune._NUMPY_CACHE_PROVENANCE_FILE),
                encoding="utf-8",
            ) as f:
                self.assertEqual(json.load(f), {"git_commit": "abc123"})

    def test_existing_partial_cache_is_not_relabelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"GIT_COMMIT": "abc123"}):
            _touch(os.path.join(tmp, "token_ids_part_0000.npy"))
            olmo_core_finetune._record_numpy_cache_provenance(tmp)
            self.assertFalse(os.path.exists(os.path.join(tmp, olmo_core_finetune._NUMPY_CACHE_PROVENANCE_FILE)))

    def test_mismatched_source_commit_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                os.path.join(tmp, olmo_core_finetune._NUMPY_CACHE_PROVENANCE_FILE),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"git_commit": "old123"}, f)
            with (
                mock.patch.dict(os.environ, {"GIT_COMMIT": "new456"}),
                mock.patch.object(olmo_core_finetune.logger, "warning") as warning,
            ):
                olmo_core_finetune._warn_if_numpy_cache_provenance_mismatch(tmp)
            warning.assert_called_once()
            message = warning.call_args.args[0]
            self.assertIn("old123", message)
            self.assertIn("new456", message)


class IsHfCheckpointTest(unittest.TestCase):
    def test_local_dir_with_config_json_is_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "config.json"))
            self.assertTrue(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_local_dir_without_config_json_is_olmo_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "model.pt"))
            self.assertFalse(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_relative_local_olmo_core_dir_is_olmo_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                os.makedirs("ckpt")
                _touch(os.path.join("ckpt", "model.pt"))
                self.assertFalse(olmo_core_utils.is_hf_checkpoint("ckpt"))
            finally:
                os.chdir(cwd)

    @parameterized.expand([("allenai/Olmo-3-1025-7B",), ("allenai/OLMo-2-1124-7B",), ("Qwen/Qwen3-0.6B",)])
    def test_nonexistent_hub_id_is_hf(self, path: str) -> None:
        self.assertFalse(os.path.exists(path))
        self.assertTrue(olmo_core_utils.is_hf_checkpoint(path))

    def test_hf_marker_in_absolute_path(self) -> None:
        # Path doesn't exist on disk, but contains '-hf'.
        self.assertTrue(olmo_core_utils.is_hf_checkpoint("/weka/checkpoints/some-model-hf/step1"))


class TestCheckpointerDefaults(unittest.TestCase):
    def test_default_intervals_build_a_checkpointer(self) -> None:
        """The two defaults must not collide: olmo-core requires ephemeral < save_interval."""
        callback = olmo_core_utils.build_checkpointer_callback(
            olmo_core_utils.CheckpointConfig.checkpointing_steps, olmo_core_finetune._DEFAULT_EPHEMERAL_SAVE_INTERVAL
        )
        self.assertEqual(callback.ephemeral_save_interval, olmo_core_finetune._DEFAULT_EPHEMERAL_SAVE_INTERVAL)

    def test_non_positive_interval_disables_ephemeral_checkpoints(self) -> None:
        for interval in (-1, 0):
            with self.subTest(interval=interval):
                callback = olmo_core_utils.build_checkpointer_callback(345, interval)
                self.assertIsNone(callback.ephemeral_save_interval)


if __name__ == "__main__":
    unittest.main()
