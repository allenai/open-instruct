"""Unit tests for cache-validation and checkpoint-detection helpers."""

import os
import tempfile
import unittest

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


class IsHfCheckpointTest(unittest.TestCase):
    def test_local_dir_with_config_json_is_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "config.json"))
            self.assertTrue(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_local_dir_without_config_json_is_olmo_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "model.pt"))
            self.assertFalse(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_converted_olmo_core_dir_with_config_json_is_not_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "config.json"))
            _touch(os.path.join(tmp, "model_and_optim", ".metadata"))
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


class RetainedCheckpointerCallbackTest(unittest.TestCase):
    def test_permanent_checkpoint_removes_ephemeral_and_old_permanent(self) -> None:
        callback = olmo_core_utils.RetainedCheckpointerCallback(keep_last_n_checkpoints=1)
        callback._checkpoints = ["step100", "step200"]
        callback._ephemeral_checkpoints = ["step150"]
        callback._latest_checkpoint_path = "step200"

        callback._apply_retention()

        self.assertEqual(["step200"], callback._checkpoints)
        self.assertEqual([], callback._ephemeral_checkpoints)
        self.assertEqual(["step150", "step100"], callback._checkpoints_to_remove)

    def test_final_ephemeral_checkpoint_replaces_permanent_history(self) -> None:
        callback = olmo_core_utils.RetainedCheckpointerCallback(keep_last_n_checkpoints=1)
        callback._checkpoints = ["step100"]
        callback._ephemeral_checkpoints = ["step150"]
        callback._latest_checkpoint_path = "step150"

        callback._apply_retention(training_complete=True)

        self.assertEqual([], callback._checkpoints)
        self.assertEqual(["step150"], callback._ephemeral_checkpoints)
        self.assertEqual(["step100"], callback._checkpoints_to_remove)


if __name__ == "__main__":
    unittest.main()
