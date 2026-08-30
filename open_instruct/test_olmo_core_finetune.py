"""Unit tests for OLMo-core SFT helpers."""

import os
import tempfile
import unittest

import torch
from olmo_core import data as oc_data
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


class DataCollatorVocabularyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer_config = oc_data.TokenizerConfig(vocab_size=248064, eos_token_id=248069, pad_token_id=248077)
        self.collator = olmo_core_finetune._build_data_collator(self.tokenizer_config, model_vocab_size=248320)

    def test_accepts_added_tokens_within_model_vocabulary(self) -> None:
        self.assertEqual(self.tokenizer_config.padded_vocab_size(), 248064)
        self.assertEqual(self.collator.vocab_size, 248320)

        batch = self.collator([torch.tensor([248068, 248069]), torch.tensor([1])])

        torch.testing.assert_close(batch["input_ids"], torch.tensor([[248068, 248069], [1, 248077]]))

    def test_rejects_token_at_model_vocabulary_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, r"248320.*\[0, 248320\)"):
            self.collator([torch.tensor([248320])])


if __name__ == "__main__":
    unittest.main()
