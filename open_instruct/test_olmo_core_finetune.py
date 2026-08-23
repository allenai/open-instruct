"""Unit tests for cache-validation and checkpoint-detection helpers."""

import os
import tempfile
import unittest
import unittest.mock

from parameterized import parameterized

from open_instruct import olmo_core_finetune, olmo_core_utils, utils


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


class TestBuildWandbConfig(unittest.TestCase):
    _BEAKER_CONFIG = utils.BeakerRuntimeConfig(
        beaker_workload_id="01TEST",
        beaker_node_hostname=["node-1"],
        beaker_experiment_url=["https://beaker.org/ex/01TEST/"],
        beaker_dataset_ids=["ds-1"],
        beaker_dataset_id_urls=["https://beaker.org/ds/ds-1"],
    )

    def test_beaker_fields_are_merged_into_the_config(self) -> None:
        with unittest.mock.patch.object(utils, "maybe_get_beaker_config", return_value=self._BEAKER_CONFIG):
            config = olmo_core_utils.build_wandb_config({"learning_rate": 1e-5})

        self.assertEqual(config["learning_rate"], 1e-5)
        self.assertEqual(config["beaker_workload_id"], "01TEST")
        self.assertEqual(config["beaker_experiment_url"], ["https://beaker.org/ex/01TEST/"])

    def test_config_is_unchanged_off_beaker(self) -> None:
        with unittest.mock.patch.object(utils, "maybe_get_beaker_config", return_value=None):
            config = olmo_core_utils.build_wandb_config({"learning_rate": 1e-5})

        self.assertEqual(config, {"learning_rate": 1e-5})

    def test_the_caller_dict_is_not_mutated(self) -> None:
        """The same dict is passed to BeakerCallbackV2 and ConfigSaverCallback, so it must be copied."""
        original = {"learning_rate": 1e-5}
        with unittest.mock.patch.object(utils, "maybe_get_beaker_config", return_value=self._BEAKER_CONFIG):
            config = olmo_core_utils.build_wandb_config(original)

        self.assertEqual(original, {"learning_rate": 1e-5})
        self.assertIsNot(config, original)

    def test_build_base_callbacks_passes_beaker_fields_to_wandb(self) -> None:
        """The wiring the issue is about: the wandb callback, not just the helper, gets the fields."""
        with unittest.mock.patch.object(utils, "maybe_get_beaker_config", return_value=self._BEAKER_CONFIG):
            callbacks = olmo_core_utils.build_base_callbacks(
                config_dict={"learning_rate": 1e-5},
                run_name="test-run",
                checkpointing_steps=500,
                ephemeral_save_interval=250,
                with_tracking=True,
                wandb_project="test-project",
            )

        self.assertEqual(callbacks["wandb"].config["beaker_workload_id"], "01TEST")
        self.assertEqual(callbacks["wandb"].config["learning_rate"], 1e-5)

    def test_non_zero_ranks_skip_the_beaker_cli_lookup(self) -> None:
        """maybe_get_beaker_config shells out to the Beaker CLI, so only rank 0 should call it."""
        with (
            unittest.mock.patch.object(olmo_core_utils, "get_rank", return_value=1),
            unittest.mock.patch.object(utils, "maybe_get_beaker_config") as mock_get_config,
        ):
            config = olmo_core_utils.build_wandb_config({"learning_rate": 1e-5})

        mock_get_config.assert_not_called()
        self.assertEqual(config, {"learning_rate": 1e-5})


if __name__ == "__main__":
    unittest.main()
