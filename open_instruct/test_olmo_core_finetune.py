"""Unit tests for cache-validation and checkpoint-detection helpers."""

import gzip
import importlib
import importlib.util
import json
import os
import shlex
import sys
import tempfile
import types
import unittest
from unittest import mock

from olmo_core import data as oc_data
from parameterized import parameterized

from open_instruct import olmo_core_finetune, olmo_core_utils


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w"):
        pass


def _write(path: str, contents: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(contents)


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


class DocumentBoundariesFromMetadataTest(unittest.TestCase):
    def test_default_cache_dir_is_unchanged(self) -> None:
        # The Dolci-Think cache that H038 and H045 trained on.
        self.assertEqual(
            olmo_core_finetune._numpy_cache_dir("/cache", "1edd51f3e3", 33333, 65536, False),
            "/cache/numpy_sft/1edd51f3e3-6068a350",
        )

    def test_row_aligned_cache_gets_its_own_dir(self) -> None:
        self.assertEqual(
            olmo_core_finetune._numpy_cache_dir("/cache", "1edd51f3e3", 33333, 65536, True),
            "/cache/numpy_sft/1edd51f3e3-6068a350-rowaligned",
        )

    def test_default_adds_no_dataset_config_arguments(self) -> None:
        self.assertEqual(olmo_core_finetune._document_boundary_kwargs(False, False), {})
        # The EOS arm of a row-aligned cache reads it exactly as today.
        self.assertEqual(olmo_core_finetune._document_boundary_kwargs(False, True), {})

    def test_flag_reads_boundaries_from_metadata(self) -> None:
        with mock.patch.object(olmo_core_finetune, "_olmo_core_supports_metadata_boundaries", return_value=True):
            self.assertEqual(olmo_core_finetune._document_boundary_kwargs(True, True), {"use_array_if_local": False})

    def test_flag_requires_row_aligned_parts(self) -> None:
        with (
            mock.patch.object(olmo_core_finetune, "_olmo_core_supports_metadata_boundaries", return_value=True),
            self.assertRaisesRegex(ValueError, "--row_aligned_parts"),
        ):
            olmo_core_finetune._document_boundary_kwargs(True, False)

    def test_flag_requires_supporting_olmo_core(self) -> None:
        with (
            mock.patch.object(olmo_core_finetune, "_olmo_core_supports_metadata_boundaries", return_value=False),
            self.assertRaisesRegex(ValueError, "use_array_if_local"),
        ):
            olmo_core_finetune._document_boundary_kwargs(True, True)

    def test_default_dataset_config_matches_previous_arguments(self) -> None:
        tokenizer = oc_data.TokenizerConfig(vocab_size=128, eos_token_id=2, pad_token_id=1)
        config = olmo_core_finetune._numpy_dataset_config(
            "/data", tokenizer, work_dir="/work", sequence_length=64, document_boundary_kwargs={}
        )
        expected = oc_data.NumpyPackedFSLDatasetConfig(
            tokenizer=tokenizer,
            work_dir="/work",
            paths=["/data/token_ids_part_*.npy"],
            expand_glob=True,
            label_mask_paths=["/data/labels_mask_part_*.npy"],
            generate_doc_lengths=True,
            long_doc_strategy=oc_data.LongDocStrategy.truncate,
            sequence_length=64,
        )
        self.assertEqual(config, expected)


# EMO routing exists only in newer OLMo-core.
emo = importlib.import_module("olmo_core.nn.moe.emo") if importlib.util.find_spec("olmo_core.nn.moe.emo") else None


def _emo_model_config(emo_factory):
    """A TransformerConfig-shaped stand-in: an EMO block, an EMO override and a plain override."""

    def block(emo):
        return types.SimpleNamespace(routed_experts_router=types.SimpleNamespace(emo=emo))

    return types.SimpleNamespace(
        block=block(emo_factory()), block_overrides={"7": block(emo_factory()), "15": block(None)}
    )


class UseDocLensForEmoSegmentsTest(unittest.TestCase):
    def test_sets_every_emo_router(self) -> None:
        config = _emo_model_config(lambda: types.SimpleNamespace(segment_ids_from="eos"))
        self.assertEqual(olmo_core_utils.use_doc_lens_for_emo_segments(config), 2)
        self.assertEqual(config.block.routed_experts_router.emo.segment_ids_from, "doc_lens")
        self.assertEqual(config.block_overrides["7"].routed_experts_router.emo.segment_ids_from, "doc_lens")
        self.assertIsNone(config.block_overrides["15"].routed_experts_router.emo)

    def test_model_without_emo_is_untouched(self) -> None:
        config = types.SimpleNamespace(block=types.SimpleNamespace(), block_overrides=None)
        self.assertEqual(olmo_core_utils.use_doc_lens_for_emo_segments(config), 0)

    def test_old_olmo_core_emo_config_fails(self) -> None:
        config = _emo_model_config(types.SimpleNamespace)
        with self.assertRaisesRegex(ValueError, "segment_ids_from"):
            olmo_core_utils.use_doc_lens_for_emo_segments(config)

    @unittest.skipUnless(
        hasattr(getattr(emo, "EmoRouterConfig", None), "segment_ids_from"),
        "installed OLMo-core's EmoRouterConfig has no segment_ids_from",
    )
    def test_real_emo_router_config(self) -> None:
        config = _emo_model_config(
            lambda: emo.EmoRouterConfig(eos_token_id=0, min_document_expert_pool=2, max_document_expert_pool=4)
        )
        self.assertEqual(config.block.routed_experts_router.emo.segment_ids_from, "eos")
        olmo_core_utils.use_doc_lens_for_emo_segments(config)
        self.assertEqual(config.block.routed_experts_router.emo.segment_ids_from, "doc_lens")


class CheckRowAlignedCacheTest(unittest.TestCase):
    def _write_cache(self, tmp: str, parts: list[str], total_instances: int, marker: bool) -> None:
        configuration = {"row_aligned_parts": True} if marker else {}
        stats = {"configuration": configuration, "overall_statistics": {"total_instances": total_instances}}
        _write(os.path.join(tmp, "dataset_statistics.json"), json.dumps(stats))
        for i, lines in enumerate(parts):
            with gzip.open(os.path.join(tmp, f"token_ids_part_{i:04d}.csv.gz"), "wt") as f:
                f.write(lines)

    def test_row_aligned_cache_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._write_cache(tmp, ["0,3\n3,8\n", "0,4\n"], total_instances=3, marker=True)
            olmo_core_finetune._check_row_aligned_cache(tmp)

    def test_cache_without_marker_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._write_cache(tmp, ["0,3\n3,8\n", "0,4\n"], total_instances=3, marker=False)
            with self.assertRaisesRegex(ValueError, "--row_aligned_parts"):
                olmo_core_finetune._check_row_aligned_cache(tmp)

    def test_row_cut_across_parts_fails(self) -> None:
        # The legacy layout: the second row is cut at the part boundary and appears in both files.
        with tempfile.TemporaryDirectory() as tmp:
            self._write_cache(tmp, ["0,3\n3,8\n", "0,4\n"], total_instances=2, marker=True)
            with self.assertRaisesRegex(ValueError, "spans two parts"):
                olmo_core_finetune._check_row_aligned_cache(tmp)


class IsHfCheckpointTest(unittest.TestCase):
    def test_local_dir_with_hf_config_json_is_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write(os.path.join(tmp, "config.json"), json.dumps({"model_type": "olmo3"}))
            self.assertTrue(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_local_dir_with_olmo_core_config_json_is_olmo_core(self) -> None:
        # An olmo-core checkpoint directory also has a config.json (the experiment
        # config), so presence alone must not mark it as HF.
        with tempfile.TemporaryDirectory() as tmp:
            _write(os.path.join(tmp, "config.json"), json.dumps({"model": {"d_model": 4096}}))
            os.makedirs(os.path.join(tmp, "model_and_optim"))
            self.assertFalse(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_local_dir_with_unreadable_config_json_is_olmo_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write(os.path.join(tmp, "config.json"), "not json{")
            self.assertFalse(olmo_core_utils.is_hf_checkpoint(tmp))

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

    def test_gs_url_is_olmo_core(self) -> None:
        self.assertFalse(olmo_core_utils.is_hf_checkpoint("gs://ai2-llm/checkpoints/olmo3/step100/model_and_optim"))

    def test_gs_url_with_hf_marker_is_hf(self) -> None:
        self.assertTrue(olmo_core_utils.is_hf_checkpoint("gs://ai2-llm/checkpoints/olmo3-hf/step100"))


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


class WriteProvenanceReadmeTest(unittest.TestCase):
    def test_writes_readme_with_tracking_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            olmo_core_utils.write_provenance_readme(
                output_dir=tmp,
                run_name="my-run",
                model_name_or_path="/weka/some/base/step63802",
                tracking_url="https://github.com/allenai/open-instruct/issues/1859",
                wandb_project="open_instruct_internal",
            )
            with open(os.path.join(tmp, "README.md")) as f:
                content = f.read()
            self.assertIn("# my-run", content)
            self.assertIn("https://github.com/allenai/open-instruct/issues/1859", content)
            self.assertIn("/weka/some/base/step63802", content)
            self.assertIn("ai2-llm/open_instruct_internal", content)

    def test_command_is_shell_quoted(self) -> None:
        """A pasted command must re-parse to the original argv, even with spaces or `;`."""
        argv = ["olmo_core_finetune.py", "--run_name", "kda think; seed 1", "--output_dir", "/weka/a b/out"]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sys, "argv", argv):
            olmo_core_utils.write_provenance_readme(
                output_dir=tmp, run_name="r", model_name_or_path="base", tracking_url=None
            )
            with open(os.path.join(tmp, "README.md")) as f:
                content = f.read()
        command = content.split("```")[1].strip()
        self.assertEqual(shlex.split(command), argv)
        self.assertNotIn("\n" + " ".join(argv) + "\n", content)

    def test_does_not_overwrite_existing_readme(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "README.md")
            with open(path, "w") as f:
                f.write("hand-written notes\n")
            olmo_core_utils.write_provenance_readme(
                output_dir=tmp, run_name="my-run", model_name_or_path="base", tracking_url=None
            )
            with open(path) as f:
                self.assertEqual(f.read(), "hand-written notes\n")

    def test_unwritable_output_dir_does_not_raise(self) -> None:
        olmo_core_utils.write_provenance_readme(
            output_dir="/nonexistent-dir/for-sure", run_name="r", model_name_or_path="b", tracking_url=None
        )
