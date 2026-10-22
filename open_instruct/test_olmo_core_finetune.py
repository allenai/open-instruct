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

import numpy as np
from olmo_core import data as oc_data
from parameterized import parameterized

from open_instruct import olmo_core_finetune, olmo_core_utils, utils


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

    def test_missing_statistics_is_not_populated_when_required(self) -> None:
        # Conversion writes dataset_statistics.json last; a run killed before it left a broken cache.
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "token_ids_part_0000.npy"))
            _touch(os.path.join(tmp, "labels_mask_part_0000.npy"))
            _touch(os.path.join(tmp, "token_ids_part_0000.csv.gz"))
            self.assertTrue(olmo_core_finetune._numpy_dir_is_populated(tmp))
            self.assertFalse(olmo_core_finetune._numpy_dir_is_populated(tmp, require_statistics=True))
            _touch(os.path.join(tmp, "dataset_statistics.json"))
            self.assertTrue(olmo_core_finetune._numpy_dir_is_populated(tmp, require_statistics=True))

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


_EXTRA_EOS = 2
_EXTRA_PAD = 1
# Above 2**16, so OLMo-core reads uint32 tokens, as for the Olmo tokenizer.
_EXTRA_VOCAB = 100_352


def _write_parts(numpy_dir: str, docs_per_part: list[list[tuple[list[int], bool]]], token_dtype=np.uint32) -> None:
    """Write converter-layout parts: each doc is (EOS-terminated tokens, loss on every token)."""
    os.makedirs(numpy_dir, exist_ok=True)
    for index, docs in enumerate(docs_per_part):
        tokens = np.array([t for doc, _ in docs for t in doc], dtype=token_dtype)
        mask = np.array([loss for doc, loss in docs for _ in doc], dtype=np.bool_)
        tokens.tofile(os.path.join(numpy_dir, f"token_ids_part_{index:04d}.npy"))
        mask.tofile(os.path.join(numpy_dir, f"labels_mask_part_{index:04d}.npy"))
        with gzip.open(os.path.join(numpy_dir, f"token_ids_part_{index:04d}.csv.gz"), "wt") as f:
            start = 0
            for doc, _ in docs:
                f.write(f"{start},{start + len(doc)}\n")
                start += len(doc)


class ExtraNumpyDirsTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.tokenizer = oc_data.TokenizerConfig(
            vocab_size=_EXTRA_VOCAB, eos_token_id=_EXTRA_EOS, pad_token_id=_EXTRA_PAD
        )
        # Chat rows: prompt tokens (10-19) unmasked, so the cache mask is mixed; two parts.
        self.cache = os.path.join(self.tmp, "cache")
        _write_parts(self.cache, [[([10, 11, 12, _EXTRA_EOS], False)] * 3, [([13, 14, _EXTRA_EOS], False)] * 2])
        # Raw documents (tokens >= 50,000), loss on every token; the second dir has two parts.
        self.extra_a = os.path.join(self.tmp, "extra_a")
        _write_parts(self.extra_a, [[([50_000, 50_001, 50_002, 50_003, 50_004, _EXTRA_EOS], True)] * 2])
        self.extra_b = os.path.join(self.tmp, "extra_b")
        _write_parts(
            self.extra_b, [[([60_000, 60_001, _EXTRA_EOS], True)], [([60_002, 60_003, 60_004, _EXTRA_EOS], True)] * 3]
        )

    def _config(self, extra_numpy_dirs=(), sequence_length=8):
        return olmo_core_finetune._numpy_dataset_config(
            self.cache,
            self.tokenizer,
            work_dir=os.path.join(self.tmp, "work"),
            sequence_length=sequence_length,
            document_boundary_kwargs={},
            extra_numpy_dirs=extra_numpy_dirs,
        )

    def _check(self, extra_numpy_dirs, sequence_length=8):
        return olmo_core_finetune._check_extra_numpy_dirs(
            extra_numpy_dirs, self.cache, self.tokenizer, sequence_length=sequence_length, require_metadata=False
        )

    def test_empty_flag_leaves_config_unchanged(self) -> None:
        self.assertEqual(self._config([]), self._config())
        self.assertEqual(
            self._config(),
            olmo_core_finetune._numpy_dataset_config(
                self.cache,
                self.tokenizer,
                work_dir=os.path.join(self.tmp, "work"),
                sequence_length=8,
                document_boundary_kwargs={},
            ),
        )
        self.assertEqual(self._check([]), {})

    def test_paths_and_masks_are_aligned_cache_first(self) -> None:
        config = self._config([self.extra_b, self.extra_a])
        dirs = [self.cache, self.extra_b, self.extra_a]
        self.assertEqual(config.paths, [os.path.join(d, "token_ids_part_*.npy") for d in dirs])
        self.assertEqual(config.label_mask_paths, [os.path.join(d, "labels_mask_part_*.npy") for d in dirs])
        # What OLMo-core actually pairs after expanding the globs.
        paths, _, masks = config._resolve_paths_metadata(allow_mix=True, label_mask_paths=config.label_mask_paths)
        expected = [
            (d, f"{i:04d}") for d, n in ((self.cache, 2), (self.extra_b, 2), (self.extra_a, 1)) for i in range(n)
        ]
        self.assertEqual(len(paths), len(expected))
        for path, mask, (d, index) in zip(paths, masks, expected):
            self.assertEqual(path, os.path.join(d, f"token_ids_part_{index}.npy"))
            self.assertEqual(mask, os.path.join(d, f"labels_mask_part_{index}.npy"))

    def test_extras_pack_apart_from_chat_rows_with_full_loss(self) -> None:
        self._check([self.extra_a, self.extra_b])
        dataset = self._config([self.extra_a, self.extra_b]).build()
        dataset.prepare()
        seen_extra_tokens = 0
        for i in range(len(dataset)):
            item = dataset[i]
            ids, mask = item["input_ids"], item["label_mask"]
            real = ids != _EXTRA_PAD
            is_extra = ids >= 50_000
            # One source file per instance: never chat rows and raw documents together.
            self.assertTrue(bool(is_extra[real & (ids != _EXTRA_EOS)].all()) or not bool(is_extra.any()))
            if is_extra.any():
                self.assertTrue(bool(mask[real].all()))
                seen_extra_tokens += int(real.sum())
            else:
                self.assertFalse(bool(mask.any()))
        self.assertEqual(seen_extra_tokens, 2 * 6 + 3 + 3 * 4)

    def test_logs_counts_for_cache_and_extras(self) -> None:
        with self.assertLogs(olmo_core_finetune.logger.name, level="INFO") as logs:
            stats = self._check([self.extra_a, self.extra_b], sequence_length=5)
        self.assertEqual(stats[self.extra_a].num_tokens, 12)
        self.assertEqual(stats[self.extra_a].num_loss_tokens, 12)
        self.assertEqual(stats[self.extra_a].num_docs_over_length, 2)
        self.assertEqual(stats[self.extra_a].num_tokens_truncated, 2)
        self.assertEqual(stats[self.extra_b].num_parts, 2)
        self.assertEqual(stats[self.extra_b].num_tokens, 15)
        joined = "\n".join(logs.output)
        self.assertIn("Chat cache", joined)
        self.assertIn("18 tokens, 0 loss tokens", joined)
        self.assertIn("27 tokens, 27 loss tokens", joined)

    def test_missing_dir(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "not a local directory"):
            self._check([os.path.join(self.tmp, "nope")])

    def test_no_token_parts(self) -> None:
        os.makedirs(os.path.join(self.tmp, "empty"))
        with self.assertRaisesRegex(FileNotFoundError, "no token_ids_part"):
            self._check([os.path.join(self.tmp, "empty")])

    def test_missing_mask_part(self) -> None:
        os.remove(os.path.join(self.extra_b, "labels_mask_part_0001.npy"))
        with self.assertRaisesRegex(ValueError, r"token parts without a label mask: \['0001'\]"):
            self._check([self.extra_b])

    def test_mask_length_mismatch(self) -> None:
        np.ones(5, dtype=np.bool_).tofile(os.path.join(self.extra_a, "labels_mask_part_0000.npy"))
        with self.assertRaisesRegex(ValueError, "one-byte elements"):
            self._check([self.extra_a])

    def test_wrong_token_dtype(self) -> None:
        # uint16 tokens: the mask then has twice as many entries as OLMo-core would read tokens.
        _write_parts(self.extra_a, [[([500, 501, _EXTRA_EOS], True)] * 2], token_dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "uint32"):
            self._check([self.extra_a])

    def test_non_binary_mask(self) -> None:
        np.full(12, 7, dtype=np.uint8).tofile(os.path.join(self.extra_a, "labels_mask_part_0000.npy"))
        with self.assertRaisesRegex(ValueError, "not a bool/uint8 label mask"):
            self._check([self.extra_a])

    def test_part_must_end_with_eos(self) -> None:
        _write_parts(self.extra_a, [[([50_000, 50_001], True)]])
        with self.assertRaisesRegex(ValueError, "does not end with EOS"):
            self._check([self.extra_a])

    def test_metadata_required_with_metadata_boundaries(self) -> None:
        os.remove(os.path.join(self.extra_a, "token_ids_part_0000.csv.gz"))
        self._check([self.extra_a])
        with self.assertRaisesRegex(ValueError, "csv.gz"):
            olmo_core_finetune._check_extra_numpy_dirs(
                [self.extra_a], self.cache, self.tokenizer, sequence_length=8, require_metadata=True
            )


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

    def test_sets_emo_routers_in_named_blocks(self) -> None:
        # olmo_core_hybrid builds `block` as a dict of named blocks.
        def block(emo):
            return types.SimpleNamespace(routed_experts_router=types.SimpleNamespace(emo=emo))

        config = types.SimpleNamespace(
            block={"gdn": block(types.SimpleNamespace(segment_ids_from="eos")), "attn": block(None)}
        )
        self.assertEqual(olmo_core_utils.use_doc_lens_for_emo_segments(config), 1)
        self.assertEqual(config.block["gdn"].routed_experts_router.emo.segment_ids_from, "doc_lens")

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


class TestBuildWandbConfig(unittest.TestCase):
    _BEAKER_CONFIG = utils.BeakerRuntimeConfig(
        beaker_workload_id="01TEST",
        beaker_node_hostname=["node-1"],
        beaker_experiment_url=["https://beaker.org/ex/01TEST/"],
        beaker_dataset_ids=["ds-1"],
        beaker_dataset_id_urls=["https://beaker.org/ds/ds-1"],
    )

    def test_beaker_fields_are_merged_into_the_config(self) -> None:
        with mock.patch.object(utils, "maybe_get_beaker_config", return_value=self._BEAKER_CONFIG):
            config = olmo_core_utils.build_wandb_config({"learning_rate": 1e-5})

        self.assertEqual(config["learning_rate"], 1e-5)
        self.assertEqual(config["beaker_workload_id"], "01TEST")
        self.assertEqual(config["beaker_experiment_url"], ["https://beaker.org/ex/01TEST/"])

    def test_config_is_unchanged_off_beaker(self) -> None:
        with mock.patch.object(utils, "maybe_get_beaker_config", return_value=None):
            config = olmo_core_utils.build_wandb_config({"learning_rate": 1e-5})

        self.assertEqual(config, {"learning_rate": 1e-5})

    def test_the_caller_dict_is_not_mutated(self) -> None:
        """The same dict is passed to BeakerCallbackV2 and ConfigSaverCallback, so it must be copied."""
        original = {"learning_rate": 1e-5}
        with mock.patch.object(utils, "maybe_get_beaker_config", return_value=self._BEAKER_CONFIG):
            config = olmo_core_utils.build_wandb_config(original)

        self.assertEqual(original, {"learning_rate": 1e-5})
        self.assertIsNot(config, original)

    def test_build_base_callbacks_passes_beaker_fields_to_wandb(self) -> None:
        """The wiring the issue is about: the wandb callback, not just the helper, gets the fields."""
        with mock.patch.object(utils, "maybe_get_beaker_config", return_value=self._BEAKER_CONFIG):
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
            mock.patch.object(olmo_core_utils, "get_rank", return_value=1),
            mock.patch.object(utils, "maybe_get_beaker_config") as mock_get_config,
        ):
            config = olmo_core_utils.build_wandb_config({"learning_rate": 1e-5})

        mock_get_config.assert_not_called()
        self.assertEqual(config, {"learning_rate": 1e-5})


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
