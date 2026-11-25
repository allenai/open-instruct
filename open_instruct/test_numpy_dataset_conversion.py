"""Tests for open_instruct.numpy_dataset_conversion.

Run from project root:
    uv run pytest open_instruct/test_numpy_dataset_conversion.py -v
"""

import gc
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
import unittest
import unittest.mock

import numpy as np
import torch
from olmo_core import data as oc_data
from parameterized import parameterized
from transformers import AutoTokenizer

from open_instruct import dataset_transformation, numpy_dataset_conversion, olmo_core_finetune

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def _get_tokenizer_path():
    src_dir = os.path.join(TEST_DATA_DIR, "tokenizer")
    dst_dir = tempfile.mkdtemp(prefix="test_tokenizer_")
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        if name.endswith(".gz"):
            dst = os.path.join(dst_dir, name[:-3])
            with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(src, dst_dir)
    return dst_dir


TOKENIZER_PATH = _get_tokenizer_path()


class TestSelectTokenDtype(unittest.TestCase):
    @parameterized.expand(
        [
            ("uint8_small", 2, np.uint8),
            ("uint8_max", 256, np.uint8),
            ("uint16_min", 257, np.uint16),
            ("uint16_max", 65536, np.uint16),
            ("uint32_min", 65537, np.uint32),
            ("uint32_max", 2**32, np.uint32),
            ("uint64_min", 2**32 + 1, np.uint64),
        ]
    )
    def test_selects_expected_dtype(self, _name, vocab_size, expected_dtype):
        result = numpy_dataset_conversion._select_token_dtype(vocab_size)
        self.assertEqual(result, expected_dtype)

    def test_raises_for_vocab_too_big(self):
        with self.assertRaises(ValueError):
            numpy_dataset_conversion._select_token_dtype(2**64 + 1)


class TestWriteMemmapChunked(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.base = pathlib.Path(self.tmp_dir.name) / "token_ids"
        self.source_path = pathlib.Path(self.tmp_dir.name) / "source.bin"

    def _write_source(self, data, dtype):
        np.asarray(data, dtype=dtype).tofile(self.source_path)

    def _read_chunk(self, chunk_idx, dtype, length):
        filename = f"{self.base}_part_{chunk_idx:04d}.npy"
        return list(np.memmap(filename, mode="r", dtype=dtype, shape=(length,)))

    def test_empty_data(self):
        self.source_path.touch()
        result = numpy_dataset_conversion._write_memmap_chunked_from_file(
            self.base, self.source_path, 0, np.uint16, max_size_gb=1
        )
        self.assertEqual(result, [])
        self.assertFalse(os.path.exists(f"{self.base}_part_0000.npy"))

    def test_single_chunk(self):
        data = list(range(8))
        self._write_source(data, np.uint16)
        result = numpy_dataset_conversion._write_memmap_chunked_from_file(
            self.base, self.source_path, len(data), np.uint16, max_size_gb=1
        )
        self.assertEqual(result, [(0, 8)])
        self.assertEqual(self._read_chunk(0, np.uint16, 8), data)

    def test_three_chunks(self):
        data = list(range(17))
        self._write_source(data, np.uint16)
        result = numpy_dataset_conversion._write_memmap_chunked_from_file(
            self.base, self.source_path, len(data), np.uint16, max_size_gb=16 / 1024**3
        )
        self.assertEqual(result, [(0, 8), (8, 16), (16, 17)])
        self.assertEqual(self._read_chunk(0, np.uint16, 8), data[0:8])
        self.assertEqual(self._read_chunk(1, np.uint16, 8), data[8:16])
        self.assertEqual(self._read_chunk(2, np.uint16, 1), data[16:17])

    def test_row_aligned_chunks_end_only_at_row_ends(self):
        data = list(range(17))
        self._write_source(data, np.uint16)
        result = numpy_dataset_conversion._write_memmap_chunked_from_file(
            self.base,
            self.source_path,
            len(data),
            np.uint16,
            max_size_gb=16 / 1024**3,
            document_ends=np.array([3, 7, 10, 15, 17]),
        )
        self.assertEqual(result, [(0, 7), (7, 15), (15, 17)])
        self.assertEqual(self._read_chunk(0, np.uint16, 7), data[0:7])
        self.assertEqual(self._read_chunk(1, np.uint16, 8), data[7:15])
        self.assertEqual(self._read_chunk(2, np.uint16, 2), data[15:17])

    def test_row_aligned_row_longer_than_limit_gets_its_own_chunk(self):
        data = list(range(17))
        self._write_source(data, np.uint16)
        result = numpy_dataset_conversion._write_memmap_chunked_from_file(
            self.base,
            self.source_path,
            len(data),
            np.uint16,
            max_size_gb=16 / 1024**3,
            document_ends=np.array([2, 2, 13, 17]),
        )
        self.assertEqual(result, [(0, 2), (2, 13), (13, 17)])

    def test_row_aligned_rejects_row_ends_short_of_the_data(self):
        self._write_source(list(range(8)), np.uint16)
        with self.assertRaises(ValueError):
            numpy_dataset_conversion._write_memmap_chunked_from_file(
                self.base, self.source_path, 8, np.uint16, document_ends=np.array([3, 7])
            )


class TestWriteMetadataForChunks(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.base = pathlib.Path(self.tmp_dir.name) / "token_ids"

    def _read_chunk_rows(self, chunk_idx):
        path = f"{self.base}_part_{chunk_idx:04d}.csv.gz"
        with gzip.open(path, "rt") as f:
            return [line.strip() for line in f if line.strip()]

    def test_doc_spans_two_chunks(self):
        doc_boundaries = [(5, 12)]
        chunk_boundaries = [(0, 8), (8, 16)]
        numpy_dataset_conversion._write_metadata_for_chunks(self.base, doc_boundaries, chunk_boundaries)
        self.assertEqual(self._read_chunk_rows(0), ["5,8"])
        self.assertEqual(self._read_chunk_rows(1), ["0,4"])

    def test_row_aligned_chunks_give_one_line_per_row(self):
        doc_boundaries = [(0, 3), (3, 7), (7, 10), (10, 15), (15, 17)]
        chunk_boundaries = [(0, 7), (7, 15), (15, 17)]
        numpy_dataset_conversion._write_metadata_for_chunks(self.base, doc_boundaries, chunk_boundaries)
        self.assertEqual(self._read_chunk_rows(0), ["0,3", "3,7"])
        self.assertEqual(self._read_chunk_rows(1), ["0,3", "3,8"])
        self.assertEqual(self._read_chunk_rows(2), ["0,2"])

    def test_doc_touching_boundary_is_excluded(self):
        doc_boundaries = [(0, 8)]
        chunk_boundaries = [(0, 8), (8, 16)]
        numpy_dataset_conversion._write_metadata_for_chunks(self.base, doc_boundaries, chunk_boundaries)
        self.assertEqual(self._read_chunk_rows(0), ["0,8"])
        self.assertEqual(self._read_chunk_rows(1), [])


class TestSaveTokenizer(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.tc = dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=True,
            chat_template_name="tulu",
            add_bos=False,
        )

    def test_saves_tokenizer_files(self):
        numpy_dataset_conversion._save_tokenizer(self.tc, pathlib.Path(self.tmp_dir.name))
        tokenizer_dir = os.path.join(self.tmp_dir.name, "tokenizer")
        self.assertTrue(os.path.isdir(tokenizer_dir))
        self.assertTrue(os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json")))


class TestWriteDatasetStatistics(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

    def test_happy_path(self):
        dataset_stats = {
            "per_dataset_stats": [
                {
                    "dataset_name": "ds_a",
                    "dataset_split": "train",
                    "initial_instances": 10,
                    "final_instances": 8,
                    "instances_filtered": 2,
                    "frac_or_num_samples": 1.0,
                    "original_dataset_size": 10,
                    "is_upsampled": False,
                    "upsampling_factor": 1.0,
                },
                {
                    "dataset_name": "ds_b",
                    "dataset_split": "train",
                    "initial_instances": 5,
                    "final_instances": 5,
                    "instances_filtered": 0,
                    "frac_or_num_samples": 2.0,
                    "original_dataset_size": 5,
                    "is_upsampled": True,
                    "upsampling_factor": 2.0,
                },
            ]
        }
        numpy_dataset_conversion.write_dataset_statistics(
            output_dir=pathlib.Path(self.tmp_dir.name),
            dataset_statistics=dataset_stats,
            total_instances=13,
            total_tokens=1000,
            total_trainable_tokens=700,
            num_samples_skipped=1,
            tokenizer_name="test-tokenizer",
            max_seq_length=4096,
            chat_template_name="tulu",
            per_dataset_counts={"ds_a": 8, "ds_b": 5},
            per_dataset_tokens={"ds_a": 600, "ds_b": 400},
            per_dataset_trainable_tokens={"ds_a": 400, "ds_b": 300},
            per_dataset_filtered={"ds_a": 1, "ds_b": 0},
        )

        json_path = os.path.join(self.tmp_dir.name, "dataset_statistics.json")
        with open(json_path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded["overall_statistics"]["total_instances"], 13)
        self.assertEqual(loaded["overall_statistics"]["total_tokens"], 1000)
        self.assertEqual(loaded["overall_statistics"]["trainable_tokens"], 700)
        self.assertEqual(len(loaded["per_dataset_statistics"]), 2)
        names = {s["dataset_name"] for s in loaded["per_dataset_statistics"]}
        self.assertEqual(names, {"ds_a", "ds_b"})

        txt_path = os.path.join(self.tmp_dir.name, "dataset_statistics.txt")
        with open(txt_path) as f:
            txt = f.read()
        self.assertIn("ds_a", txt)
        self.assertIn("ds_b", txt)
        self.assertIn("Overall Statistics", txt)

    def test_zero_totals_does_not_divide_by_zero(self):
        numpy_dataset_conversion.write_dataset_statistics(
            output_dir=pathlib.Path(self.tmp_dir.name),
            dataset_statistics={"per_dataset_stats": []},
            total_instances=0,
            total_tokens=0,
            total_trainable_tokens=0,
            num_samples_skipped=0,
            tokenizer_name="test-tokenizer",
            max_seq_length=None,
            chat_template_name=None,
            per_dataset_counts={"ds_a": 0},
            per_dataset_tokens={"ds_a": 0},
            per_dataset_trainable_tokens={"ds_a": 0},
            per_dataset_filtered={"ds_a": 0},
        )
        with open(os.path.join(self.tmp_dir.name, "dataset_statistics.json")) as f:
            loaded = json.load(f)
        overall = loaded["overall_statistics"]
        self.assertEqual(overall["trainable_percentage"], 0)
        self.assertEqual(overall["average_sequence_length"], 0)
        self.assertEqual(loaded["per_dataset_statistics"][0]["avg_tokens_per_instance"], 0)


class _NumpySftTestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(gc.collect)

        patcher = unittest.mock.patch.dict(
            os.environ,
            {
                "HF_HOME": self.temp_dir.name,
                "HF_DATASETS_CACHE": os.path.join(self.temp_dir.name, "datasets"),
                "TRANSFORMERS_CACHE": os.path.join(self.temp_dir.name, "transformers"),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _make_tc(self):
        return dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=True,
            chat_template_name="tulu",
            add_bos=False,
        )

    def _parts(self, output_dir):
        """(tokens, labels_mask, metadata rows) per part, in order."""
        dtype = numpy_dataset_conversion._select_token_dtype(self._make_tc().tokenizer.vocab_size)
        parts = []
        for token_path in sorted(output_dir.glob("token_ids_part_*.npy")):
            labels_path = output_dir / token_path.name.replace("token_ids", "labels_mask")
            with gzip.open(token_path.with_name(token_path.name.replace(".npy", ".csv.gz")), "rt") as f:
                spans = [tuple(int(x) for x in line.split(",")) for line in f]
            parts.append((np.fromfile(token_path, dtype=dtype), np.fromfile(labels_path, dtype=np.bool_), spans))
        return parts


class TestConvertHfToNumpySft(_NumpySftTestBase):
    def test_end_to_end_small(self):
        output_dir = pathlib.Path(self.temp_dir.name) / "out_e2e"
        numpy_dataset_conversion.convert_hf_to_numpy_sft(
            output_dir=output_dir,
            dataset_mixer_list=[os.path.join(TEST_DATA_DIR, "sft_sample.jsonl"), "1.0"],
            dataset_mixer_list_splits=["train"],
            tc=self._make_tc(),
            dataset_transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
            transform_fn_args=[{"max_seq_length": 4096}, {}],
            dataset_target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
            num_examples=2,
        )
        token_file = output_dir / "token_ids_part_0000.npy"
        labels_file = output_dir / "labels_mask_part_0000.npy"
        metadata_file = output_dir / "token_ids_part_0000.csv.gz"
        stats_json = output_dir / "dataset_statistics.json"

        self.assertTrue(token_file.exists())
        self.assertTrue(labels_file.exists())
        self.assertTrue(metadata_file.exists())
        self.assertTrue(stats_json.exists())
        self.assertTrue((output_dir / "tokenizer").is_dir())
        for partial in ("_tokens.partial.bin", "_labels.partial.bin", "_boundaries.partial.bin"):
            self.assertFalse((output_dir / partial).exists())

        self.assertGreater(token_file.stat().st_size, 0)
        self.assertGreater(labels_file.stat().st_size, 0)

        with open(stats_json) as f:
            stats = json.load(f)
        self.assertEqual(stats["overall_statistics"]["total_instances"], 2)
        self.assertGreater(stats["overall_statistics"]["total_tokens"], 0)
        self.assertEqual(stats["configuration"]["chat_template"], "tulu")
        self.assertEqual(stats["configuration"]["chat_template_source"], "registry:tulu")
        self.assertEqual(len(stats["configuration"]["chat_template_hash"]), 64)

    @unittest.mock.patch("open_instruct.dataset_transformation.get_cached_dataset_tulu_with_statistics")
    def test_statistics_preserve_cached_template_provenance(self, get_cached_dataset):
        get_cached_dataset.return_value = (
            dataset_transformation.Dataset.from_dict(
                {
                    dataset_transformation.INPUT_IDS_KEY: [[1, 2]],
                    dataset_transformation.ATTENTION_MASK_KEY: [[1, 1]],
                    dataset_transformation.LABELS_KEY: [[-100, 2]],
                    dataset_transformation.DATASET_ORIGIN_KEY: ["cached"],
                }
            ),
            {
                "per_dataset_stats": [],
                "chat_template_name": "original",
                "chat_template_source": "tokenizer:original",
                "chat_template_hash": "original-hash",
            },
        )
        output_dir = pathlib.Path(self.temp_dir.name) / "cached_provenance"

        numpy_dataset_conversion.convert_hf_to_numpy_sft(
            output_dir=output_dir,
            dataset_mixer_list=[],
            dataset_mixer_list_splits=[],
            tc=self._make_tc(),
            dataset_transform_fn=[],
            transform_fn_args=[],
            dataset_target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
            dataset_config_hash="explicit-hash",
        )

        with open(output_dir / "dataset_statistics.json") as f:
            configuration = json.load(f)["configuration"]
        self.assertEqual(configuration["chat_template"], "original")
        self.assertEqual(configuration["chat_template_source"], "tokenizer:original")
        self.assertEqual(configuration["chat_template_hash"], "original-hash")


_EOS = 2
_PAD = 1
# (token_ids, labels) per row; -100 marks prompt tokens. The multi-turn row has an EOS after its
# first assistant turn, and the tool-call row ends in `<|im_end|>\n` (here 11, 12) with no EOS.
_MULTI_TURN = ([5, 6, _EOS, 7, 8, _EOS], [-100, 6, _EOS, -100, 8, _EOS])
_TOOL_CALL = ([9, 10, 11, 12], [-100, 10, 11, -100])
_SINGLE_TURN = ([13, 14, _EOS], [-100, 14, _EOS])
_CHAT_ROWS = [_MULTI_TURN, _TOOL_CALL, _SINGLE_TURN] * 3
# Tokens per part, small enough that the 1 GiB rule would cut rows.
_PART_TOKENS = 8


class TestRowAlignedParts(_NumpySftTestBase):
    """Parts of a cache written with `row_aligned_parts` hold whole rows only."""

    def _convert(self, name, row_aligned_parts, rows=None):
        return _convert_rows(self, name, _CHAT_ROWS if rows is None else rows, row_aligned_parts, _PART_TOKENS)

    def _shuffled_rows(self):
        order = dataset_transformation.Dataset.from_dict({"i": list(range(len(_CHAT_ROWS)))}).shuffle(seed=0)["i"]
        return [_CHAT_ROWS[i] for i in order]

    def test_empty_rows_fail_at_write_time(self):
        for i, rows in enumerate(([([], [])], [_CHAT_ROWS[0], ([], []), _CHAT_ROWS[1]])):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "row_aligned_parts.*empty row"):
                self._convert(f"empty-{i}", row_aligned_parts=True, rows=rows)

    def test_each_row_is_one_metadata_line_of_one_part(self):
        parts = self._parts(self._convert("aligned", row_aligned_parts=True))
        self.assertGreater(len(parts), 1)
        rows = []
        for tokens, labels, spans in parts:
            self.assertLessEqual(len(tokens), _PART_TOKENS)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], len(tokens))
            for (_, end), (start, _) in zip(spans, spans[1:]):
                self.assertEqual(end, start)
            rows += [(tokens[a:b].tolist(), labels[a:b].tolist()) for a, b in spans]
        expected = [(tokens, [label != -100 for label in labels]) for tokens, labels in self._shuffled_rows()]
        self.assertEqual(rows, expected)
        self.assertEqual(sum(len(spans) for _, _, spans in parts), len(_CHAT_ROWS))

    def test_rewriting_the_same_rows_gives_identical_metadata_files(self):
        # gzip stamps the write time into its header unless told otherwise; OLMo-core fingerprints
        # metadata-backed datasets by these files, so a rewrite must not change a byte.
        outputs = []
        for name, now in (("first", 1_000_000_000.0), ("second", 1_500_000_000.0)):
            with unittest.mock.patch("gzip.time.time", return_value=now):
                outputs.append(self._convert(name, row_aligned_parts=True))
        names = sorted(p.name for p in outputs[0].glob("token_ids_part_*.csv.gz"))
        self.assertGreater(len(names), 1)
        for name in names:
            self.assertEqual((outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes(), msg=name)

    def test_default_cuts_every_part_size_mid_row(self):
        default = self._parts(self._convert("default", row_aligned_parts=False))
        aligned = self._parts(self._convert("aligned", row_aligned_parts=True))
        self.assertEqual([len(tokens) for tokens, _, _ in default[:-1]], [_PART_TOKENS] * (len(default) - 1))
        # Same token and label stream either way; only the cut points differ.
        for field in (0, 1):
            self.assertEqual(
                np.concatenate([part[field] for part in default]).tolist(),
                np.concatenate([part[field] for part in aligned]).tolist(),
            )
        self.assertGreater(sum(len(spans) for _, _, spans in default), len(_CHAT_ROWS))

    def test_both_arms_read_identical_rows_from_one_cache(self):
        """The EOS and metadata arms of H049 share one row-aligned cache. Placing every token each
        arm trains on back at its offset in the cache, both arms read the same token ids and label
        masks row for row; only the document boundaries differ."""
        output_dir = self._convert("aligned", row_aligned_parts=True)
        parts = self._parts(output_dir)
        tokenizer = _oc_tokenizer(self._make_tc(), eos_token_id=_EOS, pad_token_id=_PAD)
        arms = {
            from_metadata: _build_packed(output_dir, tokenizer, 16, from_metadata) for from_metadata in (False, True)
        }
        placed = {from_metadata: _place_instances(dataset) for from_metadata, dataset in arms.items()}

        for part, (tokens, labels, spans) in enumerate(parts):
            eos_tokens, eos_labels, eos_seen = placed[False][part]
            meta_tokens, meta_labels, meta_seen = placed[True][part]
            # The metadata arm trains on every token of the cache, as stored.
            self.assertTrue(meta_seen.all())
            self.assertEqual(meta_tokens.tolist(), tokens.tolist())
            self.assertEqual(meta_labels.tolist(), labels.tolist())
            for start, end in spans:
                if eos_seen[start:end].all():
                    self.assertEqual(eos_tokens[start:end].tolist(), meta_tokens[start:end].tolist())
                    self.assertEqual(eos_labels[start:end].tolist(), meta_labels[start:end].tolist())
                else:
                    # The EOS scan never yields what follows a part's last EOS, so it drops a part's
                    # final rows when they end without EOS (tool calls). Nothing else is missing.
                    self.assertFalse(eos_seen[start:].any())
                    self.assertNotIn(_EOS, tokens[start:].tolist())

        documents = {from_metadata: _documents(dataset) for from_metadata, dataset in arms.items()}
        expected = sorted((tokens, [label != -100 for label in labels]) for tokens, labels in _CHAT_ROWS)
        self.assertEqual(documents[True], expected)
        # The EOS scan splits the multi-turn rows and merges the tool-call rows into their neighbours.
        self.assertNotEqual(documents[False], expected)

    def test_legacy_layout_splits_cut_rows_under_metadata_boundaries(self):
        """Why --document_boundaries_from_metadata requires --row_aligned_parts: in the legacy layout
        a cut row is two metadata lines, one per file, so it trains as two documents."""
        output_dir = self._convert("legacy", row_aligned_parts=False)
        parts = self._parts(output_dir)
        tokenizer = _oc_tokenizer(self._make_tc(), eos_token_id=_EOS, pad_token_id=_PAD)
        dataset = _build_packed(output_dir, tokenizer, 16, from_metadata=True)
        num_lines = sum(len(spans) for _, _, spans in parts)
        self.assertGreater(num_lines, len(_CHAT_ROWS))
        self.assertEqual(len(_documents(dataset)), num_lines)
        # Every token still reaches training; the cut rows are just split in two.
        for _, _, seen in _place_instances(dataset):
            self.assertTrue(seen.all())


def _convert_rows(test, name, rows, row_aligned_parts, part_tokens=None):
    """Write (token_ids, labels) rows to a numpy SFT cache under the test's temp dir, in parts of
    `part_tokens` tokens if given (the 1 GiB rule otherwise)."""
    dataset = dataset_transformation.Dataset.from_dict(
        {
            dataset_transformation.INPUT_IDS_KEY: [tokens for tokens, _ in rows],
            dataset_transformation.ATTENTION_MASK_KEY: [[1] * len(tokens) for tokens, _ in rows],
            dataset_transformation.LABELS_KEY: [labels for _, labels in rows],
            dataset_transformation.DATASET_ORIGIN_KEY: ["chat"] * len(rows),
        }
    )
    output_dir = pathlib.Path(test.temp_dir.name) / name
    real_write = numpy_dataset_conversion._write_memmap_chunked_from_file

    def write_parts(base_filename, source_path, total_items, dtype, **kwargs):
        if part_tokens is not None:
            kwargs["max_size_gb"] = part_tokens * np.dtype(dtype).itemsize / 1024**3
        return real_write(base_filename, source_path, total_items, dtype, **kwargs)

    with (
        unittest.mock.patch.object(
            dataset_transformation, "get_cached_dataset_tulu_with_statistics", return_value=(dataset, {})
        ),
        unittest.mock.patch.object(
            numpy_dataset_conversion, "_write_memmap_chunked_from_file", side_effect=write_parts
        ),
    ):
        numpy_dataset_conversion.convert_hf_to_numpy_sft(
            output_dir=output_dir,
            dataset_mixer_list=[],
            dataset_mixer_list_splits=[],
            tc=test._make_tc(),
            dataset_transform_fn=[],
            transform_fn_args=[],
            dataset_target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
            dataset_config_hash="rows",
            shuffle_seed=0,
            row_aligned_parts=row_aligned_parts,
        )
    return output_dir


def _oc_tokenizer(tc, eos_token_id, pad_token_id):
    return oc_data.TokenizerConfig(vocab_size=len(tc.tokenizer), eos_token_id=eos_token_id, pad_token_id=pad_token_id)


def _build_packed(output_dir, tokenizer, sequence_length, from_metadata):
    config = olmo_core_finetune._numpy_dataset_config(
        str(output_dir),
        tokenizer,
        work_dir=str(output_dir / f"work-{from_metadata}-{sequence_length}"),
        sequence_length=sequence_length,
        document_boundary_kwargs=olmo_core_finetune._document_boundary_kwargs(from_metadata, row_aligned_parts=True),
    )
    dataset = config.build()
    dataset.prepare()
    return dataset


def _segments(item):
    """(token_ids, label_mask) per doc_lens segment of one instance, padding included."""
    out, start = [], 0
    for length in item["doc_lens"].tolist():
        out.append((item["input_ids"][start : start + length].tolist(), item["label_mask"][start : start + length]))
        start += length
    assert start == item["input_ids"].numel()
    return out


def _documents(dataset):
    """Every non-padding segment of every instance, sorted. Padding must be one trailing segment."""
    pad = dataset.pad_token_id
    out = []
    for i in range(len(dataset)):
        segments = _segments(dataset[i])
        if segments[-1][0][0] == pad:
            tokens, mask = segments.pop()
            assert set(tokens) == {pad} and not mask.any()
        assert all(pad not in tokens for tokens, _ in segments)
        out += [(tokens, mask.tolist()) for tokens, mask in segments]
    return sorted(out)


def _place_instances(dataset):
    """Per part, (token_ids, label_mask, seen): each instance's tokens put back at their offsets in
    the part file, using OLMo-core's own packing indices, and which offsets any instance covered."""
    dtype = dataset.indices_dtype
    placed = []
    for group, path in enumerate(dataset.paths):
        size = dataset.source_size_groups[group][0]
        tokens, labels, seen = np.zeros(size, np.int64), np.zeros(size, np.bool_), np.zeros(size, np.bool_)
        doc_spans = np.fromfile(dataset._get_document_indices_path(path), dtype=dtype).reshape(-1, 2)
        instance_spans = np.fromfile(dataset._get_instance_offsets_path(path), dtype=dtype).reshape(-1, 2)
        docs_by_instance = np.fromfile(dataset._get_docs_by_instance_path(path), dtype=dtype)
        first_instance = dataset.source_instance_offsets[group][0]
        for j, (lo, hi) in enumerate(instance_spans):
            item = dataset[first_instance + j]
            cursor = 0
            for doc_id in docs_by_instance[lo:hi]:
                start, end = (int(x) for x in doc_spans[doc_id])
                assert not seen[start:end].any()
                tokens[start:end] = item["input_ids"][cursor : cursor + end - start].numpy()
                labels[start:end] = item["label_mask"][cursor : cursor + end - start].numpy()
                seen[start:end] = True
                cursor += end - start
        placed.append((tokens, labels, seen))
    return placed


class TestFreshRender(_NumpySftTestBase):
    """A row-aligned cache, read back row by row, equals a fresh render of the source rows."""

    MAX_SEQ_LENGTH = 256

    def _convert(self):
        output_dir = pathlib.Path(self.temp_dir.name) / "fresh"
        real_write = numpy_dataset_conversion._write_memmap_chunked_from_file

        def write_small_parts(base_filename, source_path, total_items, dtype, **kwargs):
            max_size_gb = 3000 * np.dtype(dtype).itemsize / 1024**3
            return real_write(base_filename, source_path, total_items, dtype, max_size_gb=max_size_gb, **kwargs)

        with unittest.mock.patch.object(
            numpy_dataset_conversion, "_write_memmap_chunked_from_file", side_effect=write_small_parts
        ):
            numpy_dataset_conversion.convert_hf_to_numpy_sft(
                output_dir=output_dir,
                dataset_mixer_list=[os.path.join(TEST_DATA_DIR, "sft_sample.jsonl"), "1.0"],
                dataset_mixer_list_splits=["train"],
                tc=self._make_tc(),
                dataset_transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
                transform_fn_args=[{"max_seq_length": self.MAX_SEQ_LENGTH}, {}],
                dataset_target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
                dataset_skip_cache=True,
                dataset_local_cache_dir=self.temp_dir.name,
                row_aligned_parts=True,
            )
        return output_dir

    def _fresh_rows(self, tc):
        """Render each source conversation directly: token ids from the chat template, truncated as
        the tokenizer does; labels from the SFT transform applied to the row on its own."""
        rows = []
        with open(os.path.join(TEST_DATA_DIR, "sft_sample.jsonl")) as f:
            for line in f:
                row = json.loads(line)
                rendered = tc.tokenizer.apply_chat_template(row["messages"], tokenize=True, return_dict=False)
                labels = dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
                    dict(row), tc.tokenizer, self.MAX_SEQ_LENGTH
                )[dataset_transformation.LABELS_KEY]
                if not dataset_transformation.sft_tulu_filter_v1(
                    {dataset_transformation.LABELS_KEY: labels}, tc.tokenizer
                ):
                    continue
                rows.append((list(rendered[: self.MAX_SEQ_LENGTH]), [label != -100 for label in labels]))
        return rows

    def test_cache_rows_equal_fresh_render(self):
        tc = self._make_tc()
        output_dir = self._convert()
        parts = self._parts(output_dir)
        self.assertGreater(len(parts), 1)
        cached = sorted(
            (tokens[a:b].tolist(), labels[a:b].tolist()) for tokens, labels, spans in parts for a, b in spans
        )
        fresh = self._fresh_rows(tc)
        self.assertEqual(cached, sorted(fresh))
        # The sample includes conversations longer than the limit: kept, truncated, without EOS.
        eos = tc.tokenizer.eos_token_id
        truncated = [tokens for tokens, _ in fresh if len(tokens) == self.MAX_SEQ_LENGTH and tokens[-1] != eos]
        self.assertGreater(len(truncated), 0)
        olmo_core_finetune._check_row_aligned_cache(str(output_dir))

    def test_full_length_rows_stay_one_document_under_metadata_boundaries(self):
        """Rows truncated to exactly the sequence length have no EOS. With metadata boundaries each
        is a document of its own; the EOS scan merges it into the next row and truncates the merged
        span, so that next row never trains."""
        tc = self._make_tc()
        output_dir = self._convert()
        tokenizer = _oc_tokenizer(tc, eos_token_id=tc.tokenizer.eos_token_id, pad_token_id=tc.tokenizer.pad_token_id)
        fresh = sorted(self._fresh_rows(tc))
        meta = _build_packed(output_dir, tokenizer, self.MAX_SEQ_LENGTH, from_metadata=True)
        self.assertEqual(_documents(meta), fresh)
        eos = _build_packed(output_dir, tokenizer, self.MAX_SEQ_LENGTH, from_metadata=False)
        trained = sum(int(eos[i]["label_mask"].sum()) for i in range(len(eos)))
        self.assertLess(trained, sum(sum(mask) for _, mask in fresh))


class TestOlmo35MultiTurnRow(_NumpySftTestBase):
    """A two-turn olmo35 row has an EOS after its first assistant turn. By default it trains as one
    document; the opt-out (legacy layout, EOS scan) splits it there, as training did before."""

    SEQUENCE_LENGTH = 256

    def _row(self):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.add_tokens(["<|im_start|>", "<|im_end|>"], special_tokens=True)
        with open(os.path.join(TEST_DATA_DIR, "olmo35_chat_template.jinja")) as f:
            tokenizer.chat_template = f.read()
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "6"},
        ]
        ids, _, labels, _ = dataset_transformation._tokenize_tulu_sft_with_assistant_labels(
            messages, tokenizer, None, self.SEQUENCE_LENGTH
        )
        tokens, labels = ids.flatten().tolist(), labels.flatten().tolist()
        eos_positions = [i for i, token in enumerate(tokens) if token == tokenizer.eos_token_id]
        self.assertEqual(len(eos_positions), 2)
        self.assertEqual(eos_positions[1], len(tokens) - 1)
        oc_tokenizer = oc_data.TokenizerConfig(
            vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
        )
        return tokens, labels, eos_positions[0] + 1, oc_tokenizer

    def _pack(self, name, sft_config, tokens, labels, oc_tokenizer):
        output_dir = _convert_rows(self, name, [(tokens, labels)], sft_config.row_aligned_parts)
        config = olmo_core_finetune._numpy_dataset_config(
            str(output_dir),
            oc_tokenizer,
            work_dir=str(output_dir / "work"),
            sequence_length=self.SEQUENCE_LENGTH,
            document_boundary_kwargs=olmo_core_finetune._document_boundary_kwargs(
                sft_config.document_boundaries_from_metadata, sft_config.row_aligned_parts
            ),
        )
        dataset = config.build()
        dataset.prepare()
        self.assertEqual(len(dataset), 1)
        positions = np.concatenate([np.arange(length) for length in dataset[0]["doc_lens"].tolist()])
        return _documents(dataset), positions[: len(tokens)].tolist()

    def test_default_keeps_the_row_one_document(self):
        tokens, labels, _, oc_tokenizer = self._row()
        mask = [label != -100 for label in labels]
        documents, positions = self._pack("default", olmo_core_finetune.SFTConfig(), tokens, labels, oc_tokenizer)
        self.assertEqual(documents, [(tokens, mask)])
        # Positions run on across the turn boundary instead of restarting after the first EOS.
        self.assertEqual(positions, list(range(len(tokens))))

    def test_opt_out_splits_the_row_at_the_first_assistant_eos(self):
        tokens, labels, split, oc_tokenizer = self._row()
        mask = [label != -100 for label in labels]
        opt_out = olmo_core_finetune.SFTConfig(row_aligned_parts=False, document_boundaries_from_metadata=False)
        documents, positions = self._pack("opt-out", opt_out, tokens, labels, oc_tokenizer)
        self.assertEqual(documents, sorted([(tokens[:split], mask[:split]), (tokens[split:], mask[split:])]))
        self.assertEqual(positions, list(range(split)) + list(range(len(tokens) - split)))


class TestResumeEquivalence(_NumpySftTestBase):
    def test_row_aligned_resume_rejects_existing_empty_rows(self):
        root = pathlib.Path(self.temp_dir.name)
        tokens, labels, boundaries = (root / name for name in ("tokens.bin", "labels.bin", "boundaries.bin"))
        np.asarray([1, 2], dtype=np.uint16).tofile(tokens)
        np.asarray([1, 1], dtype=np.uint8).tofile(labels)
        np.asarray([[0, 2], [2, 2]], dtype=np.int64).tofile(boundaries)
        with self.assertRaisesRegex(ValueError, "row_aligned_parts.*empty row"):
            numpy_dataset_conversion._recover_partial_state(tokens, labels, boundaries, 2, reject_empty_rows=True)
        self.assertEqual(numpy_dataset_conversion._recover_partial_state(tokens, labels, boundaries, 2), (2, 2))

    def _run(self, output_dir, resume, batch_size=10):
        numpy_dataset_conversion.convert_hf_to_numpy_sft(
            output_dir=output_dir,
            dataset_mixer_list=[os.path.join(TEST_DATA_DIR, "sft_sample.jsonl"), "1.0"],
            dataset_mixer_list_splits=["train"],
            tc=self._make_tc(),
            dataset_transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
            transform_fn_args=[{"max_seq_length": 4096}, {}],
            dataset_target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
            num_examples=50,
            resume=resume,
            batch_size=batch_size,
        )

    def test_one_shot_matches_interrupt_plus_resume(self):
        golden = pathlib.Path(self.temp_dir.name) / "golden"
        self._run(golden, resume=False)

        interrupted = pathlib.Path(self.temp_dir.name) / "interrupted"
        real_flush = numpy_dataset_conversion._flush_partial_files
        call_count = {"n": 0}

        def fail_after_two(*args, **kwargs):
            result = real_flush(*args, **kwargs)
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated interrupt")
            return result

        with (
            unittest.mock.patch.object(numpy_dataset_conversion, "_flush_partial_files", side_effect=fail_after_two),
            self.assertRaises(RuntimeError),
        ):
            self._run(interrupted, resume=False)

        self.assertTrue(
            (interrupted / "_boundaries.partial.bin").exists(), "interrupted run should have left partial files behind"
        )

        self._run(interrupted, resume=True)

        artifacts = sorted(
            p.name
            for p in golden.iterdir()
            if p.name.startswith("token_ids_part_") or p.name.startswith("labels_mask_part_")
        )
        self.assertGreater(len(artifacts), 0, "golden run produced no artifacts")
        for name in artifacts:
            if name.endswith(".gz"):
                with gzip.open(golden / name, "rb") as f1, gzip.open(interrupted / name, "rb") as f2:
                    self.assertEqual(f1.read(), f2.read(), msg=f"mismatch in {name}")
            else:
                with (golden / name).open("rb") as f1, (interrupted / name).open("rb") as f2:
                    self.assertEqual(f1.read(), f2.read(), msg=f"mismatch in {name}")

    def test_resume_after_every_row_was_collected(self):
        """A run preempted after its collect loop (here, in the statistics pass) resumes with nothing left to collect."""
        golden = pathlib.Path(self.temp_dir.name) / "golden"
        self._run(golden, resume=False)

        interrupted = pathlib.Path(self.temp_dir.name) / "interrupted"
        with (
            unittest.mock.patch.object(
                numpy_dataset_conversion, "_aggregate_stats", side_effect=RuntimeError("simulated interrupt")
            ),
            self.assertRaises(RuntimeError),
        ):
            self._run(interrupted, resume=False)
        self.assertTrue((interrupted / "_boundaries.partial.bin").exists())

        self._run(interrupted, resume=True)

        artifacts = sorted(
            p.name for p in golden.iterdir() if p.name.startswith(("token_ids_part_", "labels_mask_part_"))
        )
        self.assertGreater(len(artifacts), 0, "golden run produced no artifacts")
        for name in artifacts:
            opener = gzip.open if name.endswith(".gz") else open
            with opener(golden / name, "rb") as f1, opener(interrupted / name, "rb") as f2:
                self.assertEqual(f1.read(), f2.read(), msg=f"mismatch in {name}")

    def test_resume_rejects_partial_files_longer_than_the_dataset(self):
        golden = pathlib.Path(self.temp_dir.name) / "golden"
        with (
            unittest.mock.patch.object(
                numpy_dataset_conversion, "_aggregate_stats", side_effect=RuntimeError("simulated interrupt")
            ),
            self.assertRaises(RuntimeError),
        ):
            self._run(golden, resume=False)
        with (
            unittest.mock.patch.object(
                numpy_dataset_conversion, "_recover_partial_state", return_value=(10**6, 10**8)
            ),
            self.assertRaisesRegex(ValueError, "belong to another dataset"),
        ):
            self._run(golden, resume=True)


# Toy ids for the replay EOS fix: uint32 tokens (vocab above 2**16), as for the Olmo tokenizer.
_R_PAD, _R_EOS, _R_IM, _R_VOCAB = 1, 2, 3, 100_352


def _write_replay_dir(numpy_dir, parts, manifest=True):
    """Write a replay dir as the ledger's c11_replay.py build does: per part, token ids, a one-byte mask and a
    .csv.gz row per document, plus `manifest.jsonl` (part, start, end, length_with_eos, tokens_sha256) and a
    `stats.json` but no `dataset_statistics.json`. Each document is (tokens, mask): one bool, or one per token."""
    os.makedirs(numpy_dir)
    records = []
    for index, docs in enumerate(parts):
        tokens = np.array([t for doc, _ in docs for t in doc], dtype=np.uint32)
        mask = np.array(
            [m for doc, ms in docs for m in (ms if isinstance(ms, list) else [ms] * len(doc))], dtype=np.bool_
        )
        tokens.tofile(os.path.join(numpy_dir, f"token_ids_part_{index:04d}.npy"))
        mask.tofile(os.path.join(numpy_dir, f"labels_mask_part_{index:04d}.npy"))
        with gzip.open(os.path.join(numpy_dir, f"token_ids_part_{index:04d}.csv.gz"), "wt") as f:
            start = 0
            for doc, _ in docs:
                f.write(f"{start},{start + len(doc)}\n")
                sha = hashlib.sha256(np.array(doc[:-1], dtype=np.uint32).tobytes()).hexdigest()
                records.append(
                    {
                        "part": index,
                        "start": start,
                        "end": start + len(doc),
                        "id": f"doc{len(records)}",
                        "length_with_eos": len(doc),
                        "tokens_sha256": sha,
                    }
                )
                start += len(doc)
    if manifest:
        with open(os.path.join(numpy_dir, "manifest.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    with open(os.path.join(numpy_dir, "stats.json"), "w") as f:
        json.dump({"docs": len(records)}, f)


def _file_bytes(numpy_dir):
    out = {}
    for name in sorted(os.listdir(numpy_dir)):
        with open(os.path.join(numpy_dir, name), "rb") as f:
            out[name] = f.read()
    return out


def _chat_row(*content):
    """A chat row: prompt untrained, the assistant's tokens and its closing EOS trained."""
    return [_R_IM, 10, 11, _R_IM, *content, _R_EOS], [False] * 4 + [True] * (len(content) + 1)


class TestReplayFinalEosPositions(unittest.TestCase):
    def test_selects_only_the_final_eos_of_replay_documents(self):
        docs = [
            [_R_IM, 10, 11, _R_EOS],  # chat stop: kept
            [20, 21, _R_EOS],  # replay: its final EOS
            [22, _R_EOS, 23, _R_EOS],  # replay with an interior EOS: only the final one
            [24, 25],  # replay ending in another token, e.g. cut: nothing
            [_R_EOS],  # one-token documents: their target comes from the previous document
            [26],
            [27, _R_EOS],  # two tokens: selected
            [_R_IM, 12, _R_EOS, _R_IM, 13, _R_EOS],  # multi-turn chat row: every stop kept
            [_R_IM],
        ]
        tokens = np.array([t for doc in docs for t in doc], dtype=np.uint32)
        ends = np.cumsum([len(doc) for doc in docs])
        bounds = np.stack([ends - [len(doc) for doc in docs], ends], axis=1)
        positions = numpy_dataset_conversion.replay_final_eos_positions(tokens, bounds, _R_EOS, _R_IM)
        self.assertEqual(positions.tolist(), [ends[1] - 1, ends[2] - 1, ends[6] - 1])
        self.assertTrue((tokens[positions] == _R_EOS).all())


class TestWriteReplayEosMaskedDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.tokenizer = oc_data.TokenizerConfig(vocab_size=_R_VOCAB, eos_token_id=_R_EOS, pad_token_id=_R_PAD)
        self.cache = os.path.join(self.tmp, "cache")
        _write_replay_dir(self.cache, [[_chat_row(12), _chat_row(13, 14)]], manifest=False)

    def _masked(self, src, name="masked", manifest_sha256=None):
        """Mask `src`, pinning its manifest's current sha256 unless another is given."""
        if manifest_sha256 is None:
            with open(os.path.join(src, "manifest.jsonl"), "rb") as f:
                manifest_sha256 = hashlib.sha256(f.read()).hexdigest()
        out = os.path.join(self.tmp, name)
        return out, numpy_dataset_conversion.write_replay_eos_masked_dir(src, out, manifest_sha256, _R_EOS, _R_IM)

    def _startup_check(self, numpy_dir, sequence_length=8):
        """Current main's extra-dir validation at startup, under metadata boundaries (the default)."""
        return olmo_core_finetune._check_extra_numpy_dirs(
            [numpy_dir], self.cache, self.tokenizer, sequence_length=sequence_length, require_metadata=True, log=False
        )[numpy_dir]

    def test_clears_exactly_the_replay_final_eos_labels(self):
        src = os.path.join(self.tmp, "src")
        parts = [
            [([20, 21, _R_EOS], True), ([24, 25], True), ([30, 31, _R_EOS], [True, True, False])],
            [([22, _R_EOS, 23, _R_EOS], True), _chat_row(13, _R_EOS, 14), ([_R_EOS], True), ([27, _R_EOS], True)],
        ]
        _write_replay_dir(src, parts)
        before = _file_bytes(src)
        out, stats = self._masked(src)
        self.assertEqual(_file_bytes(src), before)
        # Part 0: the doc at 0-2 (the one at 5-7 is already untrained; 3-4 ends in another token). Part 1: the
        # interior-EOS doc's final token (3), not its interior EOS (1), and the two-token doc's (14); the chat
        # row and the one-token doc keep theirs.
        for index, positions in {"0000": [2], "0001": [3, 14]}.items():
            src_mask = np.fromfile(os.path.join(src, f"labels_mask_part_{index}.npy"), dtype=np.bool_)
            out_mask = np.fromfile(os.path.join(out, f"labels_mask_part_{index}.npy"), dtype=np.bool_)
            self.assertEqual(np.flatnonzero(src_mask != out_mask).tolist(), positions)
            for name in (f"token_ids_part_{index}.npy", f"token_ids_part_{index}.csv.gz"):
                self.assertEqual(os.path.realpath(os.path.join(out, name)), os.path.realpath(os.path.join(src, name)))
        self.assertEqual(
            os.path.realpath(os.path.join(out, "manifest.jsonl")),
            os.path.realpath(os.path.join(src, "manifest.jsonl")),
        )
        self.assertFalse(os.path.exists(os.path.join(out, "stats.json")))  # the source's counts are stale here
        change = stats["configuration"]["replay_eos_mask"]
        self.assertEqual((change["documents"], change["dropped"], change["already_untrained"]), (7, 3, 1))
        # 23 tokens, less one already untrained, four chat prompt tokens and three dropped.
        self.assertEqual(stats["overall_statistics"]["trainable_tokens"], 23 - 1 - 4 - 3)
        with open(os.path.join(out, "dataset_statistics.json")) as f:
            self.assertEqual(json.load(f), stats)

    def test_masked_dir_passes_startup_validation_on_main(self):
        """The c11 layout has no dataset_statistics.json, so main refuses a multi-part source dir; the masked
        dir's statistics carry the row-alignment marker and the new loss-token count."""
        src = os.path.join(self.tmp, "src")
        _write_replay_dir(src, [[([20, 21, _R_EOS], True), ([22, _R_EOS], True)], [([23, 24, 25, _R_EOS], True)]])
        with self.assertRaisesRegex(ValueError, "dataset_statistics.json"):
            self._startup_check(src)
        out, stats = self._masked(src)
        checked = self._startup_check(out)
        self.assertEqual((checked.num_parts, checked.num_docs, checked.num_tokens), (2, 3, 9))
        self.assertEqual(checked.num_loss_tokens, 9 - 3)
        self.assertEqual(stats["overall_statistics"]["trainable_tokens"], checked.num_loss_tokens)
        self.assertTrue(stats["configuration"][numpy_dataset_conversion.ROW_ALIGNED_PARTS_KEY])

    def test_dir_whose_documents_all_start_with_im_start_is_byte_identical(self):
        src = os.path.join(self.tmp, "chat")
        _write_replay_dir(src, [[_chat_row(12), _chat_row(13, _R_EOS, 14)]])
        out, stats = self._masked(src)
        self.assertEqual(stats["configuration"]["replay_eos_mask"]["dropped"], 0)
        with (
            open(os.path.join(src, "labels_mask_part_0000.npy"), "rb") as a,
            open(os.path.join(out, "labels_mask_part_0000.npy"), "rb") as b,
        ):
            self.assertEqual(a.read(), b.read())

    def _refused(self, src, pattern, manifest_sha256=None):
        with self.assertRaisesRegex(ValueError, pattern):
            self._masked(src, manifest_sha256=manifest_sha256)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "masked")))

    def test_refuses_a_manifest_that_is_not_the_pinned_one(self):
        """A rewritten manifest and CSV that cut [44, EOS, 55, EOS] at its interior EOS agree with each other."""
        src = os.path.join(self.tmp, "src")
        _write_replay_dir(src, [[([44, _R_EOS, 55, _R_EOS], True)]])
        with open(os.path.join(src, "manifest.jsonl"), "rb") as f:
            pinned = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(src, "manifest.jsonl"), "w") as f:
            f.writelines(json.dumps({"part": 0, "start": a, "end": b}) + "\n" for a, b in ((0, 2), (2, 4)))
        with gzip.open(os.path.join(src, "token_ids_part_0000.csv.gz"), "wt") as f:
            f.write("0,2\n2,4\n")
        self._refused(src, "not the pinned", manifest_sha256=pinned)

    def test_refuses_a_blank_csv_line(self):
        """OLMo-core cannot parse it, and main's startup check would count it as a row."""
        src = os.path.join(self.tmp, "src")
        _write_replay_dir(src, [[([20, _R_EOS], True), ([21, _R_EOS], True)]])
        with gzip.open(os.path.join(src, "token_ids_part_0000.csv.gz"), "wt") as f:
            f.write("0,2\n\n2,4\n")
        self._refused(src, "not a start,end row")

    def test_refuses_a_dir_without_the_build_manifest(self):
        src = os.path.join(self.tmp, "src")
        _write_replay_dir(src, [[([20, 21, _R_EOS], True)]], manifest=False)
        self._refused(src, "manifest.jsonl", manifest_sha256="0" * 64)

    def test_refuses_csv_rows_that_differ_from_the_manifest(self):
        """Rows split at an interior EOS, as an EOS scan gives, would make that EOS look final."""
        src = os.path.join(self.tmp, "src")
        _write_replay_dir(src, [[([22, _R_EOS, 23, _R_EOS], True)]])
        with gzip.open(os.path.join(src, "token_ids_part_0000.csv.gz"), "wt") as f:
            f.write("0,2\n2,4\n")
        self._refused(src, "differ from the manifest")

    def test_refuses_a_document_cut_between_parts(self):
        src = os.path.join(self.tmp, "src")
        _write_replay_dir(src, [[([20, 21], True)], [([22, _R_EOS], True)]])
        with open(os.path.join(src, "manifest.jsonl"), "w") as f:  # one document over both parts
            f.write(json.dumps({"part": 0, "start": 0, "end": 4, "length_with_eos": 4}) + "\n")
        self._refused(src, "index set")

    def test_refuses_orphan_parts(self):
        src = os.path.join(self.tmp, "src")
        _write_replay_dir(src, [[([20, _R_EOS], True)]])
        np.ones(2, dtype=np.bool_).tofile(os.path.join(src, "labels_mask_part_0001.npy"))
        self._refused(src, "index set")

    def test_refuses_an_existing_output_dir(self):
        src = os.path.join(self.tmp, "src")
        _write_replay_dir(src, [[([20, _R_EOS], True)]])
        os.makedirs(os.path.join(self.tmp, "masked"))
        with self.assertRaises(FileExistsError):
            self._masked(src)
        self.assertEqual(os.listdir(os.path.join(self.tmp, "masked")), [])

    def test_packed_training_drops_only_whole_replay_final_eos(self):
        """Through OLMo-core's packing: same instances and tokens; one label fewer per replay document that
        fits, none for chat rows or for a replay document truncated at the sequence length."""
        sequence_length = 8
        src = os.path.join(self.tmp, "replay")
        fitting = [[20, 21, 22, _R_EOS], [23, _R_EOS], [24, 25, 26, 27, 28, 29, 30, _R_EOS]]
        too_long = [40, 41, 42, 43, 44, 45, 46, 47, 48, _R_EOS]  # truncated to its first 8 tokens
        _write_replay_dir(src, [[(doc, True) for doc in [*fitting, too_long]]])
        out, stats = self._masked(src)
        self.assertEqual(stats["configuration"]["replay_eos_mask"]["dropped"], 4)  # too_long's is cut by packing
        boundaries = olmo_core_finetune._document_boundary_kwargs(True, row_aligned_parts=True)
        datasets = []
        for name, replay in (("src", src), ("out", out)):
            config = olmo_core_finetune._numpy_dataset_config(
                self.cache,
                self.tokenizer,
                work_dir=os.path.join(self.tmp, f"work-{name}"),
                sequence_length=sequence_length,
                document_boundary_kwargs=boundaries,
                extra_numpy_dirs=[replay],
            )
            datasets.append(config.build())
            datasets[-1].prepare()
        before, after = datasets
        self.assertEqual(len(before), len(after))
        dropped = []
        for i in range(len(before)):
            a, b = before[i], after[i]
            self.assertTrue(torch.equal(a["input_ids"], b["input_ids"]))
            self.assertTrue(torch.equal(a["doc_lens"], b["doc_lens"]))
            self.assertFalse((b["label_mask"] & ~a["label_mask"]).any())
            for position in torch.nonzero(a["label_mask"] & ~b["label_mask"]).flatten().tolist():
                dropped.append(a["input_ids"][: position + 1].tolist()[-2:])
        # Exactly one per fitting replay document: its final EOS, predicted from its own previous token.
        self.assertEqual(sorted(dropped), sorted([doc[-2], _R_EOS] for doc in fitting))


if __name__ == "__main__":
    unittest.main()
