"""Tests for open_instruct.numpy_dataset_conversion.

Run from project root:
    uv run pytest open_instruct/test_numpy_dataset_conversion.py -v
"""

import functools
import gc
import gzip
import json
import os
import pathlib
import shutil
import tempfile
import time
import unittest
import unittest.mock

import datasets
import numpy as np
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


class TestSegmentSums(unittest.TestCase):
    @staticmethod
    def _reference(values, starts):
        """np.add.reduceat with empty segments set to 0, which is what _aggregate_stats computed before."""
        out = np.zeros(len(starts), dtype=np.int64)
        lengths = np.diff(np.append(starts, len(values)))
        inside = starts < len(values)
        if inside.any():
            out[inside] = np.add.reduceat(values, starts[inside], dtype=np.int64)
        out[lengths == 0] = 0
        return out

    @parameterized.expand([(1,), (3,), (64,), (1 << 27,)])
    def test_matches_reduceat(self, chunk_size):
        rng = np.random.default_rng(chunk_size)
        for trial in range(20):
            lengths = rng.choice([0, 0, 1, 2, 5, 17, 130], size=int(rng.integers(1, 40)))
            values = rng.integers(0, 2, size=int(lengths.sum()), dtype=np.uint8)
            starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
            with self.subTest(trial=trial):
                result = numpy_dataset_conversion._segment_sums(values, starts, chunk_size=chunk_size)
                np.testing.assert_array_equal(result, self._reference(values, starts))
                np.testing.assert_array_equal(result, [values[s : s + n].sum() for s, n in zip(starts, lengths)])

    def test_trailing_empty_segment(self):
        # np.add.reduceat rejects a start equal to len(values), which a trailing empty row has.
        values = np.array([1, 0, 1], dtype=np.uint8)
        result = numpy_dataset_conversion._segment_sums(values, np.array([0, 3, 3]), chunk_size=2)
        np.testing.assert_array_equal(result, [2, 0, 0])

    def test_no_values(self):
        values = np.zeros(0, dtype=np.uint8)
        np.testing.assert_array_equal(numpy_dataset_conversion._segment_sums(values, np.array([0, 0])), [0, 0])

    def test_rejects_decreasing_starts(self):
        with self.assertRaises(ValueError):
            numpy_dataset_conversion._segment_sums(np.ones(4, dtype=np.uint8), np.array([0, 3, 2]))


class TestPrefetchInOrder(unittest.TestCase):
    def test_yields_in_input_order(self):
        def slow_for_even(x):
            time.sleep(0.002 * (x % 2 == 0))
            return x * x

        for workers, depth in ((1, 1), (4, 2), (16, 32)):
            with self.subTest(workers=workers, depth=depth):
                result = list(
                    numpy_dataset_conversion._prefetch_in_order(range(50), slow_for_even, workers=workers, depth=depth)
                )
                self.assertEqual(result, [x * x for x in range(50)])

    def test_reraises_worker_errors_in_order(self):
        def fail_on_three(x):
            if x == 3:
                raise KeyError(x)
            return x

        seen = []
        with self.assertRaises(KeyError):
            for x in numpy_dataset_conversion._prefetch_in_order(range(10), fail_on_three, workers=4, depth=4):
                seen.append(x)
        self.assertEqual(seen, [0, 1, 2])

    def test_closing_early_stops_reading_input(self):
        consumed = []

        def items():
            for i in range(1000):
                consumed.append(i)
                yield i

        gen = numpy_dataset_conversion._prefetch_in_order(items(), lambda x: x, workers=2, depth=4)
        self.assertEqual(next(gen), 0)
        gen.close()
        self.assertLessEqual(len(consumed), 4)


def _synthetic_tokenized_dataset(num_rows, seed, value_type="int64", num_chunks=4):
    """Tokenized rows like a real SFT mix, in `num_chunks` Arrow chunks: empty, one-token and long rows, -100 spans,
    rows with nothing trainable, token ids up to the dolma2 vocabulary, and four sources."""
    rng = np.random.default_rng(seed)
    columns = {key: [] for key in (*dataset_transformation.TOKENIZED_SFT_DATASET_KEYS, "source")}
    for i in range(num_rows):
        length = int(rng.choice([0, 1, 2, int(rng.integers(3, 400)), int(rng.integers(400, 3000))]))
        if i == 7:
            length = 20000
        tokens = rng.integers(0, 100278, size=length)
        labels = tokens.copy()
        if length:
            labels[: int(rng.integers(0, length + 1))] = -100
            if i % 11 == 0:
                labels[:] = -100
        columns[dataset_transformation.INPUT_IDS_KEY].append(tokens.tolist())
        columns[dataset_transformation.LABELS_KEY].append(labels.tolist())
        columns[dataset_transformation.ATTENTION_MASK_KEY].append([1] * length)
        columns["source"].append(f"source-{i % 4}")
    columns[dataset_transformation.DATASET_ORIGIN_KEY] = columns.pop("source")
    sequence = datasets.Sequence(datasets.Value(value_type))
    features = datasets.Features(
        {
            dataset_transformation.INPUT_IDS_KEY: sequence,
            dataset_transformation.LABELS_KEY: sequence,
            dataset_transformation.ATTENTION_MASK_KEY: sequence,
            dataset_transformation.DATASET_ORIGIN_KEY: datasets.Value("string"),
        }
    )
    bounds = np.linspace(0, num_rows, num_chunks + 1).astype(int)
    return datasets.concatenate_datasets(
        [
            datasets.Dataset.from_dict({k: v[lo:hi] for k, v in columns.items()}, features=features)
            for lo, hi in zip(bounds[:-1], bounds[1:])
        ]
    )


def _row_loop_partial_files(dataset, token_dtype, reject_empty_rows=False):
    """The three partial files' bytes as the row-by-row collect loop wrote them before it was batched, and the
    error it raised, if any. It checked a row for emptiness before writing it and its attention mask after."""
    tokens, labels, boundaries = bytearray(), bytearray(), bytearray()
    position = 0
    for idx, row in enumerate(dataset.with_format("numpy")):
        sample_tokens = row[dataset_transformation.INPUT_IDS_KEY]
        if reject_empty_rows and len(sample_tokens) == 0:
            return (bytes(tokens), bytes(labels), bytes(boundaries)), ValueError(f"empty row {idx}")
        tokens += np.asarray(sample_tokens, dtype=token_dtype).tobytes()
        labels += (np.asarray(row[dataset_transformation.LABELS_KEY]) != -100).astype(np.uint8).tobytes()
        boundaries += np.array([position, position + len(sample_tokens)], dtype=np.int64).tobytes()
        position += len(sample_tokens)
        if not (row[dataset_transformation.ATTENTION_MASK_KEY] == 1).all():
            return (bytes(tokens), bytes(labels), bytes(boundaries)), AssertionError(f"attention mask in row {idx}")
    return (bytes(tokens), bytes(labels), bytes(boundaries)), None


_PARTIAL_NAMES = ("_tokens.partial.bin", "_labels.partial.bin", "_boundaries.partial.bin")


class TestBatchedCollect(_NumpySftTestBase):
    """The batched, threaded collect loop writes the partial files byte for byte as the row-by-row loop did."""

    SHUFFLE_SEED = 3

    def _convert(self, dataset, output_dir, batch_size=1000, workers=None, resume=False, row_aligned_parts=False):
        """Run convert_hf_to_numpy_sft on an already tokenized dataset. Returns the partial files' bytes as the
        statistics stage saw them, or None if it was not reached."""
        snapshot = {}
        real_aggregate = numpy_dataset_conversion._aggregate_stats

        def aggregate(**kwargs):
            snapshot["files"] = tuple((output_dir / name).read_bytes() for name in _PARTIAL_NAMES)
            return real_aggregate(**kwargs)

        real_prefetch = numpy_dataset_conversion._prefetch_in_order
        prefetch = (
            real_prefetch if workers is None else functools.partial(real_prefetch, workers=workers, depth=2 * workers)
        )
        with (
            unittest.mock.patch.object(
                dataset_transformation, "get_cached_dataset_tulu_with_statistics", return_value=(dataset, {})
            ),
            unittest.mock.patch.object(numpy_dataset_conversion, "_aggregate_stats", side_effect=aggregate),
            unittest.mock.patch.object(numpy_dataset_conversion, "_prefetch_in_order", side_effect=prefetch),
        ):
            numpy_dataset_conversion.convert_hf_to_numpy_sft(
                output_dir=output_dir,
                dataset_mixer_list=[],
                dataset_mixer_list_splits=[],
                tc=self._make_tc(),
                dataset_transform_fn=[],
                transform_fn_args=[],
                dataset_target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
                dataset_config_hash="synthetic",
                shuffle_seed=self.SHUFFLE_SEED,
                batch_size=batch_size,
                resume=resume,
                row_aligned_parts=row_aligned_parts,
            )
        return snapshot.get("files")

    def _expected(self, dataset, reject_empty_rows=False):
        token_dtype = numpy_dataset_conversion._select_token_dtype(self._make_tc().tokenizer.vocab_size)
        return _row_loop_partial_files(
            dataset.shuffle(seed=self.SHUFFLE_SEED), token_dtype, reject_empty_rows=reject_empty_rows
        )

    def _outputs(self, output_dir):
        """Every part file's bytes, and the statistics without their timestamp and directory."""
        out = {
            path.name: path.read_bytes()
            for path in sorted(output_dir.iterdir())
            if path.name.startswith(("token_ids_part_", "labels_mask_part_"))
        }
        stats = json.loads((output_dir / "dataset_statistics.json").read_text())
        del stats["timestamp"], stats["output_directory"]
        out["dataset_statistics.json"] = stats
        return out

    @parameterized.expand(
        [
            ("int64_default", "int64", 1000, None),
            ("int32_default", "int32", 1000, None),
            ("one_row_batches", "int64", 1, 3),
            ("odd_batches_one_worker", "int64", 7, 1),
            ("large_batches", "int32", 4096, 16),
            ("batches_of_999", "int64", 999, 8),
        ]
    )
    def test_partial_files_match_row_loop(self, _name, value_type, batch_size, workers):
        dataset = _synthetic_tokenized_dataset(1500, seed=1, value_type=value_type)
        output_dir = pathlib.Path(self.temp_dir.name) / "out"
        files = self._convert(dataset, output_dir, batch_size=batch_size, workers=workers)
        expected, error = self._expected(dataset)
        self.assertIsNone(error)
        for name, got, want in zip(_PARTIAL_NAMES, files, expected):
            self.assertEqual(got, want, msg=f"mismatch in {name}")

        stats = json.loads((output_dir / "dataset_statistics.json").read_text())["overall_statistics"]
        trainable = [sum(label != -100 for label in row) for row in dataset[dataset_transformation.LABELS_KEY]]
        self.assertEqual(stats["total_tokens"], sum(len(row) for row in dataset[dataset_transformation.INPUT_IDS_KEY]))
        self.assertEqual(stats["trainable_tokens"], sum(trainable))
        self.assertEqual(stats["instances_filtered"], sum(count == 0 for count in trainable))

    def test_bad_attention_mask_raises_after_writing_the_same_rows(self):
        dataset = _synthetic_tokenized_dataset(300, seed=2)
        shuffled_position = 123
        original_index = dataset.shuffle(seed=self.SHUFFLE_SEED)._indices.column(0)[shuffled_position].as_py()
        masks = list(dataset[dataset_transformation.ATTENTION_MASK_KEY])
        masks[original_index] = [1] * 5 + [0]
        dataset = dataset.remove_columns(dataset_transformation.ATTENTION_MASK_KEY).add_column(
            dataset_transformation.ATTENTION_MASK_KEY, masks
        )
        expected, error = self._expected(dataset)
        self.assertIsInstance(error, AssertionError)
        self.assertEqual(len(expected[2]) // 16, shuffled_position + 1)
        for batch_size in (1, 50, 1000):
            output_dir = pathlib.Path(self.temp_dir.name) / f"bad-mask-{batch_size}"
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(AssertionError, "Expected all attention mask values to be 1"):
                    self._convert(dataset, output_dir, batch_size=batch_size)
                for name, want in zip(_PARTIAL_NAMES, expected):
                    self.assertEqual((output_dir / name).read_bytes(), want, msg=f"mismatch in {name}")

    def test_row_aligned_empty_row_raises_after_writing_the_rows_before_it(self):
        dataset = _synthetic_tokenized_dataset(300, seed=4)
        expected, error = self._expected(dataset, reject_empty_rows=True)
        self.assertIsInstance(error, ValueError)
        first_empty = len(expected[2]) // 16
        for batch_size in (1, 64, 1000):
            output_dir = pathlib.Path(self.temp_dir.name) / f"empty-{batch_size}"
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, f"cannot write empty row {first_empty}:"):
                    self._convert(dataset, output_dir, batch_size=batch_size, row_aligned_parts=True)
                for name, want in zip(_PARTIAL_NAMES, expected):
                    self.assertEqual((output_dir / name).read_bytes(), want, msg=f"mismatch in {name}")

    def test_first_failure_in_row_order_wins_within_a_batch(self):
        dataset = _synthetic_tokenized_dataset(300, seed=4)
        order = dataset.shuffle(seed=self.SHUFFLE_SEED)._indices.column(0).to_pylist()
        lengths = [len(row) for row in dataset[dataset_transformation.INPUT_IDS_KEY]]
        first_empty = next(p for p, i in enumerate(order) if lengths[i] == 0)
        before = max(p for p in range(first_empty) if lengths[order[p]] > 0)
        after = min(p for p in range(first_empty + 1, len(order)) if lengths[order[p]] > 0)
        for bad_position, error_type in ((before, AssertionError), (after, ValueError)):
            masks = list(dataset[dataset_transformation.ATTENTION_MASK_KEY])
            masks[order[bad_position]] = [0] * lengths[order[bad_position]]
            bad = dataset.remove_columns(dataset_transformation.ATTENTION_MASK_KEY).add_column(
                dataset_transformation.ATTENTION_MASK_KEY, masks
            )
            expected, error = self._expected(bad, reject_empty_rows=True)
            self.assertIsInstance(error, error_type)
            output_dir = pathlib.Path(self.temp_dir.name) / f"mixed-{bad_position}"
            with self.subTest(bad_position=bad_position), self.assertRaises(error_type):
                self._convert(bad, output_dir, batch_size=1000, row_aligned_parts=True)
            for name, want in zip(_PARTIAL_NAMES, expected):
                self.assertEqual((output_dir / name).read_bytes(), want, msg=f"mismatch in {name}")

    def test_resume_after_every_row_was_collected(self):
        dataset = _synthetic_tokenized_dataset(400, seed=5)
        golden = pathlib.Path(self.temp_dir.name) / "golden"
        self._convert(dataset, golden)

        interrupted = pathlib.Path(self.temp_dir.name) / "interrupted"
        with (
            unittest.mock.patch.object(
                numpy_dataset_conversion, "_aggregate_stats", side_effect=RuntimeError("simulated interrupt")
            ),
            self.assertRaises(RuntimeError),
        ):
            self._convert(dataset, interrupted)
        self.assertTrue((interrupted / "_boundaries.partial.bin").exists())

        files = self._convert(dataset, interrupted, resume=True)
        self.assertEqual(files, self._expected(dataset)[0])
        self.assertEqual(self._outputs(interrupted), self._outputs(golden))

    def test_resume_from_a_torn_tail(self):
        dataset = _synthetic_tokenized_dataset(400, seed=6)
        golden = pathlib.Path(self.temp_dir.name) / "golden"
        self._convert(dataset, golden, batch_size=64)

        interrupted = pathlib.Path(self.temp_dir.name) / "interrupted"
        real_flush = numpy_dataset_conversion._flush_partial_files
        calls = {"n": 0}

        def fail_on_third(*args):
            real_flush(*args)
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated interrupt")

        with (
            unittest.mock.patch.object(numpy_dataset_conversion, "_flush_partial_files", side_effect=fail_on_third),
            self.assertRaises(RuntimeError),
        ):
            self._convert(dataset, interrupted, batch_size=64)
        # Half a row in the token and label files and half a boundary pair, as a write cut short leaves them.
        with (interrupted / "_tokens.partial.bin").open("ab") as f:
            f.write(b"\x07" * 10)
        with (interrupted / "_labels.partial.bin").open("ab") as f:
            f.write(b"\x01" * 5)
        with (interrupted / "_boundaries.partial.bin").open("ab") as f:
            f.write(b"\x00" * 8)

        files = self._convert(dataset, interrupted, batch_size=100, resume=True)
        self.assertEqual(files, self._expected(dataset)[0])
        self.assertEqual(self._outputs(interrupted), self._outputs(golden))


if __name__ == "__main__":
    unittest.main()
