"""Tests for open_instruct.numpy_dataset_conversion.

Run from project root:
    uv run pytest open_instruct/test_numpy_dataset_conversion.py -v
"""

import gc
import gzip
import json
import os
import pathlib
import shutil
import tempfile
import unittest
import unittest.mock

import numpy as np
from olmo_core import data as oc_data
from parameterized import parameterized

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


_requires_metadata_boundaries = unittest.skipUnless(
    olmo_core_finetune._olmo_core_supports_metadata_boundaries(),
    "installed OLMo-core lacks use_array_if_local (allenai/OLMo-core#843)",
)


class TestRowAlignedParts(_NumpySftTestBase):
    """Parts of a cache written with `row_aligned_parts` hold whole rows only."""

    def _convert(self, name, row_aligned_parts):
        dataset = dataset_transformation.Dataset.from_dict(
            {
                dataset_transformation.INPUT_IDS_KEY: [tokens for tokens, _ in _CHAT_ROWS],
                dataset_transformation.ATTENTION_MASK_KEY: [[1] * len(tokens) for tokens, _ in _CHAT_ROWS],
                dataset_transformation.LABELS_KEY: [labels for _, labels in _CHAT_ROWS],
                dataset_transformation.DATASET_ORIGIN_KEY: ["chat"] * len(_CHAT_ROWS),
            }
        )
        output_dir = pathlib.Path(self.temp_dir.name) / name
        real_write = numpy_dataset_conversion._write_memmap_chunked_from_file

        def write_small_parts(base_filename, source_path, total_items, dtype, **kwargs):
            max_size_gb = _PART_TOKENS * np.dtype(dtype).itemsize / 1024**3
            return real_write(base_filename, source_path, total_items, dtype, max_size_gb=max_size_gb, **kwargs)

        with (
            unittest.mock.patch.object(
                dataset_transformation, "get_cached_dataset_tulu_with_statistics", return_value=(dataset, {})
            ),
            unittest.mock.patch.object(
                numpy_dataset_conversion, "_write_memmap_chunked_from_file", side_effect=write_small_parts
            ),
        ):
            numpy_dataset_conversion.convert_hf_to_numpy_sft(
                output_dir=output_dir,
                dataset_mixer_list=[],
                dataset_mixer_list_splits=[],
                tc=self._make_tc(),
                dataset_transform_fn=[],
                transform_fn_args=[],
                dataset_target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
                dataset_config_hash="rows",
                shuffle_seed=0,
                row_aligned_parts=row_aligned_parts,
            )
        return output_dir

    def _shuffled_rows(self):
        order = dataset_transformation.Dataset.from_dict({"i": list(range(len(_CHAT_ROWS)))}).shuffle(seed=0)["i"]
        return [_CHAT_ROWS[i] for i in order]

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

    @_requires_metadata_boundaries
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

    @_requires_metadata_boundaries
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

    @_requires_metadata_boundaries
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


class TestResumeEquivalence(_NumpySftTestBase):
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


if __name__ == "__main__":
    unittest.main()
