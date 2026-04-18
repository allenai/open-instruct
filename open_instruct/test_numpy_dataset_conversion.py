"""Tests for open_instruct.numpy_dataset_conversion.

Run from project root:
    uv run pytest open_instruct/test_numpy_dataset_conversion.py -v
"""

import gc
import gzip
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from parameterized import parameterized

from open_instruct import dataset_transformation, numpy_dataset_conversion

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
        self.base = os.path.join(self.tmp_dir.name, "token_ids")

    def _read_chunk(self, chunk_idx, dtype, length):
        filename = f"{self.base}_part_{chunk_idx:04d}.npy"
        return list(np.memmap(filename, mode="r", dtype=dtype, shape=(length,)))

    def test_empty_data(self):
        result = numpy_dataset_conversion._write_memmap_chunked(self.base, [], np.uint16, max_size_bytes=16)
        self.assertEqual(result, [])
        self.assertFalse(os.path.exists(f"{self.base}_part_0000.npy"))

    def test_single_chunk_exact(self):
        data = list(range(8))
        result = numpy_dataset_conversion._write_memmap_chunked(self.base, data, np.uint16, max_size_bytes=16)
        self.assertEqual(result, [(0, 8)])
        self.assertEqual(self._read_chunk(0, np.uint16, 8), data)
        self.assertFalse(os.path.exists(f"{self.base}_part_0001.npy"))

    def test_single_chunk_partial(self):
        data = [10, 20, 30, 40, 50]
        result = numpy_dataset_conversion._write_memmap_chunked(self.base, data, np.uint16, max_size_bytes=16)
        self.assertEqual(result, [(0, 5)])
        self.assertEqual(self._read_chunk(0, np.uint16, 5), data)

    def test_three_chunks(self):
        data = list(range(17))
        result = numpy_dataset_conversion._write_memmap_chunked(self.base, data, np.uint16, max_size_bytes=16)
        self.assertEqual(result, [(0, 8), (8, 16), (16, 17)])
        self.assertEqual(self._read_chunk(0, np.uint16, 8), data[0:8])
        self.assertEqual(self._read_chunk(1, np.uint16, 8), data[8:16])
        self.assertEqual(self._read_chunk(2, np.uint16, 1), data[16:17])


class TestWriteMetadataForChunks(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.base = os.path.join(self.tmp_dir.name, "token_ids")

    def _read_chunk_rows(self, chunk_idx):
        path = f"{self.base}_part_{chunk_idx:04d}.csv.gz"
        with gzip.open(path, "rt") as f:
            return [line.strip() for line in f if line.strip()]

    def test_empty_document_boundaries(self):
        numpy_dataset_conversion._write_metadata_for_chunks(self.base, [], [(0, 8), (8, 16)])
        self.assertEqual(self._read_chunk_rows(0), [])
        self.assertEqual(self._read_chunk_rows(1), [])

    def test_docs_entirely_within_single_chunk(self):
        doc_boundaries = [(0, 3), (3, 8)]
        chunk_boundaries = [(0, 8)]
        numpy_dataset_conversion._write_metadata_for_chunks(self.base, doc_boundaries, chunk_boundaries)
        self.assertEqual(self._read_chunk_rows(0), ["0,3", "3,8"])

    def test_doc_spans_two_chunks(self):
        doc_boundaries = [(5, 12)]
        chunk_boundaries = [(0, 8), (8, 16)]
        numpy_dataset_conversion._write_metadata_for_chunks(self.base, doc_boundaries, chunk_boundaries)
        self.assertEqual(self._read_chunk_rows(0), ["5,8"])
        self.assertEqual(self._read_chunk_rows(1), ["0,4"])

    def test_doc_touching_boundary_is_excluded(self):
        doc_boundaries = [(0, 8)]
        chunk_boundaries = [(0, 8), (8, 16)]
        numpy_dataset_conversion._write_metadata_for_chunks(self.base, doc_boundaries, chunk_boundaries)
        self.assertEqual(self._read_chunk_rows(0), ["0,8"])
        self.assertEqual(self._read_chunk_rows(1), [])

    def test_doc_aligned_to_chunk_start(self):
        doc_boundaries = [(8, 12)]
        chunk_boundaries = [(0, 8), (8, 16)]
        numpy_dataset_conversion._write_metadata_for_chunks(self.base, doc_boundaries, chunk_boundaries)
        self.assertEqual(self._read_chunk_rows(0), [])
        self.assertEqual(self._read_chunk_rows(1), ["0,4"])


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
        numpy_dataset_conversion._save_tokenizer(self.tc, self.tmp_dir.name)
        tokenizer_dir = os.path.join(self.tmp_dir.name, "tokenizer")
        self.assertTrue(os.path.isdir(tokenizer_dir))
        self.assertTrue(os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json")))

    def test_creates_output_dir_if_missing(self):
        nested = os.path.join(self.tmp_dir.name, "does", "not", "exist")
        numpy_dataset_conversion._save_tokenizer(self.tc, nested)
        self.assertTrue(os.path.isdir(os.path.join(nested, "tokenizer")))


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
            output_dir=self.tmp_dir.name,
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
            output_dir=self.tmp_dir.name,
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

    def test_missing_pre_transform_stats_falls_back(self):
        numpy_dataset_conversion.write_dataset_statistics(
            output_dir=self.tmp_dir.name,
            dataset_statistics={"per_dataset_stats": []},
            total_instances=5,
            total_tokens=100,
            total_trainable_tokens=50,
            num_samples_skipped=0,
            tokenizer_name="test-tokenizer",
            max_seq_length=2048,
            chat_template_name="tulu",
            per_dataset_counts={"ds_x": 5},
            per_dataset_tokens={"ds_x": 100},
            per_dataset_trainable_tokens={"ds_x": 50},
            per_dataset_filtered={"ds_x": 0},
        )
        with open(os.path.join(self.tmp_dir.name, "dataset_statistics.json")) as f:
            loaded = json.load(f)
        stat = loaded["per_dataset_statistics"][0]
        self.assertEqual(stat["dataset_split"], "unknown")
        self.assertEqual(stat["initial_instances"], "N/A")
        self.assertEqual(stat["instances_after_transformation"], "N/A")


class TestConvertHfToNumpySft(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.original_hf_home = os.environ.get("HF_HOME")
        self.original_hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        self.original_transformers_cache = os.environ.get("TRANSFORMERS_CACHE")

        os.environ["HF_HOME"] = self.temp_dir.name
        os.environ["HF_DATASETS_CACHE"] = os.path.join(self.temp_dir.name, "datasets")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.temp_dir.name, "transformers")

    def tearDown(self):
        for key, original in [
            ("HF_HOME", self.original_hf_home),
            ("HF_DATASETS_CACHE", self.original_hf_datasets_cache),
            ("TRANSFORMERS_CACHE", self.original_transformers_cache),
        ]:
            if original is not None:
                os.environ[key] = original
            else:
                os.environ.pop(key, None)
        gc.collect()

    def _make_tc(self):
        return dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=True,
            chat_template_name="tulu",
            add_bos=False,
        )

    def test_tokenizer_config_only(self):
        output_dir = os.path.join(self.temp_dir.name, "out")
        numpy_dataset_conversion.convert_hf_to_numpy_sft(
            output_dir=output_dir,
            dataset_mixer_list=[os.path.join(TEST_DATA_DIR, "sft_sample.jsonl"), "1.0"],
            dataset_mixer_list_splits=["train"],
            tc=self._make_tc(),
            dataset_transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
            transform_fn_args=[{"max_seq_length": 4096}, {}],
            dataset_target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
            tokenizer_config_only=True,
        )
        self.assertTrue(os.path.isdir(os.path.join(output_dir, "tokenizer")))
        self.assertFalse(os.path.exists(os.path.join(output_dir, "token_ids_part_0000.npy")))
        self.assertFalse(os.path.exists(os.path.join(output_dir, "dataset_statistics.json")))

    def test_end_to_end_small(self):
        output_dir = os.path.join(self.temp_dir.name, "out_e2e")
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
        token_file = os.path.join(output_dir, "token_ids_part_0000.npy")
        labels_file = os.path.join(output_dir, "labels_mask_part_0000.npy")
        metadata_file = os.path.join(output_dir, "token_ids_part_0000.csv.gz")
        stats_json = os.path.join(output_dir, "dataset_statistics.json")
        checkpoint_file = os.path.join(output_dir, "_checkpoint.json")

        self.assertTrue(os.path.exists(token_file))
        self.assertTrue(os.path.exists(labels_file))
        self.assertTrue(os.path.exists(metadata_file))
        self.assertTrue(os.path.exists(stats_json))
        self.assertTrue(os.path.isdir(os.path.join(output_dir, "tokenizer")))
        self.assertFalse(os.path.exists(checkpoint_file))

        token_size = os.path.getsize(token_file)
        labels_size = os.path.getsize(labels_file)
        self.assertGreater(token_size, 0)
        self.assertGreater(labels_size, 0)

        with open(stats_json) as f:
            stats = json.load(f)
        self.assertEqual(stats["overall_statistics"]["total_instances"], 2)
        self.assertGreater(stats["overall_statistics"]["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
