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
