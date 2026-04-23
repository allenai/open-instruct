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


class TestIncrementalCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.output_dir = pathlib.Path(self.tmp_dir.name)

    def _scalar_state(self):
        return {
            "current_position": 0,
            "num_samples_skipped": 0,
            "per_dataset_counts": {},
            "per_dataset_tokens": {},
            "per_dataset_trainable_tokens": {},
            "per_dataset_filtered": {},
        }

    def test_incremental_save_appends_only_new_data(self):
        token_ids = [1, 2, 3]
        labels_mask = [1, 0, 1]
        document_boundaries = [(0, 3)]
        tw1, sw1, _ = numpy_dataset_conversion.save_checkpoint(
            self.output_dir,
            samples_processed=1,
            token_ids=token_ids,
            labels_mask=labels_mask,
            document_boundaries=document_boundaries,
            scalar_state=self._scalar_state(),
            prev_tokens_written=0,
            prev_samples_written=0,
        )

        tokens_path = self.output_dir / "_checkpoint_token_ids.bin"
        size_after_first = tokens_path.stat().st_size

        token_ids.extend([4, 5, 6, 7])
        labels_mask.extend([0, 0, 1, 1])
        document_boundaries.append((3, 7))

        tw2, sw2, _ = numpy_dataset_conversion.save_checkpoint(
            self.output_dir,
            samples_processed=2,
            token_ids=token_ids,
            labels_mask=labels_mask,
            document_boundaries=document_boundaries,
            scalar_state=self._scalar_state(),
            prev_tokens_written=tw1,
            prev_samples_written=sw1,
        )
        size_after_second = tokens_path.stat().st_size

        self.assertEqual(tw2, 7)
        self.assertEqual(sw2, 2)
        self.assertEqual(size_after_second - size_after_first, 4 * np.dtype(np.uint32).itemsize)

        loaded = numpy_dataset_conversion.load_checkpoint(self.output_dir)
        self.assertEqual(loaded["token_ids"], token_ids)
        self.assertEqual(loaded["labels_mask"], labels_mask)
        self.assertEqual(loaded["document_boundaries"], document_boundaries)

    def test_load_raises_on_truncated_binary(self):
        token_ids = [1, 2, 3, 4, 5]
        labels_mask = [1, 0, 1, 0, 1]
        document_boundaries = [(0, 5)]
        numpy_dataset_conversion.save_checkpoint(
            self.output_dir,
            samples_processed=1,
            token_ids=token_ids,
            labels_mask=labels_mask,
            document_boundaries=document_boundaries,
            scalar_state=self._scalar_state(),
            prev_tokens_written=0,
            prev_samples_written=0,
        )

        tokens_path = self.output_dir / "_checkpoint_token_ids.bin"
        with open(tokens_path, "r+b") as f:
            f.truncate(np.dtype(np.uint32).itemsize)

        with self.assertRaises(RuntimeError):
            numpy_dataset_conversion.load_checkpoint(self.output_dir)

    def test_load_raises_on_missing_binary(self):
        token_ids = [1, 2, 3]
        labels_mask = [1, 0, 1]
        document_boundaries = [(0, 3)]
        numpy_dataset_conversion.save_checkpoint(
            self.output_dir,
            samples_processed=1,
            token_ids=token_ids,
            labels_mask=labels_mask,
            document_boundaries=document_boundaries,
            scalar_state=self._scalar_state(),
            prev_tokens_written=0,
            prev_samples_written=0,
        )

        (self.output_dir / "_checkpoint_token_ids.bin").unlink()

        with self.assertRaises(RuntimeError):
            numpy_dataset_conversion.load_checkpoint(self.output_dir)


class TestWriteMemmapChunked(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.base = os.path.join(self.tmp_dir.name, "token_ids")

    def _read_chunk(self, chunk_idx, dtype, length):
        filename = f"{self.base}_part_{chunk_idx:04d}.npy"
        return list(np.memmap(filename, mode="r", dtype=dtype, shape=(length,)))

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

    def test_doc_spans_two_chunks(self):
        doc_boundaries = [(5, 12)]
        chunk_boundaries = [(0, 8), (8, 16)]
        numpy_dataset_conversion._write_metadata_for_chunks(self.base, doc_boundaries, chunk_boundaries)
        self.assertEqual(self._read_chunk_rows(0), ["5,8"])
        self.assertEqual(self._read_chunk_rows(1), ["0,4"])


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


class TestResumeEquivalence(unittest.TestCase):
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

    def _run(self, output_dir, checkpoint_interval, resume):
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
            checkpoint_interval=checkpoint_interval,
            resume=resume,
        )

    def test_one_shot_matches_interrupt_plus_resume(self):
        golden = os.path.join(self.temp_dir.name, "golden")
        self._run(golden, checkpoint_interval=1000, resume=False)

        interrupted = os.path.join(self.temp_dir.name, "interrupted")
        real_save = numpy_dataset_conversion.save_checkpoint
        call_count = {"n": 0}

        def fail_after_first(*args, **kwargs):
            result = real_save(*args, **kwargs)
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated interrupt")
            return result

        with (
            unittest.mock.patch.object(numpy_dataset_conversion, "save_checkpoint", side_effect=fail_after_first),
            self.assertRaises(RuntimeError),
        ):
            self._run(interrupted, checkpoint_interval=10, resume=False)

        self._run(interrupted, checkpoint_interval=10, resume=True)

        artifacts = sorted(
            name
            for name in os.listdir(golden)
            if name.startswith("token_ids_part_") or name.startswith("labels_mask_part_")
        )
        self.assertGreater(len(artifacts), 0, "golden run produced no artifacts")
        for name in artifacts:
            golden_path = os.path.join(golden, name)
            interrupted_path = os.path.join(interrupted, name)
            if name.endswith(".gz"):
                with gzip.open(golden_path, "rb") as f1, gzip.open(interrupted_path, "rb") as f2:
                    self.assertEqual(f1.read(), f2.read(), msg=f"mismatch in {name}")
            else:
                with open(golden_path, "rb") as f1, open(interrupted_path, "rb") as f2:
                    self.assertEqual(f1.read(), f2.read(), msg=f"mismatch in {name}")


if __name__ == "__main__":
    unittest.main()
