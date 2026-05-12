import csv
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import parameterized

from open_instruct import benchmark_generators, utils


def _result():
    return {
        "mfu": 0.1,
        "mbu": 0.2,
        "tokens_per_second": 50.0,
        "generation_time": 0.5,
        "weight_sync_time": 0.25,
        "num_new_tokens": 100,
        "finish_reasons": {"stop": 1},
        "response_lengths": [10],
        "prompt_lengths": [5],
    }


class TestBenchmark(unittest.TestCase):
    @parameterized.parameterized.expand(
        [("NVIDIA H100 80GB HBM3", "h100"), ("NVIDIA L40S", "l40s"), ("NVIDIA RTX A6000", "a6000")]
    )
    def test_get_device_name(self, device_name, expected):
        self.assertEqual(utils.get_device_name(device_name), expected)


class TestSaveCompletionLengths(unittest.TestCase):
    """Regression test for PR #1623: duplicate headers on every batch."""

    def test_header_written_once_across_batches(self):
        batches = [
            [{"response_lengths": [10, 20]}],
            [{"response_lengths": [30]}],
            [{"response_lengths": [40, 50, 60]}],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            with mock.patch.object(benchmark_generators, "DATA_DIR", tmp_path):
                for i, batch in enumerate(batches):
                    benchmark_generators.save_completion_lengths(batch, 1234567890, i)
            rows = list(csv.reader((tmp_path / "completion_lengths_1234567890.csv").open()))
        self.assertEqual(rows[0], ["batch_num", "prompt_num", "completion_length"])
        self.assertEqual(len(rows), 1 + 6)


class TestSaveBenchmarkResultsToCsv(unittest.TestCase):
    """Regression test for PR #1619: header never written because the
    Path('a') open call created the file before the existence check."""

    def test_header_written_exactly_once_across_calls(self):
        streaming = SimpleNamespace(
            num_unique_prompts_rollout=4, num_samples_per_prompt_rollout=2, response_length=128
        )
        model = SimpleNamespace(model_name_or_path="fake/model")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            with mock.patch.object(benchmark_generators, "DATA_DIR", tmp_path):
                for _ in range(3):
                    benchmark_generators.save_benchmark_results_to_csv(
                        results=[_result()], total_time=10.0, streaming_config=streaming, model_config=model
                    )
            rows = list(csv.reader((tmp_path / "generator_benchmark_results.csv").open()))
        self.assertEqual(rows[0][0], "git_commit")
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
