import csv
import pathlib
import tempfile
import unittest
from unittest import mock

import parameterized

from open_instruct import benchmark_generators, utils


class TestBenchmark(unittest.TestCase):
    @parameterized.parameterized.expand(
        [("NVIDIA H100 80GB HBM3", "h100"), ("NVIDIA L40S", "l40s"), ("NVIDIA RTX A6000", "a6000")]
    )
    def test_get_device_name(self, device_name, expected):
        result = utils.get_device_name(device_name)
        self.assertEqual(result, expected)


class TestSaveCompletionLengths(unittest.TestCase):
    """Regression test for PR #1623: duplicate headers on every batch."""

    def test_header_written_once_across_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            timestamp = 1234567890
            batches = [
                [{"response_lengths": [10, 20]}],
                [{"response_lengths": [30]}],
                [{"response_lengths": [40, 50, 60]}],
            ]
            with mock.patch.object(benchmark_generators, "DATA_DIR", tmp_path):
                for batch_idx, batch_results in enumerate(batches):
                    benchmark_generators.save_completion_lengths(batch_results, timestamp, batch_idx)

            csv_path = tmp_path / f"completion_lengths_{timestamp}.csv"
            with csv_path.open() as f:
                rows = list(csv.reader(f))

            self.assertEqual(rows[0], ["batch_num", "prompt_num", "completion_length"])
            for row in rows[1:]:
                self.assertNotEqual(row, ["batch_num", "prompt_num", "completion_length"])
            self.assertEqual(len(rows), 1 + 2 + 1 + 3)


class TestSaveBenchmarkResultsToCsv(unittest.TestCase):
    """Regression test for PR #1619: header never written because the
    Path('a') open call created the file before the existence check."""

    def _streaming_config(self):
        return mock.MagicMock(num_unique_prompts_rollout=4, num_samples_per_prompt_rollout=2, response_length=128)

    def _model_config(self):
        return mock.MagicMock(model_name_or_path="fake/model")

    def _agg(self):
        return {
            "total_generation_time": 1.0,
            "total_weight_sync_time": 0.5,
            "total_num_new_tokens": 100,
            "avg_tokens_per_second": 50.0,
            "avg_mfu": 0.1,
            "avg_mbu": 0.2,
            "avg_generation_time": 0.5,
            "avg_weight_sync_time": 0.25,
        }

    def test_header_written_exactly_once_across_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            with (
                mock.patch.object(benchmark_generators, "DATA_DIR", tmp_path),
                mock.patch.object(benchmark_generators, "aggregate_results", return_value=self._agg()),
                mock.patch.object(benchmark_generators.utils, "get_git_commit", return_value="abc"),
            ):
                for _ in range(3):
                    benchmark_generators.save_benchmark_results_to_csv(
                        results=[{}],
                        total_time=10.0,
                        streaming_config=self._streaming_config(),
                        model_config=self._model_config(),
                    )

            csv_path = tmp_path / "generator_benchmark_results.csv"
            with csv_path.open() as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0][0], "git_commit")
            self.assertEqual(len(rows), 4)
            for row in rows[1:]:
                self.assertEqual(row[0], "abc")


if __name__ == "__main__":
    unittest.main()
