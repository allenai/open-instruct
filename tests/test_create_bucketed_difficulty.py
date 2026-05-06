"""Unit tests for posterior-aware bucketing in open_instruct.rlvr_difficulty."""

import importlib
import json
import math
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from unittest.mock import patch

import numpy as np


def _load_create_bucketed_difficulty_module():
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = type("Dataset", (), {})
    fake_datasets.load_dataset = lambda *_args, **_kwargs: None

    fake_scipy = types.ModuleType("scipy")
    fake_scipy_optimize = types.ModuleType("scipy.optimize")
    fake_scipy_optimize.minimize = lambda *_args, **_kwargs: types.SimpleNamespace(
        success=False, message="not implemented in test stub", x=(0.0, 0.0)
    )
    fake_scipy_special = types.ModuleType("scipy.special")
    fake_scipy_special.betaln = lambda alpha, beta: math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    fake_scipy_stats = types.ModuleType("scipy.stats")

    class _ApproximateBetaDistribution:
        @staticmethod
        def _mean(alpha, beta):
            return alpha / (alpha + beta)

        @staticmethod
        def _sigma(alpha, beta):
            total = alpha + beta
            variance = np.maximum(alpha * beta / (total * total * (total + 1.0)), 1e-6)
            return np.sqrt(variance)

        @classmethod
        def cdf(cls, x, alpha, beta):
            x_array, alpha_array, beta_array = np.broadcast_arrays(
                np.asarray(x, dtype=float), np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
            )
            mean = cls._mean(alpha_array, beta_array)
            sigma = cls._sigma(alpha_array, beta_array)
            z_scores = (x_array - mean) / (sigma * math.sqrt(2.0))
            return 0.5 * (1.0 + np.vectorize(math.erf)(z_scores))

        @classmethod
        def pdf(cls, x, alpha, beta):
            x_array, alpha_array, beta_array = np.broadcast_arrays(
                np.asarray(x, dtype=float), np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
            )
            mean = cls._mean(alpha_array, beta_array)
            sigma = cls._sigma(alpha_array, beta_array)
            z_scores = (x_array - mean) / sigma
            normalizer = sigma * math.sqrt(2.0 * math.pi)
            return np.exp(-0.5 * z_scores * z_scores) / normalizer

        @classmethod
        def ppf(cls, q, alpha, beta):
            q_array, alpha_array, beta_array = np.broadcast_arrays(
                np.asarray(q, dtype=float), np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
            )
            quantiles = np.empty_like(q_array, dtype=float)
            for index in np.ndindex(q_array.shape):
                mean = float(cls._mean(alpha_array[index], beta_array[index]))
                sigma = float(cls._sigma(alpha_array[index], beta_array[index]))
                quantiles[index] = np.clip(NormalDist(mu=mean, sigma=sigma).inv_cdf(float(q_array[index])), 0.0, 1.0)
            return quantiles

    fake_scipy_stats.beta = _ApproximateBetaDistribution

    modules = {
        "datasets": fake_datasets,
        "scipy": fake_scipy,
        "scipy.optimize": fake_scipy_optimize,
        "scipy.special": fake_scipy_special,
        "scipy.stats": fake_scipy_stats,
    }

    with patch.dict(sys.modules, modules):
        sys.modules.pop("open_instruct.rlvr_difficulty", None)
        return importlib.import_module("open_instruct.rlvr_difficulty")


MODULE = _load_create_bucketed_difficulty_module()


class TestCreateBucketedDifficulty(unittest.TestCase):
    class FakeHFDataset:
        def __init__(self, rows):
            self._rows = [dict(row) for row in rows]

        def __getitem__(self, index):
            return self._rows[index]

        def __iter__(self):
            return iter(self._rows)

        def __len__(self):
            return len(self._rows)

        @property
        def column_names(self):
            return list(self._rows[0].keys()) if self._rows else []

        def select(self, indices):
            return TestCreateBucketedDifficulty.FakeHFDataset([self._rows[index] for index in indices])

        def remove_columns(self, column_names):
            names = {column_names} if isinstance(column_names, str) else set(column_names)
            return TestCreateBucketedDifficulty.FakeHFDataset(
                [{key: value for key, value in row.items() if key not in names} for row in self._rows]
            )

        def add_column(self, name, values):
            return TestCreateBucketedDifficulty.FakeHFDataset(
                [{**row, name: value} for row, value in zip(self._rows, values, strict=True)]
            )

    def test_discover_rollout_sources_resolves_directory_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "demo_run_metadata.jsonl").write_text(
                json.dumps({"run_name": "demo_run", "model_name": "demo-model"}) + "\n"
            )
            (root / "demo_run_rollouts_000000.jsonl").write_text(
                json.dumps(
                    {
                        "prompt_tokens": [1, 2, 3],
                        "reward": 1.0,
                        "finish_reason": "stop",
                        "dataset": "math",
                        "ground_truth": "4",
                    }
                )
                + "\n"
            )

            sources = MODULE.discover_rollout_sources([str(root)])

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].run_name, "demo_run")
        self.assertEqual(sources[0].metadata_path.name, "demo_run_metadata.jsonl")
        self.assertEqual([path.name for path in sources[0].rollout_paths], ["demo_run_rollouts_000000.jsonl"])

    def test_rollout_contributions_aggregate_and_normalize_constant_rewards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "demo_run_metadata.jsonl").write_text(
                json.dumps({"run_name": "demo_run", "model_name": "Qwen/Qwen3-4B-Base"}) + "\n"
            )
            shard = root / "demo_run_rollouts_000000.jsonl"
            shard.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "prompt_tokens": [11, 12, 13],
                                "reward": 10.0,
                                "finish_reason": "stop",
                                "dataset": "math",
                                "ground_truth": {"answer": "4"},
                                "request_info": {"timeouts": 0, "tool_errors": ""},
                            }
                        ),
                        json.dumps(
                            {
                                "prompt_tokens": [11, 12, 13],
                                "reward": 0.0,
                                "finish_reason": "length",
                                "dataset": "math",
                                "ground_truth": {"answer": "4"},
                                "request_info": {"timeouts": 1, "tool_errors": ""},
                            }
                        ),
                        json.dumps(
                            {
                                "prompt_tokens": [21, 22, 23],
                                "reward": 10.0,
                                "finish_reason": "stop",
                                "dataset": "math",
                                "ground_truth": {"answer": "9"},
                                "request_info": {"timeouts": 0, "tool_errors": ""},
                            }
                        ),
                    ]
                )
                + "\n"
            )

            source = MODULE.discover_rollout_sources([str(root)])[0]
            contributions, malformed_records = MODULE.build_contributions_for_source(
                source_run=source, task_filters=set(), strict=True
            )

        self.assertEqual(malformed_records, 0)

        rows = MODULE.aggregate_contributions(contributions)
        self.assertEqual(len(rows), 2)

        rows_by_group = MODULE.group_rows_by_task_and_model(rows)
        group_rows, score_processing, skipped_nonunit = MODULE.normalize_attempt_scores_for_group(
            rows_by_group[("math", "Qwen/Qwen3-4B-Base")], allow_nonunit_scores=False
        )

        self.assertEqual(skipped_nonunit, 0)
        self.assertEqual(score_processing["normalization"], "binary_zero_or_constant")
        self.assertEqual(score_processing["positive_reward_value"], 10.0)

        easy_row = next(row for row in group_rows if row["ground_truth"] == {"answer": "4"})
        self.assertEqual(easy_row["attempt_scores"], [1.0, 0.0])
        self.assertEqual(easy_row["prompt_tokens"], [11, 12, 13])
        self.assertEqual(easy_row["finish_reasons"], ["stop", "length"])
        self.assertEqual(easy_row["score_sources"], ["math"])
        self.assertEqual(easy_row["experiment_metadata"]["model_name"], "Qwen/Qwen3-4B-Base")
        self.assertIn("timeout", easy_row["warnings"])

    def test_normalize_attempt_scores_for_group_marks_unsupported_rewards(self):
        rows = [
            {
                "instance_id": "example",
                "task_name": "math",
                "base_task_name": "math",
                "prompt_tokens": [1, 2, 3],
                "ground_truth": "4",
                "attempt_scores": [10.0, 5.0],
                "finish_reasons": ["stop", "stop"],
                "experiment_metadata": {
                    "source_root": "/tmp/example-rollouts",
                    "model_name": "demo-model",
                    "experiment_id": None,
                    "experiment_name": "demo-run",
                },
                "score_sources": ["math"],
                "warnings": [],
            }
        ]

        kept_rows, score_processing, skipped_nonunit = MODULE.normalize_attempt_scores_for_group(
            rows, allow_nonunit_scores=True
        )

        self.assertEqual(skipped_nonunit, 0)
        self.assertFalse(score_processing["supports_binary_difficulty"])
        self.assertEqual(kept_rows[0]["attempt_scores"], [10.0, 5.0])
        self.assertIn("nonbinary_reward_scores", kept_rows[0]["warnings"])

        dropped_rows, _, dropped_count = MODULE.normalize_attempt_scores_for_group(rows, allow_nonunit_scores=False)

        self.assertEqual(dropped_rows, [])
        self.assertEqual(dropped_count, 1)

    def test_build_dataset_metadata_captures_difficulty_generation_details(self):
        rows = [
            {
                "instance_id": "easy",
                "difficulty": {
                    "value": 0.1,
                    "posterior_mean": 0.2,
                    "posterior_lower_bound": 0.9,
                    "expected_quantile": 0.2,
                    "bucket_index": 0,
                    "bucket_count": 3,
                },
            },
            {
                "instance_id": "hard",
                "difficulty": {
                    "value": 0.8,
                    "posterior_mean": 0.7,
                    "posterior_lower_bound": 0.2,
                    "expected_quantile": 0.9,
                    "bucket_index": 2,
                    "bucket_count": 3,
                },
            },
            {"instance_id": "nonbinary", "difficulty": MODULE.make_empty_difficulty_payload()},
        ]

        metadata = MODULE.build_dataset_metadata(
            rows=rows,
            task_name="math",
            model_name="demo-model",
            requested_prior_mode="empirical-bayes",
            requested_bucket_count=5,
            lower_quantile=0.1,
            prior=MODULE.BetaPrior(alpha=0.75, beta=1.25, source="empirical_bayes"),
            binary_row_count=2,
            score_processing={
                "source_field": "reward",
                "output_field": "attempt_scores",
                "normalization": "binary_zero_or_constant",
                "positive_reward_value": 10.0,
                "supports_binary_difficulty": True,
            },
            source_format=MODULE.build_rollout_source_format_metadata(),
        )

        self.assertEqual(metadata["task_name"], "math")
        self.assertEqual(metadata["model_name"], "demo-model")
        self.assertEqual(metadata["row_count"], 3)
        self.assertEqual(metadata["source_format"]["kind"], "open_instruct_rollout_traces")
        self.assertEqual(metadata["score_processing"]["normalization"], "binary_zero_or_constant")
        self.assertEqual(metadata["score_processing"]["positive_reward_value"], 10.0)
        self.assertEqual(metadata["difficulty_generation"]["method"], "beta_binomial_posterior_quantiles")
        self.assertEqual(metadata["difficulty_generation"]["posterior_lower_quantile"], 0.1)
        self.assertEqual(metadata["difficulty_generation"]["bucket_count_requested"], 5)
        self.assertEqual(metadata["difficulty_generation"]["bucket_count_effective"], 3)
        self.assertEqual(metadata["difficulty_generation"]["beta_prior_used"]["source"], "empirical_bayes")
        self.assertEqual(metadata["difficulty_generation"]["beta_prior_used"]["alpha"], 0.75)
        self.assertEqual(metadata["difficulty_generation"]["beta_prior_used"]["beta"], 1.25)
        self.assertEqual(metadata["difficulty_generation"]["binary_instance_count"], 2)
        self.assertEqual(metadata["difficulty_generation"]["nonbinary_instance_count"], 1)

    def test_build_hf_dataset_row_parses_pass_rate_counts(self):
        row = MODULE.build_hf_dataset_row(
            source_row={
                "dataset": "math",
                "extra_info": {"index": "row-7"},
                "pass_count": 3,
                "num_samples": 5,
                "pass_rate": "3/5",
                "generator_model": "Qwen/Qwen3-4B-Base",
            },
            source_row_index=7,
            dataset_name="mnoukhov/demo",
            config_name=None,
            split="train",
            row_id_field="extra_info.index",
            task_field="dataset",
            model_field="generator_model",
            pass_count_field="pass_count",
            attempt_count_field="num_samples",
            pass_rate_field="pass_rate",
        )

        self.assertEqual(row["instance_id"], "mnoukhov/demo::row-7")
        self.assertEqual(row["source_row_id"], "row-7")
        self.assertEqual(row["attempt_scores"], [1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertEqual(row["experiment_metadata"]["model_name"], "Qwen/Qwen3-4B-Base")
        self.assertEqual(row["experiment_metadata"]["source_root"], "hf://mnoukhov/demo/default/train")

    def test_load_hf_dataset_rows_builds_bundle_and_filters_tasks(self):
        fake_dataset = self.FakeHFDataset(
            [
                {
                    "dataset": "math",
                    "extra_info": {"index": "math-1"},
                    "pass_count": 2,
                    "num_samples": 4,
                    "pass_rate": "2/4",
                    "generator_model": "Qwen/Qwen3-4B-Base",
                },
                {
                    "dataset": "gsm8k",
                    "extra_info": {"index": "gsm-1"},
                    "pass_count": 1,
                    "num_samples": 4,
                    "pass_rate": "1/4",
                    "generator_model": "Qwen/Qwen3-4B-Base",
                },
            ]
        )

        with patch.object(MODULE, "load_dataset", return_value=fake_dataset):
            bundle = MODULE.load_hf_dataset_rows(
                dataset_name="mnoukhov/demo",
                config_name=None,
                split="train",
                task_filters={"math"},
                strict=True,
                row_id_field="extra_info.index",
                task_field="dataset",
                model_field="generator_model",
                pass_count_field="pass_count",
                attempt_count_field="num_samples",
                pass_rate_field="pass_rate",
            )

        self.assertEqual(bundle.malformed_records, 0)
        self.assertEqual(bundle.source_format["kind"], MODULE.HF_SOURCE_FORMAT_KIND)
        self.assertEqual(bundle.source_format["dataset_repo_id"], "mnoukhov/demo")
        self.assertEqual(len(bundle.rows), 1)
        self.assertEqual(bundle.rows[0]["instance_id"], "mnoukhov/demo::math-1")
        self.assertEqual(bundle.rows[0]["attempt_scores"], [1.0, 1.0, 0.0, 0.0])

    def test_build_hf_output_dataset_preserves_source_rows_and_order(self):
        source_dataset = self.FakeHFDataset(
            [
                {"prompt": "first", "extra_info": {"index": "row-0"}},
                {"prompt": "second", "extra_info": {"index": "row-1"}},
            ]
        )
        rows = [
            {
                MODULE.HF_SOURCE_ROW_INDEX_FIELD: 1,
                "instance_id": "mnoukhov/demo::row-1",
                "task_name": "math",
                "base_task_name": "math",
                "source_dataset": "mnoukhov/demo",
                "source_row_id": "row-1",
                "attempt_scores": [0.0, 0.0],
                "finish_reasons": [],
                "experiment_metadata": {
                    "source_root": "hf://mnoukhov/demo/default/train",
                    "model_name": "Qwen/Qwen3-4B-Base",
                    "experiment_id": None,
                    "experiment_name": "mnoukhov/demo",
                },
                "score_sources": ["math"],
                "warnings": [],
                "difficulty": {
                    "value": 0.9,
                    "posterior_mean": 0.1,
                    "posterior_lower_bound": 0.1,
                    "expected_quantile": 0.9,
                    "bucket_index": 1,
                    "bucket_count": 2,
                },
            },
            {
                MODULE.HF_SOURCE_ROW_INDEX_FIELD: 0,
                "instance_id": "mnoukhov/demo::row-0",
                "task_name": "math",
                "base_task_name": "math",
                "source_dataset": "mnoukhov/demo",
                "source_row_id": "row-0",
                "attempt_scores": [1.0, 1.0],
                "finish_reasons": [],
                "experiment_metadata": {
                    "source_root": "hf://mnoukhov/demo/default/train",
                    "model_name": "Qwen/Qwen3-4B-Base",
                    "experiment_id": None,
                    "experiment_name": "mnoukhov/demo",
                },
                "score_sources": ["math"],
                "warnings": [],
                "difficulty": {
                    "value": 0.1,
                    "posterior_mean": 0.9,
                    "posterior_lower_bound": 0.9,
                    "expected_quantile": 0.1,
                    "bucket_index": 0,
                    "bucket_count": 2,
                },
            },
        ]

        dataset = MODULE.build_hf_output_dataset(source_dataset, rows)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.column_names, ["prompt", "extra_info", "difficulty"])
        self.assertEqual(dataset[0]["prompt"], "first")
        self.assertEqual(dataset[0]["difficulty"]["bucket_index"], 0)
        self.assertEqual(dataset[1]["prompt"], "second")
        self.assertEqual(dataset[1]["difficulty"]["bucket_index"], 1)

    def test_annotate_dataset_metadata_stores_json_description(self):
        class FakeInfo:
            description = ""

        class FakeDataset:
            def __init__(self):
                self.info = FakeInfo()

        dataset = FakeDataset()
        dataset_metadata = {"task_name": "math", "difficulty_generation": {"bucket_count_requested": 5}}

        MODULE.annotate_dataset_metadata(dataset, dataset_metadata)

        self.assertEqual(json.loads(dataset.info.description), dataset_metadata)

    def test_normalize_experiment_metadata_uses_canonical_source_root_only(self):
        normalized = MODULE.normalize_experiment_metadata(
            {
                "source_root": "/tmp/example-rollouts",
                "source_input": "/tmp/example-rollouts/demo_run_metadata.jsonl",
                "model_name": "demo-model",
                "experiment_id": "exp-123",
                "experiment_name": "demo-run",
            }
        )

        self.assertEqual(
            normalized,
            {
                "source_root": "/tmp/example-rollouts",
                "model_name": "demo-model",
                "experiment_id": "exp-123",
                "experiment_name": "demo-run",
            },
        )

    def test_apply_beta_binomial_difficulty_orders_rows_by_expected_quantile(self):
        rows = [
            {"instance_id": "easy", "attempt_scores": [1.0, 1.0, 1.0, 1.0]},
            {"instance_id": "medium", "attempt_scores": [1.0, 1.0, 0.0, 0.0]},
            {"instance_id": "hard", "attempt_scores": [0.0, 0.0, 0.0, 0.0]},
        ]

        result = MODULE.apply_beta_binomial_difficulty(
            rows, prior=MODULE.BetaPrior(alpha=0.5, beta=0.5, source="test"), lower_quantile=0.1, num_buckets=3
        )
        difficulties = {row["instance_id"]: row["difficulty"] for row in result}

        self.assertLess(difficulties["easy"]["expected_quantile"], difficulties["medium"]["expected_quantile"])
        self.assertLess(difficulties["medium"]["expected_quantile"], difficulties["hard"]["expected_quantile"])
        self.assertEqual(difficulties["easy"]["bucket_index"], 0)
        self.assertEqual(difficulties["medium"]["bucket_index"], 1)
        self.assertEqual(difficulties["hard"]["bucket_index"], 2)
        self.assertTrue(all(difficulty["bucket_count"] == 3 for difficulty in difficulties.values()))

    def test_apply_beta_binomial_difficulty_balances_bucket_sizes(self):
        rows = [
            {"instance_id": "easiest", "attempt_scores": [1.0, 1.0, 1.0, 1.0]},
            {"instance_id": "easy", "attempt_scores": [1.0, 1.0, 1.0, 0.0]},
            {"instance_id": "mid", "attempt_scores": [1.0, 1.0, 0.0, 0.0]},
            {"instance_id": "hard", "attempt_scores": [1.0, 0.0, 0.0, 0.0]},
            {"instance_id": "hardest", "attempt_scores": [0.0, 0.0, 0.0, 0.0]},
        ]

        result = MODULE.apply_beta_binomial_difficulty(
            rows, prior=MODULE.BetaPrior(alpha=0.5, beta=0.5, source="test"), lower_quantile=0.1, num_buckets=2
        )
        bucket_counts = Counter(row["difficulty"]["bucket_index"] for row in result)

        self.assertEqual(bucket_counts[0], 3)
        self.assertEqual(bucket_counts[1], 2)

    def test_apply_beta_binomial_difficulty_leaves_nonbinary_rows_unbucketed(self):
        rows = [
            {"instance_id": "easy", "attempt_scores": [1.0, 1.0]},
            {"instance_id": "nonbinary", "attempt_scores": [0.5, 1.0]},
            {"instance_id": "hard", "attempt_scores": [0.0, 0.0]},
        ]

        result = MODULE.apply_beta_binomial_difficulty(
            rows, prior=MODULE.BetaPrior(alpha=0.5, beta=0.5, source="test"), lower_quantile=0.1, num_buckets=2
        )
        difficulties = {row["instance_id"]: row["difficulty"] for row in result}

        self.assertIsNone(difficulties["nonbinary"]["value"])
        self.assertIsNone(difficulties["nonbinary"]["expected_quantile"])
        self.assertIsNone(difficulties["nonbinary"]["bucket_index"])
        self.assertIsNone(difficulties["nonbinary"]["bucket_count"])


if __name__ == "__main__":
    unittest.main()
