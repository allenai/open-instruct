"""Unit tests for posterior-aware bucketing in create_difficulty_map.py."""

import importlib.util
import json
import math
import sys
import types
import unittest
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from unittest.mock import patch

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/data/difficulty_sampling/create_difficulty_map.py"


def _load_create_difficulty_map_module():
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
    module_name = "test_create_difficulty_map_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with patch.dict(sys.modules, modules):
        sys.modules.pop(module_name, None)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


MODULE = _load_create_difficulty_map_module()


class TestCreateDifficultyMap(unittest.TestCase):
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
            return TestCreateDifficultyMap.FakeHFDataset([self._rows[index] for index in indices])

        def remove_columns(self, column_names):
            names = {column_names} if isinstance(column_names, str) else set(column_names)
            return TestCreateDifficultyMap.FakeHFDataset(
                [{key: value for key, value in row.items() if key not in names} for row in self._rows]
            )

        def add_column(self, name, values):
            return TestCreateDifficultyMap.FakeHFDataset(
                [{**row, name: value} for row, value in zip(self._rows, values, strict=True)]
            )

    def make_row(
        self,
        *,
        source_row_index=0,
        instance_id="row-0",
        task_name="math",
        source_dataset="mnoukhov/demo",
        source_row_id="row-0",
        attempt_scores=None,
        model_name="demo-model",
        warnings=None,
        difficulty=None,
    ):
        return MODULE.DifficultyRow(
            source_row_index=source_row_index,
            instance_id=instance_id,
            task_name=task_name,
            base_task_name=MODULE.get_base_task_name(task_name),
            source_dataset=source_dataset,
            source_row_id=source_row_id,
            attempt_scores=list(attempt_scores or []),
            finish_reasons=[],
            experiment_metadata=MODULE.ExperimentMetadata(
                source_root=f"hf://{source_dataset}/default/train",
                model_name=model_name,
                experiment_id=None,
                experiment_name=source_dataset,
            ),
            score_sources=[task_name],
            warnings=list(warnings or []),
            difficulty=difficulty or MODULE.DifficultyPayload(),
        )

    def test_parser_requires_hf_dataset_and_rejects_source(self):
        with self.assertRaises(SystemExit):
            MODULE.make_parser().parse_args(["--output", "/tmp/difficulty"])

        with self.assertRaises(SystemExit):
            MODULE.make_parser().parse_args(["--source", "/tmp/rollouts", "--output", "/tmp/difficulty"])

    def test_normalize_attempt_scores_for_group_marks_unsupported_rewards(self):
        rows = [self.make_row(instance_id="example", source_row_id="example", attempt_scores=[10.0, 5.0])]

        kept_rows, score_processing, skipped_nonunit = MODULE.normalize_attempt_scores_for_group(
            rows, allow_nonunit_scores=True
        )

        self.assertEqual(skipped_nonunit, 0)
        self.assertFalse(score_processing.supports_binary_difficulty)
        self.assertEqual(kept_rows[0].attempt_scores, [10.0, 5.0])
        self.assertIn("nonbinary_reward_scores", kept_rows[0].warnings)

        dropped_rows, _, dropped_count = MODULE.normalize_attempt_scores_for_group(rows, allow_nonunit_scores=False)

        self.assertEqual(dropped_rows, [])
        self.assertEqual(dropped_count, 1)

    def test_build_dataset_metadata_captures_difficulty_generation_details(self):
        rows = [
            self.make_row(
                instance_id="easy",
                source_row_id="easy",
                difficulty=MODULE.DifficultyPayload(
                    value=0.1,
                    posterior_mean=0.2,
                    posterior_lower_bound=0.9,
                    expected_quantile=0.2,
                    bucket_index=0,
                    bucket_count=3,
                ),
            ),
            self.make_row(
                source_row_index=1,
                instance_id="hard",
                source_row_id="hard",
                difficulty=MODULE.DifficultyPayload(
                    value=0.8,
                    posterior_mean=0.7,
                    posterior_lower_bound=0.2,
                    expected_quantile=0.9,
                    bucket_index=2,
                    bucket_count=3,
                ),
            ),
            self.make_row(source_row_index=2, instance_id="nonbinary", source_row_id="nonbinary"),
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
            score_processing=MODULE.ScoreProcessingMetadata(
                source_field="reward",
                output_field="attempt_scores",
                normalization="binary_zero_or_constant",
                positive_reward_value=10.0,
                supports_binary_difficulty=True,
            ),
            source_format=MODULE.build_hf_source_format_metadata(
                dataset_name="mnoukhov/demo",
                config_name=None,
                split="train",
                row_id_field="extra_info.index",
                task_field="dataset",
                model_field="generator_model",
                pass_count_field="pass_count",
                attempt_count_field="num_samples",
                pass_rate_field="pass_rate",
            ),
        )

        self.assertEqual(metadata.task_name, "math")
        self.assertEqual(metadata.model_name, "demo-model")
        self.assertEqual(metadata.row_count, 3)
        self.assertEqual(metadata.source_format.kind, MODULE.HF_SOURCE_FORMAT_KIND)
        self.assertEqual(metadata.score_processing.normalization, "binary_zero_or_constant")
        self.assertEqual(metadata.score_processing.positive_reward_value, 10.0)
        self.assertEqual(metadata.difficulty_generation.method, "beta_binomial_posterior_quantiles")
        self.assertEqual(metadata.difficulty_generation.posterior_lower_quantile, 0.1)
        self.assertEqual(metadata.difficulty_generation.bucket_count_requested, 5)
        self.assertEqual(metadata.difficulty_generation.bucket_count_effective, 3)
        self.assertEqual(metadata.difficulty_generation.beta_prior_used.source, "empirical_bayes")
        self.assertEqual(metadata.difficulty_generation.beta_prior_used.alpha, 0.75)
        self.assertEqual(metadata.difficulty_generation.beta_prior_used.beta, 1.25)
        self.assertEqual(metadata.difficulty_generation.binary_instance_count, 2)
        self.assertEqual(metadata.difficulty_generation.nonbinary_instance_count, 1)

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

        self.assertEqual(row.instance_id, "mnoukhov/demo::row-7")
        self.assertEqual(row.source_row_id, "row-7")
        self.assertEqual(row.attempt_scores, [1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertEqual(row.experiment_metadata.model_name, "Qwen/Qwen3-4B-Base")
        self.assertEqual(row.experiment_metadata.source_root, "hf://mnoukhov/demo/default/train")

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
        self.assertEqual(bundle.source_format.kind, MODULE.HF_SOURCE_FORMAT_KIND)
        self.assertEqual(bundle.source_format.dataset_repo_id, "mnoukhov/demo")
        self.assertEqual(len(bundle.rows), 1)
        self.assertEqual(bundle.rows[0].instance_id, "mnoukhov/demo::math-1")
        self.assertEqual(bundle.rows[0].attempt_scores, [1.0, 1.0, 0.0, 0.0])

    def test_build_hf_output_dataset_preserves_source_rows_and_order(self):
        source_dataset = self.FakeHFDataset(
            [
                {"prompt": "first", "extra_info": {"index": "row-0"}},
                {"prompt": "second", "extra_info": {"index": "row-1"}},
            ]
        )
        rows = [
            self.make_row(
                source_row_index=1,
                instance_id="mnoukhov/demo::row-1",
                source_row_id="row-1",
                attempt_scores=[0.0, 0.0],
                model_name="Qwen/Qwen3-4B-Base",
                difficulty=MODULE.DifficultyPayload(
                    value=0.9,
                    posterior_mean=0.1,
                    posterior_lower_bound=0.1,
                    expected_quantile=0.9,
                    bucket_index=1,
                    bucket_count=2,
                ),
            ),
            self.make_row(
                source_row_index=0,
                instance_id="mnoukhov/demo::row-0",
                source_row_id="row-0",
                attempt_scores=[1.0, 1.0],
                model_name="Qwen/Qwen3-4B-Base",
                difficulty=MODULE.DifficultyPayload(
                    value=0.1,
                    posterior_mean=0.9,
                    posterior_lower_bound=0.9,
                    expected_quantile=0.1,
                    bucket_index=0,
                    bucket_count=2,
                ),
            ),
        ]

        output_rows, dataset = MODULE.build_hf_output_dataset(source_dataset, rows)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.column_names, ["prompt", "extra_info", "difficulty"])
        self.assertEqual(output_rows[0]["source_row_id"], "row-0")
        self.assertEqual(output_rows[1]["source_row_id"], "row-1")
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
        dataset_metadata = MODULE.build_dataset_metadata(
            rows=[self.make_row(difficulty=MODULE.DifficultyPayload(bucket_index=0, bucket_count=5))],
            task_name="math",
            model_name="demo-model",
            requested_prior_mode="empirical-bayes",
            requested_bucket_count=5,
            lower_quantile=0.1,
            prior=MODULE.BetaPrior(alpha=0.5, beta=0.5, source="empirical_bayes"),
            binary_row_count=1,
            score_processing=MODULE.ScoreProcessingMetadata(
                source_field="pass_count,num_samples,pass_rate",
                output_field="attempt_scores",
                normalization="identity_binary",
                positive_reward_value=1.0,
                supports_binary_difficulty=True,
            ),
            source_format=MODULE.build_hf_source_format_metadata(
                dataset_name="mnoukhov/demo",
                config_name=None,
                split="train",
                row_id_field="extra_info.index",
                task_field="dataset",
                model_field="generator_model",
                pass_count_field="pass_count",
                attempt_count_field="num_samples",
                pass_rate_field="pass_rate",
            ),
        )

        MODULE.annotate_dataset_metadata(dataset, dataset_metadata)

        self.assertEqual(json.loads(dataset.info.description), MODULE.make_jsonable(dataset_metadata))

    def test_apply_beta_binomial_difficulty_orders_rows_by_expected_quantile(self):
        rows = [
            self.make_row(instance_id="easy", source_row_id="easy", attempt_scores=[1.0, 1.0, 1.0, 1.0]),
            self.make_row(
                source_row_index=1, instance_id="medium", source_row_id="medium", attempt_scores=[1.0, 1.0, 0.0, 0.0]
            ),
            self.make_row(
                source_row_index=2, instance_id="hard", source_row_id="hard", attempt_scores=[0.0, 0.0, 0.0, 0.0]
            ),
        ]

        result = MODULE.apply_beta_binomial_difficulty(
            rows, prior=MODULE.BetaPrior(alpha=0.5, beta=0.5, source="test"), lower_quantile=0.1, num_buckets=3
        )
        difficulties = {row.instance_id: row.difficulty for row in result}

        self.assertLess(difficulties["easy"].expected_quantile, difficulties["medium"].expected_quantile)
        self.assertLess(difficulties["medium"].expected_quantile, difficulties["hard"].expected_quantile)
        self.assertEqual(difficulties["easy"].bucket_index, 0)
        self.assertEqual(difficulties["medium"].bucket_index, 1)
        self.assertEqual(difficulties["hard"].bucket_index, 2)
        self.assertTrue(all(difficulty.bucket_count == 3 for difficulty in difficulties.values()))

    def test_apply_beta_binomial_difficulty_balances_bucket_sizes(self):
        rows = [
            self.make_row(instance_id="easiest", source_row_id="easiest", attempt_scores=[1.0, 1.0, 1.0, 1.0]),
            self.make_row(
                source_row_index=1, instance_id="easy", source_row_id="easy", attempt_scores=[1.0, 1.0, 1.0, 0.0]
            ),
            self.make_row(
                source_row_index=2, instance_id="mid", source_row_id="mid", attempt_scores=[1.0, 1.0, 0.0, 0.0]
            ),
            self.make_row(
                source_row_index=3, instance_id="hard", source_row_id="hard", attempt_scores=[1.0, 0.0, 0.0, 0.0]
            ),
            self.make_row(
                source_row_index=4, instance_id="hardest", source_row_id="hardest", attempt_scores=[0.0, 0.0, 0.0, 0.0]
            ),
        ]

        result = MODULE.apply_beta_binomial_difficulty(
            rows, prior=MODULE.BetaPrior(alpha=0.5, beta=0.5, source="test"), lower_quantile=0.1, num_buckets=2
        )
        bucket_counts = Counter(row.difficulty.bucket_index for row in result)

        self.assertEqual(bucket_counts[0], 3)
        self.assertEqual(bucket_counts[1], 2)

    def test_apply_beta_binomial_difficulty_leaves_nonbinary_rows_unbucketed(self):
        rows = [
            self.make_row(instance_id="easy", source_row_id="easy", attempt_scores=[1.0, 1.0]),
            self.make_row(
                source_row_index=1, instance_id="nonbinary", source_row_id="nonbinary", attempt_scores=[0.5, 1.0]
            ),
            self.make_row(source_row_index=2, instance_id="hard", source_row_id="hard", attempt_scores=[0.0, 0.0]),
        ]

        result = MODULE.apply_beta_binomial_difficulty(
            rows, prior=MODULE.BetaPrior(alpha=0.5, beta=0.5, source="test"), lower_quantile=0.1, num_buckets=2
        )
        difficulties = {row.instance_id: row.difficulty for row in result}

        self.assertIsNone(difficulties["nonbinary"].value)
        self.assertIsNone(difficulties["nonbinary"].expected_quantile)
        self.assertIsNone(difficulties["nonbinary"].bucket_index)
        self.assertIsNone(difficulties["nonbinary"].bucket_count)


if __name__ == "__main__":
    unittest.main()
