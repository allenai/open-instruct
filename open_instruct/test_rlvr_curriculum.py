import sys
import tempfile
import types
import unittest

from datasets import Dataset

if "vllm" not in sys.modules:
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.SamplingParams = object
    sys.modules["vllm"] = vllm_stub

from open_instruct import data_loader, rlvr_curriculum


class ListDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_difficulty_row(index: int, bucket_index: int, posterior_mean: float, bucket_count: int = 5) -> dict:
    return {
        "index": index,
        "difficulty": {
            "value": 1.0 - posterior_mean,
            "posterior_mean": posterior_mean,
            "posterior_lower_bound": 0.0,
            "expected_quantile": bucket_index / max(bucket_count - 1, 1),
            "bucket_index": bucket_index,
            "bucket_count": bucket_count,
        },
    }


def make_bucket_dataset(bucket_count: int = 5) -> ListDataset:
    rows = [
        make_difficulty_row(index=0, bucket_index=0, posterior_mean=0.95, bucket_count=bucket_count),
        make_difficulty_row(index=1, bucket_index=1, posterior_mean=0.80, bucket_count=bucket_count),
        make_difficulty_row(index=2, bucket_index=2, posterior_mean=0.50, bucket_count=bucket_count),
        make_difficulty_row(index=3, bucket_index=3, posterior_mean=0.20, bucket_count=bucket_count),
        make_difficulty_row(index=4, bucket_index=4, posterior_mean=0.003, bucket_count=bucket_count),
    ]
    return ListDataset(rows)


def make_plain_hf_dataset(num_examples: int) -> Dataset:
    return Dataset.from_dict(
        {"text": [f"example_{index}" for index in range(num_examples)], "index": list(range(num_examples))}
    )


class TestDifficultyCurriculumSampler(unittest.TestCase):
    def _make_config(self, **overrides) -> rlvr_curriculum.DifficultyCurriculumConfig:
        return rlvr_curriculum.DifficultyCurriculumConfig(
            enabled=True, easy_focus_steps=100, warmup_steps=120, total_curriculum_steps=200, seed=13, **overrides
        )

    def _make_sampler(self, dataset, **config_overrides) -> rlvr_curriculum.BetaBinomialDifficultySampler:
        config = self._make_config(**config_overrides)
        return rlvr_curriculum.BetaBinomialDifficultySampler(
            dataset=dataset, num_samples=max(len(dataset), 1), config=config, global_step_getter=lambda: 0
        )

    def test_missing_metadata_raises_when_strict_metadata(self):
        dataset = ListDataset([{"index": 0}])
        with self.assertRaises(ValueError):
            self._make_sampler(dataset, strict_metadata=True)

    def test_missing_metadata_falls_back_when_not_strict(self):
        dataset = ListDataset(
            [make_difficulty_row(index=0, bucket_index=0, posterior_mean=0.9, bucket_count=5), {"index": 1}]
        )
        sampler = self._make_sampler(dataset, strict_metadata=False)
        self.assertEqual(sampler.metadata_fallback_count, 1)
        self.assertIn(1, sampler.bucket_to_indices[2])

    def test_bucket_grouping_works(self):
        sampler = self._make_sampler(make_bucket_dataset())
        self.assertEqual(sampler.bucket_to_indices, ((0,), (1,), (2,), (3,), (4,)))

    def test_bootstrap_curriculum_heavily_samples_easy_buckets(self):
        sampler = self._make_sampler(make_bucket_dataset())
        early_probs = sampler.get_static_bucket_probs(step=0)
        self.assertGreater(early_probs[0] + early_probs[1], 0.75)
        self.assertGreater(early_probs[0], early_probs[2])
        self.assertGreater(early_probs[1], early_probs[2])
        self.assertLessEqual(early_probs[4], sampler.config.min_hard_frac + 1e-6)

    def test_post_bootstrap_curriculum_returns_to_medium_buckets(self):
        sampler = self._make_sampler(make_bucket_dataset())
        post_bootstrap_probs = sampler.get_static_bucket_probs(step=sampler.config.easy_focus_steps)
        self.assertEqual(int(post_bootstrap_probs.argmax()), 2)
        self.assertGreater(post_bootstrap_probs[2], post_bootstrap_probs[1])
        self.assertGreater(post_bootstrap_probs[2], post_bootstrap_probs[3])

    def test_late_curriculum_increases_hard_bucket_probability(self):
        sampler = self._make_sampler(make_bucket_dataset())
        early_probs = sampler.get_bucket_probs(step=0)
        late_step = sampler.config.warmup_steps + sampler.config.total_curriculum_steps
        late_probs = sampler.get_bucket_probs(step=late_step)
        self.assertGreater(late_probs[4], early_probs[4])
        self.assertGreater(late_probs[4], late_probs[2])
        self.assertGreater(late_probs[3], late_probs[2])

    def test_extremely_hard_example_is_rare_early_but_more_likely_late(self):
        sampler = self._make_sampler(make_bucket_dataset())
        early_probability = sampler.get_example_probability(4, step=0)
        late_step = sampler.config.warmup_steps + sampler.config.total_curriculum_steps
        late_probability = sampler.get_example_probability(4, step=late_step)
        self.assertLess(early_probability, 0.1)
        self.assertGreater(late_probability, early_probability)

    def test_probabilities_always_sum_to_one(self):
        sampler = self._make_sampler(make_bucket_dataset())
        for step in (
            0,
            sampler.config.warmup_steps + 5,
            sampler.config.warmup_steps + sampler.config.total_curriculum_steps,
        ):
            self.assertAlmostEqual(float(sampler.get_static_bucket_probs(step=step).sum()), 1.0, places=6)
            self.assertAlmostEqual(float(sampler.get_bucket_probs(step=step).sum()), 1.0, places=6)

    def test_adaptive_stats_increase_sampling_probability_for_high_signal_bucket(self):
        sampler = self._make_sampler(
            make_bucket_dataset(), adaptive_enabled=True, adaptive_update_every=1, adaptive_blend_weight=0.5
        )
        static_probs = sampler.get_bucket_probs(step=0)
        sampler.record_observations(
            dataset_indices=[4, 4, 4, 4, 4, 2, 2],
            rewards=[0.3, 0.35, 0.25, 0.4, 0.3, 0.95, 0.9],
            advantages=[1.2, 1.1, 1.0, 0.9, 1.3, 0.05, 0.02],
        )
        adaptive_probs = sampler.get_bucket_probs(step=1)
        self.assertGreater(adaptive_probs[4], static_probs[4])

    def test_bootstrap_distribution_is_tunable(self):
        default_sampler = self._make_sampler(make_bucket_dataset())
        tuned_sampler = self._make_sampler(
            make_bucket_dataset(),
            bootstrap_target_bucket_ratio=0.0,
            warmup_target_bucket_ratio=0.4,
            easy_focus_sigma=0.5,
        )

        default_probs = default_sampler.get_static_bucket_probs(step=0)
        tuned_probs = tuned_sampler.get_static_bucket_probs(step=0)

        self.assertGreater(tuned_probs[0], default_probs[0])
        self.assertLess(tuned_probs[2], default_probs[2])


class TestDifficultyCurriculumLoaderIntegration(unittest.TestCase):
    def test_existing_behavior_is_unchanged_when_curriculum_disabled(self):
        dataset = make_plain_hf_dataset(20)
        config = data_loader.StreamingDataLoaderConfig(difficulty_curriculum_enabled=False)

        built_loader = data_loader.build_data_preparation_prompt_dataloader(
            dataset=dataset, seed=7, work_dir=tempfile.gettempdir(), config=config
        )
        baseline_loader = data_loader.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=7,
            dp_rank=0,
            dp_world_size=1,
            work_dir=tempfile.gettempdir(),
            automatic_reshuffle=True,
            collator=data_loader.single_example_collator,
        )

        self.assertIs(type(built_loader), data_loader.HFDataLoader)

        built_indices = [batch["index"].item() for batch in built_loader]
        baseline_indices = [batch["index"].item() for batch in baseline_loader]
        self.assertEqual(built_indices, baseline_indices)


if __name__ == "__main__":
    unittest.main()
