import unittest
from unittest import mock

from transformers import HfArgumentParser

from open_instruct import difficulty_curriculum


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


class TestDifficultyCurriculumSampler(unittest.TestCase):
    def _make_metadata(self, **overrides) -> difficulty_curriculum.DifficultyCurriculumMetadataConfig:
        return difficulty_curriculum.DifficultyCurriculumMetadataConfig(**overrides)

    def _make_schedule(self, **overrides) -> difficulty_curriculum.DifficultyCurriculumScheduleConfig:
        return difficulty_curriculum.DifficultyCurriculumScheduleConfig(
            bootstrap_steps=100, warmup_steps=120, total_steps=200, **overrides
        )

    def _make_adaptive(self, **overrides) -> difficulty_curriculum.DifficultyCurriculumAdaptiveConfig:
        return difficulty_curriculum.DifficultyCurriculumAdaptiveConfig(**overrides)

    def _make_config(self, **overrides) -> difficulty_curriculum.DifficultyCurriculumConfig:
        return difficulty_curriculum.DifficultyCurriculumConfig(
            metadata=overrides.pop("metadata", self._make_metadata()),
            schedule=overrides.pop("schedule", self._make_schedule()),
            adaptive=overrides.pop("adaptive", self._make_adaptive()),
            seed=13,
            **overrides,
        )

    def _make_sampler(self, dataset, **config_overrides) -> difficulty_curriculum.DifficultyCurriculumSampler:
        config = self._make_config(**config_overrides)
        return difficulty_curriculum.DifficultyCurriculumSampler(
            dataset=dataset, num_samples=max(len(dataset), 1), config=config, global_step_getter=lambda: 0
        )

    def test_missing_metadata_raises_when_strict_metadata(self):
        dataset = ListDataset([{"index": 0}])
        with self.assertRaises(ValueError):
            self._make_sampler(dataset, metadata=self._make_metadata(strict=True))

    def test_missing_metadata_falls_back_when_not_strict(self):
        dataset = ListDataset(
            [make_difficulty_row(index=0, bucket_index=0, posterior_mean=0.9, bucket_count=5), {"index": 1}]
        )
        sampler = self._make_sampler(dataset, metadata=self._make_metadata(strict=False))
        self.assertEqual(sampler.metadata_fallback_count, 1)
        self.assertIn(1, sampler.bucket_to_indices[2])

    def test_quantile_filter_keeps_only_hardest_half(self):
        sampler = self._make_sampler(make_bucket_dataset(), min_quantile=0.5)
        self.assertEqual(sampler.bucket_to_indices, ((), (), (2,), (3,), (4,)))
        self.assertEqual(sampler.filtered_out_count, 2)
        self.assertEqual(sampler.get_example_probability(0, step=0), 0.0)
        self.assertTrue(all(sampler.sample_index(step=0) in {2, 3, 4} for _ in range(20)))

    def test_quantile_filter_reweights_only_available_buckets(self):
        sampler = self._make_sampler(make_bucket_dataset(), min_quantile=0.5)
        early_probs = sampler.get_static_bucket_probs(step=0)
        self.assertAlmostEqual(float(early_probs.sum()), 1.0, places=6)
        self.assertEqual(float(early_probs[0]), 0.0)
        self.assertEqual(float(early_probs[1]), 0.0)
        self.assertGreater(early_probs[2], early_probs[3])

    def test_bucket_grouping_works(self):
        sampler = self._make_sampler(make_bucket_dataset())
        self.assertEqual(sampler.bucket_to_indices, ((0,), (1,), (2,), (3,), (4,)))

    def test_bootstrap_curriculum_heavily_samples_easy_buckets(self):
        sampler = self._make_sampler(make_bucket_dataset())
        early_probs = sampler.get_static_bucket_probs(step=0)
        self.assertGreater(early_probs[0] + early_probs[1], 0.75)
        self.assertGreater(early_probs[0], early_probs[2])
        self.assertGreater(early_probs[1], early_probs[2])
        self.assertLessEqual(early_probs[4], sampler.config.schedule.min_hard_frac + 1e-6)

    def test_post_bootstrap_curriculum_returns_to_medium_buckets(self):
        sampler = self._make_sampler(make_bucket_dataset())
        post_bootstrap_probs = sampler.get_static_bucket_probs(step=sampler.config.schedule.bootstrap_steps)
        self.assertEqual(int(post_bootstrap_probs.argmax()), 2)
        self.assertGreater(post_bootstrap_probs[2], post_bootstrap_probs[1])
        self.assertGreater(post_bootstrap_probs[2], post_bootstrap_probs[3])

    def test_late_curriculum_increases_hard_bucket_probability(self):
        sampler = self._make_sampler(make_bucket_dataset())
        early_probs = sampler.get_bucket_probs(step=0)
        late_step = sampler.config.schedule.warmup_steps + sampler.config.schedule.total_steps
        late_probs = sampler.get_bucket_probs(step=late_step)
        self.assertGreater(late_probs[4], early_probs[4])
        self.assertGreater(late_probs[4], late_probs[2])
        self.assertGreater(late_probs[3], late_probs[2])

    def test_extremely_hard_example_is_rare_early_but_more_likely_late(self):
        sampler = self._make_sampler(make_bucket_dataset())
        early_probability = sampler.get_example_probability(4, step=0)
        late_step = sampler.config.schedule.warmup_steps + sampler.config.schedule.total_steps
        late_probability = sampler.get_example_probability(4, step=late_step)
        self.assertLess(early_probability, 0.1)
        self.assertGreater(late_probability, early_probability)

    def test_probabilities_always_sum_to_one(self):
        sampler = self._make_sampler(make_bucket_dataset())
        for step in (
            0,
            sampler.config.schedule.warmup_steps + 5,
            sampler.config.schedule.warmup_steps + sampler.config.schedule.total_steps,
        ):
            self.assertAlmostEqual(float(sampler.get_static_bucket_probs(step=step).sum()), 1.0, places=6)
            self.assertAlmostEqual(float(sampler.get_bucket_probs(step=step).sum()), 1.0, places=6)

    def test_static_bucket_probs_are_cached_per_step(self):
        sampler = self._make_sampler(make_bucket_dataset())

        with mock.patch.object(sampler._schedule, "build_probs", wraps=sampler._schedule.build_probs) as build_probs:
            sampler.get_bucket_probs(step=7)
            sampler.get_bucket_probs(step=7)
            sampler.get_static_bucket_probs(step=7)
            self.assertEqual(build_probs.call_count, 1)

            sampler.get_bucket_probs(step=8)
            self.assertEqual(build_probs.call_count, 2)

    def test_excluding_index_invalidates_cached_bucket_probs(self):
        sampler = self._make_sampler(make_bucket_dataset())

        with mock.patch.object(sampler._schedule, "build_probs", wraps=sampler._schedule.build_probs) as build_probs:
            sampler.get_static_bucket_probs(step=0)
            self.assertEqual(build_probs.call_count, 1)

            sampler.exclude_index(4)
            excluded_probs = sampler.get_static_bucket_probs(step=0)
            self.assertEqual(build_probs.call_count, 2)

        self.assertEqual(float(excluded_probs[4]), 0.0)

    def test_adaptive_stats_increase_sampling_probability_for_high_signal_bucket(self):
        sampler = self._make_sampler(
            make_bucket_dataset(), adaptive=self._make_adaptive(enabled=True, update_every=1, blend=0.5)
        )
        static_probs = sampler.get_bucket_probs(step=0)
        sampler.record_observations(
            dataset_indices=[4, 4, 4, 4, 4, 2, 2],
            rewards=[0.3, 0.35, 0.25, 0.4, 0.3, 0.95, 0.9],
            advantages=[1.2, 1.1, 1.0, 0.9, 1.3, 0.05, 0.02],
        )
        adaptive_probs = sampler.get_bucket_probs(step=1)
        self.assertGreater(adaptive_probs[4], static_probs[4])

    def test_excluded_index_has_zero_probability_and_persists_after_restore(self):
        dataset = make_bucket_dataset()
        sampler = self._make_sampler(dataset)
        sampler.exclude_index(4)

        self.assertEqual(sampler.get_example_probability(4, step=0), 0.0)
        self.assertEqual(float(sampler.get_bucket_probs(step=0)[4]), 0.0)
        self.assertTrue(all(sampler.sample_index(step=0) != 4 for _ in range(20)))

        restored_sampler = self._make_sampler(dataset)
        restored_sampler.load_state_dict(sampler.state_dict())
        self.assertEqual(restored_sampler.get_example_probability(4, step=0), 0.0)
        self.assertEqual(float(restored_sampler.get_bucket_probs(step=0)[4]), 0.0)
        self.assertTrue(all(restored_sampler.sample_index(step=0) != 4 for _ in range(20)))

    def test_bootstrap_distribution_is_tunable(self):
        default_sampler = self._make_sampler(make_bucket_dataset())
        tuned_sampler = self._make_sampler(
            make_bucket_dataset(),
            schedule=self._make_schedule(bootstrap_target=0.0, warmup_target=0.4, bootstrap_sigma=0.5),
        )

        default_probs = default_sampler.get_static_bucket_probs(step=0)
        tuned_probs = tuned_sampler.get_static_bucket_probs(step=0)

        self.assertGreater(tuned_probs[0], default_probs[0])
        self.assertLess(tuned_probs[2], default_probs[2])

    def test_curriculum_parser_builds_grouped_config(self):
        parser = HfArgumentParser((difficulty_curriculum.DifficultyCurriculumArgs,))
        (curriculum,) = parser.parse_args_into_dataclasses(
            [
                "--curriculum",
                "difficulty",
                "--curriculum_bootstrap_steps",
                "12",
                "--curriculum_warmup_steps",
                "34",
                "--curriculum_total_steps",
                "56",
                "--curriculum_adaptive",
                "true",
                "--curriculum_adaptive_blend",
                "0.25",
                "--curriculum_min_quantile",
                "0.5",
            ]
        )

        curriculum.verify()
        curriculum_config = curriculum.build_curriculum_config(seed=17)

        self.assertIsNotNone(curriculum_config)
        assert curriculum_config is not None
        self.assertEqual(curriculum_config.schedule.bootstrap_steps, 12)
        self.assertEqual(curriculum_config.schedule.warmup_steps, 34)
        self.assertEqual(curriculum_config.schedule.total_steps, 56)
        self.assertTrue(curriculum_config.adaptive.enabled)
        self.assertEqual(curriculum_config.adaptive.blend, 0.25)
        self.assertEqual(curriculum_config.min_quantile, 0.5)
        self.assertEqual(curriculum_config.seed, 17)

    def test_curriculum_verify_accepts_none_mode(self):
        difficulty_curriculum.DifficultyCurriculumArgs().verify()


if __name__ == "__main__":
    unittest.main()
