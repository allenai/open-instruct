"""Difficulty-aware curriculum sampling for RLVR / GRPO prompt selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Sampler

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

_DEFAULT_POSTERIOR_MEAN = 0.5


def _resolve_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return None


def _normalize_probs(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values.astype(np.float64) / total


@dataclass
class DifficultyCurriculumMetadataConfig:
    field: str = "difficulty"
    posterior_mean_field: str = "posterior_mean"
    bucket_index_field: str = "bucket_index"
    bucket_count_field: str = "bucket_count"
    strict: bool = True


@dataclass
class DifficultyCurriculumScheduleConfig:
    bootstrap_steps: int = 100
    warmup_steps: int = 500
    total_steps: int = 10000
    bootstrap_target: float = 0.125
    warmup_target: float = 0.5
    final_target: float = 1.0
    min_hard_frac: float = 0.05
    max_hard_frac: float = 0.50
    bucket_sigma: float = 0.0
    bootstrap_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.bootstrap_steps < 0:
            raise ValueError("bootstrap_steps must be >= 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if not 0.0 <= self.bootstrap_target <= 1.0:
            raise ValueError("bootstrap_target must be in [0, 1]")
        if not 0.0 <= self.warmup_target <= 1.0:
            raise ValueError("warmup_target must be in [0, 1]")
        if not 0.0 <= self.final_target <= 1.0:
            raise ValueError("final_target must be in [0, 1]")
        if not 0.0 <= self.min_hard_frac <= 1.0:
            raise ValueError("min_hard_frac must be in [0, 1]")
        if not 0.0 <= self.max_hard_frac <= 1.0:
            raise ValueError("max_hard_frac must be in [0, 1]")
        if self.min_hard_frac > self.max_hard_frac:
            raise ValueError("min_hard_frac must be <= max_hard_frac")
        if self.bucket_sigma < 0:
            raise ValueError("bucket_sigma must be >= 0")
        if self.bootstrap_sigma < 0:
            raise ValueError("bootstrap_sigma must be >= 0")


@dataclass
class DifficultyCurriculumAdaptiveConfig:
    enabled: bool = False
    update_every: int = 50
    learning_weight: float = 0.7
    exploration_weight: float = 0.3
    blend: float = 0.5

    def __post_init__(self) -> None:
        if self.update_every <= 0:
            raise ValueError("update_every must be > 0")
        if not 0.0 <= self.learning_weight <= 1.0:
            raise ValueError("learning_weight must be in [0, 1]")
        if not 0.0 <= self.exploration_weight <= 1.0:
            raise ValueError("exploration_weight must be in [0, 1]")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")


@dataclass
class DifficultyCurriculumConfig:
    metadata: DifficultyCurriculumMetadataConfig = field(default_factory=DifficultyCurriculumMetadataConfig)
    schedule: DifficultyCurriculumScheduleConfig = field(default_factory=DifficultyCurriculumScheduleConfig)
    adaptive: DifficultyCurriculumAdaptiveConfig = field(default_factory=DifficultyCurriculumAdaptiveConfig)
    uncertainty_weight: float = 0.5
    seed: int = 0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 <= self.uncertainty_weight <= 1.0:
            raise ValueError("uncertainty_weight must be in [0, 1]")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")


@dataclass
class DifficultyCurriculumArgs:
    curriculum: Literal["none", "difficulty"] = "none"
    curriculum_metadata_field: str = "difficulty"
    curriculum_posterior_mean_field: str = "posterior_mean"
    curriculum_bucket_index_field: str = "bucket_index"
    curriculum_bucket_count_field: str = "bucket_count"
    curriculum_strict_metadata: bool = True
    curriculum_bootstrap_steps: int = 100
    curriculum_warmup_steps: int = 500
    curriculum_total_steps: int = 10000
    curriculum_bootstrap_target: float = 0.125
    curriculum_warmup_target: float = 0.5
    curriculum_final_target: float = 1.0
    curriculum_min_hard_frac: float = 0.05
    curriculum_max_hard_frac: float = 0.50
    curriculum_bucket_sigma: float = 0.0
    curriculum_bootstrap_sigma: float = 0.0
    curriculum_uncertainty_weight: float = 0.5
    curriculum_adaptive: bool = False
    curriculum_adaptive_update_every: int = 50
    curriculum_adaptive_learning_weight: float = 0.7
    curriculum_adaptive_exploration_weight: float = 0.3
    curriculum_adaptive_blend: float = 0.5

    def build_metadata_config(self) -> DifficultyCurriculumMetadataConfig:
        return DifficultyCurriculumMetadataConfig(
            field=self.curriculum_metadata_field,
            posterior_mean_field=self.curriculum_posterior_mean_field,
            bucket_index_field=self.curriculum_bucket_index_field,
            bucket_count_field=self.curriculum_bucket_count_field,
            strict=self.curriculum_strict_metadata,
        )

    def build_schedule_config(self) -> DifficultyCurriculumScheduleConfig:
        return DifficultyCurriculumScheduleConfig(
            bootstrap_steps=self.curriculum_bootstrap_steps,
            warmup_steps=self.curriculum_warmup_steps,
            total_steps=self.curriculum_total_steps,
            bootstrap_target=self.curriculum_bootstrap_target,
            warmup_target=self.curriculum_warmup_target,
            final_target=self.curriculum_final_target,
            min_hard_frac=self.curriculum_min_hard_frac,
            max_hard_frac=self.curriculum_max_hard_frac,
            bucket_sigma=self.curriculum_bucket_sigma,
            bootstrap_sigma=self.curriculum_bootstrap_sigma,
        )

    def build_adaptive_config(self) -> DifficultyCurriculumAdaptiveConfig:
        return DifficultyCurriculumAdaptiveConfig(
            enabled=self.curriculum_adaptive,
            update_every=self.curriculum_adaptive_update_every,
            learning_weight=self.curriculum_adaptive_learning_weight,
            exploration_weight=self.curriculum_adaptive_exploration_weight,
            blend=self.curriculum_adaptive_blend,
        )

    def verify(self) -> None:
        if self.curriculum == "none":
            return
        if self.curriculum != "difficulty":
            raise ValueError(f"Unsupported curriculum type: {self.curriculum}")

        self.build_metadata_config()
        self.build_schedule_config()
        self.build_adaptive_config()

    def build_curriculum_config(self, *, seed: int) -> DifficultyCurriculumConfig | None:
        self.verify()
        if self.curriculum == "none":
            return None

        return DifficultyCurriculumConfig(
            metadata=self.build_metadata_config(),
            schedule=self.build_schedule_config(),
            adaptive=self.build_adaptive_config(),
            uncertainty_weight=self.curriculum_uncertainty_weight,
            seed=seed,
        )


class AdaptiveBucketStats:
    """Tracks per-bucket learning signal statistics for adaptive sampling."""

    def __init__(self, learning_signal_weight: float, exploration_weight: float, epsilon: float) -> None:
        self.learning_signal_weight = learning_signal_weight
        self.exploration_weight = exploration_weight
        self.epsilon = epsilon

        self.total_count = 0
        self._count_by_bucket: dict[int, int] = {}
        self._reward_sum_by_bucket: dict[int, float] = {}
        self._reward_sq_sum_by_bucket: dict[int, float] = {}
        self._abs_advantage_sum_by_bucket: dict[int, float] = {}
        self._advantage_count_by_bucket: dict[int, int] = {}

    def update(
        self,
        bucket_indices: list[int] | np.ndarray,
        rewards: list[float] | np.ndarray,
        advantages: list[float] | np.ndarray | None = None,
    ) -> None:
        if len(bucket_indices) != len(rewards):
            raise ValueError("bucket_indices and rewards must have the same length")
        if advantages is not None and len(advantages) != len(rewards):
            raise ValueError("advantages and rewards must have the same length")

        reward_values = np.clip(np.asarray(rewards, dtype=np.float64), 0.0, 1.0)
        advantage_values = None if advantages is None else np.abs(np.asarray(advantages, dtype=np.float64))

        for position, bucket_index in enumerate(bucket_indices):
            bucket = int(bucket_index)
            reward = float(reward_values[position])

            self.total_count += 1
            self._count_by_bucket[bucket] = self._count_by_bucket.get(bucket, 0) + 1
            self._reward_sum_by_bucket[bucket] = self._reward_sum_by_bucket.get(bucket, 0.0) + reward
            self._reward_sq_sum_by_bucket[bucket] = self._reward_sq_sum_by_bucket.get(bucket, 0.0) + reward * reward

            if advantage_values is not None:
                advantage = float(advantage_values[position])
                self._abs_advantage_sum_by_bucket[bucket] = (
                    self._abs_advantage_sum_by_bucket.get(bucket, 0.0) + advantage
                )
                self._advantage_count_by_bucket[bucket] = self._advantage_count_by_bucket.get(bucket, 0) + 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "count_by_bucket": self._count_by_bucket,
            "reward_sum_by_bucket": self._reward_sum_by_bucket,
            "reward_sq_sum_by_bucket": self._reward_sq_sum_by_bucket,
            "abs_advantage_sum_by_bucket": self._abs_advantage_sum_by_bucket,
            "advantage_count_by_bucket": self._advantage_count_by_bucket,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.total_count = int(state_dict.get("total_count", 0))
        self._count_by_bucket = {int(k): int(v) for k, v in state_dict.get("count_by_bucket", {}).items()}
        self._reward_sum_by_bucket = {int(k): float(v) for k, v in state_dict.get("reward_sum_by_bucket", {}).items()}
        self._reward_sq_sum_by_bucket = {
            int(k): float(v) for k, v in state_dict.get("reward_sq_sum_by_bucket", {}).items()
        }
        self._abs_advantage_sum_by_bucket = {
            int(k): float(v) for k, v in state_dict.get("abs_advantage_sum_by_bucket", {}).items()
        }
        self._advantage_count_by_bucket = {
            int(k): int(v) for k, v in state_dict.get("advantage_count_by_bucket", {}).items()
        }

    def get_count(self, bucket_index: int) -> int:
        return self._count_by_bucket.get(bucket_index, 0)

    def get_mean_reward(self, bucket_index: int) -> float:
        count = self.get_count(bucket_index)
        if count == 0:
            return 0.0
        return self._reward_sum_by_bucket.get(bucket_index, 0.0) / count

    def get_mean_abs_advantage(self, bucket_index: int) -> float:
        count = self._advantage_count_by_bucket.get(bucket_index, 0)
        if count == 0:
            return 0.0
        return self._abs_advantage_sum_by_bucket.get(bucket_index, 0.0) / count

    def _get_reward_variance(self, bucket_index: int) -> float:
        count = self.get_count(bucket_index)
        if count == 0:
            return 0.0
        mean_reward = self.get_mean_reward(bucket_index)
        mean_reward_sq = self._reward_sq_sum_by_bucket.get(bucket_index, 0.0) / count
        return max(0.0, mean_reward_sq - mean_reward * mean_reward)

    def get_bucket_scores(self, bucket_count: int) -> np.ndarray:
        scores = np.zeros(bucket_count, dtype=np.float64)
        total_count = max(self.total_count, 0)

        for bucket_index in range(bucket_count):
            count = self.get_count(bucket_index)
            mean_reward = self.get_mean_reward(bucket_index)
            mean_abs_advantage = self.get_mean_abs_advantage(bucket_index)

            if self._advantage_count_by_bucket.get(bucket_index, 0) > 0:
                learning_signal = mean_abs_advantage * max(0.0, 1.0 - mean_reward)
            else:
                reward_variance = self._get_reward_variance(bucket_index)
                non_saturation = max(0.0, 1.0 - 2.0 * abs(mean_reward - 0.5))
                learning_signal = 0.5 * reward_variance + 0.5 * non_saturation

            exploration_bonus = math.sqrt(math.log(total_count + 1.0) / (count + 1.0))
            scores[bucket_index] = (
                self.learning_signal_weight * learning_signal
                + self.exploration_weight * exploration_bonus
                + self.epsilon
            )

        return scores


@dataclass(frozen=True)
class _ParsedDifficultyMetadata:
    bucket_index: int | None
    bucket_count: int | None
    posterior_mean: float | None
    error: str | None


@dataclass(frozen=True)
class _DifficultyBucketIndex:
    index_to_bucket: dict[int, int]
    index_to_bucket_position: dict[int, int]
    bucket_to_indices: tuple[tuple[int, ...], ...]
    bucket_weights: tuple[torch.Tensor, ...]
    bucket_count: int
    metadata_fallback_count: int


class _DifficultyCurriculumSchedule:
    def __init__(self, config: DifficultyCurriculumScheduleConfig, bucket_count: int) -> None:
        self.config = config
        self.bucket_count = bucket_count

    def get_progress(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return 0.0
        return min(1.0, (step - self.config.warmup_steps) / self.config.total_steps)

    def _smooth_progress(self, step: int) -> float:
        progress = self.get_progress(step)
        return progress * progress * (3.0 - 2.0 * progress)

    def _get_default_bucket_sigma(self) -> float:
        return max(0.85, 0.25 * max(self.bucket_count - 1, 1))

    def _get_bucket_sigma(self, step: int) -> float:
        sigma = self.config.bucket_sigma if self.config.bucket_sigma > 0 else self._get_default_bucket_sigma()
        if step < self.config.bootstrap_steps and self.config.bootstrap_sigma > 0:
            return self.config.bootstrap_sigma
        return sigma

    def _bucket_ratio_to_bucket_index(self, bucket_ratio: float) -> float:
        return float(self.bucket_count - 1) * bucket_ratio

    def _get_target_bucket(self, step: int) -> float:
        warmup_target_bucket = self._bucket_ratio_to_bucket_index(self.config.warmup_target)
        final_target_bucket = self._bucket_ratio_to_bucket_index(self.config.final_target)

        if self.config.bootstrap_steps > 0 and step < self.config.bootstrap_steps:
            bootstrap_progress = min(1.0, step / self.config.bootstrap_steps)
            bootstrap_target_bucket = self._bucket_ratio_to_bucket_index(self.config.bootstrap_target)
            return bootstrap_target_bucket + (warmup_target_bucket - bootstrap_target_bucket) * bootstrap_progress

        smooth_progress = self._smooth_progress(step)
        return warmup_target_bucket + (final_target_bucket - warmup_target_bucket) * smooth_progress

    def build_probs(self, step: int, available_mask: np.ndarray) -> np.ndarray:
        if self.bucket_count == 1:
            return np.ones(1, dtype=np.float64)

        smooth_progress = self._smooth_progress(step)
        target_bucket = self._get_target_bucket(step)
        hard_bucket_frac = (
            self.config.min_hard_frac + (self.config.max_hard_frac - self.config.min_hard_frac) * smooth_progress
        )

        bucket_ids = np.arange(self.bucket_count - 1, dtype=np.float64)
        sigma = self._get_bucket_sigma(step)
        gaussian_logits = np.exp(-0.5 * ((bucket_ids - target_bucket) / sigma) ** 2)
        non_hard_probs = _normalize_probs(gaussian_logits)

        static_probs = np.zeros(self.bucket_count, dtype=np.float64)
        static_probs[:-1] = (1.0 - hard_bucket_frac) * non_hard_probs
        static_probs[-1] = hard_bucket_frac

        static_probs *= available_mask
        if available_mask.sum() == 0:
            return np.ones(self.bucket_count, dtype=np.float64) / self.bucket_count
        if static_probs.sum() <= 0:
            return _normalize_probs(available_mask)
        return _normalize_probs(static_probs)


def _parse_difficulty_metadata(
    example: dict[str, Any], index: int, metadata_config: DifficultyCurriculumMetadataConfig
) -> _ParsedDifficultyMetadata:
    difficulty_blob = _resolve_path(example, metadata_config.field)
    if not isinstance(difficulty_blob, dict):
        return _ParsedDifficultyMetadata(
            bucket_index=None,
            bucket_count=None,
            posterior_mean=None,
            error=f"missing '{metadata_config.field}' metadata for dataset index {index}",
        )

    bucket_index = _coerce_int(_resolve_path(difficulty_blob, metadata_config.bucket_index_field))
    bucket_count = _coerce_int(_resolve_path(difficulty_blob, metadata_config.bucket_count_field))
    posterior_mean = _coerce_float(_resolve_path(difficulty_blob, metadata_config.posterior_mean_field))

    if bucket_index is None or bucket_index < 0:
        return _ParsedDifficultyMetadata(
            bucket_index=None,
            bucket_count=bucket_count,
            posterior_mean=posterior_mean,
            error=f"invalid bucket_index for dataset index {index}",
        )
    if bucket_count is None or bucket_count <= 0:
        return _ParsedDifficultyMetadata(
            bucket_index=bucket_index,
            bucket_count=None,
            posterior_mean=posterior_mean,
            error=f"invalid bucket_count for dataset index {index}",
        )
    if posterior_mean is None:
        return _ParsedDifficultyMetadata(
            bucket_index=bucket_index,
            bucket_count=bucket_count,
            posterior_mean=None,
            error=f"invalid posterior_mean for dataset index {index}",
        )
    return _ParsedDifficultyMetadata(
        bucket_index=bucket_index, bucket_count=bucket_count, posterior_mean=posterior_mean, error=None
    )


def _compute_example_weight(posterior_mean: float, uncertainty_weight: float, epsilon: float) -> float:
    probability = float(np.clip(posterior_mean, 0.0, 1.0))
    uncertainty = 4.0 * probability * (1.0 - probability)
    hardness = 1.0 - probability
    return uncertainty_weight * uncertainty + (1.0 - uncertainty_weight) * hardness + epsilon


def _build_difficulty_bucket_index(
    dataset, metadata_config: DifficultyCurriculumMetadataConfig, uncertainty_weight: float, epsilon: float
) -> _DifficultyBucketIndex:
    parsed_rows: list[_ParsedDifficultyMetadata] = []
    observed_bucket_counts: set[int] = set()
    max_bucket_index = -1

    for dataset_index in range(len(dataset)):
        parsed = _parse_difficulty_metadata(dataset[dataset_index], dataset_index, metadata_config)
        if parsed.error is not None and metadata_config.strict:
            raise ValueError(parsed.error)
        if parsed.bucket_count is not None:
            observed_bucket_counts.add(parsed.bucket_count)
        if parsed.bucket_index is not None:
            max_bucket_index = max(max_bucket_index, parsed.bucket_index)
        parsed_rows.append(parsed)

    if observed_bucket_counts:
        if metadata_config.strict and len(observed_bucket_counts) > 1:
            raise ValueError(f"inconsistent difficulty bucket_count values found: {sorted(observed_bucket_counts)}")
        bucket_count = max(observed_bucket_counts)
    elif max_bucket_index >= 0:
        bucket_count = max_bucket_index + 1
    else:
        bucket_count = 1

    bucket_to_indices: list[list[int]] = [[] for _ in range(bucket_count)]
    bucket_weight_lists: list[list[float]] = [[] for _ in range(bucket_count)]
    index_to_bucket: dict[int, int] = {}
    index_to_bucket_position: dict[int, int] = {}
    metadata_fallback_count = 0
    fallback_bucket = min(bucket_count - 1, bucket_count // 2)

    for dataset_index, parsed in enumerate(parsed_rows):
        if parsed.error is not None:
            bucket_index = fallback_bucket
            posterior_mean = _DEFAULT_POSTERIOR_MEAN
            metadata_fallback_count += 1
        else:
            assert parsed.bucket_index is not None
            bucket_index = int(np.clip(parsed.bucket_index, 0, bucket_count - 1))
            posterior_mean = parsed.posterior_mean

        if posterior_mean is None:
            posterior_mean = _DEFAULT_POSTERIOR_MEAN
        example_weight = _compute_example_weight(
            posterior_mean=float(np.clip(posterior_mean, 0.0, 1.0)),
            uncertainty_weight=uncertainty_weight,
            epsilon=epsilon,
        )

        index_to_bucket[dataset_index] = bucket_index
        index_to_bucket_position[dataset_index] = len(bucket_to_indices[bucket_index])
        bucket_to_indices[bucket_index].append(dataset_index)
        bucket_weight_lists[bucket_index].append(example_weight)

    return _DifficultyBucketIndex(
        index_to_bucket=index_to_bucket,
        index_to_bucket_position=index_to_bucket_position,
        bucket_to_indices=tuple(tuple(indices) for indices in bucket_to_indices),
        bucket_weights=tuple(torch.tensor(weight_list, dtype=torch.float64) for weight_list in bucket_weight_lists),
        bucket_count=bucket_count,
        metadata_fallback_count=metadata_fallback_count,
    )


class DifficultyCurriculumSampler(Sampler[int]):
    """Bucket-aware curriculum sampler for difficulty-annotated prompt datasets."""

    def __init__(
        self,
        dataset,
        num_samples: int,
        config: DifficultyCurriculumConfig,
        global_step_getter: Callable[[], int] | None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        self.dataset = dataset
        self.num_samples = num_samples
        self.config = config
        self.global_step_getter = global_step_getter

        self._generator = torch.Generator()
        self._generator.manual_seed(self.config.seed)

        bucket_index = _build_difficulty_bucket_index(
            dataset=dataset,
            metadata_config=self.config.metadata,
            uncertainty_weight=self.config.uncertainty_weight,
            epsilon=self.config.epsilon,
        )
        self._index_to_bucket = dict(bucket_index.index_to_bucket)
        self._index_to_bucket_position = dict(bucket_index.index_to_bucket_position)
        self.bucket_count = bucket_index.bucket_count
        self.metadata_fallback_count = bucket_index.metadata_fallback_count
        self._schedule = _DifficultyCurriculumSchedule(self.config.schedule, self.bucket_count)

        self._excluded_indices: set[int] = set()
        self._base_bucket_indices = [list(indices) for indices in bucket_index.bucket_to_indices]
        self._base_bucket_weights = [weights.clone() for weights in bucket_index.bucket_weights]
        self._active_bucket_weights = [weights.clone() for weights in self._base_bucket_weights]
        self._active_bucket_counts = [len(indices) for indices in self._base_bucket_indices]
        self._active_bucket_weight_sums = [float(weights.sum().item()) for weights in self._active_bucket_weights]

        self.adaptive_stats = None
        if self.config.adaptive.enabled:
            self.adaptive_stats = AdaptiveBucketStats(
                learning_signal_weight=self.config.adaptive.learning_weight,
                exploration_weight=self.config.adaptive.exploration_weight,
                epsilon=self.config.epsilon,
            )

        self._cached_adaptive_probs: np.ndarray | None = None
        self._last_adaptive_refresh_step = -1

        if self.metadata_fallback_count > 0 and not self.config.metadata.strict:
            logger.warning(
                "Difficulty curriculum fell back to conservative defaults for %s/%s rows because metadata was missing "
                "or invalid.",
                self.metadata_fallback_count,
                len(self.dataset),
            )

    def __len__(self) -> int:
        return self.num_samples

    @property
    def bucket_to_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(indices) for indices in self._base_bucket_indices)

    def _get_current_step(self) -> int:
        step = self.global_step_getter() if self.global_step_getter is not None else 0
        return max(int(step), 0)

    def get_progress(self, step: int | None = None) -> float:
        if step is None:
            step = self._get_current_step()
        return self._schedule.get_progress(step)

    def _available_bucket_mask(self) -> np.ndarray:
        return np.array([1.0 if count > 0 else 0.0 for count in self._active_bucket_counts], dtype=np.float64)

    def get_static_bucket_probs(self, step: int | None = None) -> np.ndarray:
        if step is None:
            step = self._get_current_step()
        return self._schedule.build_probs(step, self._available_bucket_mask())

    def get_adaptive_bucket_probs(self, step: int | None = None) -> np.ndarray | None:
        if not self.config.adaptive.enabled or self.adaptive_stats is None or self.adaptive_stats.total_count == 0:
            return None

        refresh_step = self._get_current_step() if step is None else step
        if (
            self._cached_adaptive_probs is not None
            and refresh_step - self._last_adaptive_refresh_step < self.config.adaptive.update_every
        ):
            return self._cached_adaptive_probs.copy()

        adaptive_scores = self.adaptive_stats.get_bucket_scores(self.bucket_count)
        adaptive_scores *= self._available_bucket_mask()
        if adaptive_scores.sum() <= 0:
            return None

        self._cached_adaptive_probs = _normalize_probs(adaptive_scores)
        self._last_adaptive_refresh_step = refresh_step
        return self._cached_adaptive_probs.copy()

    def get_bucket_probs(self, step: int | None = None) -> np.ndarray:
        static_probs = self.get_static_bucket_probs(step)
        adaptive_probs = self.get_adaptive_bucket_probs(step)
        if adaptive_probs is None:
            return static_probs

        final_probs = (1.0 - self.config.adaptive.blend) * static_probs + self.config.adaptive.blend * adaptive_probs
        final_probs *= self._available_bucket_mask()
        if final_probs.sum() <= 0:
            return static_probs
        return _normalize_probs(final_probs)

    def bucket_for_dataset_index(self, dataset_index: int) -> int:
        return self._index_to_bucket[int(dataset_index)]

    def get_example_probability(self, dataset_index: int, step: int | None = None) -> float:
        if int(dataset_index) in self._excluded_indices:
            return 0.0
        bucket_index = self.bucket_for_dataset_index(dataset_index)
        if self._active_bucket_counts[bucket_index] == 0:
            return 0.0
        local_index = self._index_to_bucket_position.get(int(dataset_index))
        if local_index is None:
            return 0.0
        bucket_weight = self._active_bucket_weights[bucket_index]
        dataset_weight = float(bucket_weight[local_index].item())
        if dataset_weight <= 0:
            return 0.0
        weight_total = self._active_bucket_weight_sums[bucket_index]
        if weight_total <= 0:
            return 0.0
        bucket_probs = self.get_bucket_probs(step)
        return float(bucket_probs[bucket_index] * dataset_weight / weight_total)

    def sample_index(self, step: int | None = None) -> int:
        if self._available_bucket_mask().sum() == 0:
            raise RuntimeError("All dataset examples have been excluded. Cannot continue iteration.")
        bucket_probs = torch.tensor(self.get_bucket_probs(step), dtype=torch.float64)
        bucket_index = int(torch.multinomial(bucket_probs, 1, generator=self._generator).item())

        example_weights = self._active_bucket_weights[bucket_index]
        if self._active_bucket_counts[bucket_index] == 0 or self._active_bucket_weight_sums[bucket_index] <= 0:
            raise RuntimeError("attempted to sample from an empty curriculum bucket")

        sampled_offset = int(torch.multinomial(example_weights, 1, generator=self._generator).item())
        return self._base_bucket_indices[bucket_index][sampled_offset]

    def __iter__(self):
        for _ in range(self.num_samples):
            yield self.sample_index()

    def exclude_index(self, dataset_index: int) -> None:
        dataset_index = int(dataset_index)
        if dataset_index in self._excluded_indices:
            return

        bucket_index = self._index_to_bucket.get(dataset_index)
        if bucket_index is None:
            return

        position = self._index_to_bucket_position.get(dataset_index)
        if position is None:
            self._excluded_indices.add(dataset_index)
            return

        weights = self._active_bucket_weights[bucket_index]
        current_weight = float(weights[position].item())
        if current_weight <= 0:
            self._excluded_indices.add(dataset_index)
            return

        weights[position] = 0.0
        self._active_bucket_counts[bucket_index] -= 1
        self._active_bucket_weight_sums[bucket_index] = max(
            0.0, self._active_bucket_weight_sums[bucket_index] - current_weight
        )
        self._excluded_indices.add(dataset_index)

    def record_observations(
        self,
        dataset_indices: list[int] | np.ndarray,
        rewards: list[float] | np.ndarray,
        advantages: list[float] | np.ndarray | None = None,
    ) -> None:
        if not self.config.adaptive.enabled or self.adaptive_stats is None:
            return
        if len(dataset_indices) == 0:
            return

        bucket_indices = [self.bucket_for_dataset_index(int(dataset_index)) for dataset_index in dataset_indices]
        self.adaptive_stats.update(bucket_indices, rewards, advantages)
        self._cached_adaptive_probs = None

    def build_metrics(self, prompt_dataset_indices: list[int], step: int | None = None) -> dict[str, float]:
        metrics: dict[str, float] = {"curriculum/progress": self.get_progress(step)}

        static_probs = self.get_static_bucket_probs(step)
        for bucket_index, probability in enumerate(static_probs):
            metrics[f"curriculum/static_bucket_prob_{bucket_index}"] = float(probability)

        adaptive_probs = self.get_adaptive_bucket_probs(step)
        if self.config.adaptive.enabled:
            if adaptive_probs is None:
                adaptive_probs = np.zeros(self.bucket_count, dtype=np.float64)
            for bucket_index, probability in enumerate(adaptive_probs):
                metrics[f"curriculum/adaptive_bucket_prob_{bucket_index}"] = float(probability)

        final_probs = self.get_bucket_probs(step)
        for bucket_index, probability in enumerate(final_probs):
            metrics[f"curriculum/bucket_prob_{bucket_index}"] = float(probability)

        sampled_counts = np.zeros(self.bucket_count, dtype=np.float64)
        for dataset_index in prompt_dataset_indices:
            sampled_counts[self.bucket_for_dataset_index(int(dataset_index))] += 1.0
        for bucket_index, count in enumerate(sampled_counts):
            metrics[f"curriculum/sampled_bucket_count_{bucket_index}"] = float(count)

        if self.config.adaptive.enabled and self.adaptive_stats is not None:
            for bucket_index in range(self.bucket_count):
                metrics[f"curriculum/bucket_reward_mean_{bucket_index}"] = float(
                    self.adaptive_stats.get_mean_reward(bucket_index)
                )
                metrics[f"curriculum/bucket_abs_advantage_mean_{bucket_index}"] = float(
                    self.adaptive_stats.get_mean_abs_advantage(bucket_index)
                )

        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self._generator.get_state(),
            "excluded_indices": sorted(self._excluded_indices),
            "adaptive_stats": None if self.adaptive_stats is None else self.adaptive_stats.state_dict(),
            "last_adaptive_refresh_step": self._last_adaptive_refresh_step,
            "cached_adaptive_probs": None
            if self._cached_adaptive_probs is None
            else self._cached_adaptive_probs.tolist(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        generator_state = state_dict.get("generator_state")
        if generator_state is not None:
            self._generator.set_state(generator_state)

        self._excluded_indices = {int(index) for index in state_dict.get("excluded_indices", [])}
        self._active_bucket_weights = [weights.clone() for weights in self._base_bucket_weights]
        self._active_bucket_counts = [len(indices) for indices in self._base_bucket_indices]
        self._active_bucket_weight_sums = [float(weights.sum().item()) for weights in self._active_bucket_weights]
        for dataset_index in self._excluded_indices:
            bucket_index = self._index_to_bucket.get(dataset_index)
            position = self._index_to_bucket_position.get(dataset_index)
            if bucket_index is None or position is None:
                continue

            current_weight = float(self._active_bucket_weights[bucket_index][position].item())
            if current_weight <= 0:
                continue

            self._active_bucket_weights[bucket_index][position] = 0.0
            self._active_bucket_counts[bucket_index] -= 1
            self._active_bucket_weight_sums[bucket_index] = max(
                0.0, self._active_bucket_weight_sums[bucket_index] - current_weight
            )

        if self.adaptive_stats is not None and state_dict.get("adaptive_stats") is not None:
            self.adaptive_stats.load_state_dict(state_dict["adaptive_stats"])

        self._last_adaptive_refresh_step = int(state_dict.get("last_adaptive_refresh_step", -1))
        cached_adaptive_probs = state_dict.get("cached_adaptive_probs")
        self._cached_adaptive_probs = None if cached_adaptive_probs is None else np.array(cached_adaptive_probs)
