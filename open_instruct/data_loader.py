# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any, Literal

import numpy as np
import ray
import torch
import vllm
from datasets import Dataset
from olmo_core.data import data_loader
from ray.util import queue as ray_queue
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from open_instruct import data_types, padding_free_collator, utils
from open_instruct.data_types import EnvConfig, EnvConfigEntry
from open_instruct.dataset_transformation import (
    ENV_CONFIG_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    TOOLS_COLUMN_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.environments.tools.utils import EnvStatistics
from open_instruct.model_utils import Batch
from open_instruct.rl_utils import (
    PackedSequences,
    pack_sequences,
    save_filtered_rollouts_to_disk,
    save_rollout_metadata,
    save_rollouts_to_disk,
)
from open_instruct.rubrics import RubricManager
from open_instruct.utils import combine_reward_metrics, repeat_each

logger = logging.getLogger(__name__)

DATA_PREP_ACTOR_NAME = "data_prep_singleton"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default %s", name, value, default)
        return default


@ray.remote
class ImagePrewarmActor:
    """Best-effort per-node image prewarmer for Docker/Podman-backed sandboxes."""

    def __init__(self, concurrency: int | None = None, max_queued: int | None = None):
        self.concurrency = concurrency or _env_int("SWERL_DOCKER_PREWARM_CONCURRENCY", 4)
        self.max_queued = max_queued or _env_int("SWERL_DOCKER_PREWARM_MAX_QUEUED", 4096)
        self.enabled = _env_flag("SWERL_DOCKER_PREWARM_ENABLED", False)
        self._executor = ThreadPoolExecutor(max_workers=max(1, self.concurrency), thread_name_prefix="ImagePrewarm")
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._inflight: set[str] = set()
        self._stats = {"scheduled": 0, "skipped": 0, "dropped": 0, "ok": 0, "failed": 0}
        logger.info(
            "[ImagePrewarmActor] initialized enabled=%s concurrency=%s max_queued=%s",
            self.enabled,
            self.concurrency,
            self.max_queued,
        )

    def prewarm(self, images: list[str]) -> dict[str, int]:
        if not self.enabled:
            return dict(self._stats)
        scheduled = 0
        with self._lock:
            for image in images:
                if not image or image in self._seen:
                    self._stats["skipped"] += 1
                    continue
                if len(self._inflight) >= self.max_queued:
                    self._stats["dropped"] += 1
                    continue
                self._seen.add(image)
                self._inflight.add(image)
                self._stats["scheduled"] += 1
                scheduled += 1
                future = self._executor.submit(self._pull_image, image)
                future.add_done_callback(lambda fut, image=image: self._finish_pull(image, fut))
        if scheduled:
            logger.info("[ImagePrewarmActor] scheduled %s images; stats=%s", scheduled, self._stats)
        return dict(self._stats)

    def get_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats) | {"seen": len(self._seen), "inflight": len(self._inflight)}

    def _pull_image(self, image: str) -> None:
        import docker as docker_sdk

        client = docker_sdk.from_env(timeout=300)
        try:
            client.images.get(image)
            return
        except docker_sdk.errors.ImageNotFound:
            client.images.pull(image)

    def _finish_pull(self, image: str, future) -> None:
        try:
            future.result()
        except Exception as e:
            logger.warning("[ImagePrewarmActor] failed to prewarm image=%s: %s", image, e)
            ok = False
        else:
            ok = True
        with self._lock:
            self._inflight.discard(image)
            self._stats["ok" if ok else "failed"] += 1


def _images_from_env_config(env_config: EnvConfig) -> list[str]:
    images: list[str] = []
    for entry in env_config.env_configs.values():
        image = entry.kwargs.get("image")
        if isinstance(image, str) and image:
            images.append(image)
    return images


def _prewarm_env_images(env_config: EnvConfig, image_prewarm_actors: list[ray.actor.ActorHandle] | None) -> None:
    if not image_prewarm_actors:
        return
    images = sorted(set(_images_from_env_config(env_config)))
    if not images:
        return
    for actor in image_prewarm_actors:
        actor.prewarm.remote(images)


def concave_length_penalty(x: np.ndarray, k: float, q: float) -> np.ndarray:
    """Box-Cox-style concave length penalty.

        C_{k,q}(x) = [(1 + kx)^(1-q) - 1] / [k(1-q)]

    With C(0) = 0 and C'(0) = 1 regardless of (k, q). For q > 1 the penalty
    saturates at 1 / (k * (q - 1)); for q == 1 the limit is log(1 + kx) / k;
    for q < 1 it grows polynomially as x^(1-q).
    """
    k = float(k)
    q = float(q)
    if k <= 0.0:
        raise ValueError(f"concave_length_penalty requires k > 0, got {k}")
    if abs(1.0 - q) < 1e-8:
        return np.log1p(k * x) / k
    return ((1.0 + k * x) ** (1.0 - q) - 1.0) / (k * (1.0 - q))


def _sample_non_submitting_unmask_idxes(
    submitting_count: int, non_submitting_idxes: list[int], target_fraction: float, rng: np.random.Generator
) -> set[int]:
    """Sample non-submitting rollouts so they are at most target_fraction of the retained batch."""
    if target_fraction <= 0.0 or submitting_count <= 0 or not non_submitting_idxes:
        return set()

    target_count = int(np.floor(target_fraction * submitting_count / (1.0 - target_fraction)))
    target_count = min(target_count, len(non_submitting_idxes))
    if target_count <= 0:
        return set()

    sampled = rng.choice(np.array(non_submitting_idxes, dtype=np.int64), size=target_count, replace=False)
    return set(int(i) for i in sampled)


def to_device(batch: dict[str, Any], device: torch.device | None) -> dict[str, Any]:
    """Move all tensors in a batch dictionary to the specified device.

    Args:
        batch: Dictionary potentially containing torch.Tensor values.
        device: Target device. If None, tensors are not moved.

    Returns:
        Dictionary with the same keys, but tensor values moved to the target device.
    """
    if device is None:
        return batch
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class HFDataLoader(data_loader.DataLoaderBase):
    """A DataLoader that wraps a HuggingFace Dataset for use with olmo_core's Trainer.

    This class implements the DataLoaderBase interface, providing iteration over
    a HuggingFace Dataset with support for sharding across distributed workers,
    shuffling, checkpointing, and optional collation.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        seed: int,
        dp_rank: int,
        dp_world_size: int,
        work_dir: str,
        automatic_reshuffle: bool = False,
        collator: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
        device: torch.device | None = None,
        drop_last: bool = True,
        fs_local_rank: int | None = None,
        max_seq_length: int = 1,
    ) -> None:
        """Initialize the HFDataLoader.

        Args:
            dataset: The HuggingFace Dataset to load data from. Must have an 'index' column.
            batch_size: The global batch size (in sequences).
            seed: Random seed for shuffling.
            dp_rank: The rank of the current process in the distributed setup.
            dp_world_size: Total number of data-parallel processes in the distributed setup.
            work_dir: Working directory for the data loader (required by DataLoaderBase).
            automatic_reshuffle: If True, automatically reshuffle at epoch boundaries.
            collator: Optional collation function for batching examples. If None, batches will be
                dictionaries of the form `{'examples': [example_1, example_2, ...]}`.
            device: Device to move tensors to.
            drop_last: If True, drop the last incomplete batch. If False, pad the last batch
                with repeated indices to fill a complete batch.
            fs_local_rank: File system local rank. Defaults to dp_rank when None.
            max_seq_length: Maximum sequence length. Used to report global_batch_size in tokens
                to the trainer for batch-size validation.

        Note:
            The dataset must have an 'index' column for tracking samples across epochs.
            This is automatically added by get_cached_dataset_tulu(). For custom datasets,
            add it with: dataset.add_column('index', range(len(dataset)))
        """
        # OLMo-core's trainer expects global_batch_size in tokens, not sequences.
        super().__init__(
            work_dir=work_dir,
            global_batch_size=batch_size * max_seq_length,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank if fs_local_rank is not None else dp_rank,
        )

        if "index" not in dataset.column_names:
            raise ValueError(
                "Dataset must have an 'index' column. This is typically added by get_cached_dataset_tulu(). "
                "If using a custom dataset, add it with: dataset.add_column('index', range(len(dataset)))"
            )
        self._full_dataset = dataset
        self.seed = seed
        self._batch_size = batch_size
        if batch_size < dp_world_size:
            raise ValueError(
                f"Global batch size ({batch_size}) must be >= world size ({dp_world_size}). "
                f"Each rank needs at least one example per batch."
            )
        if batch_size % dp_world_size != 0:
            logger.warning(
                f"Global batch size {batch_size} is not divisible by world size {dp_world_size}. "
                f"The effective global batch size will be {batch_size // dp_world_size * dp_world_size}."
            )
        self._per_rank_batch_size = batch_size // dp_world_size
        self._collator = collator if collator is not None else (lambda x: {"examples": x})
        self._automatic_reshuffle = automatic_reshuffle
        self._drop_last = drop_last
        self._excluded_indices: set[int] = set()
        self._overflow: list[dict[str, Any]] = []
        self._precomputed_batch_sizes: list[int] | None = None
        self._num_padding_batches: int = 0
        self._epoch: int = 0
        self._current_iter: Iterator[dict[str, Any]] | None = None
        self._device = device

        self._reshard(epoch=0)

    def __next__(self) -> dict[str, Any]:
        if self._current_iter is None:
            self._current_iter = iter(self)
        try:
            return next(self._current_iter)
        except StopIteration:
            self._current_iter = None
            if self._automatic_reshuffle:
                self.reshuffle()
                if self.effective_size == 0:
                    raise RuntimeError("All dataset examples have been excluded. Cannot continue iteration.") from None
                self._current_iter = iter(self)
                return next(self._current_iter)
            self._epoch += 1
            self.batches_processed = 0
            raise

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        """Return an iterable over all batches in the epoch."""
        # World-aware packing: batch boundaries were precomputed by
        # _reshard_with_packing so that every rank has the same number of
        # batches. Each entry in _precomputed_batch_sizes is the number of
        # examples in that batch (variable due to packing).
        if self._precomputed_batch_sizes is not None:
            num_real = len(self._precomputed_batch_sizes) - self._num_padding_batches
            offset = 0
            for batch_idx, batch_size in enumerate(self._precomputed_batch_sizes):
                if batch_idx < self.batches_processed:
                    offset += batch_size
                    continue
                examples = []
                for i in range(offset, offset + batch_size):
                    example = self.dataset[i]
                    examples.append(example | {"prompt_id": f"{self._epoch}_{example['index']}"})
                batch = to_device(self._collator(examples), self._device) | {"is_padding": batch_idx >= num_real}
                offset += batch_size
                yield batch
            return

        start_example = self.batches_processed * self._per_rank_batch_size
        batch_examples: list[dict[str, Any]] = []
        for i in range(start_example, self.effective_size):
            example = self.dataset[i]
            batch_examples.append(example | {"prompt_id": f"{self._epoch}_{example['index']}"})
            if len(batch_examples) == self._per_rank_batch_size:
                all_examples = self._overflow + batch_examples
                batch = to_device(self._collator(all_examples), self._device)
                self._overflow = all_examples[len(batch["index"]) :]
                yield batch
                batch_examples = []
        while self._overflow:
            batch = to_device(self._collator(self._overflow), self._device)
            assert len(batch["index"]) > 0, (
                f"Collator consumed 0 examples from {len(self._overflow)} overflow examples"
            )
            self._overflow = self._overflow[len(batch["index"]) :]
            yield batch

    @property
    def total_batches(self) -> int:
        """Return the total number of batches in an epoch."""
        if self._precomputed_batch_sizes is not None:
            return len(self._precomputed_batch_sizes)
        return self.effective_size // self._per_rank_batch_size

    def state_dict(self) -> dict[str, Any]:
        """Return a state dictionary for checkpointing."""
        return {
            "epoch": self._epoch,
            "batches_processed": self.batches_processed,
            "excluded_indices": list(self._excluded_indices),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a state dictionary to restore the data loader's state."""
        self._excluded_indices = set(state_dict.get("excluded_indices", []))
        # Set epoch to one less than target since reshuffle() increments it
        self._epoch = state_dict["epoch"] - 1
        self.reshuffle()
        assert self._epoch == state_dict["epoch"]
        self.batches_processed = state_dict["batches_processed"]
        self._current_iter = None

    def exclude_index(self, index: int) -> None:
        """Exclude a dataset index from future iterations.

        Args:
            index: The index to exclude.
        """
        self._excluded_indices.add(index)

    def reshuffle(self, epoch: int | None = None, **kwargs: Any) -> None:
        """Reshuffle and reshard the dataset for a new epoch.

        Args:
            epoch: The epoch number to use for shuffling seed. If None, increments internal counter.
            **kwargs: Additional keyword arguments (unused, for API compatibility).
        """
        self._epoch = self._epoch + 1 if epoch is None else epoch
        self.batches_processed = 0
        self._reshard(self._epoch)

    def _reshard(self, epoch: int) -> None:
        """Reshard the dataset for a given epoch.

        Uses index-based shuffling to avoid copying the dataset.
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        dataset_len = len(self._full_dataset)
        all_indices = torch.randperm(dataset_len, generator=generator).numpy()
        if self._excluded_indices:
            mask = np.isin(all_indices, list(self._excluded_indices), invert=True)
            all_indices = all_indices[mask]

        packing_enabled = hasattr(self._collator, "max_seq_length") and self._collator.max_seq_length is not None
        if packing_enabled:
            self._reshard_with_packing(all_indices)
            return

        self._precomputed_batch_sizes = None
        self._num_padding_batches = 0
        self._overflow = []

        global_size = len(all_indices)
        total_batches = global_size // self._batch_size
        usable_size = total_batches * self._batch_size

        if not self._drop_last and usable_size < global_size:
            remainder = global_size - usable_size
            pad_indices = all_indices[: self._batch_size - remainder]
            all_indices = np.concatenate([all_indices, pad_indices])
            total_batches += 1
            usable_size = total_batches * self._batch_size

        # Distribute examples from global batches to ranks. This is a form of strided sampling where each
        # rank gets a subset of examples from each global batch, ensuring a diverse set of examples.
        rank_indices = all_indices[:usable_size].reshape(total_batches, self._batch_size)
        rank_indices = rank_indices[:, self.dp_rank :: self.dp_world_size].flatten()

        self.effective_size = len(rank_indices)
        self.dataset = self._full_dataset.select(rank_indices.tolist())

    def _reshard_with_packing(self, all_indices: np.ndarray) -> None:
        """Reshard with world-aware packing so all ranks get the same batch count.

        Instead of distributing examples to ranks and letting each rank pack
        independently (which can produce different batch counts due to variable
        overflow), this packs globally first and then distributes packed batches
        round-robin to ranks.
        """
        max_seq_length = self._collator.max_seq_length
        column_names = self._full_dataset.column_names
        subset = self._full_dataset.select(all_indices.tolist())
        if "chosen_input_ids" in column_names:
            lengths = [[len(c), len(r)] for c, r in zip(subset["chosen_input_ids"], subset["rejected_input_ids"])]
        else:
            lengths = [[len(x)] for x in subset["input_ids"]]

        num_streams = len(lengths[0])
        batches: list[list[int]] = []
        current_batch: list[int] = []
        running_totals = [0] * num_streams

        for i in range(len(all_indices)):
            new_totals = [running_totals[s] + lengths[i][s] for s in range(num_streams)]
            would_exceed = len(current_batch) > 0 and any(t > max_seq_length for t in new_totals)
            at_max_samples = len(current_batch) >= self._per_rank_batch_size

            if would_exceed or at_max_samples:
                batches.append(current_batch)
                current_batch = [i]
                running_totals = list(lengths[i])
            else:
                current_batch.append(i)
                running_totals = new_totals

        if current_batch:
            batches.append(current_batch)

        num_batches = len(batches)
        padding_start = num_batches
        if self._drop_last:
            num_batches = (num_batches // self.dp_world_size) * self.dp_world_size
            batches = batches[:num_batches]
        else:
            if (remainder := num_batches % self.dp_world_size) > 0:
                for _ in range(self.dp_world_size - remainder):
                    batches.append(batches[-1])

        rank_global_indices = list(range(self.dp_rank, len(batches), self.dp_world_size))
        self._num_padding_batches = sum(1 for gi in rank_global_indices if gi >= padding_start)

        rank_batches = batches[self.dp_rank :: self.dp_world_size]

        rank_indices: list[int] = []
        self._precomputed_batch_sizes = []
        for batch in rank_batches:
            for pos in batch:
                rank_indices.append(int(all_indices[pos]))
            self._precomputed_batch_sizes.append(len(batch))

        self.effective_size = len(rank_indices)
        self.dataset = self._full_dataset.select(rank_indices)

    def get_mock_batch(self) -> dict[str, Any]:
        """Return a batch with arbitrary data for dry-run testing.

        Used by the trainer to do a dry-run of the
        forward and backward pass before training officially starts.
        """
        num_examples = min(self._per_rank_batch_size, len(self.dataset))
        examples = [self.dataset[i] for i in range(num_examples)]
        return to_device(self._collator(examples), self._device)

    def global_num_tokens_in_batch(self, batch: dict[str, Any]) -> int:
        """Return the total number of tokens in the batch across all ranks.

        Counts tokens from all keys containing 'input_ids' that are torch tensors.

        Args:
            batch: A batch dictionary containing input tensors.

        Returns:
            Total number of tokens across all ranks.

        Raises:
            ValueError: If no input_ids tensors are found in the batch.
        """
        num_tokens = padding_free_collator.get_num_tokens(batch)
        return num_tokens * self.dp_world_size


@dataclass
class VLLMConfig:
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    vllm_attention_backend: str | None = None
    vllm_gdn_prefill_backend: str | None = None
    vllm_sync_backend: str = "nccl"
    vllm_gpu_memory_utilization: float = 0.9
    vllm_enable_prefix_caching: bool = False
    vllm_top_p: float = 1.0
    # Manually override the per-engine concurrency (number of in-flight requests).
    # If None (default), it's auto-computed from KV cache capacity via `get_kv_cache_info()`.
    vllm_inference_batch_size: int | None = None


@dataclass
class StreamingDataLoaderConfig:
    # Data loading/packing
    max_prompt_token_length: int = 256
    response_length: int = 256
    pack_length: int = 512

    # Batching
    async_steps: int = 8
    num_samples_per_prompt_rollout: int = 4
    num_unique_prompts_rollout: int = 16

    # GRPO sampling/filtering
    active_sampling: bool = False
    filter_zero_std_samples: bool = True
    no_resampling_pass_rate: float | None = None
    advantage_normalization_type: Literal["standard", "centered", "maxrl"] = "centered"
    """How to normalize per-prompt rollout rewards into policy advantages.

    ``standard`` is GRPO-style (centered and divided by group std), ``centered``
    is REINFORCE/RLOO-style with a per-prompt baseline, and ``maxrl`` uses the
    MaxRL estimator from arXiv:2602.02710: ``(reward - mean_reward) / mean_reward``.
    Prompt groups with zero mean reward get zero advantages.
    """
    mask_truncated_completions: bool = False
    mask_non_submitting_completions: bool = False
    """Drop rollouts where the env state never reached `done=True`.

    Cleaner alternative to `mask_truncated_completions`: instead of inferring
    truncation from finish_reason / response length, look at whether the model
    actually finished its trajectory (i.e. submitted / declared task done).
    Catches both token-cap'd and env-max-step'd rollouts uniformly. Defaults to
    `False` so existing runs are unchanged.
    """
    mask_non_submitting_completions_percent: float = 0.0
    """Fraction of the retained batch that may be randomly kept from non-submitting rollouts.

    Only applies when `mask_non_submitting_completions=True`. For example, 0.1
    keeps enough randomly selected non-submitting rollouts for them to make up
    at most 10% of the post-filter batch. Defaults to 0.0, preserving the
    existing behavior of dropping all non-submitting rollouts.
    """
    mask_tool_use: bool = True

    # Concave length penalty (Box-Cox style) applied to raw scores before advantage normalization.
    # Penalty is:  alpha * [ (1 + k*x)^(1-q) - 1 ] / [ k*(1-q) ]
    # With defaults below, x is simply the total response token count divided by the normalizer
    # (i.e. response length in kilo-tokens). Per-category weights are exposed for ablations but
    # default to "count all response tokens equally".
    add_concave_length_penalty: bool = False
    concave_length_penalty_alpha: float = 0.015
    concave_length_penalty_k: float = 0.02
    concave_length_penalty_q: float = 2.0
    concave_length_penalty_w_model: float = 1.0
    """Weight on model-emitted response tokens (think + tool-call + final)."""
    concave_length_penalty_w_obs: float = 1.0
    """Weight on tool-output (observation) tokens. Defaults to 1 — all response tokens count equally."""
    concave_length_penalty_w_call: float = 0.0
    """Per-tool-call cost, in token-equivalent units. Defaults to 0 — per-call term disabled."""
    concave_length_penalty_normalizer: float = 1000.0
    """Divisor applied to the weighted sum so the input x is in kilo-token units."""

    # Dataset
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    dataset_mixer_eval_list: list[str] = field(default_factory=list)
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    dataset_mixer_eval_list_splits: list[str] = field(default_factory=lambda: ["test"])
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_max_length_filter_v1"])
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: str | None = None
    dataset_config_eval_hash: str | None = None
    dataset_skip_cache: bool = False
    system_prompt_override_file: str | None = None

    # Generation
    temperature: float = 0.7
    stop_strings: list[str] | None = None
    inflight_updates: bool = True
    inflight_updates_recompute_kv_cache: bool = False
    """When using in-flight weight updates, clear vLLM caches so paused requests recompute KV after resume."""
    eval_response_length: int | None = None
    """Local eval max tokens in GRPO `grpo_fast`. Defaults to `response_length` (see `__post_init__`)."""

    # Reward - R1 style format reward
    apply_r1_style_format_reward: bool = False
    r1_style_format_reward: float = 1.0
    additive_format_reward: bool = False

    # Reward - Verifiable reward
    apply_verifiable_reward: bool = True
    verification_reward: float = 10.0
    remap_verifier: str | None = None

    # Reward aggregation
    reward_aggregator: Literal["last", "sum"] = "last"
    """How to combine per-turn rewards: 'last' (use last turn reward) or 'sum' (sum all rewards across turns)."""

    # LLM judge verifier
    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    llm_judge_max_tokens: int = 2048
    llm_judge_max_context_length: int = 8192
    llm_judge_temperature: float = 1.0
    llm_judge_timeout: int = 60

    # Code verifier
    code_api_url: str = field(
        default_factory=lambda: os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program"
    )
    code_max_execution_time: float = 1.0
    code_pass_rate_reward_threshold: float = 0.0
    code_apply_perf_penalty: bool = False

    # Max length verifier
    max_length_verifier_max_length: int = 32768

    # Non stop penalty
    non_stop_penalty: bool = False
    non_stop_penalty_value: float = 0.0

    # Evolving rubric reward
    apply_evolving_rubric_reward: bool = False
    """Whether to generate and apply evolving rubrics for reward computation.
    When enabled, a rubric buffer is automatically maintained across training steps."""
    max_active_rubrics: int = 5
    """Maximum number of active evolving rubrics per query."""
    cache_evolving_rubric_data_dir: str | None = None
    """Directory to cache evolving rubric generation data for debugging/analysis. If set, rubric data will be saved."""

    # Rollout saving
    save_traces: bool = False
    save_filtered_rollouts: bool = False
    """Save prompt groups dropped by zero-std filtering for debugging active sampling."""
    save_trainer_logprobs: bool = True
    rollouts_save_path: str = "/weka/oe-adapt-default/allennlp/deletable_rollouts/"

    # Computed at post_init
    max_possible_score: float = 1.0

    # Runtime value-model flags, copied from GRPOExperimentConfig.
    use_value_model: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.eval_response_length is None:
            self.eval_response_length = self.response_length

        assert self.pack_length >= self.max_prompt_token_length + self.response_length, (
            "The `pack_length` needs to be greater than the sum of `max_prompt_token_length` and `response_length`!"
        )
        assert self.num_samples_per_prompt_rollout > 0, "Number of samples per prompt must be greater than 0!"
        if self.num_samples_per_prompt_rollout == 1:
            logger.warning("num_samples_per_prompt_rollout is 1. This reduces GRPO to REINFORCE.")

        if self.active_sampling:
            assert self.async_steps > 1, (
                "With active_sampling, you should set async_steps > 1 to account for filtering of the first batch. "
                "Otherwise, your generator only generates only one batch worth of prompts and a single filtered "
                "prompt will cause the trainer to stall waiting for more data  . "
            )
            assert self.filter_zero_std_samples, (
                "filter_zero_std_samples must be True when active_sampling is True. "
                "Active sampling requires filtering to work correctly."
            )
        if self.num_samples_per_prompt_rollout == 1 and self.filter_zero_std_samples:
            raise ValueError(
                "`filter_zero_std_samples` cannot be True when `num_samples_per_prompt_rollout` is 1, "
                "as the reward standard deviation will always be 0, causing all samples to be filtered."
            )
        if self.async_steps < 1:
            raise ValueError("`async_steps` must be greater than 0. Fully synchronous training is not supported.")
        if not 0.0 <= self.mask_non_submitting_completions_percent < 1.0:
            raise ValueError("`mask_non_submitting_completions_percent` must be in [0.0, 1.0).")
        if self.mask_non_submitting_completions_percent > 0.0 and not self.mask_non_submitting_completions:
            raise ValueError(
                "`mask_non_submitting_completions_percent` only applies when "
                "`mask_non_submitting_completions` is True."
            )

        assert (
            self.apply_verifiable_reward
            or self.apply_r1_style_format_reward
            or self.non_stop_penalty
            or self.apply_evolving_rubric_reward
        ), "At least one reward must be applied!"

        if self.stop_strings is None:
            self.stop_strings = []

        self.max_possible_score = 0.0
        if self.apply_verifiable_reward:
            self.max_possible_score += self.verification_reward
        if self.apply_r1_style_format_reward and self.additive_format_reward:
            self.max_possible_score += self.r1_style_format_reward

        if self.save_traces and not self.rollouts_save_path:
            raise ValueError("`rollouts_save_path` must be provided when `save_traces` is True.")
        if self.save_filtered_rollouts and not self.rollouts_save_path:
            raise ValueError("`rollouts_save_path` must be provided when `save_filtered_rollouts` is True.")

    def build_dataloader(
        self,
        tokenizer: PreTrainedTokenizer,
        dp_rank: int,
        fs_local_rank: int,
        num_training_steps: int,
        work_dir: Path | str,
        dp_world_size: int,
    ) -> "StreamingDataLoader":
        """Build a thin wrapper dataloader that pulls from the DataPreparationActor singleton."""
        return StreamingDataLoader(
            tokenizer=tokenizer,
            work_dir=work_dir,
            global_batch_size=self.num_unique_prompts_rollout,
            num_training_steps=num_training_steps,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )


class StreamingDataLoader(data_loader.DataLoaderBase):
    """Thin wrapper dataloader that pulls pre-prepared data from the DataPreparationActor singleton."""

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizer,
        work_dir: Path | str,
        global_batch_size: int,
        num_training_steps: int = 0,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
    ):
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )

        self.data_prep_actor = ray.get_actor(DATA_PREP_ACTOR_NAME)
        self.tokenizer = tokenizer
        self.num_training_steps = num_training_steps
        self.training_step = 0
        self.current_epoch = 0

    @property
    def total_batches(self) -> int | None:
        return self.num_training_steps

    def state_dict(self) -> dict[str, Any]:
        return {"training_step": self.training_step, "current_epoch": self.current_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.training_step = state_dict["training_step"]
        self.current_epoch = state_dict.get("current_epoch", 0)

    def reshuffle(self, epoch: int | None = None, **kwargs):
        if epoch is not None:
            self.current_epoch = epoch

    def get_mock_batch(self) -> dict[str, Any]:
        dummy_qr = torch.tensor([[self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]], dtype=torch.long)
        dummy_attention = torch.tensor([[1, 1]], dtype=torch.long)
        dummy_position_ids = torch.arange(dummy_qr.shape[-1], dtype=torch.long).unsqueeze(0)
        dummy_response_mask = torch.tensor([[0, 1]], dtype=torch.long)
        dummy_advantage = torch.tensor([[0.0, 1.0]], dtype=torch.float)

        batch = data_types.CollatedBatchData(
            query_responses=[dummy_qr],
            attention_masks=[dummy_attention],
            position_ids=[dummy_position_ids],
            advantages=[dummy_advantage],
            response_masks=[dummy_response_mask],
            vllm_logprobs=[torch.zeros_like(dummy_qr, dtype=torch.float)],
            prompt_masks=[torch.tensor([[1, 0]], dtype=torch.long)],
            rollout_sample_ids=[torch.tensor([[0, 0]], dtype=torch.long)],
        )
        return {"batch": batch, "metrics": {}}

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        for step in range(self.training_step, self.num_training_steps):
            wait_start_time = time.perf_counter()
            batch_data = ray.get(self.data_prep_actor.get_data.remote(rank=self.dp_rank, step=step))
            trainer_idle_wait_time = time.perf_counter() - wait_start_time
            batch_data.setdefault("metrics", {})["time/trainer_idle_waiting_for_inference"] = trainer_idle_wait_time
            self.training_step = step + 1
            yield batch_data


def collate_fn(tensors_list: list[torch.Tensor], pad_token_id: int, pin_memory: bool = True) -> torch.Tensor:
    padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors_list, batch_first=True, padding_value=pad_token_id)
    padded_tensor = torch.atleast_2d(padded_tensor)
    if pin_memory and torch.cuda.is_available():
        padded_tensor = padded_tensor.pin_memory()
    return padded_tensor


@dataclass
class BatchStatistics:
    prompt_lengths: list[int]
    response_lengths: list[int]
    filtered_prompts: int
    filtered_prompts_zero: int
    filtered_prompts_solved: int
    filtered_prompts_nonzero: int
    percent_solved_mean: float
    percent_solved_hist: np.ndarray
    no_resampled_prompts: int
    total_prompts: int


def _compute_avg_group_performance(n_solved: int, n_zero: int, n_kept: int, batch_avg_score: float) -> float:
    total_groups = n_solved + n_zero + n_kept
    if total_groups == 0:
        return 0.0
    return float((n_solved + n_kept * batch_avg_score) / total_groups)


def compute_group_advantages(
    scores: np.ndarray, num_samples_per_prompt: int, advantage_normalization_type: str
) -> np.ndarray:
    if num_samples_per_prompt <= 0:
        raise ValueError(f"num_samples_per_prompt must be positive, got {num_samples_per_prompt}.")
    scores = np.asarray(scores)
    if scores.size % num_samples_per_prompt != 0:
        raise ValueError(
            f"Number of scores ({scores.size}) must be divisible by num_samples_per_prompt ({num_samples_per_prompt})."
        )

    score_dtype = np.result_type(scores.dtype, np.float32)
    scores_per_prompt = scores.astype(score_dtype, copy=False).reshape(-1, num_samples_per_prompt)
    mean_grouped_rewards = scores_per_prompt.mean(axis=-1, keepdims=True)

    if advantage_normalization_type == "standard":
        std_grouped_rewards = scores_per_prompt.std(axis=-1, keepdims=True)
        advantages = (scores_per_prompt - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
    elif advantage_normalization_type == "centered":
        advantages = scores_per_prompt - mean_grouped_rewards
    elif advantage_normalization_type == "maxrl":
        advantages = np.zeros_like(scores_per_prompt, dtype=score_dtype)
        np.divide(
            scores_per_prompt - mean_grouped_rewards,
            mean_grouped_rewards,
            out=advantages,
            where=mean_grouped_rewards > 0.0,
        )
    else:
        raise ValueError(f"Invalid advantage normalization type: {advantage_normalization_type}")
    return advantages.reshape(-1)


def single_example_collator(examples: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(examples) == 1, f"Expected 1 example, got {len(examples)}"
    example = examples[0]
    return example | {"index": torch.tensor([example["index"]])}


def _merge_env_config(base_env_config: EnvConfig, sample_env_config: dict[str, Any] | None) -> EnvConfig:
    """Merge base and sample env config into canonical payload.
    Sample env_config overrides any base env_configs with the same name.
    """
    if sample_env_config is None:
        return base_env_config

    max_steps = sample_env_config.get("max_steps", base_env_config.max_steps)

    merged = dict(base_env_config.env_configs)
    sample_entries = list(sample_env_config.get("env_configs", []))
    # Backward compatibility: support flat per-sample env_config shape
    # like {"env_name": "swerl_sandbox", "image": "...", ...}.
    if not sample_entries and "env_name" in sample_env_config:
        sample_entries = [sample_env_config]

    for sample_entry in sample_entries:
        env_name = sample_entry["env_name"]
        base = merged.get(env_name)
        is_text_env = sample_entry.get("is_text_env", base.is_text_env if base else False)
        extra = {k: v for k, v in sample_entry.items() if k not in ("env_name", "is_text_env")}
        merged_kwargs = {**(base.kwargs if base else {}), **extra}
        merged[env_name] = EnvConfigEntry(env_name=env_name, is_text_env=is_text_env, kwargs=merged_kwargs)

    return EnvConfig(max_steps=max_steps, env_configs=merged)


def _aggregate_env_metrics(rollout_states: list[dict]) -> dict[str, float]:
    env_metrics: dict[str, dict[str, list[float]]] = {}
    for rs in rollout_states:
        info = rs.get("info", {})
        multi_env_metrics = info.get("env_metrics")
        if multi_env_metrics is not None:
            for ename, per_env in multi_env_metrics.items():
                bucket = env_metrics.setdefault(ename, {})
                for k, v in per_env.items():
                    bucket.setdefault(k, []).append(float(v))
            continue

        ename = info.get("env_name", "unknown")
        bucket = env_metrics.setdefault(ename, {})
        for k, v in info.items():
            if k != "env_name" and isinstance(v, (int, float)):
                bucket.setdefault(k, []).append(float(v))

    return {
        f"env/{ename}/{k}": float(np.mean(vals))
        for ename, metrics in env_metrics.items()
        for k, vals in metrics.items()
    }


def add_prompt_to_generator(
    example: dict[str, Any],
    epoch_number: int,
    param_prompt_Q: ray_queue.Queue,
    generation_config,
    is_eval: bool,
    base_env_config: EnvConfig,
    ground_truth_overrides: dict[int, Any] | None = None,
    image_prewarm_actors: list[ray.actor.ActorHandle] | None = None,
) -> None:
    index = int(example["index"])

    sample_env_config = example.get(ENV_CONFIG_KEY)
    env_config = _merge_env_config(base_env_config, sample_env_config)
    _prewarm_env_images(env_config, image_prewarm_actors)

    ground_truth = ground_truth_overrides.get(index) if ground_truth_overrides else None

    param_prompt_Q.put(
        data_types.PromptRequest(
            prompt=example[INPUT_IDS_PROMPT_KEY],
            generation_config=generation_config,
            index=index,
            prompt_id=f"{epoch_number}_{index}",
            is_eval=is_eval,
            active_tools=example.get(TOOLS_COLUMN_KEY),
            env_config=env_config,
            ground_truth=ground_truth,
        )
    )


def accumulate_inference_batches(
    inference_results_Q: ray_queue.Queue,
    generation_config: vllm.SamplingParams,
    num_prompts: int,
    model_dims: utils.ModelDims,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    base_env_config: EnvConfig,
    actor_manager=None,
    timeout: float | None = None,
    active_sampling: bool = False,
    filter_zero_std_samples: bool = False,
    replenish_prompts: bool = False,
    no_resampling_pass_rate: float | None = None,
    iter_dataloader: HFDataLoader | None = None,
    param_prompt_Q: ray_queue.Queue | None = None,
    training_step: int | None = None,
    verbose: bool = False,
    max_possible_score: float = 1.0,
    requeue_on_timeout: bool = True,
    max_result_age_steps: int | None = None,
    ground_truth_overrides: dict[int, Any] | None = None,
    image_prewarm_actors: list[ray.actor.ActorHandle] | None = None,
    save_filtered_rollouts: bool = False,
    filtered_rollouts_save_path: str | None = None,
    run_name: str | None = None,
) -> (
    tuple[data_types.GenerationResult, Batch, dict, BatchStatistics]
    | tuple[data_types.ShutdownSentinel | None, None, None, None]
):
    if no_resampling_pass_rate is not None:
        assert iter_dataloader is not None, "no_resampling requires the iter_dataloader passed"

    if replenish_prompts:
        assert param_prompt_Q is not None and iter_dataloader is not None and dataset is not None, (
            "replenish_prompts requires param_prompt_Q and iter_dataloader and dataset"
        )

    results = []
    all_queries = []
    all_ground_truths = []
    all_datasets = []
    all_raw_queries = []
    all_decoded_responses = []
    all_reward_metrics = []
    all_active_tools = []
    all_response_model_steps = []
    all_scores = []
    all_indices = []
    all_percent_solved = []
    all_model_steps = []
    total_filtered_prompts = 0
    filtered_prompt_zero = 0
    filtered_prompt_solved = 0
    filtered_prompt_nonzero = 0
    total_no_resampled = 0
    stale_results_dropped = 0
    progress_bar = tqdm(
        total=num_prompts,
        desc=f"Accumulating Responses and Rewarding {num_prompts} prompts",
        bar_format="{l_bar}{bar}{r_bar}\n",
        disable=not verbose,
    )
    logger.info(
        f"[accumulate_inference_batches] Starting to accumulate {num_prompts} prompts, training_step={training_step}"
    )
    num_prompts_sampled = 0
    collected_results = []  # Track results for potential requeue on timeout
    while num_prompts_sampled < num_prompts:
        logger.info(
            f"[accumulate_inference_batches] Waiting for result {num_prompts_sampled + 1}/{num_prompts} from inference_results_Q"
        )
        try:
            result = inference_results_Q.get(timeout=timeout)
        except Empty:
            if requeue_on_timeout and collected_results:
                logger.info(
                    f"[accumulate_inference_batches] Timeout with {len(collected_results)}/{num_prompts} results, requeuing"
                )
                for r in collected_results:
                    inference_results_Q.put(r)
            raise
        logger.info(
            f"[accumulate_inference_batches] Got result {num_prompts_sampled + 1}/{num_prompts}, type: {type(result).__name__}"
        )

        if isinstance(result, data_types.ShutdownSentinel):
            return result, None, None, None

        if (
            max_result_age_steps is not None
            and training_step is not None
            and result.model_step is not None
            and training_step - result.model_step > max_result_age_steps
        ):
            lag = training_step - result.model_step
            stale_results_dropped += 1
            logger.warning(
                "[accumulate_inference_batches] Dropping stale result for index=%s prompt_id=%s "
                "at training_step=%s: model_step=%s lag=%s max_result_age_steps=%s",
                result.index,
                result.prompt_id,
                training_step,
                result.model_step,
                lag,
                max_result_age_steps,
            )
            if replenish_prompts:
                assert iter_dataloader is not None
                assert param_prompt_Q is not None
                example = next(iter_dataloader)
                add_prompt_to_generator(
                    example,
                    iter_dataloader._epoch,
                    param_prompt_Q,
                    generation_config,
                    is_eval=False,
                    base_env_config=base_env_config,
                    ground_truth_overrides=ground_truth_overrides,
                    image_prewarm_actors=image_prewarm_actors,
                )
            continue

        collected_results.append(result)

        assert len(result.responses) == generation_config.n, (
            f"Mismatch: individual prompt result has {len(result.responses)} responses "
            f"but expected {generation_config.n} samples per prompt. "
            f"Index: {result.index}, Prompt ID: {result.prompt_id}"
        )

        example = dataset[result.index]
        query = example[INPUT_IDS_PROMPT_KEY]
        ground_truth = example[GROUND_TRUTHS_KEY]
        dataset_name = example[VERIFIER_SOURCE_KEY]
        raw_query = example[RAW_PROMPT_KEY]
        sample_active_tools = example.get(TOOLS_COLUMN_KEY)

        if replenish_prompts:
            assert iter_dataloader is not None
            assert param_prompt_Q is not None
            example = next(iter_dataloader)
            add_prompt_to_generator(
                example,
                iter_dataloader._epoch,
                param_prompt_Q,
                generation_config,
                is_eval=False,
                base_env_config=base_env_config,
                ground_truth_overrides=ground_truth_overrides,
                image_prewarm_actors=image_prewarm_actors,
            )

        for i in range(len(result.finish_reasons)):
            if result.finish_reasons[i] == "stop" and len(result.responses[i]) == 0:
                result.responses[i].append(tokenizer.eos_token_id)
                result.masks[i].append(1)
                result.logprobs[i].append(float("nan"))

        decoded_responses = tokenizer.batch_decode(result.responses, skip_special_tokens=False)

        k_queries = repeat_each([query], generation_config.n)
        k_ground_truths = repeat_each([ground_truth], generation_config.n)
        k_datasets = repeat_each([dataset_name], generation_config.n)
        k_raw_queries = repeat_each([raw_query], generation_config.n)
        k_active_tools = repeat_each([sample_active_tools], generation_config.n)
        k_indices = repeat_each([result.index], generation_config.n)
        k_model_steps = (
            result.model_steps
            if result.model_steps is not None
            else repeat_each([result.model_step], generation_config.n)
        )

        percent_solved = np.mean(result.reward_scores).item() / max_possible_score
        if no_resampling_pass_rate is not None and percent_solved >= no_resampling_pass_rate:
            assert iter_dataloader is not None
            iter_dataloader.exclude_index(result.index)
            total_no_resampled += 1
            logging.debug(
                f"[Data Preparation Thread] Prompt solved at {percent_solved}, will be excluded from resampling, total no resampled: {total_no_resampled}"
            )

        if filter_zero_std_samples and np.std(result.reward_scores) == 0:
            if not active_sampling:
                num_prompts_sampled += 1
                progress_bar.update(1)

            if save_filtered_rollouts and filtered_rollouts_save_path and run_name is not None:
                filtered_batch = Batch(
                    queries=k_queries,
                    ground_truths=k_ground_truths,
                    datasets=k_datasets,
                    raw_queries=k_raw_queries,
                    decoded_responses=decoded_responses,
                    indices=k_indices,
                    scores=result.reward_scores,
                    active_tools=k_active_tools if k_active_tools else None,
                )
                save_filtered_rollouts_to_disk(
                    filtered_rollouts_save_path,
                    run_name,
                    training_step if training_step is not None else -1,
                    "zero_std_reward",
                    filtered_batch,
                    result,
                    generation_config.n,
                    total_filtered_prompts * generation_config.n,
                )

            total_filtered_prompts += 1
            if result.reward_scores[0] == 0:
                filtered_prompt_zero += 1
            elif result.reward_scores[0] >= max_possible_score - 1e-8:
                filtered_prompt_solved += 1
            else:
                filtered_prompt_nonzero += 1
            logging.debug(
                f"[Data Preparation Thread] Filtered prompt with reward std 0, total filtered {total_filtered_prompts}"
            )
            continue
        else:
            num_prompts_sampled += 1
            progress_bar.update(1)

        results.append(result)
        all_queries.extend(k_queries)
        all_ground_truths.extend(k_ground_truths)
        all_datasets.extend(k_datasets)
        all_raw_queries.extend(k_raw_queries)
        all_active_tools.extend(k_active_tools)
        all_response_model_steps.extend(k_model_steps)
        all_indices.extend(k_indices)
        all_decoded_responses.extend(decoded_responses)
        all_scores.extend(result.reward_scores)
        all_reward_metrics.append(result.reward_metrics)
        all_percent_solved.append(percent_solved)
        if result.model_step is not None:
            all_model_steps.append(result.model_step)

    if len(results) == 0:
        logging.warning(
            "[Data Preparation Thread] All prompts were filtered during accumulation. "
            f"Filtered: {total_filtered_prompts} (zero std: {filtered_prompt_zero}, "
            f"solved: {filtered_prompt_solved}, nonzero: {filtered_prompt_nonzero})"
        )
        return None, None, None, None

    combined_responses = []
    combined_finish_reasons = []
    combined_masks = []
    combined_num_calls = []
    combined_timeouts = []
    combined_tool_errors = []
    combined_tool_outputs = []
    combined_tool_runtimes = []
    combined_tool_calleds = []
    combined_tool_call_stats = []
    combined_rollout_states = []
    combined_logprobs = []

    earliest_start_time = float("inf")
    prompt_lengths = []
    response_lengths = []

    total_prompt_tokens = 0
    total_response_tokens = 0
    max_generation_time = 0

    for i, result in enumerate(results):
        combined_responses.extend(result.responses)
        combined_finish_reasons.extend(result.finish_reasons)
        combined_masks.extend(result.masks)
        combined_num_calls.extend(result.request_info.num_calls)
        combined_timeouts.extend(result.request_info.timeouts)
        combined_tool_errors.extend(result.request_info.tool_errors)
        combined_tool_outputs.extend(result.request_info.tool_outputs)
        combined_tool_runtimes.extend(result.request_info.tool_runtimes)
        combined_tool_calleds.extend(result.request_info.tool_calleds)
        combined_tool_call_stats.extend(result.request_info.tool_call_stats)
        combined_rollout_states.extend(result.request_info.rollout_states)

        combined_logprobs.extend(result.logprobs)

        earliest_start_time = min(earliest_start_time, result.start_time)

        prompt_lengths.append(len(all_queries[i * generation_config.n]))

        for response in result.responses:
            response_lengths.append(len(response))

        total_prompt_tokens += result.token_statistics.num_prompt_tokens
        total_response_tokens += result.token_statistics.num_response_tokens
        max_generation_time = max(max_generation_time, result.token_statistics.generation_time)

    accumulated_stats = data_types.TokenStatistics(
        num_prompt_tokens=total_prompt_tokens,
        num_response_tokens=total_response_tokens,
        generation_time=max_generation_time,
        earliest_start_time=earliest_start_time,
    )

    combined_request_info = data_types.RequestInfo(
        num_calls=combined_num_calls,
        timeouts=combined_timeouts,
        tool_errors=combined_tool_errors,
        tool_outputs=combined_tool_outputs,
        tool_runtimes=combined_tool_runtimes,
        tool_calleds=combined_tool_calleds,
        tool_call_stats=combined_tool_call_stats,
        rollout_states=combined_rollout_states,
    )

    combined_result = data_types.GenerationResult(
        responses=combined_responses,
        finish_reasons=combined_finish_reasons,
        masks=combined_masks,
        request_info=combined_request_info,
        index=None,
        prompt_id=results[0].prompt_id,
        token_statistics=accumulated_stats,
        logprobs=combined_logprobs,
        model_steps=all_response_model_steps,
    )

    if actor_manager is not None:
        ray.get(actor_manager.report_token_statistics.remote(accumulated_stats))

    batch = Batch(
        queries=all_queries,
        ground_truths=all_ground_truths,
        datasets=all_datasets,
        raw_queries=all_raw_queries,
        decoded_responses=all_decoded_responses,
        indices=all_indices,
        scores=all_scores,
        active_tools=all_active_tools if all_active_tools else None,
    )

    combined_reward_metrics = combine_reward_metrics(all_reward_metrics)
    combined_reward_metrics["stale_results_dropped"] = float(stale_results_dropped)
    if all_model_steps:
        model_steps_array = np.array(all_model_steps, dtype=float)
        combined_reward_metrics["model_step_min"] = float(model_steps_array.min())
        combined_reward_metrics["model_step_max"] = float(model_steps_array.max())
        combined_reward_metrics["model_step_mean"] = float(model_steps_array.mean())
    percent_solved_mean = np.mean(all_percent_solved) if all_percent_solved else 0.0

    batch_stats = BatchStatistics(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        filtered_prompts=total_filtered_prompts,
        filtered_prompts_zero=filtered_prompt_zero,
        filtered_prompts_solved=filtered_prompt_solved,
        filtered_prompts_nonzero=filtered_prompt_nonzero,
        percent_solved_mean=percent_solved_mean,
        percent_solved_hist=np.array(all_percent_solved),
        no_resampled_prompts=total_no_resampled,
        total_prompts=len(results),
    )
    return combined_result, batch, combined_reward_metrics, batch_stats


def populate_value_model_fields(packed_sequences: PackedSequences, scores: np.ndarray) -> None:
    """Populate per-token PPO rewards from per-rollout scalar scores."""
    assert packed_sequences.dones is not None
    lookup_rewards = np.zeros(len(scores) + 1, dtype=np.float32)
    lookup_rewards[1:] = scores.astype(np.float32)
    packed_sequences.rewards = [
        torch.tensor(lookup_rewards[packed_dones.cpu().numpy().astype(np.int64)], dtype=torch.float32)
        for packed_dones in packed_sequences.dones
    ]


def prepare_collated_data_for_workers(
    packed_sequences: PackedSequences,
    dp_world_size: int,
    per_device_train_batch_size: int,
    pad_token_id: int,
    pin_memory: bool = True,
) -> list[data_types.CollatedBatchData]:
    """Distributes and collates packed sequences for distributed training.

    Splits packed sequences across workers, randomly shuffles each worker's data,
    and collates into micro-batches for training.

    Args:
        packed_sequences: Packed training sequences containing query responses,
            attention masks, position IDs, advantages, response masks,
            and vllm logprobs.
        dp_world_size: Number of distributed workers.
        per_device_train_batch_size: Batch size for each device's micro-batch.
        pad_token_id: Token ID used for padding sequences.
        pin_memory: Whether to pin memory for faster data transfer to GPU.

    Returns:
        List of CollatedBatchData, one per worker, each containing collated tensors
        for query_responses, attention_masks, position_ids,
        advantages, response_masks, and vllm_logprobs.
    """
    total_sequences = len(packed_sequences.query_responses)
    if total_sequences % dp_world_size != 0:
        new_total = (total_sequences // dp_world_size) * dp_world_size
        logger.warning(
            f"Total packed sequences ({total_sequences}) is not evenly divisible by dp_world_size ({dp_world_size}). "
            f"Truncating to {new_total} sequences (dropping {total_sequences - new_total})."
        )
    B = total_sequences // dp_world_size
    collated_data = []
    assert packed_sequences.position_ids is not None
    assert packed_sequences.advantages is not None
    assert packed_sequences.vllm_logprobs is not None
    has_rewards = packed_sequences.rewards is not None
    has_dones = packed_sequences.dones is not None
    for i in range(dp_world_size):
        per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
        per_device_packed_attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
        per_device_packed_position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
        per_device_packed_advantages = packed_sequences.advantages[B * i : B * (i + 1)]
        per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]
        per_device_packed_vllm_logprobs = packed_sequences.vllm_logprobs[B * i : B * (i + 1)]
        per_device_packed_rewards = packed_sequences.rewards[B * i : B * (i + 1)] if has_rewards else None
        per_device_packed_dones = packed_sequences.dones[B * i : B * (i + 1)] if has_dones else None
        if packed_sequences.prompt_masks is None:
            per_device_packed_prompt_masks = [
                torch.zeros_like(t, dtype=torch.long) for t in per_device_packed_query_responses
            ]
        else:
            per_device_packed_prompt_masks = packed_sequences.prompt_masks[B * i : B * (i + 1)]
        if packed_sequences.rollout_sample_ids is None:
            per_device_packed_rollout_sample_ids = [
                torch.full_like(t, -1, dtype=torch.long) for t in per_device_packed_query_responses
            ]
        else:
            per_device_packed_rollout_sample_ids = packed_sequences.rollout_sample_ids[B * i : B * (i + 1)]
        if packed_sequences.model_steps is None:
            per_device_packed_model_steps = [
                torch.full_like(t, -1, dtype=torch.long) for t in per_device_packed_query_responses
            ]
        else:
            per_device_packed_model_steps = packed_sequences.model_steps[B * i : B * (i + 1)]

        # Shuffle the batch and collate the data
        b_inds = np.random.permutation(len(per_device_packed_query_responses))
        collated_query_responses = []
        collated_attention_masks = []
        collated_position_ids = []
        collated_response_masks = []
        collated_prompt_masks = []
        collated_rollout_sample_ids = []
        collated_model_steps = []
        collated_advantages = []
        collated_vllm_logprobs = []
        collated_rewards: list[torch.Tensor] | None = [] if has_rewards else None
        collated_dones: list[torch.Tensor] | None = [] if has_dones else None
        for j in range(0, len(per_device_packed_query_responses), per_device_train_batch_size):
            micro_range = b_inds[j : j + per_device_train_batch_size]
            collated_query_responses.append(
                collate_fn([per_device_packed_query_responses[idx] for idx in micro_range], pad_token_id, pin_memory)
            )
            collated_attention_masks.append(
                collate_fn([per_device_packed_attention_masks[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_position_ids.append(
                collate_fn([per_device_packed_position_ids[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_response_masks.append(
                collate_fn([per_device_packed_response_masks[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_prompt_masks.append(
                collate_fn([per_device_packed_prompt_masks[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_rollout_sample_ids.append(
                collate_fn([per_device_packed_rollout_sample_ids[idx] for idx in micro_range], -1, pin_memory)
            )
            collated_model_steps.append(
                collate_fn([per_device_packed_model_steps[idx] for idx in micro_range], -1, pin_memory)
            )
            collated_advantages.append(
                collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_vllm_logprobs.append(
                collate_fn([per_device_packed_vllm_logprobs[idx] for idx in micro_range], 0, pin_memory)
            )
            if collated_rewards is not None:
                assert per_device_packed_rewards is not None
                collated_rewards.append(
                    collate_fn([per_device_packed_rewards[idx] for idx in micro_range], 0, pin_memory)
                )
            if collated_dones is not None:
                assert per_device_packed_dones is not None
                collated_dones.append(collate_fn([per_device_packed_dones[idx] for idx in micro_range], 0, pin_memory))
        collated_data.append(
            data_types.CollatedBatchData(
                query_responses=collated_query_responses,
                attention_masks=collated_attention_masks,
                position_ids=collated_position_ids,
                advantages=collated_advantages,
                response_masks=collated_response_masks,
                vllm_logprobs=collated_vllm_logprobs,
                rewards=collated_rewards,
                dones=collated_dones,
                prompt_masks=collated_prompt_masks,
                rollout_sample_ids=collated_rollout_sample_ids,
                model_steps=collated_model_steps,
            )
        )
    return collated_data


@ray.remote
class DataPreparationActor:
    """Ray actor singleton that handles centralized data preparation for all ranks.

    This actor runs a background thread that continuously prepares training data,
    ensuring all ranks receive the same number of micro-batches (preventing deadlock
    from uneven filtering).
    """

    def __init__(
        self,
        dataset: Dataset,
        inference_results_Q: ray_queue.Queue,
        param_prompt_Q: ray_queue.Queue,
        tokenizer: PreTrainedTokenizer,
        config: StreamingDataLoaderConfig,
        generation_config,
        num_training_steps: int,
        seed: int,
        per_device_train_batch_size: int,
        global_batch_size: int,
        dp_world_size: int,
        max_possible_score: float,
        actor_manager,
        model_dims: utils.ModelDims,
        verbose: bool,
        work_dir: str,
        tool_names: list[str],
        run_name: str,
        model_name: str | None,
        base_env_config: EnvConfig,
        image_prewarm_actors: list[ray.actor.ActorHandle] | None = None,
        initial_state: dict | None = None,
    ):
        self.inference_results_Q = inference_results_Q
        self.param_prompt_Q = param_prompt_Q
        self.tokenizer = tokenizer
        self.config = config
        self.config.max_possible_score = max_possible_score
        self.generation_config = generation_config
        self.seed = seed
        self.num_training_steps = num_training_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.global_batch_size = global_batch_size
        self.dp_world_size = dp_world_size
        self.actor_manager = actor_manager
        self.model_dims = model_dims
        self.verbose = verbose
        self.dataset = dataset
        self.tool_names = tool_names
        self.run_name = run_name
        self.model_name = model_name
        self.base_env_config = base_env_config
        self.image_prewarm_actors = image_prewarm_actors or []

        self.iter_dataloader = HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=seed,
            dp_rank=0,
            dp_world_size=1,
            work_dir=work_dir,
            automatic_reshuffle=True,
            collator=single_example_collator,
        )

        self.prepared_data: dict[int, list[data_types.CollatedBatchData]] = {}
        self.metrics: dict[int, dict] = {}
        self.current_prepared_step = -1
        self._last_consumed_step = -1
        self.lock = threading.Lock()
        self.training_step = 0
        self.total_samples_written = 0
        self.metadata_saved = False
        self._executor: ThreadPoolExecutor | None = None
        self._prep_future = None

        self.rubric_manager: RubricManager | None = None
        self.ground_truth_overrides: dict[int, Any] = {}
        if self.config.apply_evolving_rubric_reward:
            self.rubric_manager = RubricManager(self.config, dataset[GROUND_TRUTHS_KEY])

        if initial_state is not None:
            logger.info("[DataPreparationActor] Given initial state, setting state and starting preparation loop")
            self.set_state(initial_state)
            self.start()

    def start(self):
        if self._prep_future is not None:
            return
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DataPrepActor")
        self._prep_future = self._executor.submit(self._data_preparation_loop)
        logger.info(f"[DataPreparationActor] Started preparation loop from training_step={self.training_step}")

    def _data_preparation_loop(self):
        logger.info("[DataPreparationActor] Starting _data_preparation_loop")

        should_save_rollout_metadata = self.config.save_traces or self.config.save_filtered_rollouts
        if should_save_rollout_metadata and self.config.rollouts_save_path and not self.metadata_saved:
            save_rollout_metadata(self.config.rollouts_save_path, self.run_name, self.model_name)
            self.metadata_saved = True

        num_initial_prompts = self.config.async_steps * self.global_batch_size
        logger.info(f"[DataPreparationActor] Pushing {num_initial_prompts} initial prompts to param_prompt_Q")
        for _ in range(num_initial_prompts):
            add_prompt_to_generator(
                next(self.iter_dataloader),
                self.iter_dataloader._epoch,
                self.param_prompt_Q,
                self.generation_config,
                is_eval=False,
                base_env_config=self.base_env_config,
                ground_truth_overrides=self.ground_truth_overrides,
                image_prewarm_actors=self.image_prewarm_actors,
            )

        for step in range(self.training_step, self.num_training_steps):
            generation_idle_wait_start_time = time.perf_counter()
            while step - self._last_consumed_step > self.config.async_steps:
                logger.info(
                    f"[DataPreparationActor] Step {step}: waiting for step {self._last_consumed_step + self.config.async_steps} to be consumed. Consider increasing training compute."
                )
                time.sleep(0.1)
            generation_idle_wait_time = time.perf_counter() - generation_idle_wait_start_time

            logger.info(
                f"[DataPreparationActor] Step {step}: calling accumulate_inference_batches for {self.global_batch_size} prompts"
            )
            result, batch, reward_metrics, batch_stats = accumulate_inference_batches(
                self.inference_results_Q,
                self.generation_config,
                num_prompts=self.global_batch_size,
                model_dims=self.model_dims,
                tokenizer=self.tokenizer,
                dataset=self.dataset,
                actor_manager=self.actor_manager,
                active_sampling=self.config.active_sampling,
                filter_zero_std_samples=self.config.filter_zero_std_samples,
                replenish_prompts=True,
                no_resampling_pass_rate=self.config.no_resampling_pass_rate,
                iter_dataloader=self.iter_dataloader,
                param_prompt_Q=self.param_prompt_Q,
                training_step=step,
                verbose=self.verbose,
                max_possible_score=self.config.max_possible_score,
                max_result_age_steps=self.config.async_steps,
                base_env_config=self.base_env_config,
                ground_truth_overrides=self.ground_truth_overrides,
                image_prewarm_actors=self.image_prewarm_actors,
                save_filtered_rollouts=self.config.save_filtered_rollouts,
                filtered_rollouts_save_path=os.path.join(self.config.rollouts_save_path, "filtered"),
                run_name=self.run_name,
            )
            logger.info(
                f"[DataPreparationActor] Step {step}: accumulate_inference_batches returned, result type: {type(result).__name__}"
            )

            if isinstance(result, data_types.ShutdownSentinel):
                return

            if result is None:
                empty_data = [
                    data_types.CollatedBatchData(
                        query_responses=[],
                        attention_masks=[],
                        position_ids=[],
                        advantages=[],
                        response_masks=[],
                        vllm_logprobs=[],
                        prompt_masks=[],
                        rollout_sample_ids=[],
                    )
                    for _ in range(self.dp_world_size)
                ]
                with self.lock:
                    self.prepared_data[step] = empty_data
                    self.metrics[step] = {"time/generation_idle_waiting_for_trainer": generation_idle_wait_time}
                    self.current_prepared_step = step
                continue

            assert batch is not None
            assert batch_stats is not None

            if self.rubric_manager and batch.decoded_responses:
                rubric_metrics, new_overrides = self.rubric_manager.run_step(
                    decoded_responses=batch.decoded_responses,
                    ground_truths=batch.ground_truths,
                    indices=batch.indices,
                    step=step,
                )
                reward_metrics.update(rubric_metrics)
                self.ground_truth_overrides.update(new_overrides)

            scores = np.array(batch.scores)
            raw_scores = scores.copy()
            pre_filtering_batch_avg_score = float(raw_scores.mean() / self.config.max_possible_score)

            concave_length_metrics: dict[str, Any] = {}
            if self.config.add_concave_length_penalty and len(scores) > 0:
                response_token_counts = np.array([len(r) for r in result.responses], dtype=np.float64)
                model_token_counts = np.array([float(sum(m)) for m in result.masks], dtype=np.float64)
                tool_output_token_counts = np.maximum(response_token_counts - model_token_counts, 0.0)
                num_calls_arr = np.array(result.request_info.num_calls, dtype=np.float64)
                concave_length_x = (
                    self.config.concave_length_penalty_w_model * model_token_counts
                    + self.config.concave_length_penalty_w_obs * tool_output_token_counts
                    + self.config.concave_length_penalty_w_call * num_calls_arr
                ) / self.config.concave_length_penalty_normalizer
                concave_length_penalties = self.config.concave_length_penalty_alpha * concave_length_penalty(
                    concave_length_x, k=self.config.concave_length_penalty_k, q=self.config.concave_length_penalty_q
                )
                scores = scores - concave_length_penalties

                solved_mask_raw = raw_scores >= (self.config.max_possible_score - 1e-8)
                concave_length_metrics = {
                    "concave_length_penalty/x_mean": float(concave_length_x.mean()),
                    "concave_length_penalty/x_max": float(concave_length_x.max()),
                    "concave_length_penalty/x_min": float(concave_length_x.min()),
                    "concave_length_penalty/x_hist": concave_length_x,
                    "concave_length_penalty/penalty_mean": float(concave_length_penalties.mean()),
                    "concave_length_penalty/penalty_max": float(concave_length_penalties.max()),
                    "concave_length_penalty/penalty_min": float(concave_length_penalties.min()),
                    "concave_length_penalty/penalty_hist": concave_length_penalties,
                    "concave_length_penalty/shaped_score_mean": float(scores.mean()),
                    "concave_length_penalty/raw_score_mean": float(raw_scores.mean()),
                }
                if solved_mask_raw.any():
                    concave_length_metrics["concave_length_penalty/penalty_solved_mean"] = float(
                        concave_length_penalties[solved_mask_raw].mean()
                    )
                    concave_length_metrics["concave_length_penalty/penalty_solved_max"] = float(
                        concave_length_penalties[solved_mask_raw].max()
                    )
                if (~solved_mask_raw).any():
                    concave_length_metrics["concave_length_penalty/penalty_unsolved_mean"] = float(
                        concave_length_penalties[~solved_mask_raw].mean()
                    )

                # Within-group spread of penalties among solved rollouts — measures the
                # gradient signal this penalty adds between multiple successes on the same prompt.
                k_per_group = self.config.num_samples_per_prompt_rollout
                if len(concave_length_penalties) % k_per_group == 0 and k_per_group > 1:
                    pen_per_group = concave_length_penalties.reshape(-1, k_per_group)
                    solved_per_group = solved_mask_raw.reshape(-1, k_per_group)
                    group_gaps = []
                    for grp_pen, grp_solved in zip(pen_per_group, solved_per_group):
                        if int(grp_solved.sum()) >= 2:
                            grp_solved_pens = grp_pen[grp_solved]
                            group_gaps.append(float(grp_solved_pens.max() - grp_solved_pens.min()))
                    if group_gaps:
                        concave_length_metrics["concave_length_penalty/group_success_penalty_gap_mean"] = float(
                            np.mean(group_gaps)
                        )
                        concave_length_metrics["concave_length_penalty/group_success_penalty_gap_max"] = float(
                            max(group_gaps)
                        )
                        concave_length_metrics["concave_length_penalty/groups_with_multi_success"] = len(group_gaps)

            advantages = compute_group_advantages(
                scores=scores,
                num_samples_per_prompt=self.config.num_samples_per_prompt_rollout,
                advantage_normalization_type=self.config.advantage_normalization_type,
            )

            if self.config.save_traces and self.config.rollouts_save_path:
                save_rollouts_to_disk(
                    self.config.rollouts_save_path,
                    self.run_name,
                    step,
                    batch,
                    result,
                    advantages,
                    self.config.num_samples_per_prompt_rollout,
                    self.total_samples_written,
                )
                self.total_samples_written += len(batch.queries)

            rollout_sample_ids = list(range(len(batch.queries)))
            rollout_model_steps = result.model_steps or [result.model_step for _ in result.responses]

            # Truncated-completion stats are computed on the unfiltered batch so
            # they stay meaningful regardless of whether `mask_truncated_completions`
            # is actually dropping the rollouts downstream.
            #
            # A rollout counts as truncated if EITHER:
            #  - finish_reason != "stop" (vLLM hit max_tokens on the final turn), OR
            #  - len(response) >= response_length (budget exhausted inside the tool agent
            #    loop so the loop exited without another vLLM call — the final stored
            #    finish_reason can still be "stop" from an earlier turn). Only checking
            #    finish_reason misses this second path, which is common in multi-turn
            #    tool-using rollouts.
            num_before_filter = len(result.finish_reasons)
            response_length_cap = self.config.response_length

            def _is_truncated(i: int) -> bool:
                return result.finish_reasons[i] != "stop" or len(result.responses[i]) >= response_length_cap

            def _is_non_submitting(i: int) -> bool:
                states = result.request_info.rollout_states
                if i >= len(states):
                    return False
                return not states[i].get("done", True)

            truncated_idxes = [i for i in range(num_before_filter) if _is_truncated(i)]
            num_truncated_completion = len(truncated_idxes)
            truncated_completion_correct_count = (
                int(sum(1 for i in truncated_idxes if raw_scores[i] > 0)) if num_truncated_completion else 0
            )
            truncated_completion_lengths = np.array(
                [len(result.responses[i]) for i in truncated_idxes], dtype=np.int64
            )

            non_submitting_idxes = [i for i in range(num_before_filter) if _is_non_submitting(i)]
            num_non_submitting_completion = len(non_submitting_idxes)
            num_unmasked_non_submitting_completion = 0

            do_mask_filter = self.config.mask_truncated_completions or self.config.mask_non_submitting_completions
            if do_mask_filter:
                truncated_drop_idxes = {
                    i for i in range(num_before_filter) if self.config.mask_truncated_completions and _is_truncated(i)
                }
                non_submitting_drop_idxes = [
                    i
                    for i in range(num_before_filter)
                    if self.config.mask_non_submitting_completions
                    and _is_non_submitting(i)
                    and i not in truncated_drop_idxes
                ]
                unmasked_non_submitting_idxes: set[int] = set()
                if self.config.mask_non_submitting_completions_percent > 0.0:
                    submitting_keep_count = sum(
                        1
                        for i in range(num_before_filter)
                        if i not in truncated_drop_idxes and not _is_non_submitting(i)
                    )
                    unmasked_non_submitting_idxes = _sample_non_submitting_unmask_idxes(
                        submitting_count=submitting_keep_count,
                        non_submitting_idxes=non_submitting_drop_idxes,
                        target_fraction=self.config.mask_non_submitting_completions_percent,
                        rng=np.random.default_rng(self.seed + step),
                    )
                    num_unmasked_non_submitting_completion = len(unmasked_non_submitting_idxes)

                drop_idxes = truncated_drop_idxes | (set(non_submitting_drop_idxes) - unmasked_non_submitting_idxes)
                keep_idxes_list = [i for i in range(num_before_filter) if i not in drop_idxes]
                num_dropped = num_before_filter - len(keep_idxes_list)
                if num_dropped > 0:
                    logger.info(
                        f"[DataPreparationActor] Filtered {num_dropped} rollouts "
                        f"(mask_truncated={self.config.mask_truncated_completions}, "
                        f"mask_non_submitting={self.config.mask_non_submitting_completions}, "
                        f"mask_non_submitting_percent={self.config.mask_non_submitting_completions_percent}, "
                        f"unmasked_non_submitting={num_unmasked_non_submitting_completion}). "
                        f"Retention rate: {len(keep_idxes_list) / num_before_filter:.2%}"
                    )
                scores = scores[keep_idxes_list]
                raw_scores = raw_scores[keep_idxes_list]
                advantages = advantages[keep_idxes_list]
                batch = batch[keep_idxes_list]
                result.responses = [result.responses[i] for i in keep_idxes_list]
                result.masks = [result.masks[i] for i in keep_idxes_list]
                result.finish_reasons = [result.finish_reasons[i] for i in keep_idxes_list]
                rollout_sample_ids = [rollout_sample_ids[i] for i in keep_idxes_list]
                rollout_model_steps = [rollout_model_steps[i] for i in keep_idxes_list]
                assert result.logprobs is not None
                result.logprobs = [result.logprobs[i] for i in keep_idxes_list]

            assert result.logprobs is not None
            packed_sequences = pack_sequences(
                queries=batch.queries,
                responses=result.responses,
                masks=result.masks,
                pack_length=self.config.pack_length,
                pad_token_id=self.tokenizer.pad_token_id,
                vllm_logprobs=result.logprobs,
                rollout_sample_ids=rollout_sample_ids,
                model_steps=rollout_model_steps,
                mask_tool_use=self.config.mask_tool_use,
                min_num_batches=self.dp_world_size,
            )
            lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
            lookup_advantages[1:] = advantages
            packed_advantages = [
                torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
                for packed_mask in packed_sequences.response_masks
            ]
            packed_sequences.advantages = packed_advantages
            if self.config.use_value_model:
                populate_value_model_fields(packed_sequences, scores)

            collated_data = prepare_collated_data_for_workers(
                packed_sequences, self.dp_world_size, self.per_device_train_batch_size, self.tokenizer.pad_token_id
            )

            if len(result.responses) == 0:
                step_metrics = {"time/generation_idle_waiting_for_trainer": generation_idle_wait_time}
            else:
                real_num_responses = len(result.responses)
                expected_num_responses = self.config.num_samples_per_prompt_rollout * self.global_batch_size
                # Use raw_scores (pre length-penalty) for the solved_mask so downstream metrics
                # keep measuring actual task-success rate rather than shaped scores.
                solved_mask = raw_scores >= (self.config.max_possible_score - 1e-8)
                unsolved_num_responses = (~solved_mask).sum()
                sequence_lengths = np.array([len(response) for response in result.responses])
                sequence_length_solved = (
                    np.array([]) if np.all(raw_scores == 0) else np.array(sequence_lengths[solved_mask])
                )
                sequence_length_unsolved = (
                    np.array([]) if np.all(solved_mask) else np.array(sequence_lengths[~solved_mask])
                )
                stop_rate = sum(int(fr == "stop") for fr in result.finish_reasons) / len(result.finish_reasons)

                batch_metrics_dict = asdict(batch_stats)
                batch_metrics_prefixed = {f"batch/{k}": v for k, v in batch_metrics_dict.items()}

                step_metrics = {
                    "time/generation_idle_waiting_for_trainer": generation_idle_wait_time,
                    "scores": raw_scores.mean(),
                    "val/avg_group_performance_pre_filter": _compute_avg_group_performance(
                        n_solved=batch_stats.filtered_prompts_solved,
                        n_zero=batch_stats.filtered_prompts_zero,
                        n_kept=batch_stats.total_prompts,
                        batch_avg_score=pre_filtering_batch_avg_score,
                    ),
                    "val/avg_group_performance_post_filter": _compute_avg_group_performance(
                        n_solved=batch_stats.filtered_prompts_solved,
                        n_zero=batch_stats.filtered_prompts_zero,
                        n_kept=batch_stats.total_prompts,
                        batch_avg_score=float(raw_scores.mean() / self.config.max_possible_score),
                    ),
                    "real_batch_size_ratio": real_num_responses / expected_num_responses,
                    "unsolved_batch_size_ratio": unsolved_num_responses / real_num_responses,
                    "packed_ratio": len(packed_sequences.query_responses) / real_num_responses,
                    "val/solve_rate_hist": batch_stats.percent_solved_hist,
                    "val/total_reward_groups": real_num_responses / self.config.num_samples_per_prompt_rollout,
                    "val/sequence_lengths": sequence_lengths.mean(),
                    "val/sequence_lengths_min": sequence_lengths.min(),
                    "val/sequence_lengths_max": sequence_lengths.max(),
                    "val/sequence_lengths_unsolved": (
                        0 if len(sequence_length_unsolved) == 0 else sequence_length_unsolved.mean()
                    ),
                    "val/sequence_lengths_solved": (
                        0 if len(sequence_length_solved) == 0 else sequence_length_solved.mean()
                    ),
                    "val/sequence_lengths_unsolved_hist": sequence_length_unsolved,
                    "val/sequence_lengths_solved_hist": sequence_length_solved,
                    "val/stop_rate": stop_rate,
                    "val/truncated_completion_count": num_truncated_completion,
                    "val/truncated_completion_fraction": (
                        num_truncated_completion / num_before_filter if num_before_filter else 0.0
                    ),
                    "val/truncated_completion_correct_count": truncated_completion_correct_count,
                    "val/non_submitting_completion_count": num_non_submitting_completion,
                    "val/non_submitting_completion_fraction": (
                        num_non_submitting_completion / num_before_filter if num_before_filter else 0.0
                    ),
                    "val/non_submitting_completion_unmasked_count": num_unmasked_non_submitting_completion,
                    "val/non_submitting_completion_unmasked_fraction": (
                        num_unmasked_non_submitting_completion / real_num_responses if real_num_responses else 0.0
                    ),
                    "val/truncated_completion_length_mean": (
                        float(truncated_completion_lengths.mean()) if num_truncated_completion else 0.0
                    ),
                    "val/truncated_completion_length_max": (
                        int(truncated_completion_lengths.max()) if num_truncated_completion else 0
                    ),
                    "val/advantages_mean": advantages.mean(),
                    "val/advantages_min": advantages.min(),
                    "val/advantages_max": advantages.max(),
                    "val/advantages_hist": advantages,
                    **reward_metrics,
                    **batch_metrics_prefixed,
                    **concave_length_metrics,
                }

                tool_stats = EnvStatistics(tool_names=self.tool_names)
                for rollout_stats in result.request_info.tool_call_stats:
                    tool_stats.add_rollout(rollout_stats)
                step_metrics.update(tool_stats.compute_metrics())

                step_metrics.update(_aggregate_env_metrics(result.request_info.rollout_states))

                assert result.token_statistics is not None
                total_tokens = result.token_statistics.num_prompt_tokens + result.token_statistics.num_response_tokens
                step_metrics["val/actor_tokens_per_second"] = total_tokens / result.token_statistics.generation_time
                step_metrics["time/getting_response"] = result.token_statistics.generation_time

            with self.lock:
                self.prepared_data[step] = collated_data
                self.metrics[step] = step_metrics
                self.current_prepared_step = step

    def get_data(self, rank: int, step: int) -> dict:
        """Called by each rank's StreamingDataLoader. Blocks until data ready."""
        if self._prep_future is None:
            self.start()
        logger.info(
            f"[DataPreparationActor.get_data] rank={rank} requesting step={step}, current_prepared_step={self.current_prepared_step}"
        )
        wait_count = 0
        while True:
            if self._prep_future.done():
                self._prep_future.result()
            with self.lock:
                if step <= self.current_prepared_step:
                    batch_data = self.prepared_data[step][rank]
                    result = {"batch": batch_data, "metrics": self.metrics[step]}
                    self._last_consumed_step = max(self._last_consumed_step, step)
                    self._cleanup_old_steps(step)
                    logger.info(
                        f"[DataPreparationActor.get_data] rank={rank} got data for step={step} after {wait_count} waits"
                    )
                    return result
            wait_count += 1
            if wait_count % 1000 == 0:
                logger.info(
                    f"[DataPreparationActor.get_data] rank={rank} still waiting for step={step}, current_prepared_step={self.current_prepared_step}, wait_count={wait_count}"
                )
            time.sleep(0.01)

    def _cleanup_old_steps(self, current_step: int):
        """Remove old step data to prevent memory leak."""
        steps_to_remove = [s for s in self.prepared_data if s < current_step - 1]
        for s in steps_to_remove:
            del self.prepared_data[s]
            if s in self.metrics:
                del self.metrics[s]

    def get_state(self) -> dict:
        return {
            "training_step": self.training_step,
            "last_consumed_step": self._last_consumed_step,
            "iter_dataloader_state": self.iter_dataloader.state_dict(),
        }

    def set_state(self, state: dict):
        if self._prep_future is not None:
            raise RuntimeError("Cannot update DataPreparationActor state after preparation has started")
        self.iter_dataloader.load_state_dict(state["iter_dataloader_state"])

        self._last_consumed_step = state.get("last_consumed_step", state["training_step"] - 1)
        self.training_step = self._last_consumed_step + 1

        logger.info(
            f"[DataPreparationActor] Restored state: training_step={self.training_step}, last_consumed_step={self._last_consumed_step}"
        )
