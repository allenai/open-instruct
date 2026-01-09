import logging
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from datasets import Dataset
from olmo_core.data import data_loader

logger = logging.getLogger(__name__)


def to_device(batch: dict[str, Any], device: torch.device | None) -> dict[str, Any]:
    """Move all tensors in a batch dictionary to the specified device.

    Args:
        batch: Dictionary potentially containing torch.Tensor values.
        device: Target device. If None, tensors are not moved.

    Returns:
        Dictionary with the same keys, but tensor values moved to the target device.
    """
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
        rank: int,
        world_size: int,
        work_dir: str,
        automatic_reshuffle: bool = False,
        collator: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the HFDataLoader.

        Args:
            dataset: The HuggingFace Dataset to load data from.
            batch_size: The global batch size.
            seed: Random seed for shuffling.
            rank: The rank of the current process in the distributed setup.
            world_size: Total number of processes in the distributed setup.
            work_dir: Working directory for the data loader (required by DataLoaderBase).
            automatic_reshuffle: If True, automatically reshuffle at epoch boundaries.
            collator: Optional collation function for batching examples.
            device: Device to move tensors to.
        """
        super().__init__(
            work_dir=work_dir, global_batch_size=batch_size, dp_world_size=world_size, dp_rank=rank, fs_local_rank=rank
        )

        self._full_dataset = dataset.map(lambda example, idx: example | {"dataset_index": idx}, with_indices=True)
        self.seed = seed
        self._batch_size = batch_size
        self._per_rank_batch_size = batch_size // world_size
        self._collator = collator if collator is not None else (lambda x: {"examples": x})
        self._automatic_reshuffle = automatic_reshuffle
        self._excluded_indices: set[int] = set()
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
        start_example = self.batches_processed * self._per_rank_batch_size
        batch_examples: list[dict[str, Any]] = []
        for i in range(start_example, self.effective_size):
            example = self.dataset[i]
            batch_examples.append(example | {"prompt_id": f"{self._epoch}_{example['dataset_index']}"})
            if len(batch_examples) == self._per_rank_batch_size:
                yield to_device(self._collator(batch_examples), self._device)
                batch_examples = []

    @property
    def total_batches(self) -> int:
        """Return the total number of batches in an epoch."""
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
            index: The dataset_index to exclude.
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
        """Reshard the dataset for a given epoch."""
        shuffled = self._full_dataset.shuffle(seed=self.seed + epoch)
        filtered = shuffled.filter(lambda x: x["dataset_index"] not in self._excluded_indices)
        global_size = len(filtered)
        total_batches = global_size // self._batch_size
        self.effective_size = total_batches * self._per_rank_batch_size
        self.dataset = filtered.shard(num_shards=self.dp_world_size, index=self.dp_rank)

    def get_mock_batch(self) -> dict[str, Any]:
        """Return a batch with arbitrary data for dry-run testing.

        Used by the trainer to do a dry-run of the
        forward and backward pass before training officially starts.
        """
        num_examples = min(self._per_rank_batch_size, len(self.dataset))
        examples = [self.dataset[i] for i in range(num_examples)]
        return to_device(self._collator(examples), self._device)  # type: ignore[return-value]

    def global_num_tokens_in_batch(self, batch: dict[str, Any]) -> int | None:
        """Return the total number of tokens in the batch across all ranks.

        Counts tokens from all keys containing 'input_ids' that are torch tensors.

        Args:
            batch: A batch dictionary containing input tensors.

        Returns:
            Total number of tokens across all ranks, or None if no input_ids found.
        """
        num_tokens = 0
        for key, value in batch.items():
            if "input_ids" in key and isinstance(value, torch.Tensor):
                num_tokens += value.numel()
        if num_tokens == 0:
            return None
        return num_tokens * self.dp_world_size


@dataclass
class VLLMConfig:
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    vllm_sync_backend: str = "nccl"
    vllm_gpu_memory_utilization: float = 0.9
    vllm_enable_prefix_caching: bool = False
    vllm_top_p: float = 1.0

    def __post_init__(self):
        if os.environ.get("VLLM_USE_V1") == "0":
            logger.warning("When using the v0 version of vLLM, caching is broken and will never be invalidated.")
            if self.vllm_enable_prefix_caching:
                raise ValueError("Prefix caching is currently not supported for v0.")


@dataclass
class StreamingDataLoaderConfig:
    max_prompt_token_length: int = 256
    response_length: int = 256
    pack_length: int = 512

    async_steps: int = 1
    num_samples_per_prompt_rollout: int = 4
    num_unique_prompts_rollout: int = 16

    active_sampling: bool = False
    filter_zero_std_samples: bool = True
    no_resampling_pass_rate: float | None = None
    advantage_normalization_type: str = "standard"
    mask_truncated_completions: bool = False
    mask_tool_use: bool = True

    dataset_mixer_list: list[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    dataset_mixer_eval_list: list[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    dataset_mixer_eval_list_splits: list[str] = field(default_factory=lambda: ["test"])
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_max_length_filter_v1"])
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: str | None = None
    dataset_config_eval_hash: str | None = None
    dataset_skip_cache: bool = False
    shuffle_eval_dataset: bool = False
    system_prompt_override_file: str | None = None

    temperature: float = 0.7
    stop_strings: list[str] | None = None
    inflight_updates: bool = False

    apply_r1_style_format_reward: bool = False
    r1_style_format_reward: float = 1.0
    additive_format_reward: bool = False

    apply_verifiable_reward: bool = True
    verification_reward: float = 10.0
    remap_verifier: str | None = None

    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    llm_judge_max_tokens: int = 2048
    llm_judge_max_context_length: int = 8192
    llm_judge_temperature: float = 1.0
    llm_judge_timeout: int = 60

    code_api_url: str = field(
        default_factory=lambda: os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program"
    )
    code_max_execution_time: float = 1.0
    code_pass_rate_reward_threshold: float = 0.0
    code_apply_perf_penalty: bool = False

    max_length_verifier_max_length: int = 32768

    non_stop_penalty: bool = False
    non_stop_penalty_value: float = 0.0

    tools: list[str] | None = None
    max_tool_calls: tuple[int, ...] = (5,)
    only_reward_good_outputs: bool = False

    number_documents_to_search: int = 3
    search_api_endpoint: str | None = None

    code_tool_api_endpoint: str | None = None

    max_possible_score: float = 1.0

    def __post_init__(self):
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

        assert self.apply_verifiable_reward or self.apply_r1_style_format_reward or self.non_stop_penalty, (
            "At least one reward must be applied!"
        )

        if self.stop_strings is None:
            self.stop_strings = []

        self.max_tool_calls = tuple(int(x) for x in self.max_tool_calls)

        if self.tools is not None and len(self.tools) > 0:
            for tool in self.tools:
                if tool not in ["search", "code"]:
                    raise ValueError(f"Tool {tool} is not supported. Supported tools are: search, code")
            assert len(self.tools) == len(set(self.tools)), "Duplicate tools are not allowed"

        self.max_possible_score = 0.0
        if self.apply_verifiable_reward:
            self.max_possible_score += self.verification_reward
        if self.apply_r1_style_format_reward and self.additive_format_reward:
            self.max_possible_score += self.r1_style_format_reward
