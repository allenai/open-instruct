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
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
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

from open_instruct import data_types, utils
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.model_utils import Batch
from open_instruct.rl_utils import PackedSequences, pack_sequences
from open_instruct.utils import combine_reward_metrics, repeat_each

logger = logging.getLogger(__name__)


class HFDataLoader(data_loader.DataLoaderBase):
    """A DataLoader that wraps a HuggingFace Dataset for use with olmo_core's Trainer.

    This class implements the DataLoaderBase interface, providing iteration over
    a HuggingFace Dataset with support for sharding across distributed workers,
    shuffling, and checkpointing.
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
        """
        super().__init__(
            work_dir=work_dir, global_batch_size=batch_size, dp_world_size=world_size, dp_rank=rank, fs_local_rank=0
        )

        dataset_with_indices = dataset.map(lambda example, idx: example | {"dataset_index": idx}, with_indices=True)
        self._original_dataset = dataset_with_indices.shard(num_shards=world_size, index=rank)
        self.dataset = self._original_dataset.shuffle(seed=seed)
        self.seed = seed
        self._batch_size = batch_size
        self.effective_size = len(self.dataset) - (len(self.dataset) % batch_size)
        self._automatic_reshuffle = automatic_reshuffle
        self._excluded_indices: set[int] = set()
        self._epoch: int = 0
        self._current_iter: Iterator[dict[str, Any]] | None = None

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
        for i in range(self.batches_processed, self.effective_size):
            example = self.dataset[i]
            yield example | {"prompt_id": f"{self._epoch}_{example['dataset_index']}"}

    @property
    def total_batches(self) -> int:
        """Return the total number of batches in an epoch."""
        return self.effective_size // self._batch_size

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
        """Reshuffle the dataset for a new epoch.

        Args:
            epoch: The epoch number (unused, for API compatibility).
            **kwargs: Additional keyword arguments (unused, for API compatibility).
        """
        self._epoch += 1
        self.batches_processed = 0
        shuffled = self._original_dataset.shuffle(seed=self.seed + self._epoch)
        # If this is slow, we can speed it up by making this a boolean mask.
        self.dataset = shuffled.filter(lambda x: x["dataset_index"] not in self._excluded_indices)
        self.effective_size = len(self.dataset) - (len(self.dataset) % self._batch_size)

    def get_mock_batch(self) -> dict[str, Any]:
        """Return a batch with arbitrary data for dry-run testing.

        Used by the trainer to do a dry-run of the
        forward and backward pass before training officially starts.

        Returns:
            The first item from the dataset.
        """
        return self.dataset[0]


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
    # Data loading/packing
    max_prompt_token_length: int = 256
    response_length: int = 256
    pack_length: int = 512

    # Batching
    async_steps: int = 1
    num_samples_per_prompt_rollout: int = 4
    num_unique_prompts_rollout: int = 16

    # GRPO sampling/filtering
    active_sampling: bool = False
    filter_zero_std_samples: bool = True
    no_resampling_pass_rate: float | None = None
    advantage_normalization_type: str = "standard"
    mask_truncated_completions: bool = False
    mask_tool_use: bool = True

    # Dataset
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

    # Generation
    temperature: float = 0.7
    stop_strings: list[str] | None = None
    inflight_updates: bool = False

    # Reward - R1 style format reward
    apply_r1_style_format_reward: bool = False
    r1_style_format_reward: float = 1.0
    additive_format_reward: bool = False

    # Reward - Verifiable reward
    apply_verifiable_reward: bool = True
    verification_reward: float = 10.0
    remap_verifier: str | None = None

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

    # Tools
    tools: list[str] | None = None
    max_tool_calls: tuple[int, ...] = (5,)
    only_reward_good_outputs: bool = False

    # RAG
    number_documents_to_search: int = 3
    search_api_endpoint: str | None = None

    # Code tool
    code_tool_api_endpoint: str | None = None

    # Computed at post_init
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

    def build_dataloader(
        self,
        data_prep_actor_name: str,
        tokenizer: PreTrainedTokenizer,
        dp_rank: int,
        fs_local_rank: int,
        num_training_steps: int,
        work_dir: Path | str,
        dp_world_size: int,
    ) -> "StreamingDataLoader":
        """Build a thin wrapper dataloader that pulls from the DataPreparationActor singleton."""
        return StreamingDataLoader(
            data_prep_actor_name=data_prep_actor_name,
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
        data_prep_actor_name: str,
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

        self.data_prep_actor = ray.get_actor(data_prep_actor_name)
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
        dummy_qr = torch.tensor([self.tokenizer.pad_token_id, self.tokenizer.eos_token_id], dtype=torch.long)
        dummy_attention = torch.tensor([1, 1], dtype=torch.long)
        dummy_position_ids = torch.arange(len(dummy_qr), dtype=torch.long)
        dummy_response_mask = torch.zeros_like(dummy_qr)
        dummy_advantage = torch.zeros_like(dummy_qr, dtype=torch.float)

        batch = data_types.CollatedBatchData(
            query_responses=[dummy_qr],
            attention_masks=[dummy_attention],
            position_ids=[dummy_position_ids],
            advantages=[dummy_advantage],
            response_masks=[dummy_response_mask],
            vllm_logprobs=[torch.zeros_like(dummy_qr, dtype=torch.float)],
        )
        return {"batch": batch, "metrics": {}}

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        for step in range(self.training_step, self.num_training_steps):
            batch_data = ray.get(self.data_prep_actor.get_data.remote(rank=self.dp_rank, step=step))
            self.training_step = step + 1
            yield batch_data


def collate_fn(tensors_list: list[torch.Tensor], pad_token_id: int, pin_memory: bool = True) -> torch.Tensor:
    padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors_list, batch_first=True, padding_value=pad_token_id)
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


def add_prompt_to_generator(
    example: dict[str, Any], epoch_number: int, param_prompt_Q: ray_queue.Queue, generation_config, is_eval: bool
) -> None:
    dataset_index = example["dataset_index"]
    param_prompt_Q.put(
        data_types.PromptRequest(
            prompt=example[INPUT_IDS_PROMPT_KEY],
            generation_config=generation_config,
            dataset_index=dataset_index,
            prompt_id=f"{epoch_number}_{dataset_index}",
            is_eval=is_eval,
        )
    )


def accumulate_inference_batches(
    inference_results_Q: ray_queue.Queue,
    generation_config: vllm.SamplingParams,
    num_prompts: int,
    model_dims: utils.ModelDims,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
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
    all_scores = []
    all_percent_solved = []
    total_filtered_prompts = 0
    filtered_prompt_zero = 0
    filtered_prompt_solved = 0
    filtered_prompt_nonzero = 0
    total_no_resampled = 0
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
    while num_prompts_sampled < num_prompts:
        logger.info(
            f"[accumulate_inference_batches] Waiting for result {num_prompts_sampled + 1}/{num_prompts} from inference_results_Q"
        )
        result = inference_results_Q.get(timeout=timeout)
        logger.info(
            f"[accumulate_inference_batches] Got result {num_prompts_sampled + 1}/{num_prompts}, type: {type(result).__name__}"
        )

        if isinstance(result, data_types.ShutdownSentinel):
            return result, None, None, None

        assert len(result.responses) == generation_config.n, (
            f"Mismatch: individual prompt result has {len(result.responses)} responses "
            f"but expected {generation_config.n} samples per prompt. "
            f"Dataset index: {result.dataset_index}, Prompt ID: {result.prompt_id}"
        )

        example = dataset[result.dataset_index]
        query = example[INPUT_IDS_PROMPT_KEY]
        ground_truth = example[GROUND_TRUTHS_KEY]
        dataset_name = example[VERIFIER_SOURCE_KEY]
        raw_query = example[RAW_PROMPT_KEY]

        if replenish_prompts:
            assert iter_dataloader is not None
            assert param_prompt_Q is not None
            example = next(iter_dataloader)
            add_prompt_to_generator(example, iter_dataloader._epoch, param_prompt_Q, generation_config, is_eval=False)

        for i in range(len(result.finish_reasons)):
            if result.finish_reasons[i] == "stop" and len(result.responses[i]) == 0:
                result.responses[i].append(tokenizer.eos_token_id)
                result.masks[i].append(1)
                result.logprobs[i].append(float("nan"))

        decoded_responses = tokenizer.batch_decode(result.responses, skip_special_tokens=True)

        k_queries = repeat_each([query], generation_config.n)
        k_ground_truths = repeat_each([ground_truth], generation_config.n)
        k_datasets = repeat_each([dataset_name], generation_config.n)
        k_raw_queries = repeat_each([raw_query], generation_config.n)

        percent_solved = np.mean(result.reward_scores).item() / max_possible_score
        if no_resampling_pass_rate is not None and percent_solved >= no_resampling_pass_rate:
            assert iter_dataloader is not None
            iter_dataloader.exclude_index(result.dataset_index)
            total_no_resampled += 1
            logging.debug(
                f"[Data Preparation Thread] Prompt solved at {percent_solved}, will be excluded from resampling, total no resampled: {total_no_resampled}"
            )

        if filter_zero_std_samples and np.std(result.reward_scores) == 0:
            if not active_sampling:
                num_prompts_sampled += 1
                progress_bar.update(1)

            total_filtered_prompts += 1
            if result.reward_scores[0] == 0:
                filtered_prompt_zero += 1
            elif result.reward_scores[0] == max_possible_score:
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
        all_decoded_responses.extend(decoded_responses)
        all_scores.extend(result.reward_scores)
        all_reward_metrics.append(result.reward_metrics)
        all_percent_solved.append(percent_solved)

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
    )

    combined_result = data_types.GenerationResult(
        responses=combined_responses,
        finish_reasons=combined_finish_reasons,
        masks=combined_masks,
        request_info=combined_request_info,
        dataset_index=None,
        prompt_id=results[0].prompt_id,
        token_statistics=accumulated_stats,
        logprobs=combined_logprobs,
    )

    if actor_manager is not None:
        ray.get(actor_manager.report_token_statistics.remote(accumulated_stats))

    batch = Batch(
        queries=all_queries,
        ground_truths=all_ground_truths,
        datasets=all_datasets,
        raw_queries=all_raw_queries,
        decoded_responses=all_decoded_responses,
        indices=None,
        scores=all_scores,
    )

    combined_reward_metrics = combine_reward_metrics(all_reward_metrics)
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


def pad_sequences_for_world_size(
    packed_sequences: PackedSequences, world_size: int, pad_token_id: int, eos_token_id: int
) -> None:
    """Pads packed sequences so total count is divisible by world_size.

    Mutates packed_sequences in place by appending dummy sequences.
    """
    total = len(packed_sequences.query_responses)
    remainder = total % world_size
    if remainder == 0:
        return

    shortfall = world_size - remainder
    logger.warning(f"Padding {shortfall} sequences for world size. In future, you should adjust your compute.")

    assert packed_sequences.position_ids is not None
    assert packed_sequences.advantages is not None
    assert packed_sequences.vllm_logprobs is not None

    dummy_qr = torch.tensor([pad_token_id, eos_token_id], dtype=torch.long)
    dummy_attention = torch.tensor([1, 1], dtype=torch.long)
    dummy_position_ids = torch.arange(len(dummy_qr), dtype=torch.long)
    dummy_response_mask = torch.zeros_like(dummy_qr)
    dummy_advantage = torch.zeros_like(dummy_qr, dtype=torch.float)
    dummy_logprobs = torch.zeros_like(dummy_qr, dtype=torch.float)

    for _ in range(shortfall):
        packed_sequences.query_responses.append(dummy_qr)
        packed_sequences.attention_masks.append(dummy_attention)
        packed_sequences.position_ids.append(dummy_position_ids)
        packed_sequences.response_masks.append(dummy_response_mask)
        packed_sequences.advantages.append(dummy_advantage)
        packed_sequences.vllm_logprobs.append(dummy_logprobs)


def prepare_collated_data_for_workers(
    packed_sequences: PackedSequences,
    world_size: int,
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
        world_size: Number of distributed workers.
        per_device_train_batch_size: Batch size for each device's micro-batch.
        pad_token_id: Token ID used for padding sequences.
        pin_memory: Whether to pin memory for faster data transfer to GPU.

    Returns:
        List of CollatedBatchData, one per worker, each containing collated tensors
        for query_responses, attention_masks, position_ids,
        advantages, response_masks, and vllm_logprobs.
    """
    total_sequences = len(packed_sequences.query_responses)
    if total_sequences % world_size != 0:
        new_total = (total_sequences // world_size) * world_size
        logger.warning(
            f"Total packed sequences ({total_sequences}) is not evenly divisible by world_size ({world_size}). "
            f"Truncating to {new_total} sequences (dropping {total_sequences - new_total})."
        )
    B = total_sequences // world_size
    collated_data = []
    assert packed_sequences.position_ids is not None
    assert packed_sequences.advantages is not None
    assert packed_sequences.vllm_logprobs is not None
    for i in range(world_size):
        per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
        per_device_packed_attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
        per_device_packed_position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
        per_device_packed_advantages = packed_sequences.advantages[B * i : B * (i + 1)]
        per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]
        per_device_packed_vllm_logprobs = packed_sequences.vllm_logprobs[B * i : B * (i + 1)]

        # Shuffle the batch and collate the data
        b_inds = np.random.permutation(len(per_device_packed_query_responses))
        collated_query_responses = []
        collated_attention_masks = []
        collated_position_ids = []
        collated_response_masks = []
        collated_advantages = []
        collated_vllm_logprobs = []
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
            collated_advantages.append(
                collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_vllm_logprobs.append(
                collate_fn([per_device_packed_vllm_logprobs[idx] for idx in micro_range], 0, pin_memory)
            )
        collated_data.append(
            data_types.CollatedBatchData(
                query_responses=collated_query_responses,
                attention_masks=collated_attention_masks,
                position_ids=collated_position_ids,
                advantages=collated_advantages,
                response_masks=collated_response_masks,
                vllm_logprobs=collated_vllm_logprobs,
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
        initial_state: dict | None = None,
        allow_world_padding: bool = False,
    ):
        self.allow_world_padding = allow_world_padding
        self.inference_results_Q = inference_results_Q
        self.param_prompt_Q = param_prompt_Q
        self.tokenizer = tokenizer
        self.config = config
        self.config.max_possible_score = max_possible_score
        self.generation_config = generation_config
        self.num_training_steps = num_training_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.global_batch_size = global_batch_size
        self.dp_world_size = dp_world_size
        self.actor_manager = actor_manager
        self.model_dims = model_dims
        self.verbose = verbose
        self.dataset = dataset

        self.iter_dataloader = HFDataLoader(
            dataset=dataset, batch_size=1, seed=seed, rank=0, world_size=1, work_dir=work_dir, automatic_reshuffle=True
        )

        self.prepared_data: dict[int, list[data_types.CollatedBatchData]] = {}
        self.metrics: dict[int, dict] = {}
        self.current_prepared_step = -1
        self.lock = threading.Lock()
        self.shutdown_requested = False
        self.training_step = 0

        if initial_state is not None:
            self.training_step = initial_state["training_step"]
            self.iter_dataloader.load_state_dict(initial_state["iter_dataloader_state"])
            logger.info(f"[DataPreparationActor] Restored state: training_step={self.training_step}")

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DataPrepActor")
        self._prep_future = self._executor.submit(self._data_preparation_loop)

    def _data_preparation_loop(self):
        logger.info("[DataPreparationActor] Starting _data_preparation_loop")
        num_initial_prompts = self.config.async_steps * self.global_batch_size
        logger.info(f"[DataPreparationActor] Pushing {num_initial_prompts} initial prompts to param_prompt_Q")
        for _ in range(num_initial_prompts):
            add_prompt_to_generator(
                next(self.iter_dataloader),
                self.iter_dataloader._epoch,
                self.param_prompt_Q,
                self.generation_config,
                is_eval=False,
            )

        for step in range(self.training_step, self.num_training_steps):
            if self.shutdown_requested:
                return

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
                    )
                    for _ in range(self.dp_world_size)
                ]
                with self.lock:
                    self.prepared_data[step] = empty_data
                    self.metrics[step] = {}
                    self.current_prepared_step = step
                continue

            assert batch is not None
            assert batch_stats is not None
            scores = np.array(batch.scores)
            scores_per_prompt = scores.reshape(-1, self.config.num_samples_per_prompt_rollout)
            mean_grouped_rewards = scores_per_prompt.mean(axis=-1)
            mean_grouped_rewards = np.repeat(mean_grouped_rewards, self.config.num_samples_per_prompt_rollout, axis=0)
            std_grouped_rewards = scores_per_prompt.std(axis=-1)
            std_grouped_rewards = np.repeat(std_grouped_rewards, self.config.num_samples_per_prompt_rollout, axis=0)

            if self.config.advantage_normalization_type == "standard":
                advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
            elif self.config.advantage_normalization_type == "centered":
                advantages = scores - mean_grouped_rewards
            else:
                raise ValueError(f"Invalid advantage normalization type: {self.config.advantage_normalization_type}")

            if self.config.mask_truncated_completions:
                stop_idxes = torch.tensor(
                    [i for i in range(len(result.finish_reasons)) if result.finish_reasons[i] == "stop"]
                )
                num_truncated = len(result.finish_reasons) - len(stop_idxes)
                if num_truncated > 0:
                    logger.info(
                        f"[DataPreparationActor] Filtered {num_truncated} responses that didn't finish with 'stop'. "
                        f"Retention rate: {len(stop_idxes) / len(result.finish_reasons):.2%}"
                    )
                scores = scores[stop_idxes]
                advantages = advantages[stop_idxes]
                batch = batch[stop_idxes.tolist()]
                result.responses = [result.responses[i] for i in stop_idxes]
                result.masks = [result.masks[i] for i in stop_idxes]
                result.finish_reasons = [result.finish_reasons[i] for i in stop_idxes]
                assert result.logprobs is not None
                result.logprobs = [result.logprobs[i] for i in stop_idxes]

            assert result.logprobs is not None
            packed_sequences = pack_sequences(
                queries=batch.queries,
                responses=result.responses,
                masks=result.masks,
                pack_length=self.config.pack_length,
                pad_token_id=self.tokenizer.pad_token_id,
                vllm_logprobs=result.logprobs,
                mask_tool_use=self.config.mask_tool_use,
            )
            lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
            lookup_advantages[1:] = advantages
            packed_advantages = [
                torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
                for packed_mask in packed_sequences.response_masks
            ]
            packed_sequences.advantages = packed_advantages

            if self.allow_world_padding:
                pad_sequences_for_world_size(
                    packed_sequences, self.dp_world_size, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                )

            collated_data = prepare_collated_data_for_workers(
                packed_sequences, self.dp_world_size, self.per_device_train_batch_size, self.tokenizer.pad_token_id
            )

            if len(result.responses) == 0:
                step_metrics = {}
            else:
                real_num_responses = len(result.responses)
                expected_num_responses = self.config.num_samples_per_prompt_rollout * self.global_batch_size
                unsolved_num_responses = (scores < self.config.max_possible_score).sum()
                sequence_lengths = np.array([len(response) for response in result.responses])
                sequence_length_solved = (
                    np.array([])
                    if np.all(scores == 0)
                    else np.array(sequence_lengths[scores == self.config.max_possible_score])
                )
                sequence_length_unsolved = (
                    np.array([])
                    if np.all(scores == self.config.max_possible_score)
                    else np.array(sequence_lengths[scores == 0])
                )
                stop_rate = sum(int(fr == "stop") for fr in result.finish_reasons) / len(result.finish_reasons)

                batch_metrics_dict = asdict(batch_stats)
                batch_metrics_prefixed = {f"batch/{k}": v for k, v in batch_metrics_dict.items()}

                step_metrics = {
                    "scores": scores.mean(),
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
                    "val/advantages_mean": advantages.mean(),
                    "val/advantages_min": advantages.min(),
                    "val/advantages_max": advantages.max(),
                    "val/advantages_hist": advantages,
                    "val/num_calls_rate": np.array(result.request_info.num_calls).mean(),
                    "val/timeouts_rate": np.array(result.request_info.timeouts).mean(),
                    "val/tool_errors_rate": np.array(
                        [len(item) > 0 for item in result.request_info.tool_errors]
                    ).mean(),
                    "val/tool_runtimes_rate": np.array(result.request_info.tool_runtimes).mean(),
                    "val/tool_calleds_rate": np.array(result.request_info.tool_calleds).mean(),
                    **reward_metrics,
                    **batch_metrics_prefixed,
                }

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
            "training_step": self.current_prepared_step + 1,
            "iter_dataloader_state": self.iter_dataloader.state_dict(),
        }

    def set_state(self, state: dict):
        self.training_step = state["training_step"]
        self.iter_dataloader.load_state_dict(state["iter_dataloader_state"])
