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
import threading
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue as StdQueue
from typing import Any

import numpy as np
import torch
import vllm
from datasets import Dataset
from ray.util import queue as ray_queue
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from open_instruct import data_loader as data_loader_lib
from open_instruct import utils
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.model_utils import Batch
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, ShutdownSentinel, TokenStatistics
from open_instruct.rl_utils import PackedSequences, Timer, pack_sequences
from open_instruct.utils import combine_reward_metrics, repeat_each

logger = logging.getLogger(__name__)


@dataclass
class StreamingDataLoaderConfig:
    max_prompt_token_length: int = 256
    response_length: int = 256
    async_steps: int = 1
    num_samples_per_prompt_rollout: int = 4
    active_sampling: bool = False
    filter_zero_std_samples: bool = True
    no_resampling_pass_rate: float | None = None
    advantage_normalization_type: str = "standard"
    mask_truncated_completions: bool = False
    pack_length: int = 512

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

    def build(
        self,
        dataset: Dataset,
        inference_results_Q: ray_queue.Queue,
        param_prompt_Q: ray_queue.Queue,
        tokenizer: PreTrainedTokenizer,
        generation_config: Any,
        dp_rank: int,
        fs_local_rank: int,
        num_training_steps: int,
        seed: int,
        per_device_train_batch_size: int,
        verbose: bool,
        work_dir: Path | str,
        global_batch_size: int,
        dp_world_size: int,
        max_possible_score: float,
        actor_manager=None,
        model_dims: utils.ModelDims | None = None,
    ) -> "StreamingDataLoader":
        return StreamingDataLoader(
            dataset=dataset,
            inference_results_Q=inference_results_Q,
            param_prompt_Q=param_prompt_Q,
            tokenizer=tokenizer,
            config=self,
            generation_config=generation_config,
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            num_training_steps=num_training_steps,
            seed=seed,
            per_device_train_batch_size=per_device_train_batch_size,
            verbose=verbose,
            max_possible_score=max_possible_score,
            actor_manager=actor_manager,
            model_dims=model_dims,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )


class DataLoaderBase:
    def __init__(
        self,
        *,
        work_dir: Path | str,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
    ):
        self.work_dir = Path(work_dir)
        self._global_batch_size = global_batch_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.fs_local_rank = fs_local_rank
        self.batches_processed = 0
        self.epoch: int | None = None

    @property
    def global_batch_size(self) -> int:
        return self._global_batch_size

    @global_batch_size.setter
    def global_batch_size(self, value: int):
        self._global_batch_size = value

    @property
    def rank_batch_size(self) -> int:
        return self.global_batch_size // self.dp_world_size

    @property
    @abstractmethod
    def total_batches(self) -> int | None:
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]):
        pass

    @abstractmethod
    def reshuffle(self, epoch: int | None = None, **kwargs):
        pass

    @abstractmethod
    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        pass

    @abstractmethod
    def get_mock_batch(self) -> dict[str, Any]:
        pass

    def __iter__(self):
        return self._iter_batches()

    def __next__(self):
        if not hasattr(self, "_iterator"):
            self._iterator = self._iter_batches()
        return next(self._iterator)

    def reset(self):
        if hasattr(self, "_iterator"):
            del self._iterator
        self.batches_processed = 0


class TextDataLoaderBase(DataLoaderBase):
    def __init__(
        self,
        *,
        work_dir: Path | str,
        global_batch_size: int,
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
        self.tokens_processed: int = 0

    def reset(self):
        super().reset()
        self.tokens_processed = 0

    def global_num_tokens_in_batch(self, batch: dict[str, Any]) -> int | None:
        del batch
        return self.global_batch_size


class StreamingDataLoader(TextDataLoaderBase):
    def __init__(
        self,
        *,
        dataset: Dataset,
        inference_results_Q: ray_queue.Queue,
        param_prompt_Q: ray_queue.Queue,
        tokenizer: PreTrainedTokenizer,
        config: StreamingDataLoaderConfig,
        generation_config: Any,
        work_dir: Path | str,
        global_batch_size: int,
        num_training_steps: int = 0,
        seed: int,
        per_device_train_batch_size: int,
        verbose: bool,
        max_possible_score: float,
        actor_manager=None,
        model_dims: utils.ModelDims = None,
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

        self.dataset = dataset
        self.inference_results_Q = inference_results_Q
        self.param_prompt_Q = param_prompt_Q
        self.tokenizer = tokenizer
        self.config = config
        self.config.max_possible_score = max_possible_score
        self.generation_config = generation_config
        self.num_training_steps = num_training_steps
        self.actor_manager = actor_manager
        self.model_dims = model_dims

        self.per_device_train_batch_size = per_device_train_batch_size
        self.verbose = verbose

        self.training_step = 0
        self.current_epoch = 0
        self.seed = seed

        self.iter_dataloader = data_loader_lib.HFDataLoader(
            dataset=dataset,
            batch_size=1,
            seed=seed,
            rank=dp_rank,
            world_size=dp_world_size,
            work_dir=work_dir,
            automatic_reshuffle=True,
        )

        self.local_queue = StdQueue(maxsize=config.async_steps)
        self.background_thread = None
        self.shutdown_requested = False

    @property
    def total_batches(self) -> int | None:
        return self.num_training_steps

    def state_dict(self) -> dict[str, Any]:
        return {
            "training_step": self.training_step,
            "current_epoch": self.current_epoch,
            "iter_dataloader_state": self.iter_dataloader.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.training_step = state_dict["training_step"]
        self.current_epoch = state_dict.get("current_epoch", 0)
        self.iter_dataloader.load_state_dict(state_dict["iter_dataloader_state"])

    def reshuffle(self, epoch: int | None = None, **kwargs):
        if epoch is not None:
            self.current_epoch = epoch

    def get_mock_batch(self) -> dict[str, Any]:
        dummy_qr = torch.tensor([self.tokenizer.pad_token_id, self.tokenizer.eos_token_id], dtype=torch.long)
        dummy_tool_mask = torch.zeros_like(dummy_qr)
        dummy_attention = torch.tensor([1, 1], dtype=torch.long)
        dummy_position_ids = torch.arange(len(dummy_qr), dtype=torch.long)
        dummy_response_mask = torch.zeros_like(dummy_qr)
        dummy_advantage = torch.zeros_like(dummy_qr, dtype=torch.float)

        return {
            "collated_query_responses": [dummy_qr],
            "collated_tool_masks": [dummy_tool_mask],
            "collated_attention_masks": [dummy_attention],
            "collated_position_ids": [dummy_position_ids],
            "collated_advantages": [dummy_advantage],
            "collated_response_masks": [dummy_response_mask],
            "collated_vllm_logprobs": [torch.zeros_like(dummy_qr, dtype=torch.float)],
        }

    def _iter_batches(self) -> Iterable[dict[str, Any]]:
        if self.background_thread is None:
            self._start_background_thread()

        while self.training_step < self.num_training_steps:
            batch_data = self.local_queue.get()
            self.training_step += 1
            yield batch_data

    def _start_background_thread(self):
        self.shutdown_requested = False
        self.background_thread = threading.Thread(
            target=self._data_preparation_loop, daemon=True, name=f"DataLoader-Worker-Rank{self.dp_rank}"
        )
        self.background_thread.start()

    def _data_preparation_loop(self):
        for _ in range(self.config.async_steps * self.global_batch_size // self.dp_world_size):
            example = next(self.iter_dataloader)
            dataset_index = example["dataset_index"]
            add_prompt_to_generator(
                example,
                dataset_index,
                self.iter_dataloader._epoch,
                self.training_step,
                self.param_prompt_Q,
                self.generation_config,
                is_eval=False,
            )

        for training_step in range(self.training_step, self.num_training_steps):
            if self.shutdown_requested:
                logger.info(f"[DataLoader Worker {self.dp_rank}] Shutdown requested, exiting")
                return

            with Timer("🚀 [Data Preparation Thread] Getting response ids") as timer:
                result, batch, reward_metrics, batch_stats = accumulate_inference_batches(
                    self.inference_results_Q,
                    self.generation_config,
                    num_prompts=self.rank_batch_size,
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
                    training_step=training_step,
                    verbose=self.verbose,
                    max_possible_score=self.config.max_possible_score,
                )
                if isinstance(result, ShutdownSentinel):
                    logger.info(f"[DataLoader Worker {self.dp_rank}] Received shutdown sentinel, exiting")
                    return

            getting_response_time = timer.duration
            scores = np.array(batch.scores)

            good_outputs = [
                len(result.request_info.tool_outputs[i]) > 0
                and result.request_info.tool_calleds[i]
                and not result.request_info.timeouts[i]
                and not result.request_info.tool_errors[i]
                for i in range(len(result.request_info.tool_outputs))
            ]
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
                        f"[Truncated completions filtering] Filtered {num_truncated} responses that didn't finish with 'stop'. "
                        f"Retention rate: {len(stop_idxes) / len(result.finish_reasons):.2%}"
                    )
                scores = scores[stop_idxes]
                advantages = advantages[stop_idxes]
                batch = batch[stop_idxes.tolist()]
                result.responses = [result.responses[i] for i in stop_idxes]
                result.masks = [result.masks[i] for i in stop_idxes]
                result.finish_reasons = [result.finish_reasons[i] for i in stop_idxes]
                result.logprobs = [result.logprobs[i] for i in stop_idxes]

            with Timer("📦 [Data Preparation Thread] Packing sequences"):
                packed_sequences = pack_sequences(
                    queries=batch.queries,
                    responses=result.responses,
                    masks=result.masks,
                    pack_length=self.config.pack_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    vllm_logprobs=result.logprobs,
                )
                lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
                lookup_advantages[1:] = advantages
                packed_advantages = [
                    torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
                    for packed_mask in packed_sequences.response_masks
                ]
                packed_sequences.advantages = packed_advantages

            collated_data = self._prepare_collated_data_for_self(packed_sequences)

            if len(result.responses) == 0:
                metrics = {}
                logger.warning(f"No responses in batch {training_step}.")
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
                stop_rate = sum(int(finish_reason == "stop") for finish_reason in result.finish_reasons) / len(
                    result.finish_reasons
                )

                batch_metrics = asdict(batch_stats)
                batch_metrics_prefixed = {f"batch/{k}": v for k, v in batch_metrics.items()}

                metrics = {
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
                    "val/good_outputs_rate": np.array(good_outputs).mean(),
                    "val/tool_runtimes_rate": np.array(result.request_info.tool_runtimes).mean(),
                    "val/tool_calleds_rate": np.array(result.request_info.tool_calleds).mean(),
                    "time/getting_response": getting_response_time,
                    **reward_metrics,
                    **batch_metrics_prefixed,
                }

                total_tokens = result.token_statistics.num_prompt_tokens + result.token_statistics.num_response_tokens
                metrics["val/actor_tokens_per_second"] = total_tokens / result.token_statistics.generation_time

            collated_data["metrics"] = metrics
            self.local_queue.put(collated_data)

    def _prepare_collated_data_for_self(self, packed_sequences: PackedSequences) -> dict[str, list[torch.Tensor]]:
        per_device_packed_query_responses = packed_sequences.query_responses
        per_device_packed_tool_masks = getattr(packed_sequences, "tool_masks", None)
        per_device_packed_attention_masks = packed_sequences.attention_masks
        per_device_packed_position_ids = packed_sequences.position_ids
        per_device_packed_advantages = packed_sequences.advantages
        per_device_packed_response_masks = packed_sequences.response_masks
        per_device_packed_vllm_logprobs = packed_sequences.vllm_logprobs

        b_inds = np.random.permutation(len(per_device_packed_query_responses))
        collated_query_responses = []
        collated_tool_masks = [] if per_device_packed_tool_masks is not None else None
        collated_attention_masks = []
        collated_position_ids = []
        collated_response_masks = []
        collated_advantages = []
        collated_vllm_logprobs = []
        for j in range(0, len(per_device_packed_query_responses), self.per_device_train_batch_size):
            micro_range = b_inds[j : j + self.per_device_train_batch_size]
            collated_query_responses.append(
                collate_fn(
                    [per_device_packed_query_responses[idx] for idx in micro_range], self.tokenizer.pad_token_id, True
                )
            )
            if per_device_packed_tool_masks is not None:
                collated_tool_masks.append(
                    collate_fn([per_device_packed_tool_masks[idx] for idx in micro_range], 0, True)
                )
            collated_attention_masks.append(
                collate_fn([per_device_packed_attention_masks[idx] for idx in micro_range], 0, True)
            )
            collated_position_ids.append(
                collate_fn([per_device_packed_position_ids[idx] for idx in micro_range], 0, True)
            )
            collated_response_masks.append(
                collate_fn([per_device_packed_response_masks[idx] for idx in micro_range], 0, True)
            )
            collated_advantages.append(collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0, True))
            collated_vllm_logprobs.append(
                collate_fn([per_device_packed_vllm_logprobs[idx] for idx in micro_range], 0, True)
            )

        result = {
            "collated_query_responses": collated_query_responses,
            "collated_attention_masks": collated_attention_masks,
            "collated_position_ids": collated_position_ids,
            "collated_advantages": collated_advantages,
            "collated_response_masks": collated_response_masks,
            "collated_vllm_logprobs": collated_vllm_logprobs,
        }
        if collated_tool_masks is not None:
            result["collated_tool_masks"] = collated_tool_masks
        return result

    def shutdown(self):
        self.shutdown_requested = True
        if self.background_thread is not None:
            self.background_thread.join(timeout=5.0)


def collate_fn(tensors_list: list[torch.Tensor], pad_token_id: int, pin_memory: bool = True) -> torch.Tensor:
    padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors_list, batch_first=True, padding_value=pad_token_id)
    if pin_memory:
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


class PendingQueriesMap:
    def __init__(self):
        self._map = {}
        self._lock = threading.Lock()

    def insert(self, dataset_idx, query, ground_truth, dataset, raw_query):
        with self._lock:
            if dataset_idx in self._map:
                existing_query, existing_ground_truth, existing_dataset, existing_raw_query, count = self._map[
                    dataset_idx
                ]
                self._map[dataset_idx] = (
                    existing_query,
                    existing_ground_truth,
                    existing_dataset,
                    existing_raw_query,
                    count + 1,
                )
            else:
                self._map[dataset_idx] = (query, ground_truth, dataset, raw_query, 1)

    def pop(self, dataset_idx):
        with self._lock:
            if dataset_idx not in self._map:
                raise RuntimeError(f"Dataset index {dataset_idx} not found in pending_queries_map")

            query, ground_truth, dataset, raw_query, count = self._map[dataset_idx]

            if count > 1:
                self._map[dataset_idx] = (query, ground_truth, dataset, raw_query, count - 1)
            else:
                del self._map[dataset_idx]

            return query, ground_truth, dataset, raw_query

    def __len__(self):
        with self._lock:
            return len(self._map)

    def __contains__(self, dataset_idx):
        with self._lock:
            return dataset_idx in self._map

    def __getitem__(self, dataset_idx):
        with self._lock:
            return self._map[dataset_idx]

    def keys(self):
        with self._lock:
            return list(self._map.keys())


def add_prompt_to_generator(
    example: dict[str, Any],
    example_index: int,
    epoch_number: int,
    training_step: int,
    param_prompt_Q: ray_queue.Queue,
    generation_config,
    is_eval: bool,
) -> None:
    query = example[INPUT_IDS_PROMPT_KEY]

    param_prompt_Q.put(
        PromptRequest(
            prompt=query,
            generation_config=generation_config,
            epoch_number=epoch_number,
            training_step=training_step,
            dataset_index=example_index,
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
    iter_dataloader: data_loader_lib.HFDataLoader | None = None,
    param_prompt_Q: ray_queue.Queue | None = None,
    training_step: int = None,
    verbose: bool = False,
    max_possible_score: float = 1.0,
) -> tuple[GenerationResult, Batch, dict, BatchStatistics]:
    import ray

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
    num_prompts_sampled = 0
    while num_prompts_sampled < num_prompts:
        result = inference_results_Q.get(timeout=timeout)

        if isinstance(result, ShutdownSentinel):
            return result, None, None, None

        assert len(result.responses) == generation_config.n, (
            f"Mismatch: individual prompt result has {len(result.responses)} responses "
            f"but expected {generation_config.n} samples per prompt. "
            f"Dataset index: {result.dataset_index}, Epoch: {result.epoch_number}"
        )

        example = dataset[result.dataset_index]
        query = example[INPUT_IDS_PROMPT_KEY]
        ground_truth = example[GROUND_TRUTHS_KEY]
        dataset_name = example[VERIFIER_SOURCE_KEY]
        raw_query = example[RAW_PROMPT_KEY]

        if replenish_prompts:
            example = next(iter_dataloader)
            dataset_index = example["dataset_index"]
            add_prompt_to_generator(
                example,
                dataset_index,
                iter_dataloader._epoch,
                training_step,
                param_prompt_Q,
                generation_config,
                is_eval=False,
            )

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

    total_generation_time = max_generation_time

    accumulated_stats = TokenStatistics(
        num_prompt_tokens=total_prompt_tokens,
        num_response_tokens=total_response_tokens,
        generation_time=total_generation_time,
        earliest_start_time=earliest_start_time,
    )

    combined_request_info = RequestInfo(
        num_calls=combined_num_calls,
        timeouts=combined_timeouts,
        tool_errors=combined_tool_errors,
        tool_outputs=combined_tool_outputs,
        tool_runtimes=combined_tool_runtimes,
        tool_calleds=combined_tool_calleds,
    )

    combined_result = GenerationResult(
        responses=combined_responses,
        finish_reasons=combined_finish_reasons,
        masks=combined_masks,
        request_info=combined_request_info,
        dataset_index=None,
        epoch_number=results[0].epoch_number,
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
