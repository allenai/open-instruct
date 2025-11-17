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
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue as StdQueue
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from ray.util import queue as ray_queue
from transformers import PreTrainedTokenizer

from open_instruct import utils
from open_instruct.rl_utils import PackedSequences, Timer, pack_sequences

logger = logging.getLogger(__name__)

PathOrStr = Path | str


@dataclass
class StreamingDataLoaderConfig:
    work_dir: PathOrStr
    global_batch_size: int
    dp_world_size: int
    resume_training_step: int
    num_training_steps: int
    args: Any

    def build(
        self,
        dataset: Dataset,
        reward_fn: Callable,
        inference_results_Q: ray_queue.Queue,
        param_prompt_Q: ray_queue.Queue,
        pending_queries_map: dict,
        tokenizer: PreTrainedTokenizer,
        generation_config: Any,
        dp_rank: int,
        fs_local_rank: int,
        actor_manager=None,
        model_dims: utils.ModelDims | None = None,
    ) -> "StreamingDataLoader":
        return StreamingDataLoader(
            dataset=dataset,
            reward_fn=reward_fn,
            inference_results_Q=inference_results_Q,
            param_prompt_Q=param_prompt_Q,
            pending_queries_map=pending_queries_map,
            tokenizer=tokenizer,
            args=self.args,
            generation_config=generation_config,
            work_dir=self.work_dir,
            global_batch_size=self.global_batch_size,
            resume_training_step=self.resume_training_step,
            num_training_steps=self.num_training_steps,
            actor_manager=actor_manager,
            model_dims=model_dims,
            dp_world_size=self.dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )


class DataLoaderBase:
    def __init__(
        self,
        *,
        work_dir: PathOrStr,
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
        work_dir: PathOrStr,
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


class ShufflingIterator:
    def __init__(self, data: np.ndarray, batch_size: int, seed: int | None = None):
        self.data = data.copy()
        self.batch_size = batch_size
        self.index = 0
        self.epoch_number = 0
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.data)
        self.exclude_list = []

        self._update_effective_size()

    def __iter__(self):
        return self

    def __next__(self) -> list[int] | int:
        if self.index >= self.effective_size:
            self.index = 0
            self._update_effective_size()
            self.epoch_number += 1
            self.rng.shuffle(self.data)

        end_index = self.index + self.batch_size
        batch = self.data[self.index : end_index].tolist()
        if self.batch_size == 1:
            batch = batch[0]
        self.index = end_index

        return batch

    def get_state(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "epoch_number": self.epoch_number,
            "data": self.data.copy(),
            "rng_state": self.rng.bit_generator.state,
            "exclude_list": self.exclude_list.copy(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.index = state["index"]
        self.epoch_number = state.get("epoch_number", 0)
        self.data = state["data"].copy()
        self.rng.bit_generator.state = state["rng_state"]
        self.exclude_list = state.get("exclude_list", [])
        self._update_effective_size()

    def exclude_index(self, index: int) -> None:
        self.exclude_list.append(index)

    def _update_effective_size(self) -> None:
        if self.exclude_list:
            mask = ~np.isin(self.data, self.exclude_list)
            self.data = self.data[mask]
            self.exclude_list = []

        self.effective_size = len(self.data) - (len(self.data) % self.batch_size)


class StreamingDataLoader(TextDataLoaderBase):
    def __init__(
        self,
        *,
        dataset: Dataset,
        reward_fn: Callable,
        inference_results_Q: ray_queue.Queue,
        param_prompt_Q: ray_queue.Queue,
        pending_queries_map: dict,
        tokenizer: PreTrainedTokenizer,
        args: Any,
        generation_config: Any,
        work_dir: PathOrStr,
        global_batch_size: int,
        resume_training_step: int = 0,
        num_training_steps: int = 0,
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
        self.reward_fn = reward_fn
        self.inference_results_Q = inference_results_Q
        self.param_prompt_Q = param_prompt_Q
        self.pending_queries_map = pending_queries_map
        self.tokenizer = tokenizer
        self.args = args
        self.generation_config = generation_config
        self.num_training_steps = num_training_steps
        self.actor_manager = actor_manager
        self.model_dims = model_dims

        self.training_step = resume_training_step
        self.current_epoch = 0

        dataset_indices = np.arange(len(dataset))
        self.iter_dataloader = ShufflingIterator(dataset_indices, 1, seed=args.seed + dp_rank)

        self.local_queue = StdQueue(maxsize=args.async_steps)
        self.background_thread = None
        self.shutdown_requested = False

    @property
    def total_batches(self) -> int | None:
        return self.num_training_steps

    def state_dict(self) -> dict[str, Any]:
        return {
            "training_step": self.training_step,
            "current_epoch": self.current_epoch,
            "iter_dataloader_state": self.iter_dataloader.get_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.training_step = state_dict["training_step"]
        self.current_epoch = state_dict.get("current_epoch", 0)
        self.iter_dataloader.set_state(state_dict["iter_dataloader_state"])

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
        from open_instruct.grpo_fast import accumulate_inference_batches
        from open_instruct.queue_types import ShutdownSentinel

        for training_step in range(self.training_step, self.num_training_steps):
            if self.shutdown_requested:
                logger.info(f"[DataLoader Worker {self.dp_rank}] Shutdown requested, exiting")
                return

            with Timer("🚀 [Data Preparation Thread] Getting response ids") as timer:
                result, batch, reward_metrics, batch_stats = accumulate_inference_batches(
                    self.inference_results_Q,
                    self.pending_queries_map,
                    self.args,
                    self.generation_config,
                    num_prompts=self.rank_batch_size,
                    model_dims=self.model_dims,
                    tokenizer=self.tokenizer,
                    reward_fn=self.reward_fn,
                    actor_manager=self.actor_manager,
                    active_sampling=self.args.active_sampling,
                    filter_zero_std_samples=self.args.filter_zero_std_samples,
                    replenish_prompts=True,
                    no_resampling_pass_rate=self.args.no_resampling_pass_rate,
                    iter_dataloader=self.iter_dataloader,
                    prompt_dataset=self.dataset,
                    param_prompt_Q=self.param_prompt_Q,
                    training_step=training_step,
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
            scores_per_prompt = scores.reshape(-1, self.args.num_samples_per_prompt_rollout)
            mean_grouped_rewards = scores_per_prompt.mean(axis=-1)
            mean_grouped_rewards = np.repeat(mean_grouped_rewards, self.args.num_samples_per_prompt_rollout, axis=0)
            std_grouped_rewards = scores_per_prompt.std(axis=-1)
            std_grouped_rewards = np.repeat(std_grouped_rewards, self.args.num_samples_per_prompt_rollout, axis=0)
            if self.args.advantage_normalization_type == "standard":
                advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
            elif self.args.advantage_normalization_type == "centered":
                advantages = scores - mean_grouped_rewards
            else:
                raise ValueError(f"Invalid advantage normalization type: {self.args.advantage_normalization_type}")

            if self.args.mask_truncated_completions:
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
                    pack_length=self.args.pack_length,
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
                expected_num_responses = (
                    self.args.num_samples_per_prompt_rollout * self.args.num_unique_prompts_rollout
                )

                unsolved_num_responses = (scores < self.args.max_possible_score).sum()
                sequence_lengths = np.array([len(response) for response in result.responses])
                sequence_length_solved = (
                    np.array([])
                    if np.all(scores == 0)
                    else np.array(sequence_lengths[scores == self.args.max_possible_score])
                )
                sequence_length_unsolved = (
                    np.array([])
                    if np.all(scores == self.args.max_possible_score)
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
                    "val/solve_rate_hist": None,
                    "val/total_reward_groups": real_num_responses / self.args.num_samples_per_prompt_rollout,
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

            self.local_queue.put(collated_data)

    def _prepare_collated_data_for_self(self, packed_sequences: PackedSequences) -> dict[str, list[torch.Tensor]]:
        from open_instruct.grpo_fast import collate_fn

        per_device_packed_query_responses = packed_sequences.query_responses
        per_device_packed_tool_masks = packed_sequences.tool_masks
        per_device_packed_attention_masks = packed_sequences.attention_masks
        per_device_packed_position_ids = packed_sequences.position_ids
        per_device_packed_advantages = packed_sequences.advantages
        per_device_packed_response_masks = packed_sequences.response_masks
        per_device_packed_vllm_logprobs = packed_sequences.vllm_logprobs

        b_inds = np.random.permutation(len(per_device_packed_query_responses))
        collated_query_responses = []
        collated_tool_masks = []
        collated_attention_masks = []
        collated_position_ids = []
        collated_response_masks = []
        collated_advantages = []
        collated_vllm_logprobs = []
        for j in range(0, len(per_device_packed_query_responses), self.args.per_device_train_batch_size):
            micro_range = b_inds[j : j + self.args.per_device_train_batch_size]
            collated_query_responses.append(
                collate_fn(
                    [per_device_packed_query_responses[idx] for idx in micro_range], self.tokenizer.pad_token_id, True
                )
            )
            collated_tool_masks.append(collate_fn([per_device_packed_tool_masks[idx] for idx in micro_range], 0, True))
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

        return {
            "collated_query_responses": collated_query_responses,
            "collated_tool_masks": collated_tool_masks,
            "collated_attention_masks": collated_attention_masks,
            "collated_position_ids": collated_position_ids,
            "collated_advantages": collated_advantages,
            "collated_response_masks": collated_response_masks,
            "collated_vllm_logprobs": collated_vllm_logprobs,
        }

    def shutdown(self):
        self.shutdown_requested = True
        if self.background_thread is not None:
            self.background_thread.join(timeout=5.0)
