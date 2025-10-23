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
# ---------------------------------------------------------------------
# Part of the code is adapted from https://github.com/OpenRLHF/OpenRLHF
# which has the following license:
# Copyright [yyyy] [name of copyright owner]
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
# isort: off
import os
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor

from open_instruct import utils

# isort: on
import asyncio
import json
import logging
import shutil
import threading
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import tinker
import tinker.types as tinker_types
import torch
import torch.utils
import torch.utils.data
import wandb
from rich.pretty import pprint
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from open_instruct import logger_utils
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import (
    Args,
    ShutdownSentinel,
    collate_fn,
    load_data_from_packing_thread,
    make_reward_fn,
    make_tokenizer,
    next_batch,
    setup_datasets,
    setup_experiment_tracking,
    setup_runtime_variables,
)
from open_instruct.model_utils import (
    Batch,
    ModelConfig,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
)
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.rl_utils import Timer, pack_sequences
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    is_beaker_job,
    maybe_update_beaker_description,
    repeat_each,
)

logger = logger_utils.setup_logger(__name__)

INVALID_LOGPROB = 1.0


@dataclass
class GenerationConfig:
    temperature: float
    n: int
    seed: Optional[int]


@dataclass
class TinkerArgs(Args):
    tinker_base_model: str = "meta-llama/Llama-3.2-1B"
    tinker_lora_rank: int = 32
    tinker_num_workers: int = 8

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self, "tools"):
            delattr(self, "tools")
        if hasattr(self, "mask_tool_use"):
            delattr(self, "mask_tool_use")


class MetricsTracker:
    """A simple class to prellocate all metrics in an array
    so we can do only one allreduce operation to get the metrics mean"""

    def __init__(self, max_metrics: int = 32, device: torch.device = torch.device("cuda")):
        self.metrics = torch.zeros(max_metrics, device=device)
        self.names2idx = {}
        self.current_idx = 0
        self.max_metrics = max_metrics

    def add(self, name: str, value: torch.tensor):
        if name not in self.names2idx:
            if self.current_idx >= self.max_metrics:
                raise ValueError(f"Exceeded maximum number of metrics ({self.max_metrics})")
            self.names2idx[name] = self.current_idx
            self.current_idx += 1

        self.metrics[self.names2idx[name]] = value
        return self

    def get_metrics_list(self) -> dict[str, float]:
        metrics_list = self.metrics.tolist()
        return {name: metrics_list[idx] for name, idx in self.names2idx.items()}


class ShufflingIterator:
    def __init__(self, data: np.ndarray, batch_size: int, seed: Optional[int] = None):
        self.data = data.copy()
        self.batch_size = batch_size
        self.index = 0
        self.epoch_number = 0
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.data)

        # Ensure the effective dataset size is divisible by batch_size
        self.effective_size = len(self.data) - (len(self.data) % batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        return self

    def __next__(self) -> List[int]:
        if self.index >= self.effective_size:
            self.index = 0
            self.epoch_number += 1
            self.rng.shuffle(self.data)

        end_index = self.index + self.batch_size
        batch = self.data[self.index : end_index].tolist()
        self.index = end_index

        return batch

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the iterator for checkpointing."""
        return {
            "index": self.index,
            "epoch_number": self.epoch_number,
            "data": self.data.copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the iterator state from a checkpoint."""
        self.index = state["index"]
        self.epoch_number = state["epoch_number"]
        self.data = state["data"].copy()
        self.rng.bit_generator.state = state["rng_state"]


class TinkerTrainingManager:
    def __init__(self, base_model: str, lora_rank: int = 32):
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.service_client = tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(base_model=base_model, rank=lora_rank)
        logger.info(f"Initialized TinkerTrainingManager with model {base_model}, LoRA rank {lora_rank}")

    def train_step(
        self, processed_examples: List[tinker_types.Datum], learning_rate: float, loss_fn: str = "importance_sampling"
    ):
        fwdbwd_future = self.training_client.forward_backward(processed_examples, loss_fn)
        fwdbwd_result = fwdbwd_future.result()
        optim_future = self.training_client.optim_step(tinker_types.AdamParams(learning_rate=learning_rate))
        optim_result = optim_future.result()
        return fwdbwd_result, optim_result

    def get_sampling_client(self, name: str):
        sampling_path = self.training_client.save_weights_for_sampler(name=name).result().path
        sampling_client = self.service_client.create_sampling_client(model_path=sampling_path)
        logger.info(f"Created sampling client with checkpoint {name} at path {sampling_path}")
        return sampling_client

    def save_checkpoint(self, name: str):
        resume_path = self.training_client.save_state(name=name).result().path
        logger.info(f"Saved training checkpoint {name} at path {resume_path}")
        return resume_path

    def load_checkpoint(self, path: str):
        self.training_client.load_state(path)
        logger.info(f"Loaded training checkpoint from path {path}")


class LLMTinkerActor:
    def __init__(
        self,
        sampling_client,
        num_workers: int,
        max_tokens: int,
        temperature: float,
        stop_strings: List[str],
        tokenizer: PreTrainedTokenizer,
    ):
        self.sampling_client = sampling_client
        self.num_workers = num_workers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.prompt_queue = Queue()
        self.results_queue = Queue()
        self.eval_results_queue = Queue()
        self.active_futures = {}
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.running = False
        self.worker_thread = None
        logger.info(f"Initialized LLMTinkerActor with {num_workers} workers")

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_prompts, daemon=True)
        self.worker_thread.start()
        logger.info("Started LLMTinkerActor worker thread")

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=10)
        self.executor.shutdown(wait=True)
        logger.info("Stopped LLMTinkerActor")

    def update_sampling_client(self, new_sampling_client):
        self.sampling_client = new_sampling_client
        logger.info("Updated sampling client in LLMTinkerActor")

    def submit_request(self, prompt_request):
        self.prompt_queue.put(prompt_request)

    def _sample_single(self, prompt_tokens: List[int], n: int, seed: Optional[int] = None):
        prompt_input = tinker_types.ModelInput.from_ints(prompt_tokens)
        params = tinker_types.SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature, stop=self.stop_strings or [], seed=seed
        )
        results = []
        for i in range(n):
            params_i = params
            if seed is not None:
                params_i = tinker_types.SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self.stop_strings or [],
                    seed=seed + i,
                )
            future = self.sampling_client.sample(prompt=prompt_input, sampling_params=params_i, num_samples=1)
            result = future.result()
            results.append(result)
        return results

    def _process_prompts(self):
        while self.running:
            try:
                request = self.prompt_queue.get(timeout=1.0)
                if isinstance(request, ShutdownSentinel):
                    break
                start_time = time.perf_counter()
                sampling_results = self._sample_single(
                    request.prompt, request.generation_config.n, request.generation_config.seed
                )
                responses = []
                logprobs_list = []
                finish_reasons = []
                for sampling_result in sampling_results:
                    for sample in sampling_result.samples:
                        response_tokens = sample.token_ids
                        responses.append(response_tokens)
                        logprobs = sample.logprobs if hasattr(sample, "logprobs") else [0.0] * len(response_tokens)
                        logprobs_list.append(logprobs)
                        finish_reasons.append("stop" if sample.finish_reason == "stop" else "length")
                result = GenerationResult(
                    responses=responses,
                    finish_reasons=finish_reasons,
                    masks=[[1] * len(resp) for resp in responses],
                    request_info=RequestInfo(
                        num_calls=[0] * len(responses),
                        timeouts=[False] * len(responses),
                        tool_errors=[""] * len(responses),
                        tool_outputs=[""] * len(responses),
                        tool_runtimes=[0.0] * len(responses),
                        tool_calleds=[False] * len(responses),
                    ),
                    dataset_index=request.dataset_index,
                    epoch_number=request.epoch_number,
                    token_statistics=TokenStatistics(
                        num_prompt_tokens=len(request.prompt),
                        num_response_tokens=sum(len(resp) for resp in responses),
                        generation_time=time.perf_counter() - start_time,
                    ),
                    start_time=start_time,
                    logprobs=logprobs_list,
                )
                if request.is_eval:
                    self.eval_results_queue.put(result)
                else:
                    self.results_queue.put(result)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in LLMTinkerActor worker: {e}")


def convert_to_tinker_format(
    token_ids: torch.Tensor, advantages: torch.Tensor, response_masks: torch.Tensor
) -> List[tinker_types.Datum]:
    examples = []
    for ids, adv, mask in zip(token_ids, advantages, response_masks):
        datum = tinker_types.Datum(
            loss_fn_inputs={
                "token_ids": ids.cpu().numpy(),
                "advantages": adv.cpu().numpy(),
                "weights": mask.cpu().numpy(),
            }
        )
        examples.append(datum)
    return examples


class PendingQueriesMap:
    """Thread-safe map for tracking pending queries with reference counting."""

    def __init__(self):
        self._map = {}  # dataset_idx -> (query, ground_truth, dataset, count)
        self._lock = threading.Lock()

    def insert(self, dataset_idx, query, ground_truth, dataset, raw_query):
        """Insert or increment count for a dataset index."""
        with self._lock:
            if dataset_idx in self._map:
                # Already exists - just increment count
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
                # New entry - count starts at 1
                self._map[dataset_idx] = (query, ground_truth, dataset, raw_query, 1)

    def pop(self, dataset_idx):
        """Retrieve data and decrement count. Removes entry when count reaches 0."""
        with self._lock:
            if dataset_idx not in self._map:
                raise RuntimeError(f"Dataset index {dataset_idx} not found in pending_queries_map")

            query, ground_truth, dataset, raw_query, count = self._map[dataset_idx]

            if count > 1:
                # More results expected - just decrement
                self._map[dataset_idx] = (query, ground_truth, dataset, raw_query, count - 1)
            else:
                # Last result - remove entry
                del self._map[dataset_idx]

            return query, ground_truth, dataset, raw_query

    def __len__(self):
        """Return the number of entries in the map."""
        with self._lock:
            return len(self._map)

    def __contains__(self, dataset_idx):
        """Check if a dataset index is in the map."""
        with self._lock:
            return dataset_idx in self._map

    def __getitem__(self, dataset_idx):
        """Get the value for a dataset index."""
        with self._lock:
            return self._map[dataset_idx]

    def keys(self):
        """Return a view of the keys in the map."""
        with self._lock:
            return list(self._map.keys())


def accumulate_inference_batches(
    inference_results_Q: Queue,
    pending_queries_map: PendingQueriesMap,
    args: TinkerArgs,
    generation_config: GenerationConfig,
    num_prompts: int,
    timeout: Optional[float] = None,
) -> tuple[GenerationResult, Batch]:
    """Accumulate multiple inference results into a single training batch.

    Args:
        inference_results_Q: Queue containing individual GenerationResult objects (one per prompt)
        pending_queries_map: PendingQueriesMap instance for thread-safe query tracking
        args: Arguments containing vllm_num_engines and batch size info
        generation_config: Generation config containing n (number of samples per prompt)
        num_prompts: Number of prompts to accumulate
        timeout: Optional timeout in seconds for queue get operations. If None, blocks indefinitely.

    Raises:
        queue.Empty: If timeout is specified and no data is available within timeout.

    Returns:
        Tuple of (combined_result, Batch with queries, ground_truths, datasets, prompt_lengths, response_lengths)
        or (ShutdownSentinel, None, None, None) if shutdown signal received
    """
    results = []
    all_queries = []
    all_ground_truths = []
    all_datasets = []
    all_raw_queries = []
    for i in tqdm(
        range(num_prompts),
        total=num_prompts,
        desc=f"Accumulating results from {num_prompts} prompts",
        bar_format="{l_bar}{bar}{r_bar}\n",
        disable=not args.verbose,
    ):
        result = inference_results_Q.get(timeout=timeout)

        if isinstance(result, ShutdownSentinel):
            return result, None, None, None

        # Validate that each individual result has the expected number of responses
        assert len(result.responses) == generation_config.n, (
            f"Mismatch: individual prompt result has {len(result.responses)} responses "
            f"but expected {generation_config.n} samples per prompt. "
            f"Dataset index: {result.dataset_index}, Epoch: {result.epoch_number}"
        )

        query, ground_truth, dataset, raw_query = pending_queries_map.pop(result.dataset_index)

        results.append(result)
        all_queries.append(query)
        all_ground_truths.append(ground_truth)
        all_datasets.append(dataset)
        all_raw_queries.append(raw_query)

    # Combine all results into a single GenerationResult
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

        prompt_lengths.append(len(all_queries[i]))

        for response in result.responses:
            response_lengths.append(len(response))

        total_prompt_tokens += result.token_statistics.num_prompt_tokens
        total_response_tokens += result.token_statistics.num_response_tokens
        max_generation_time = max(max_generation_time, result.token_statistics.generation_time)

    # Use the maximum generation time across engines since they work in parallel
    # This avoids including queue overhead and accumulation time in MFU/MBU calculations
    total_generation_time = max_generation_time

    accumulated_stats = TokenStatistics(
        num_prompt_tokens=total_prompt_tokens,
        num_response_tokens=total_response_tokens,
        generation_time=total_generation_time,
        earliest_start_time=earliest_start_time,
    )

    # Create combined RequestInfo
    combined_request_info = RequestInfo(
        num_calls=combined_num_calls,
        timeouts=combined_timeouts,
        tool_errors=combined_tool_errors,
        tool_outputs=combined_tool_outputs,
        tool_runtimes=combined_tool_runtimes,
        tool_calleds=combined_tool_calleds,
    )

    # Create combined GenerationResult
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

    # Note: We don't have dataset_indices here, but they're not needed for the returned batch
    batch = Batch(
        queries=all_queries,
        ground_truths=all_ground_truths,
        datasets=all_datasets,
        raw_queries=all_raw_queries,
        indices=None,  # Not meaningful for combined results
    )
    return combined_result, batch, prompt_lengths, response_lengths


def data_preparation_thread(
    reward_fn: Callable,
    inference_results_Q: Queue,
    packed_sequences_Q: Queue,
    pending_queries_map: dict,
    args: TinkerArgs,
    tokenizer: PreTrainedTokenizer,
    num_training_steps: int,
    generation_config,
    resume_training_step: int,
):
    for training_step in range(resume_training_step, num_training_steps + 1):
        # Streaming accumulation: collect results as they arrive
        with Timer("🚀 [Data Preparation Thread] Getting response ids") as timer:
            result, batch, prompt_lengths, response_lengths = accumulate_inference_batches(
                inference_results_Q,
                pending_queries_map,
                args,
                generation_config,
                num_prompts=args.num_unique_prompts_rollout,
            )
            if isinstance(result, ShutdownSentinel):
                logger.info("[Data Preparation Thread] Received shutdown sentinel, exiting")
                return

        getting_response_time = timer.duration

        # ------------------------------------------------------------------------------------------------
        # Pack sequences
        if args.num_samples_per_prompt_rollout > 1:
            batch = Batch(
                queries=repeat_each(batch.queries, args.num_samples_per_prompt_rollout),
                ground_truths=repeat_each(batch.ground_truths, args.num_samples_per_prompt_rollout),
                datasets=repeat_each(batch.datasets, args.num_samples_per_prompt_rollout),
                raw_queries=repeat_each(batch.raw_queries, args.num_samples_per_prompt_rollout),
                indices=repeat_each(batch.indices, args.num_samples_per_prompt_rollout) if batch.indices else None,
            )
            good_outputs = [
                len(result.request_info.tool_outputs[i]) > 0
                and result.request_info.tool_calleds[i]
                and not result.request_info.timeouts[i]
                and not result.request_info.tool_errors[i]
                for i in range(len(result.request_info.tool_outputs))
            ]
        for i in range(len(result.finish_reasons)):
            if result.finish_reasons[i] == "stop" and len(result.responses[i]) == 0:
                result.responses[i].append(tokenizer.eos_token_id)
                result.masks[i].append(1)
                result.logprobs[i].append(float("nan"))
        with Timer("🔥 [Data Preparation Thread] Decoding responses", noop=True):
            decoded_responses = tokenizer.batch_decode(result.responses, skip_special_tokens=True)
            decoded_queries = batch.raw_queries
            stop_rate = sum(int(finish_reason == "stop") for finish_reason in result.finish_reasons) / len(
                result.finish_reasons
            )

        with Timer("💰 [Data Preparation Thread] Calculating rewards and advantages"):
            scores, reward_metrics = asyncio.run(
                reward_fn(
                    result.responses,
                    decoded_responses,
                    batch,
                    result.finish_reasons,
                    result.request_info,
                    decoded_queries,
                )
            )
            scores = np.array(scores)
            scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
            mean_grouped_rewards = scores_per_prompt.mean(axis=-1)
            mean_grouped_rewards = np.repeat(mean_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0)
            std_grouped_rewards = scores_per_prompt.std(axis=-1)
            std_grouped_rewards = np.repeat(std_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0)
            if args.advantage_normalization_type == "standard":
                advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
            elif args.advantage_normalization_type == "centered":
                advantages = scores - mean_grouped_rewards
            else:
                raise ValueError(f"Invalid advantage normalization type: {args.advantage_normalization_type}")

        with Timer("📦 [Data Preparation Thread] Filtering sequences"):
            # Here we get the max possible score for each prompt, and see how many prompts are unsolved
            max_possible_score = 0
            if args.apply_verifiable_reward:
                max_possible_score += args.verification_reward
            if args.apply_r1_style_format_reward and args.additive_format_reward:
                max_possible_score += args.r1_style_format_reward
            unsolved_batch_size_ratio = ((scores != max_possible_score) > 0).sum() / len(scores)
            # In GRPO, if the std of grouped rewards is 0, then there is zero gradient for the batch
            # of args.num_samples_per_prompt_rollout responses, so we need to filter out those batches
            non_zero_std_mask = scores_per_prompt.std(axis=-1) != 0
            real_batch_size_ratio = non_zero_std_mask.sum() * args.num_samples_per_prompt_rollout / len(scores)
            expanded_mask = np.repeat(non_zero_std_mask, args.num_samples_per_prompt_rollout)
            non_zero_gradient_index = np.where(expanded_mask)[0]

            # Log zero-gradient filtering statistics
            num_zero_std_prompts = (~non_zero_std_mask).sum()
            num_filtered_responses = len(scores) - len(non_zero_gradient_index)
            if num_filtered_responses > 0:
                logger.info(
                    f"[Zero-gradient filtering] Filtered {num_zero_std_prompts} prompts with zero std "
                    f"({num_filtered_responses} responses). Retention rate: {len(non_zero_gradient_index) / len(scores):.2%}"
                )

            advantages = advantages[non_zero_gradient_index]
            original_batch_size = len(scores)
            scores = scores[non_zero_gradient_index]
            responses = [result.responses[i] for i in non_zero_gradient_index]
            masks = [result.masks[i] for i in non_zero_gradient_index]
            batch = batch[non_zero_gradient_index.tolist()]
            finish_reasons = [result.finish_reasons[i] for i in non_zero_gradient_index]
            vllm_logprobs = [result.logprobs[i] for i in non_zero_gradient_index]
            if args.mask_truncated_completions:
                stop_idxes = torch.tensor([i for i in range(len(finish_reasons)) if finish_reasons[i] == "stop"])
                num_truncated = len(finish_reasons) - len(stop_idxes)
                if num_truncated > 0:
                    logger.info(
                        f"[Truncated completions filtering] Filtered {num_truncated} responses that didn't finish with 'stop'. "
                        f"Retention rate: {len(stop_idxes) / len(finish_reasons):.2%}"
                    )
                scores = scores[stop_idxes]
                advantages = advantages[stop_idxes]
                responses = [responses[i] for i in stop_idxes]
                masks = [masks[i] for i in stop_idxes]
                batch = batch[stop_idxes.tolist()]
                finish_reasons = [finish_reasons[i] for i in stop_idxes]
                vllm_logprobs = [vllm_logprobs[i] for i in stop_idxes]

            if args.fill_completions:
                with Timer("⏱ [Data Preparation Thread] Refill completions"):
                    current_batch_size = len(scores)
                    original_prompt_cnt = original_batch_size // args.num_samples_per_prompt_rollout
                    current_prompt_cnt = current_batch_size // args.num_samples_per_prompt_rollout
                    need_to_fill_prompt = original_prompt_cnt - current_prompt_cnt
                    k = args.num_samples_per_prompt_rollout

                    if need_to_fill_prompt > 0 and current_prompt_cnt > 0:
                        scores_matrix = scores.reshape(current_prompt_cnt, k)
                        stds = scores_matrix.std(axis=1) + 1e-8
                        probs = stds / stds.sum()

                        logger.info(
                            f"[Refill completions] Need to fill {need_to_fill_prompt} prompts to maintain batch size. "
                            f"Original: {original_prompt_cnt}, Current: {current_prompt_cnt}"
                        )

                        sampled_prompt_ids = np.random.choice(
                            current_prompt_cnt, size=need_to_fill_prompt, replace=True, p=probs
                        )

                        sampled_indices = []
                        for pid in sampled_prompt_ids:
                            start = pid * k
                            sampled_indices.extend(range(start, start + k))

                        advantages = np.concatenate([advantages, advantages[sampled_indices]])
                        scores = np.concatenate([scores, scores[sampled_indices]])
                        responses += [responses[i] for i in sampled_indices]
                        masks += [masks[i] for i in sampled_indices]

                        sampled_batch = batch[sampled_indices]

                        batch = Batch(
                            queries=batch.queries + sampled_batch.queries,
                            ground_truths=batch.ground_truths + sampled_batch.ground_truths,
                            datasets=batch.datasets + sampled_batch.datasets,
                            indices=batch.indices + sampled_batch.indices if batch.indices is not None else None,
                        )

                        finish_reasons += [finish_reasons[i] for i in sampled_indices]
                        vllm_logprobs += [vllm_logprobs[i] for i in sampled_indices]

                        logger.info(
                            f"📊 Duplicated {need_to_fill_prompt} prompts from {len(sampled_indices)} total responses"
                        )

            # Count groups with all zero rewards
            all_zero_groups = (scores_per_prompt == 0).all(axis=-1).sum()
            total_groups = len(scores_per_prompt)
            logger.info(
                f"[Reward Summary] Groups with all zero rewards: {all_zero_groups}/{total_groups} "
                f"({all_zero_groups / total_groups:.1%})"
            )

        with Timer("📦 [Data Preparation Thread] Packing sequences"):
            packed_sequences = pack_sequences(
                queries=batch.queries,
                responses=responses,
                masks=masks,
                pack_length=args.pack_length,
                pad_token_id=tokenizer.pad_token_id,
                vllm_logprobs=vllm_logprobs,
            )
            num_new_tokens = sum(len(seq) for seq in packed_sequences.query_responses)
            # Vectorized advantage calculation: create a lookup array where each index corresponds to a response mask value
            # and each value is the corresponding advantage score: index 0 is set to 0 since response masks start from 1 (1-indexed)
            lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
            lookup_advantages[1:] = advantages
            packed_advantages = [
                torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
                for packed_mask in packed_sequences.response_masks
            ]
            packed_sequences.advantages = packed_advantages

        # if we have less batches than world size, we need to pad out so each world is fine
        # ideally, you should avoid this since its wasting computation.
        if args.allow_world_padding:
            with Timer("🤺 [Data Preparation Thread] Padding sequences for world size"):
                shortfall = args.world_size - len(packed_sequences.query_responses)
                if shortfall > 0:
                    logger.warning(
                        f"Padding {shortfall} sequences for world size. In future, you should adjust your compute this."
                    )
                    # construct "dummy" sequences for padding out the world size
                    dummy_qr = torch.tensor([tokenizer.pad_token_id, tokenizer.eos_token_id], dtype=torch.long)
                    dummy_tool_mask = torch.zeros_like(dummy_qr)
                    dummy_attention = torch.tensor([1, 1], dtype=torch.long)
                    dummy_position_ids = torch.arange(len(dummy_qr), dtype=torch.long)
                    dummy_response_mask = torch.zeros_like(dummy_qr)
                    dummy_advantage = torch.zeros_like(dummy_qr, dtype=torch.float)
                    # pad out the world size
                    for _ in range(shortfall):
                        packed_sequences.query_responses.append(dummy_qr)
                        packed_sequences.tool_masks.append(dummy_tool_mask)
                        packed_sequences.attention_masks.append(dummy_attention)
                        packed_sequences.position_ids.append(dummy_position_ids)
                        packed_sequences.response_masks.append(dummy_response_mask)
                        packed_sequences.advantages.append(dummy_advantage)

        with Timer("🔄 [Data Preparation Thread] Prepare collated data for each worker"):
            B = (
                len(packed_sequences.query_responses) // args.world_size
            )  # essentially doing `drop_last=True`, which is fine.
            collated_data = []
            for i in range(args.world_size):
                per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
                per_device_packed_tool_masks = packed_sequences.tool_masks[B * i : B * (i + 1)]
                per_device_packed_attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
                per_device_packed_position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
                per_device_packed_advantages = packed_sequences.advantages[B * i : B * (i + 1)]
                per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]
                per_device_packed_vllm_logprobs = packed_sequences.vllm_logprobs[B * i : B * (i + 1)]

                # Shuffle the batch and collate the data
                b_inds = np.random.permutation(len(per_device_packed_query_responses))
                collated_query_responses = []
                collated_tool_masks = []
                collated_attention_masks = []
                collated_position_ids = []
                collated_response_masks = []
                collated_advantages = []
                collated_vllm_logprobs = []
                for j in range(0, len(per_device_packed_query_responses), args.per_device_train_batch_size):
                    micro_range = b_inds[j : j + args.per_device_train_batch_size]
                    collated_query_responses.append(
                        collate_fn(
                            [per_device_packed_query_responses[idx] for idx in micro_range], tokenizer.pad_token_id
                        )
                    )
                    collated_tool_masks.append(
                        collate_fn([per_device_packed_tool_masks[idx] for idx in micro_range], 0)
                    )
                    collated_attention_masks.append(
                        collate_fn([per_device_packed_attention_masks[idx] for idx in micro_range], 0)
                    )
                    collated_position_ids.append(
                        collate_fn([per_device_packed_position_ids[idx] for idx in micro_range], 0)
                    )
                    collated_response_masks.append(
                        collate_fn([per_device_packed_response_masks[idx] for idx in micro_range], 0)
                    )
                    collated_advantages.append(
                        collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0)
                    )
                    collated_vllm_logprobs.append(
                        collate_fn([per_device_packed_vllm_logprobs[idx] for idx in micro_range], 0)
                    )
                collated_data.append(
                    {
                        "collated_query_responses": collated_query_responses,
                        "collated_tool_masks": collated_tool_masks,
                        "collated_attention_masks": collated_attention_masks,
                        "collated_position_ids": collated_position_ids,
                        "collated_advantages": collated_advantages,
                        "collated_response_masks": collated_response_masks,
                        "collated_vllm_logprobs": collated_vllm_logprobs,
                    }
                )

        # Create a result package with metrics and data
        if len(responses) == 0:
            # Handle empty responses case
            # in this case, we won't log metrics, so it should be fine.
            metrics = {}
        else:
            sequence_lengths = np.array([len(response) for response in responses])
            sequence_length_solved = (
                np.array([]) if np.all(scores == 0) else np.array(sequence_lengths[scores == max_possible_score])
            )
            sequence_length_unsolved = (
                np.array([]) if np.all(scores == max_possible_score) else np.array(sequence_lengths[scores == 0])
            )

            # Use the already calculated reward summary metrics for wandb
            all_zero_groups_ratio = all_zero_groups / total_groups if total_groups > 0 else 0

            metrics = {
                "scores": np.array(scores).mean(),
                "real_batch_size_ratio": real_batch_size_ratio,
                "unsolved_batch_size_ratio": unsolved_batch_size_ratio,
                "packed_ratio": len(packed_sequences.query_responses) / len(responses) if len(responses) > 0 else 0,
                "val/all_zero_reward_groups": all_zero_groups,
                "val/all_zero_reward_groups_ratio": all_zero_groups_ratio,
                "val/total_reward_groups": total_groups,
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
                "val/tool_errors_rate": np.array([len(item) > 0 for item in result.request_info.tool_errors]).mean(),
                "val/good_outputs_rate": np.array(good_outputs).mean(),
                "val/tool_runtimes_rate": np.array(result.request_info.tool_runtimes).mean(),
                "val/tool_calleds_rate": np.array(result.request_info.tool_calleds).mean(),
                "time/getting_response": getting_response_time,
                **reward_metrics,
            }

            total_tokens = result.token_statistics.num_prompt_tokens + result.token_statistics.num_response_tokens
            metrics["val/actor_tokens_per_second"] = total_tokens / result.token_statistics.generation_time

        if args.save_traces:
            traces = {
                "scores": scores.tolist(),
                "finish_reasons": finish_reasons,
                "responses": responses,
                "training_step": training_step,
                **asdict(batch),  # Unpack all batch fields
                **reward_metrics,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            with open(f"{args.output_dir}/traces_{args.run_name}.jsonl", "a") as f:
                json.dump(traces, f)
                f.write("\n")

        if len(responses) == 0:
            logger.warning(f"No responses in batch {training_step}.")

        # Put the packed sequences and metrics into the output queue
        packed_sequences_Q.put(
            {
                "packed_sequences": packed_sequences,  # for debugging purposes
                "collated_data": collated_data,
                "metrics": metrics,
                "responses_count": len(responses),
                "num_new_tokens": num_new_tokens,
                "B": B,
                "prompt_lengths": prompt_lengths,
                "response_lengths": response_lengths,
            }
        )


def create_model_and_optimizer(
    args: TinkerArgs,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    beaker_config: BeakerRuntimeConfig,
    wandb_url: str,
    tokenizer: PreTrainedTokenizer,
):
    logger.info("Creating TinkerTrainingManager and LLMTinkerActor...")
    training_manager = TinkerTrainingManager(base_model=args.tinker_base_model, lora_rank=args.tinker_lora_rank)
    sampling_client = training_manager.get_sampling_client(name="init")
    llm_actor = LLMTinkerActor(
        sampling_client=sampling_client,
        num_workers=args.tinker_num_workers,
        max_tokens=args.response_length,
        temperature=args.temperature,
        stop_strings=args.stop_strings,
        tokenizer=tokenizer,
    )
    llm_actor.start()
    resume_training_step = 1
    episode = 0
    logger.info("======== ✅ Tinker training manager and LLM actor initialized =========")
    return training_manager, llm_actor, resume_training_step, episode


def create_generation_configs(args: TinkerArgs):
    generation_config = GenerationConfig(
        temperature=args.temperature, n=args.num_samples_per_prompt_rollout, seed=args.seed
    )
    eval_generation_config = GenerationConfig(temperature=0.0, n=1, seed=args.seed)
    return {"train": generation_config, "eval": eval_generation_config}


def split_and_insert_batch(
    batch: Batch,
    epoch_number: int,
    training_step: int,
    pending_queries_map: PendingQueriesMap,
    llm_actor: LLMTinkerActor,
    generation_config,
    is_eval: bool,
) -> None:
    """Split a batch into multiple inference batches and insert individual prompts into queues and mapping."""
    for idx, query, ground_truth, dataset, raw_query in zip(
        batch.indices, batch.queries, batch.ground_truths, batch.datasets, batch.raw_queries
    ):
        pending_queries_map.insert(idx, query, ground_truth, dataset, raw_query)
        llm_actor.submit_request(
            PromptRequest(
                prompt=query,
                generation_config=generation_config,
                epoch_number=epoch_number,
                training_step=training_step,
                dataset_index=idx,
                is_eval=is_eval,
            )
        )


def weight_sync_tinker(
    training_manager: TinkerTrainingManager, llm_actor: LLMTinkerActor, training_step: int
) -> Dict[str, float]:
    with Timer("[Weight Sync]") as timer:
        logger.debug(f"[Weight Sync] Creating new sampling client for step {training_step}")
        new_sampling_client = training_manager.get_sampling_client(name=f"step_{training_step}")
        llm_actor.update_sampling_client(new_sampling_client)
    return {"time/weight_sync": timer.duration}


def one_training_step_tinker(
    args: TinkerArgs,
    training_manager: TinkerTrainingManager,
    collated_data,
    tokenizer,
    data_thread_metrics,
    episode,
    training_step,
    num_total_tokens,
    num_step_tokens,
    start_time,
    train_dataset,
    training_start_time,
):
    with Timer("[Main Thread] 🗡️ Training") as train_timer:
        processed_examples = convert_to_tinker_format(
            collated_data["query_responses"], collated_data["advantages"], collated_data["response_masks"]
        )
        current_lr = args.learning_rate
        fwdbwd_result, optim_result = training_manager.train_step(processed_examples, current_lr)
    step_time = time.perf_counter() - start_time
    total_training_time = time.perf_counter() - training_start_time
    metrics = {
        "episode": episode,
        "global_step": episode,
        "training_step": training_step,
        "val/num_total_tokens": num_total_tokens,
        "val/num_step_tokens": num_step_tokens,
        "epoch": episode / args.num_samples_per_prompt_rollout / len(train_dataset),
        "learner_tokens_per_second_overall": num_total_tokens / total_training_time,
        "learner_tokens_per_second_step": num_step_tokens / step_time,
        "time/total": step_time,
        "time/training": train_timer.duration,
        **data_thread_metrics,
    }
    return metrics


def maybe_evaluate(
    args: TinkerArgs,
    training_step: int,
    evaluation_inference_results_Q: Queue,
    tokenizer,
    reward_fn,
    episode,
    eval_pending_queries_map: PendingQueriesMap,
    eval_generation_config,
    generate_metrics_Q: Queue,
    num_eval_prompts: int,
):
    """Optionally evaluate the model."""
    try:
        # timeout 0.01 if this is the last training step or we're not evaluating
        # otherwise, wait to get the last evaluation generations (long timeout just in case)
        timeout = 0.01 if (training_step < args.num_training_steps or args.local_eval_every < 0) else 100

        # Accumulate evaluation results
        eval_result, eval_batch, _, _ = accumulate_inference_batches(
            evaluation_inference_results_Q,
            eval_pending_queries_map,
            args,
            eval_generation_config,
            num_prompts=num_eval_prompts,
            timeout=timeout,
        )

        logger.info("[Main Thread] 📊 Evaluation responses received")

        eval_generate_metrics = {}
        try:
            eval_generate_metrics = generate_metrics_Q.get_nowait()
        except Empty:
            logger.info("[Main Thread] didn't get eval generation metrics")

        eval_sequence_lengths = np.array([len(response) for response in eval_result.responses])
        eval_decoded_responses = tokenizer.batch_decode(eval_result.responses, skip_special_tokens=True)
        eval_stop_rate = sum(int(finish_reason == "stop") for finish_reason in eval_result.finish_reasons) / len(
            eval_result.finish_reasons
        )

        # get and log evaluation metrics
        eval_scores, eval_reward_metrics = asyncio.run(
            reward_fn(
                eval_result.responses,
                eval_decoded_responses,
                eval_batch if eval_batch else Batch(queries=[], ground_truths=[], datasets=[], indices=None),
                eval_result.finish_reasons,
                eval_result.request_info,
            )
        )
        eval_reward_metrics = {f"eval/{key}": val for key, val in eval_reward_metrics.items()}
        eval_metrics = {
            "eval/scores": np.array(eval_scores).mean(),
            "eval/sequence_lengths": eval_sequence_lengths.mean(),
            "eval/sequence_lengths_min": eval_sequence_lengths.min(),
            "eval/sequence_lengths_max": eval_sequence_lengths.max(),
            "eval/stop_rate": eval_stop_rate,
            **eval_reward_metrics,
        }
        if "time/generation" in eval_generate_metrics:
            eval_metrics["eval/generation_time"] = eval_generate_metrics["time/generation"]

        total_tokens = (
            eval_result.token_statistics.num_prompt_tokens + eval_result.token_statistics.num_response_tokens
        )
        eval_metrics["eval/actor_tokens_per_second"] = total_tokens / eval_result.token_statistics.generation_time

        print_rich_single_line_metrics(eval_metrics)

        table = {}
        table["prompt"] = tokenizer.batch_decode(eval_batch.queries if eval_batch else [])
        table["response"] = eval_decoded_responses
        table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
        table["scores"] = eval_scores
        table["ground_truth"] = eval_batch.ground_truths if eval_batch else []
        df = pd.DataFrame(table)

        if args.with_tracking:
            eval_metrics["sample_completions"] = wandb.Table(dataframe=df)
            wandb.log(eval_metrics, step=episode)
        else:
            print_rich_table(df.iloc[:1])
        del table
    except Empty:
        logger.warning("[Main Thread] 🙈 Evaluation responses not received")


def run_training(
    args,
    tokenizer,
    train_dataset,
    eval_batch,
    training_manager,
    llm_actor,
    generation_configs,
    iter_dataloader,
    reward_fn,
    resume_training_step,
    episode,
    wandb_url,
    tc,
    stop_event,
    executor,
    packed_sequences_Q,
    pending_queries_map,
    eval_pending_queries_map,
    generate_metrics_Q,
    weight_sync_metrics_Q,
    checkpoint_state=None,
):
    if resume_training_step > 1:
        logger.info(f"[Main Thread] Resuming training from step {resume_training_step}")

    logger.info("======== ✅ data preparation thread starts =========")
    packing_future = executor.submit(
        data_preparation_thread,
        reward_fn,
        llm_actor.results_queue,
        packed_sequences_Q,
        pending_queries_map,
        args,
        tokenizer,
        args.num_training_steps,
        generation_configs["train"],
        resume_training_step,
    )

    def health_check_fn():
        if packing_future.done():
            packing_future.result()

    # Send initial data to ensure we have a N-step offset.
    for _ in range(args.async_steps):
        dataset_indices = next(iter_dataloader)
        batch = next_batch(dataset_indices, train_dataset)
        split_and_insert_batch(
            batch,
            iter_dataloader.epoch_number,
            resume_training_step,
            pending_queries_map,
            llm_actor,
            generation_configs["train"],
            is_eval=False,
        )
    if checkpoint_state and "num_total_tokens" in checkpoint_state:
        num_total_tokens = checkpoint_state["num_total_tokens"]
        logger.info(f"Restored num_total_tokens: {num_total_tokens}")
    else:
        num_total_tokens = 0

    training_start_time = time.perf_counter()  # Track overall training start time
    for training_step in range(resume_training_step, args.num_training_steps + 1):
        start_time = time.perf_counter()

        if (
            training_step == resume_training_step
            or training_step % args.update_progress_every == 0
            or training_step == args.num_training_steps
        ):
            maybe_update_beaker_description(
                current_step=training_step,
                total_steps=args.num_training_steps,
                start_time=training_start_time,
                wandb_url=wandb_url,
            )

        # Check if any of the threads have raised an exception.
        health_check_start = time.perf_counter()
        health_check_fn()
        health_check_time = time.perf_counter() - health_check_start

        episode += args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
        batch = next_batch(next(iter_dataloader), train_dataset)
        split_and_insert_batch(
            batch,
            iter_dataloader.epoch_number,
            training_step,
            pending_queries_map,
            llm_actor,
            generation_configs["train"],
            is_eval=False,
        )
        if (
            training_step % args.local_eval_every == 0
            and eval_batch is not None
            and (args.eval_on_step_0 or training_step > 1)
        ):
            split_and_insert_batch(
                eval_batch,
                iter_dataloader.epoch_number,
                training_step,
                eval_pending_queries_map,
                llm_actor,
                generation_configs["eval"],
                is_eval=True,
            )

        collated_data, data_thread_metrics, num_total_tokens, num_step_tokens, prompt_lengths, response_lengths = (
            load_data_from_packing_thread(packed_sequences_Q, num_total_tokens, stop_event, health_check_fn)
        )
        if collated_data is None:
            continue

        for metrics_Q in [generate_metrics_Q, weight_sync_metrics_Q]:
            try:
                data_thread_metrics |= metrics_Q.get_nowait()
            except Empty:
                logger.info("[Main Thread] didn't get train generation metrics")

        data_thread_metrics["time/health_check"] = health_check_time

        metrics = one_training_step_tinker(
            args,
            training_manager,
            collated_data,
            tokenizer,
            data_thread_metrics,
            episode,
            training_step,
            num_total_tokens,
            num_step_tokens,
            start_time,
            train_dataset,
            training_start_time,
        )
        if args.with_tracking:
            wandb.log(metrics)
        weight_sync_metrics = weight_sync_tinker(training_manager, llm_actor, training_step)
        if args.with_tracking:
            wandb.log(weight_sync_metrics)

        # Checkpoint after one_training_step (or even if it was skipped)
        # This ensures we checkpoint progress even if the exact checkpoint step has no data
        if (
            args.checkpoint_state_freq > 0
            and training_step % args.checkpoint_state_freq == 0
            and args.checkpoint_state_dir is not None
        ):
            with Timer("[Main Thread] 🗡️ Saving checkpoint state"):
                # Save comprehensive client state including ShufflingIterator state
                client_state = {
                    "training_step": training_step,
                    "episode": episode,
                    "num_total_tokens": num_total_tokens,
                }

                # Save ShufflingIterator state
                if iter_dataloader is not None:
                    client_state["shuffling_iterator_state"] = iter_dataloader.get_state()

                checkpoint_name = f"checkpoint_{training_step}"
                checkpoint_path = training_manager.save_checkpoint(checkpoint_name)
                logger.info(f"Saved Tinker checkpoint state at step {training_step} to {checkpoint_path}")

                client_state_path = os.path.join(args.checkpoint_state_dir, "client_state.json")
                os.makedirs(args.checkpoint_state_dir, exist_ok=True)
                with open(client_state_path, "w") as f:
                    json.dump(client_state, f)
                logger.info(f"Saved client state to {client_state_path}")

        maybe_evaluate(
            args,
            training_step,
            llm_actor.eval_results_queue,
            tokenizer,
            reward_fn,
            episode,
            eval_pending_queries_map,
            generation_configs["eval"],
            generate_metrics_Q,
            len(eval_batch.queries) if eval_batch else 0,
        )

    if resume_training_step > args.num_training_steps:
        raise ValueError(f"Training didn't run since {resume_training_step=} > {args.num_training_steps=}")

    logger.info("Training complete! Final model saved via Tinker checkpoints.")
    return episode


def main(args: TinkerArgs, tc: TokenizerConfig, model_config: ModelConfig):
    tokenizer = make_tokenizer(tc, model_config)
    args = setup_runtime_variables(args)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

    beaker_config, wandb_url = setup_experiment_tracking(args, tc, model_config)

    train_dataset, eval_dataset = setup_datasets(args, tc, tokenizer)

    if len(train_dataset) < (needed := max(args.async_steps, 1) * args.num_unique_prompts_rollout):
        raise ValueError(
            f"Train dataset is too small! Is {len(train_dataset)} prompts, but {needed} are needed to have enough prompts for bsz and prefill. Try reducing async_steps or num_unique_prompts_rollout, or increasing the dataset size."
        )

    if args.cache_dataset_only:
        return

    pprint([args, model_config])

    training_manager, llm_actor, resume_training_step, episode = create_model_and_optimizer(
        args, tc, model_config, beaker_config, wandb_url, tokenizer
    )

    generation_configs = create_generation_configs(args)

    checkpoint_state = None
    if args.checkpoint_state_dir and os.path.exists(args.checkpoint_state_dir):
        # Try to load the checkpoint state from the first rank
        checkpoint_path = os.path.join(args.checkpoint_state_dir, "global_0", "state.pt")
        if os.path.exists(checkpoint_path):
            checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded checkpoint state from {checkpoint_path}")

            episode = checkpoint_state["episode"]
            logger.info(f"Restored episode count: {episode}")

    train_dataset_idxs = np.arange(len(train_dataset))
    iter_dataloader = ShufflingIterator(train_dataset_idxs, args.num_unique_prompts_rollout, seed=args.seed)

    if checkpoint_state and "shuffling_iterator_state" in checkpoint_state:
        iter_dataloader.set_state(checkpoint_state["shuffling_iterator_state"])
        logger.info("Restored ShufflingIterator state from checkpoint")

    # Create additional queues (main queues already created above)
    packed_sequences_Q = Queue(maxsize=args.async_steps)
    pending_queries_map = PendingQueriesMap()
    eval_pending_queries_map = PendingQueriesMap()
    generate_metrics_Q = Queue(maxsize=args.async_steps)
    weight_sync_metrics_Q = Queue(maxsize=args.async_steps)

    if eval_dataset is None:
        eval_batch = None
    else:
        eval_dataset_indices = list(range(len(eval_dataset)))
        eval_batch = next_batch(eval_dataset_indices, eval_dataset)
    reward_fn = make_reward_fn(args)

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="grpo")

    try:
        episode = run_training(
            args,
            tokenizer,
            train_dataset,
            eval_batch,
            training_manager,
            llm_actor,
            generation_configs,
            iter_dataloader,
            reward_fn,
            resume_training_step,
            episode,
            wandb_url,
            tc,
            stop_event,
            executor,
            packed_sequences_Q,
            pending_queries_map,
            eval_pending_queries_map,
            generate_metrics_Q,
            weight_sync_metrics_Q,
            checkpoint_state,
        )
    finally:
        llm_actor.stop()
        stop_event.set()
        executor.shutdown(wait=True)

    # Ai2 logic: we use /output to store the artifacts of the job, so we
    # make a copy of the model to `/output` in the end.
    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
        and os.path.isdir(args.output_dir)
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
    logger.info("finished training")

    accelerator = Namespace()
    accelerator.is_main_process = True  # hack
    if args.push_to_hub:
        logger.info("Pushing model to hub")
        push_folder_to_hub(accelerator, args.output_dir, args.hf_repo_id, args.hf_repo_revision)

    # Check for runtime leaks before exiting
    logger.info("Checking for runtime leaks...")

    utils.check_runtime_leaks()


if __name__ == "__main__":
    utils.check_oe_eval_internal()

    parser = ArgumentParserPlus((TinkerArgs, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, TinkerArgs)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)

    main(args, tokenizer_config, model_config)
