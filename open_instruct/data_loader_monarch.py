"""
Monarch-based data preparation actor for distributed GRPO training.

This module provides a Monarch actor implementation of the DataPreparationActor,
replacing the Ray-based version for use with Monarch-based training coordination.
"""

import asyncio
import logging
import threading
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from monarch.actor import Actor, endpoint
from transformers import PreTrainedTokenizer

from open_instruct import data_types, utils
from open_instruct.data_loader import (
    INPUT_IDS_PROMPT_KEY,
    BatchStatistics,
    HFDataLoader,
    StreamingDataLoaderConfig,
    combine_reward_metrics,
    pack_sequences,
    pad_sequences_for_world_size,
    prepare_collated_data_for_workers,
    repeat_each,
)
from open_instruct.dataset_transformation import GROUND_TRUTHS_KEY, RAW_PROMPT_KEY, VERIFIER_SOURCE_KEY
from open_instruct.vllm_utils import SamplingConfig

logger = logging.getLogger(__name__)


class Batch:
    """Container for batch data during accumulation."""

    def __init__(
        self,
        queries: list,
        ground_truths: list,
        datasets: list,
        raw_queries: list,
        decoded_responses: list,
        indices: list | None,
        scores: list,
    ):
        self.queries = queries
        self.ground_truths = ground_truths
        self.datasets = datasets
        self.raw_queries = raw_queries
        self.decoded_responses = decoded_responses
        self.indices = indices
        self.scores = scores

    def __getitem__(self, indices: list[int]) -> "Batch":
        return Batch(
            queries=[self.queries[i] for i in indices],
            ground_truths=[self.ground_truths[i] for i in indices],
            datasets=[self.datasets[i] for i in indices],
            raw_queries=[self.raw_queries[i] for i in indices],
            decoded_responses=[self.decoded_responses[i] for i in indices],
            indices=[self.indices[i] for i in indices] if self.indices else None,
            scores=[self.scores[i] for i in indices],
        )


class DataPreparationMonarchActor(Actor):
    """Monarch actor for centralized data preparation.

    This is a Monarch-compatible version of DataPreparationActor that uses
    Monarch's @endpoint decorator instead of Ray's remote methods.
    """

    def __init__(
        self,
        vllm_engines: list,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        config: StreamingDataLoaderConfig,
        generation_config: SamplingConfig,
        num_training_steps: int,
        seed: int,
        per_device_train_batch_size: int,
        global_batch_size: int,
        dp_world_size: int,
        max_possible_score: float,
        model_dims: utils.ModelDims,
        verbose: bool,
        work_dir: str,
        initial_state: dict | None = None,
        allow_world_padding: bool = False,
    ):
        self.vllm_engines = vllm_engines
        self.allow_world_padding = allow_world_padding
        self.tokenizer = tokenizer
        self.config = config
        self.config.max_possible_score = max_possible_score
        self.generation_config = generation_config
        self.num_training_steps = num_training_steps
        self.per_device_train_batch_size = per_device_train_batch_size
        self.global_batch_size = global_batch_size
        self.dp_world_size = dp_world_size
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
            logger.info(f"[DataPreparationMonarchActor] Restored state: training_step={self.training_step}")

        self._current_engine_idx = 0

    @endpoint
    async def start_data_preparation_loop(self) -> None:
        """Start the background data preparation loop."""
        asyncio.create_task(self._data_preparation_loop())

    @endpoint
    async def get_data(self, rank: int, step: int) -> dict:
        """Get prepared data for a specific rank and step. Called by StreamingDataLoader."""
        logger.info(
            f"[DataPreparationMonarchActor.get_data] rank={rank} requesting step={step}, "
            f"current_prepared_step={self.current_prepared_step}"
        )
        wait_count = 0
        while True:
            with self.lock:
                if step <= self.current_prepared_step:
                    batch_data = self.prepared_data[step][rank]
                    result = {"batch": batch_data, "metrics": self.metrics[step]}
                    self._cleanup_old_steps(step)
                    logger.info(
                        f"[DataPreparationMonarchActor.get_data] rank={rank} got data for step={step} "
                        f"after {wait_count} waits"
                    )
                    return result
            wait_count += 1
            if wait_count % 1000 == 0:
                logger.info(
                    f"[DataPreparationMonarchActor.get_data] rank={rank} still waiting for step={step}, "
                    f"current_prepared_step={self.current_prepared_step}, wait_count={wait_count}"
                )
            await asyncio.sleep(0.01)

    @endpoint
    async def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {
            "training_step": self.current_prepared_step + 1,
            "iter_dataloader_state": self.iter_dataloader.state_dict(),
        }

    @endpoint
    async def set_state(self, state: dict) -> None:
        """Set state from checkpoint."""
        self.training_step = state["training_step"]
        self.iter_dataloader.load_state_dict(state["iter_dataloader_state"])

    @endpoint
    async def shutdown(self) -> None:
        """Request shutdown of the data preparation loop."""
        self.shutdown_requested = True

    def _cleanup_old_steps(self, current_step: int) -> None:
        """Remove old step data to prevent memory leak."""
        steps_to_remove = [s for s in self.prepared_data if s < current_step - 1]
        for s in steps_to_remove:
            del self.prepared_data[s]
            if s in self.metrics:
                del self.metrics[s]

    def _get_next_engine(self):
        """Round-robin selection of vLLM engine."""
        engine = self.vllm_engines[self._current_engine_idx]
        self._current_engine_idx = (self._current_engine_idx + 1) % len(self.vllm_engines)
        return engine

    async def _submit_prompt(self, example: dict[str, Any], is_eval: bool = False) -> None:
        """Submit a prompt to a vLLM engine."""
        dataset_index = example["dataset_index"]
        prompt_request = data_types.PromptRequest(
            prompt=example[INPUT_IDS_PROMPT_KEY],
            generation_config=self.generation_config,
            dataset_index=dataset_index,
            prompt_id=f"{self.iter_dataloader._epoch}_{dataset_index}",
            is_eval=is_eval,
        )
        engine = self._get_next_engine()
        await engine.put_prompt.call(prompt_request)

    async def _get_result_from_any_engine(self) -> data_types.GenerationResult:
        """Get a result from any vLLM engine that has one ready."""
        while True:
            for engine in self.vllm_engines:
                queue_sizes = await engine.get_queue_sizes.call()
                if queue_sizes["results_queue"] > 0:
                    return await engine.get_result.call()
            await asyncio.sleep(0.01)

    async def _data_preparation_loop(self) -> None:
        """Main data preparation loop."""
        logger.info("[DataPreparationMonarchActor] Starting _data_preparation_loop")

        num_initial_prompts = self.config.async_steps * self.global_batch_size
        logger.info(f"[DataPreparationMonarchActor] Pushing {num_initial_prompts} initial prompts")

        for _ in range(num_initial_prompts):
            example = next(self.iter_dataloader)
            await self._submit_prompt(example, is_eval=False)

        for step in range(self.training_step, self.num_training_steps):
            if self.shutdown_requested:
                return

            logger.info(f"[DataPreparationMonarchActor] Step {step}: accumulating {self.global_batch_size} prompts")

            result, batch, reward_metrics, batch_stats = await self._accumulate_inference_batches(step)

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
                        f"[DataPreparationMonarchActor] Filtered {num_truncated} responses that didn't finish with 'stop'. "
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

    async def _accumulate_inference_batches(
        self, training_step: int
    ) -> tuple[data_types.GenerationResult | None, Batch | None, dict | None, BatchStatistics | None]:
        """Accumulate inference results from vLLM engines."""
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

        logger.info(
            f"[_accumulate_inference_batches] Starting to accumulate {self.global_batch_size} prompts, "
            f"training_step={training_step}"
        )
        num_prompts_sampled = 0
        while num_prompts_sampled < self.global_batch_size:
            result = await self._get_result_from_any_engine()

            if isinstance(result, data_types.ShutdownSentinel):
                return result, None, None, None

            assert len(result.responses) == self.generation_config.n, (
                f"Mismatch: individual prompt result has {len(result.responses)} responses "
                f"but expected {self.generation_config.n} samples per prompt."
            )

            example = self.dataset[result.dataset_index]
            query = example[INPUT_IDS_PROMPT_KEY]
            ground_truth = example[GROUND_TRUTHS_KEY]
            dataset_name = example[VERIFIER_SOURCE_KEY]
            raw_query = example[RAW_PROMPT_KEY]

            next_example = next(self.iter_dataloader)
            await self._submit_prompt(next_example, is_eval=False)

            for i in range(len(result.finish_reasons)):
                if result.finish_reasons[i] == "stop" and len(result.responses[i]) == 0:
                    result.responses[i].append(self.tokenizer.eos_token_id)
                    result.masks[i].append(1)
                    result.logprobs[i].append(float("nan"))

            decoded_responses = self.tokenizer.batch_decode(result.responses, skip_special_tokens=True)

            k_queries = repeat_each([query], self.generation_config.n)
            k_ground_truths = repeat_each([ground_truth], self.generation_config.n)
            k_datasets = repeat_each([dataset_name], self.generation_config.n)
            k_raw_queries = repeat_each([raw_query], self.generation_config.n)

            percent_solved = np.mean(result.reward_scores).item() / self.config.max_possible_score
            if (
                self.config.no_resampling_pass_rate is not None
                and percent_solved >= self.config.no_resampling_pass_rate
            ):
                self.iter_dataloader.exclude_index(result.dataset_index)
                total_no_resampled += 1

            if self.config.filter_zero_std_samples and np.std(result.reward_scores) == 0:
                if not self.config.active_sampling:
                    num_prompts_sampled += 1

                total_filtered_prompts += 1
                if result.reward_scores[0] == 0:
                    filtered_prompt_zero += 1
                elif result.reward_scores[0] == self.config.max_possible_score:
                    filtered_prompt_solved += 1
                else:
                    filtered_prompt_nonzero += 1
                continue
            else:
                num_prompts_sampled += 1

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
            logger.warning(
                "[_accumulate_inference_batches] All prompts were filtered during accumulation. "
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

            prompt_lengths.append(len(all_queries[i * self.generation_config.n]))

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
