import dataclasses
import gc
import os
import threading
import time
import unittest
from typing import Any
from unittest.mock import MagicMock, Mock

import ray
import torch
from datasets import Dataset
from parameterized import parameterized
from ray.util import queue as ray_queue
from transformers import AutoTokenizer

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_fast, rl_utils, utils
from open_instruct.data_types import CollatedBatchData, GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)


class TestGrpoFastBase(unittest.TestCase):
    """Base class with common test utilities."""

    def _get_resource_tracker_state(self):
        """Get current resource tracker state for debugging."""
        tracked_resources = {}
        try:
            # Try to access resource tracker directly
            from multiprocessing.resource_tracker import _resource_tracker

            if hasattr(_resource_tracker, "_cache"):
                for name, rtype in list(_resource_tracker._cache.items()):
                    if rtype not in tracked_resources:
                        tracked_resources[rtype] = []
                    tracked_resources[rtype].append(name)
        except Exception:
            # Alternative approach: check via resource_tracker module
            try:
                import multiprocessing.resource_tracker as rt

                if hasattr(rt, "getfd"):
                    # This is a hack to get the cache info

                    # Try to find the cache in the module
                    for attr_name in dir(rt):
                        attr = getattr(rt, attr_name)
                        if isinstance(attr, dict) and any("semaphore" in str(v) for v in attr.values()):
                            for k, v in attr.items():
                                if v not in tracked_resources:
                                    tracked_resources[v] = []
                                tracked_resources[v].append(k)
            except Exception:
                pass
        return tracked_resources

    def setUp(self):
        """Initialize Ray and check for pre-existing leaks."""
        # Record initial resource tracker state
        self._initial_resources = self._get_resource_tracker_state()

        # Track Ray queues for cleanup
        self._ray_queues = []

        utils.check_runtime_leaks()

        # Initialize Ray for this test
        ray.init(include_dashboard=False, runtime_env={"env_vars": dict(os.environ)})

    def _cleanup_ray_queues(self):
        """Clean up all Ray queues created during the test."""
        for queue in self._ray_queues:
            try:
                queue.shutdown()
            except Exception as e:
                print(f"Warning: Failed to shutdown Ray queue: {e}")
        self._ray_queues.clear()

    def tearDown(self):
        """Check for leaks and shutdown Ray."""
        # Clean up Ray queues BEFORE shutting down Ray
        self._cleanup_ray_queues()

        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()

        # Force garbage collection to clean up any lingering objects
        gc.collect()

        # Get final resource tracker state
        final_resources = self._get_resource_tracker_state()

        # Check for new resources that weren't there initially
        new_resources = {}
        for rtype, names in final_resources.items():
            initial_names = set(self._initial_resources.get(rtype, []))
            new_names = [n for n in names if n not in initial_names]
            if new_names:
                new_resources[rtype] = new_names

        utils.check_runtime_leaks()

        # Check for semaphore leaks
        if new_resources:
            # Report all new resources, especially semaphores
            leak_msg = f"Resource leaks detected after test {self._testMethodName}:\n"
            for rtype, names in new_resources.items():
                leak_msg += f"  {rtype}: {names}\n"

            # Fail if there are semaphore leaks
            if "semaphore" in new_resources:
                self.fail(leak_msg)

    def create_test_data(self, num_prompts, prefix="", start_idx=0):
        """Create test data with consistent naming."""
        indices = list(range(start_idx, start_idx + num_prompts))
        queries = [f"{prefix}query_{i}" for i in indices]
        ground_truths = [f"{prefix}truth_{i}" for i in indices]
        datasets = [f"{prefix}dataset_{i}" for i in indices]
        raw_queries = [f"{prefix}rawquery_{i}" for i in indices]
        return queries, ground_truths, datasets, raw_queries, indices

    def create_mock_args(self, num_engines=4, num_samples=1):
        """Create mock args object."""
        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.vllm_tensor_parallel_size = 1
        mock_args.num_samples_per_prompt_rollout = num_samples
        mock_args.verbose = False
        mock_args.max_possible_score = 1.0
        return mock_args

    def create_mock_model_dims(self):
        """Create mock ModelDims object for tests."""
        # Create a simple mock ModelDims with minimal attributes
        mock_dims = Mock(spec=utils.ModelDims)
        mock_dims.num_layers = 32
        mock_dims.hidden_size = 4096
        mock_dims.intermediate_size = 11008
        mock_dims.vocab_size = 32000
        mock_dims.num_attn_heads = 32
        mock_dims.num_kv_heads = 32
        mock_dims.device_name = "h100"
        mock_dims.device_flops = 989.5e12  # H100 peak FLOPs
        mock_dims.device_memory_bandwidth = 3.35e12  # H100 memory bandwidth

        # Mock the flops and memory_bytes methods
        mock_dims.flops = Mock(return_value=1e12)  # Return 1 TFlop
        mock_dims.memory_bytes = Mock(return_value=1e9)  # Return 1 GB

        return mock_dims

    def create_mock_packed_sequences(self, batch_size: int, seq_length: int, variable_length: bool = False):
        """Create mock PackedSequences for testing."""
        lengths = [seq_length - (i % 3) if variable_length else seq_length for i in range(batch_size)]
        return rl_utils.PackedSequences(
            query_responses=[torch.full((length,), i, dtype=torch.long) for i, length in enumerate(lengths)],
            attention_masks=[torch.ones(length, dtype=torch.long) for length in lengths],
            response_masks=[torch.ones(length, dtype=torch.long) for length in lengths],
            original_responses=[[i] * seq_length for i in range(batch_size)],
            advantages=[torch.randn(length) for length in lengths],
            position_ids=[torch.arange(length, dtype=torch.long) for length in lengths],
            vllm_logprobs=[torch.randn(length) for length in lengths],
        )

    def create_mock_result_from_request(self, request: PromptRequest, num_samples_per_prompt=1):
        """Create a mock GenerationResult from a PromptRequest."""
        return self.create_mock_result(request.dataset_index, request.prompt_id, num_samples_per_prompt)

    def create_mock_result(self, dataset_index: int, prompt_id: str, num_samples_per_prompt=1, reward_scores=None):
        """Create a mock GenerationResult."""
        total_responses = num_samples_per_prompt
        if reward_scores is None:
            reward_scores = [i / max(total_responses, 1) for i in range(total_responses)]

        return GenerationResult(
            responses=[[1, 2, 3] for _ in range(total_responses)],
            finish_reasons=["stop"] * total_responses,
            masks=[[1, 1, 1] for _ in range(total_responses)],
            request_info=RequestInfo(
                num_calls=[0] * total_responses,
                timeouts=[0] * total_responses,
                tool_errors=[""] * total_responses,
                tool_outputs=[""] * total_responses,
                tool_runtimes=[0.0] * total_responses,
                tool_calleds=[False] * total_responses,
            ),
            dataset_index=dataset_index,
            prompt_id=prompt_id,
            start_time=time.perf_counter(),
            token_statistics=TokenStatistics(
                num_prompt_tokens=10, num_response_tokens=3 * total_responses, generation_time=0.1
            ),
            logprobs=[[0.0, 0.0, 0.0] for _ in range(total_responses)],
            reward_scores=reward_scores,
            reward_metrics={"time/reward": 0.0},
        )

    def create_mock_tokenizer_and_reward_fn(self):
        # Set up dummy tokenizer
        tokenizer_name = "EleutherAI/pythia-14m"  # Using a small model for testing
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Set up dummy reward fn that will guarantee nonzero std
        async def reward_fn(
            responses: list[torch.Tensor],
            decoded_responses: list[str],
            ground_truths: list[Any],
            datasets: list[str],
            finish_reasons: list[str],
            infos: list[list[int]],
            queries: list[str] | None = None,
        ) -> (list[float], dict[str, Any]):
            num_responses = len(responses)
            return [i / num_responses for i in range(num_responses)], {"time/reward": 0.0}

        return tokenizer, reward_fn

    def create_mock_dataset(self, queries, ground_truths, datasets, raw_queries):
        """Create a mock dataset from test data."""
        data = {
            INPUT_IDS_PROMPT_KEY: queries,
            GROUND_TRUTHS_KEY: ground_truths,
            VERIFIER_SOURCE_KEY: datasets,
            RAW_PROMPT_KEY: raw_queries,
        }
        return Dataset.from_dict(data)

    def setup_and_add_prompts_to_generator(self, queries, ground_truths, datasets, raw_queries, indices, num_engines):
        """Setup queues and add prompts to generator - common pattern."""
        # Queue size must be at least as large as the number of queries to avoid blocking
        queue_size = max(len(queries), num_engines * 2)
        prompt_Q = ray_queue.Queue(maxsize=queue_size)
        inference_results_Q = ray_queue.Queue(maxsize=queue_size)

        # Track queues for cleanup
        self._ray_queues.extend([prompt_Q, inference_results_Q])

        mock_generation_config = MagicMock()
        mock_generation_config.n = 4

        mock_dataset = self.create_mock_dataset(queries, ground_truths, datasets, raw_queries)
        data_loader = data_loader_lib.HFDataLoader(
            dataset=mock_dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir="/tmp"
        )

        for example in data_loader:
            grpo_fast.add_prompt_to_generator(example, prompt_Q, mock_generation_config, False)

        return prompt_Q, inference_results_Q, mock_dataset


class TestGrpoFastVLLM(TestGrpoFastBase):
    @parameterized.expand([(1, 16), (2, 32), (4, 64), (8, 128)])
    def test_batch_splitting_and_engine_configurations(self, vllm_num_engines: int, num_unique_prompts_rollout: int):
        """Test batch splitting and accumulation with various engine configurations."""
        # Create test data
        queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices = self.create_test_data(
            num_unique_prompts_rollout
        )

        # Setup and split batch
        prompt_Q, inference_results_Q, mock_dataset = self.setup_and_add_prompts_to_generator(
            queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices, vllm_num_engines
        )

        # Verify that we have the expected number of items in the queue (one per prompt)
        self.assertEqual(prompt_Q.qsize(), num_unique_prompts_rollout)

        # Simulate vLLM processing
        batch_idx = 0
        while not prompt_Q.empty():
            request = prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertIsInstance(request.dataset_index, int)
            self.assertIsInstance(request.prompt_id, str)

            mock_result = self.create_mock_result_from_request(request)
            inference_results_Q.put(mock_result)
            batch_idx += 1

        # Simulate streaming accumulation (simplified version for testing)
        combined_responses = []
        combined_queries = []
        combined_raw_queries = []
        combined_ground_truths = []
        combined_datasets = []

        for _ in range(num_unique_prompts_rollout):
            result = inference_results_Q.get()
            dataset_index = result.dataset_index

            # Get query from dataset using index
            example = mock_dataset[dataset_index]
            q = example[INPUT_IDS_PROMPT_KEY]
            gt = example[GROUND_TRUTHS_KEY]
            d = example[VERIFIER_SOURCE_KEY]
            raw_q = example[RAW_PROMPT_KEY]

            combined_responses.extend(result.responses)
            combined_queries.append(q)
            combined_raw_queries.append(raw_q)
            combined_ground_truths.append(gt)
            combined_datasets.append(d)

        combined_result = GenerationResult(
            responses=combined_responses,
            finish_reasons=["stop"] * len(combined_responses),
            masks=[[1, 1, 1]] * len(combined_responses),
            request_info=RequestInfo(
                num_calls=[0] * len(combined_responses),
                timeouts=[0] * len(combined_responses),
                tool_errors=[""] * len(combined_responses),
                tool_outputs=[""] * len(combined_responses),
                tool_runtimes=[0.0] * len(combined_responses),
                tool_calleds=[False] * len(combined_responses),
            ),
            dataset_index=0,
            prompt_id="combined",
        )

        # Verify that the combined results contain the same items (order may differ due to shuffling)
        self.assertEqual(sorted(combined_queries), sorted(queries_next))
        self.assertEqual(sorted(combined_ground_truths), sorted(ground_truths_next))
        self.assertEqual(sorted(combined_datasets), sorted(datasets_next))

        # Verify that the combined result has the correct structure
        self.assertIsInstance(combined_result, GenerationResult)
        self.assertEqual(len(combined_result.responses), len(queries_next))
        self.assertEqual(len(combined_result.finish_reasons), len(queries_next))
        self.assertEqual(len(combined_result.masks), len(queries_next))

        # Verify that the inference_results_Q is empty after accumulation
        self.assertEqual(inference_results_Q.qsize(), 0)

    def test_dataset_index_preservation_through_pipeline(self):
        """Test that dataset indices are correctly preserved through the pipeline."""
        vllm_num_engines = 4
        num_unique_prompts_rollout = 32

        # Create test data
        queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices = self.create_test_data(
            num_unique_prompts_rollout
        )

        # Setup and split batch
        prompt_Q, inference_results_Q, mock_dataset = self.setup_and_add_prompts_to_generator(
            queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices, vllm_num_engines
        )

        # Simulate vLLM processing
        batch_idx = 0
        while not prompt_Q.empty():
            request = prompt_Q.get()
            mock_result = self.create_mock_result_from_request(request)
            inference_results_Q.put(mock_result)
            batch_idx += 1

        # Simulate streaming accumulation
        combined_queries = []
        combined_raw_queries = []
        combined_ground_truths = []
        combined_datasets = []

        for _ in range(num_unique_prompts_rollout):
            result = inference_results_Q.get()
            dataset_index = result.dataset_index

            example = mock_dataset[dataset_index]
            q = example[INPUT_IDS_PROMPT_KEY]
            gt = example[GROUND_TRUTHS_KEY]
            d = example[VERIFIER_SOURCE_KEY]
            raw_q = example[RAW_PROMPT_KEY]
            combined_queries.append(q)
            combined_raw_queries.append(raw_q)
            combined_ground_truths.append(gt)
            combined_datasets.append(d)

        # Verify results (order may differ due to shuffling)
        self.assertEqual(sorted(combined_queries), sorted(queries_next))
        self.assertEqual(sorted(combined_ground_truths), sorted(ground_truths_next))
        self.assertEqual(sorted(combined_datasets), sorted(datasets_next))

    @parameterized.expand([(1, 16), (2, 8), (4, 4)])
    def test_multiple_samples_per_prompt(self, vllm_num_engines: int, num_samples_per_prompt: int):
        """Test handling of multiple samples per prompt."""
        num_unique_prompts_rollout = 16

        # Create test data
        queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices = self.create_test_data(
            num_unique_prompts_rollout
        )

        # Setup and split batch
        prompt_Q, inference_results_Q, mock_dataset = self.setup_and_add_prompts_to_generator(
            queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices, vllm_num_engines
        )

        # Simulate vLLM processing with multiple samples
        batch_idx = 0
        while not prompt_Q.empty():
            request = prompt_Q.get()
            mock_result = self.create_mock_result_from_request(request, num_samples_per_prompt)
            inference_results_Q.put(mock_result)
            batch_idx += 1

        # Simulate streaming accumulation
        combined_responses = []
        combined_queries = []
        combined_raw_queries = []
        combined_ground_truths = []
        combined_datasets = []

        for _ in range(num_unique_prompts_rollout):
            result = inference_results_Q.get()
            dataset_index = result.dataset_index

            # Look up from dataset
            example = mock_dataset[dataset_index]
            q = example[INPUT_IDS_PROMPT_KEY]
            gt = example[GROUND_TRUTHS_KEY]
            d = example[VERIFIER_SOURCE_KEY]
            raw_q = example[RAW_PROMPT_KEY]

            combined_responses.extend(result.responses)
            combined_queries.append(q)
            combined_raw_queries.append(raw_q)
            combined_ground_truths.append(gt)
            combined_datasets.append(d)

        combined_result = GenerationResult(
            responses=combined_responses,
            finish_reasons=["stop"] * len(combined_responses),
            masks=[[1, 1, 1]] * len(combined_responses),
            request_info=RequestInfo(
                num_calls=[0] * len(combined_responses),
                timeouts=[0] * len(combined_responses),
                tool_errors=[""] * len(combined_responses),
                tool_outputs=[""] * len(combined_responses),
                tool_runtimes=[0.0] * len(combined_responses),
                tool_calleds=[False] * len(combined_responses),
            ),
            dataset_index=0,
            prompt_id="combined",
        )

        # Verify results - streaming accumulation should NOT replicate (order may differ due to shuffling)
        self.assertEqual(sorted(combined_queries), sorted(queries_next))
        self.assertEqual(sorted(combined_ground_truths), sorted(ground_truths_next))
        self.assertEqual(sorted(combined_datasets), sorted(datasets_next))

        # Verify correct number of responses
        expected_responses = num_unique_prompts_rollout * num_samples_per_prompt
        self.assertEqual(len(combined_result.responses), expected_responses)


class GrpoIntegrationTests(TestGrpoFastBase):
    """Integration tests for GRPO with parallel processing."""

    def test_out_of_order_processing(self):
        """Test that dataset indices can be processed out of order."""
        num_engines = 4
        num_prompts = 16
        num_samples_per_prompt = 4

        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_prompts)

        tokenizer, reward_fn = self.create_mock_tokenizer_and_reward_fn()

        prompt_Q, inference_results_Q, mock_dataset = self.setup_and_add_prompts_to_generator(
            queries, ground_truths, datasets, raw_queries, indices, num_engines
        )

        requests = []
        while not prompt_Q.empty():
            requests.append(prompt_Q.get())

        for request in reversed(requests):
            mock_result = self.create_mock_result_from_request(request, num_samples_per_prompt)
            inference_results_Q.put(mock_result)

        mock_args = self.create_mock_args(num_engines, num_samples_per_prompt)
        mock_generation_config = Mock()
        mock_generation_config.n = num_samples_per_prompt

        mock_model_dims = self.create_mock_model_dims()
        combined_result, batch, reward_metrics, batch_stats = grpo_fast.accumulate_inference_batches(
            inference_results_Q,
            mock_args,
            generation_config=mock_generation_config,
            num_prompts=num_prompts,
            model_dims=mock_model_dims,
            tokenizer=tokenizer,
            prompt_dataset=mock_dataset,
        )

        self.assertEqual(len(batch.queries), num_prompts * num_samples_per_prompt)
        self.assertEqual(len(combined_result.responses), num_prompts * num_samples_per_prompt)

    def test_accumulate_waits_for_all_engines(self):
        """Test that accumulate_inference_batches waits for all engines."""
        num_engines = 4
        num_prompts = 16

        tokenizer, reward_fn = self.create_mock_tokenizer_and_reward_fn()

        expected_results = 3 * (num_prompts // num_engines)
        inference_results_Q = ray_queue.Queue(maxsize=max(expected_results, num_engines * 2))

        self._ray_queues.append(inference_results_Q)

        queries = [f"q_{i}" for i in range(num_prompts)]
        ground_truths = [f"t_{i}" for i in range(num_prompts)]
        datasets = [f"d_{i}" for i in range(num_prompts)]
        raw_queries = [f"q_{i}" for i in range(num_prompts)]
        mock_dataset = self.create_mock_dataset(queries, ground_truths, datasets, raw_queries)

        for engine_id in range(3):
            for i in range(engine_id * 4, (engine_id + 1) * 4):
                mock_result = self.create_mock_result(i, f"0_{i}")
                inference_results_Q.put(mock_result)

        mock_args = self.create_mock_args(num_engines)

        completed = threading.Event()

        def run_accumulate():
            try:
                mock_generation_config = Mock()
                mock_generation_config.n = 1

                mock_model_dims = self.create_mock_model_dims()
                grpo_fast.accumulate_inference_batches(
                    inference_results_Q,
                    mock_args,
                    generation_config=mock_generation_config,
                    num_prompts=num_prompts,
                    model_dims=mock_model_dims,
                    tokenizer=tokenizer,
                    prompt_dataset=mock_dataset,
                )
                completed.set()
            except Exception:
                completed.set()

        thread = threading.Thread(target=run_accumulate, daemon=True)
        thread.start()

        self.assertFalse(completed.wait(timeout=1.0))
        self.assertTrue(thread.is_alive())

        self.assertEqual(inference_results_Q.qsize(), 0)


class TestStreamingAccumulation(TestGrpoFastBase):
    """Test the new streaming accumulation functionality."""

    def test_more_engines_than_queries(self):
        """Test that add_prompt_to_generator handles gracefully when engines > queries."""
        num_queries = 4

        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_queries)
        prompt_Q = ray_queue.Queue(maxsize=num_queries)

        self._ray_queues.append(prompt_Q)

        mock_generation_config = MagicMock()
        mock_generation_config.n = 1

        mock_dataset = self.create_mock_dataset(queries, ground_truths, datasets, raw_queries)
        data_loader = data_loader_lib.HFDataLoader(
            dataset=mock_dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir="/tmp"
        )

        for example in data_loader:
            grpo_fast.add_prompt_to_generator(example, prompt_Q, mock_generation_config, False)

        self.assertEqual(prompt_Q.qsize(), num_queries, f"Should have {num_queries} batches for {num_queries} queries")

        prompt_count = 0
        while not prompt_Q.empty():
            request = prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertIsNotNone(request.prompt, "Each request should have a prompt")
            prompt_count += 1

        self.assertEqual(prompt_count, num_queries, f"Should have {num_queries} PromptRequests")

    def test_uneven_distribution_no_empty_batches(self):
        """Test that uneven query distribution doesn't create empty batches."""
        num_queries = 7

        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_queries)
        prompt_Q = ray_queue.Queue(maxsize=num_queries)

        self._ray_queues.append(prompt_Q)

        mock_generation_config = MagicMock()
        mock_generation_config.n = 1

        mock_dataset = self.create_mock_dataset(queries, ground_truths, datasets, raw_queries)
        data_loader = data_loader_lib.HFDataLoader(
            dataset=mock_dataset, batch_size=1, seed=42, rank=0, world_size=1, work_dir="/tmp"
        )

        for example in data_loader:
            grpo_fast.add_prompt_to_generator(example, prompt_Q, mock_generation_config, False)

        request_count = 0
        while not prompt_Q.empty():
            request = prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertIsNotNone(request.prompt, "Each request should have a prompt")
            request_count += 1

        self.assertEqual(request_count, num_queries, "Total requests should match total queries")
        self.assertEqual(request_count, num_queries, f"Should have {num_queries} individual PromptRequests")

    def test_streaming_accumulation_basic(self):
        """Test basic streaming accumulation with in-order results."""
        num_prompts = 8

        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_prompts)

        inference_results_Q = ray_queue.Queue(maxsize=num_prompts)

        self._ray_queues.append(inference_results_Q)

        mock_dataset = self.create_mock_dataset(queries, ground_truths, datasets, raw_queries)

        for i in range(num_prompts):
            mock_result = self.create_mock_result(i, f"0_{i}")
            inference_results_Q.put(mock_result)

        results_list = []
        queries_list = []
        expected_results = num_prompts

        while len(results_list) < expected_results:
            result = inference_results_Q.get()

            results_list.append(result)

            dataset_index = result.dataset_index
            example = mock_dataset[dataset_index]
            q = example[INPUT_IDS_PROMPT_KEY]
            gt = example[GROUND_TRUTHS_KEY]
            d = example[VERIFIER_SOURCE_KEY]
            raw_q = example[RAW_PROMPT_KEY]
            queries_list.append((q, gt, d, raw_q))

        self.assertEqual(len(results_list), expected_results)

        combined_queries = []
        for i in range(num_prompts):
            q, _, _, _ = queries_list[i]
            combined_queries.append(q)

        self.assertEqual(combined_queries, queries)

    def test_streaming_with_multiple_samples(self):
        """Test streaming accumulation with multiple samples per prompt."""
        num_prompts = 4
        num_samples = 3

        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_prompts)

        inference_results_Q = ray_queue.Queue(maxsize=num_prompts)

        self._ray_queues.append(inference_results_Q)

        mock_dataset = self.create_mock_dataset(queries, ground_truths, datasets, raw_queries)

        for i in range(num_prompts):
            mock_result = self.create_mock_result(i, f"0_{i}", num_samples_per_prompt=num_samples)
            inference_results_Q.put(mock_result)

        total_responses = 0
        while not inference_results_Q.empty():
            result = inference_results_Q.get()

            expected_responses = num_samples
            self.assertEqual(len(result.responses), expected_responses)
            total_responses += len(result.responses)

            idx = result.dataset_index
            example = mock_dataset[idx]
            self.assertEqual(example[INPUT_IDS_PROMPT_KEY], queries[idx])

        self.assertEqual(total_responses, num_prompts * num_samples)


class TestAccumulateInferenceBatches(TestGrpoFastBase):
    """Test accumulate_inference_batches function."""

    def test_all_prompts_filtered_returns_none(self):
        """Test that accumulate_inference_batches returns None when all prompts are filtered."""
        num_prompts = 8
        num_samples_per_prompt = 4

        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_prompts)

        inference_results_Q = ray_queue.Queue(maxsize=num_prompts)

        self._ray_queues.append(inference_results_Q)

        mock_dataset = self.create_mock_dataset(queries, ground_truths, datasets, raw_queries)

        for i in range(num_prompts):
            constant_scores = [0.5] * num_samples_per_prompt
            mock_result = self.create_mock_result(
                i, f"0_{i}", num_samples_per_prompt=num_samples_per_prompt, reward_scores=constant_scores
            )
            inference_results_Q.put(mock_result)

        mock_args = self.create_mock_args(num_engines=4, num_samples=num_samples_per_prompt)
        mock_generation_config = Mock()
        mock_generation_config.n = num_samples_per_prompt
        mock_model_dims = self.create_mock_model_dims()

        tokenizer_name = "EleutherAI/pythia-14m"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        result, batch, reward_metrics, batch_stats = grpo_fast.accumulate_inference_batches(
            inference_results_Q,
            mock_args,
            generation_config=mock_generation_config,
            num_prompts=num_prompts,
            model_dims=mock_model_dims,
            tokenizer=tokenizer,
            prompt_dataset=mock_dataset,
            filter_zero_std_samples=True,
        )

        self.assertIsNone(result)
        self.assertIsNone(batch)
        self.assertIsNone(reward_metrics)
        self.assertIsNone(batch_stats)


class TestDataPreparation(TestGrpoFastBase):
    """Test prepare_collated_data_for_workers function."""

    @parameterized.expand(
        [
            (16, 4, 2, 10, False, 0),
            (32, 8, 4, 20, False, 0),
            (8, 2, 1, 5, False, 0),
            (17, 4, 2, 10, False, 0),
            (25, 8, 3, 15, False, 0),
            (4, 1, 4, 10, False, 0),
            (8, 2, 2, 10, True, 999),
        ]
    )
    def test_distribution_and_structure(
        self, batch_size, world_size, per_device_train_batch_size, seq_length, variable_length, pad_token_id
    ):
        """Test data distribution, structure, micro-batch collation, and padding."""
        packed_sequences = self.create_mock_packed_sequences(batch_size, seq_length, variable_length)
        result = grpo_fast.prepare_collated_data_for_workers(
            packed_sequences, world_size, per_device_train_batch_size, pad_token_id, pin_memory=False
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), world_size)

        expected_keys = {
            "query_responses",
            "attention_masks",
            "position_ids",
            "advantages",
            "response_masks",
            "vllm_logprobs",
        }

        expected_samples_per_worker = batch_size // world_size
        samples_per_worker = batch_size // world_size
        expected_num_microbatches = (
            samples_per_worker + per_device_train_batch_size - 1
        ) // per_device_train_batch_size

        for worker_data in result:
            self.assertIsInstance(worker_data, CollatedBatchData)
            self.assertEqual({f.name for f in dataclasses.fields(worker_data)}, expected_keys)

            total_samples = sum(len(batch) for batch in worker_data.query_responses)
            self.assertEqual(total_samples, expected_samples_per_worker)

            num_microbatches = len(worker_data.query_responses)
            self.assertEqual(num_microbatches, expected_num_microbatches)

            for field in dataclasses.fields(worker_data):
                value = getattr(worker_data, field.name)
                self.assertIsInstance(value, list)
                self.assertEqual(len(value), expected_num_microbatches)
                for i, tensor in enumerate(value):
                    self.assertIsInstance(tensor, torch.Tensor)
                    if i < expected_num_microbatches - 1:
                        self.assertEqual(len(tensor), per_device_train_batch_size)
                    else:
                        self.assertLessEqual(len(tensor), per_device_train_batch_size)

            if not variable_length:
                continue

            for batch in worker_data.query_responses:
                for row in batch:
                    padding_mask = row == pad_token_id
                    if not padding_mask.any():
                        continue
                    first_pad_idx = padding_mask.nonzero(as_tuple=True)[0][0].item()
                    self.assertTrue(torch.all(row[first_pad_idx:] == pad_token_id))


if __name__ == "__main__":
    unittest.main()
