import gc
import os
import random
import threading
import time
import unittest
from unittest.mock import MagicMock, Mock

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import ray
import torch
from parameterized import parameterized
from ray.util import queue as ray_queue
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct import grpo_fast, model_utils, rl_utils, utils
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.vllm_utils import create_vllm_engines

torch.set_default_device("cpu")


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
        # Save original environment variable value
        self._original_nccl_cumem = os.environ.get("NCCL_CUMEM_ENABLE")

        # Record initial resource tracker state
        self._initial_resources = self._get_resource_tracker_state()

        # Track Ray queues for cleanup
        self._ray_queues = []

        # Disable MPS to avoid pin_memory issues on macOS
        if hasattr(torch.backends, 'mps'):
            self._original_mps_available = torch.backends.mps.is_available
            torch.backends.mps.is_available = lambda: False

        utils.check_runtime_leaks()

        # Initialize Ray for this test
        ray.init(include_dashboard=False)

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

        # Restore original environment variable value
        if self._original_nccl_cumem is None:
            os.environ.pop("NCCL_CUMEM_ENABLE", None)
        else:
            os.environ["NCCL_CUMEM_ENABLE"] = self._original_nccl_cumem

        # Restore MPS availability
        if hasattr(torch.backends, 'mps') and hasattr(self, '_original_mps_available'):
            torch.backends.mps.is_available = self._original_mps_available

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
            query_responses=[torch.full((length,), i, dtype=torch.long, device="cpu") for i, length in enumerate(lengths)],
            attention_masks=[torch.ones((length, length), dtype=torch.long, device="cpu") for length in lengths],
            response_masks=[torch.ones(length, dtype=torch.long, device="cpu") for length in lengths],
            original_responses=[[i] * seq_length for i in range(batch_size)],
            tool_masks=[torch.zeros(length, dtype=torch.long, device="cpu") for length in lengths],
            advantages=[torch.randn(length, device="cpu") for length in lengths],
            position_ids=[torch.arange(length, dtype=torch.long, device="cpu") for length in lengths],
            vllm_logprobs=[torch.randn(length, device="cpu") for length in lengths],
        )

    def create_mock_result(self, dataset_index, epoch_number, num_samples_per_prompt=1):
        """Create a mock GenerationResult."""
        total_responses = num_samples_per_prompt

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
            epoch_number=epoch_number,
            start_time=time.perf_counter(),
            token_statistics=TokenStatistics(
                num_prompt_tokens=10, num_response_tokens=3 * total_responses, generation_time=0.1
            ),
            logprobs=[[0.0, 0.0, 0.0] for _ in range(total_responses)],
        )

    def setup_and_split_batch(
        self, queries, ground_truths, datasets, raw_queries, indices, num_engines, training_step=1
    ):
        """Setup queues and split batch - common pattern."""
        # Queue size must be at least as large as the number of queries to avoid blocking
        queue_size = max(len(queries), num_engines * 2)
        param_prompt_Q = ray_queue.Queue(maxsize=queue_size)
        inference_results_Q = ray_queue.Queue(maxsize=queue_size)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queues for cleanup
        self._ray_queues.extend([param_prompt_Q, inference_results_Q])

        batch = model_utils.Batch(
            queries=queries, ground_truths=ground_truths, datasets=datasets, raw_queries=raw_queries, indices=indices
        )

        mock_generation_config = MagicMock()
        mock_generation_config.n = 4

        # Create mock args with inference_batch_size
        mock_args = MagicMock()
        # Calculate inference_batch_size based on number of queries and engines
        mock_args.inference_batch_size = max(1, len(queries) // num_engines)

        grpo_fast.split_and_insert_batch(
            batch, 0, training_step, pending_queries_map, param_prompt_Q, mock_generation_config, False
        )

        return param_prompt_Q, inference_results_Q, pending_queries_map


class TestGrpoFastVLLM(TestGrpoFastBase):
    def test_vllm_queue_system_single_prompt(self):
        """Test the new queue-based vLLM system with a single prompt 'What is the capital of France?'"""
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, skipping test")

        # Set up tokenizer
        tokenizer_name = "EleutherAI/pythia-14m"  # Using a small model for testing
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Tokenize the test prompt
        test_prompt = "What is the capital of France?"
        prompt_token_ids = tokenizer.encode(test_prompt, return_tensors="pt").tolist()[0]

        # Create Ray queues
        param_prompt_Q = ray_queue.Queue(maxsize=1)
        inference_results_Q = ray_queue.Queue(maxsize=1)

        # Track queues for cleanup
        self._ray_queues.extend([param_prompt_Q, inference_results_Q])

        # Create vLLM engines with queues
        vllm_engines = create_vllm_engines(
            num_engines=1,
            tensor_parallel_size=1,
            enforce_eager=True,
            tokenizer_name_or_path=tokenizer_name,
            pretrain=tokenizer_name,
            revision="main",
            seed=42,
            enable_prefix_caching=False,
            max_model_len=512,
            vllm_gpu_memory_utilization=0.5,  # Use less GPU memory for testing
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
        )

        # Set up generation config
        generation_config = SamplingParams(
            temperature=0.0,  # Deterministic generation
            top_p=1.0,
            max_tokens=5,
            seed=42,
        )

        # Start vLLM engines to process from queues
        [e.process_from_queue.remote() for e in vllm_engines]

        # Put the test prompt in the queue using PromptRequest
        param_prompt_Q.put(
            PromptRequest(prompt=prompt_token_ids, dataset_index=0, generation_config=generation_config)
        )

        # Get the result
        result = inference_results_Q.get()

        # Verify it's a GenerationResult dataclass
        self.assertIsInstance(result, GenerationResult)

        # Check that we got a response
        self.assertGreater(len(result.responses), 0)
        response_ids = result.responses[0]

        # Decode the response
        generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), 0)

        # Send stop signal
        param_prompt_Q.put(None)

    @parameterized.expand([(1, 16), (2, 32), (4, 64), (8, 128)])
    def test_batch_splitting_and_engine_configurations(self, vllm_num_engines: int, num_unique_prompts_rollout: int):
        """Test batch splitting and accumulation with various engine configurations."""
        # Create test data
        queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices = self.create_test_data(
            num_unique_prompts_rollout
        )

        # Setup and split batch
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices, vllm_num_engines
        )

        # Verify that we have individual prompts in the map (not batches)
        self.assertEqual(len(pending_queries_map), num_unique_prompts_rollout)

        # Verify that we have the expected number of items in the queue (one per prompt)
        self.assertEqual(param_prompt_Q.qsize(), num_unique_prompts_rollout)

        # Simulate vLLM processing
        batch_idx = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertEqual(request.training_step, 1)
            self.assertIsInstance(request.dataset_index, int)

            mock_result = self.create_mock_result(request.dataset_index, request.epoch_number)
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

            # Get query from pending_queries_map
            q, gt, d, raw_q = pending_queries_map.pop(dataset_index)

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
            dataset_index=None,
        )

        # Verify that the combined results match the original input
        self.assertEqual(combined_queries, queries_next)
        self.assertEqual(combined_ground_truths, ground_truths_next)
        self.assertEqual(combined_datasets, datasets_next)

        # Verify that the combined result has the correct structure
        self.assertIsInstance(combined_result, GenerationResult)
        self.assertEqual(len(combined_result.responses), len(queries_next))
        self.assertEqual(len(combined_result.finish_reasons), len(queries_next))
        self.assertEqual(len(combined_result.masks), len(queries_next))

        # Verify that the pending_queries_map is empty after accumulation
        self.assertEqual(len(pending_queries_map), 0)

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
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices, vllm_num_engines
        )

        # Simulate vLLM processing
        batch_idx = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            mock_result = self.create_mock_result(request.dataset_index, request.epoch_number)
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

            q, gt, d, raw_q = pending_queries_map.pop(dataset_index)
            combined_queries.append(q)
            combined_raw_queries.append(raw_q)
            combined_ground_truths.append(gt)
            combined_datasets.append(d)

        # Verify results
        self.assertEqual(combined_queries, queries_next)
        self.assertEqual(combined_ground_truths, ground_truths_next)
        self.assertEqual(combined_datasets, datasets_next)
        self.assertEqual(len(pending_queries_map), 0)

    @parameterized.expand([(1, 16), (2, 8), (4, 4)])
    def test_multiple_samples_per_prompt(self, vllm_num_engines: int, num_samples_per_prompt: int):
        """Test handling of multiple samples per prompt."""
        num_unique_prompts_rollout = 16

        # Create test data
        queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices = self.create_test_data(
            num_unique_prompts_rollout
        )

        # Setup and split batch
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries_next, ground_truths_next, datasets_next, raw_queries_next, dataset_indices, vllm_num_engines
        )

        # For multiple samples, we need to add additional references to the pending_queries_map
        # The first reference is already added by setup_and_split_batch
        for _ in range(num_samples_per_prompt - 1):
            for idx, query, ground_truth, dataset, raw_query in zip(
                dataset_indices, queries_next, ground_truths_next, datasets_next, raw_queries_next
            ):
                pending_queries_map.insert(idx, query, ground_truth, dataset, raw_query)

        # Simulate vLLM processing with multiple samples
        batch_idx = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            mock_result = self.create_mock_result(request.dataset_index, request.epoch_number, num_samples_per_prompt)
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

            # Pop the query data for this specific result - pop multiple times for multiple samples
            q, gt, d, raw_q = pending_queries_map.pop(dataset_index)
            # Pop additional times to handle multiple samples per prompt
            for _ in range(num_samples_per_prompt - 1):
                pending_queries_map.pop(dataset_index)

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
            dataset_index=None,
        )

        # Verify results - streaming accumulation should NOT replicate
        self.assertEqual(combined_queries, queries_next)
        self.assertEqual(combined_ground_truths, ground_truths_next)
        self.assertEqual(combined_datasets, datasets_next)
        self.assertEqual(len(pending_queries_map), 0)

        # Verify correct number of responses
        expected_responses = num_unique_prompts_rollout * num_samples_per_prompt
        self.assertEqual(len(combined_result.responses), expected_responses)


class GrpoIntegrationTests(TestGrpoFastBase):
    """Integration tests for GRPO with parallel processing."""

    @ray.remote
    def mock_vllm_engine(engine_id, prompt_queue, results_queue, num_samples_per_prompt=1):
        """Mock vLLM engine that processes prompts from queue."""
        while True:
            # Get request from queue
            request = prompt_queue.get()
            if request is None:  # Stop signal
                break

            # Simulate processing time
            time.sleep(random.uniform(0.01, 0.05))

            # Create mock generation result
            batch_size = len(request.prompts)
            total_responses = batch_size * num_samples_per_prompt

            # Important: vLLM keeps dataset_index as the original unique indices
            mock_result = GenerationResult(
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
                dataset_index=request.dataset_index,  # Original indices, not replicated
            )

            # Push to results queue
            results_queue.put(mock_result)

    def test_out_of_order_processing(self):
        """Test that dataset indices can be processed out of order."""
        num_engines = 4
        num_prompts = 16
        num_samples_per_prompt = 4

        # Create test data
        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_prompts)

        # Setup and split batch
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries, ground_truths, datasets, raw_queries, indices, num_engines
        )

        # Get all requests and process in reverse order
        requests = []
        while not param_prompt_Q.empty():
            requests.append(param_prompt_Q.get())

        # Put results back in REVERSE order to simulate out-of-order processing
        for request in reversed(requests):
            mock_result = self.create_mock_result(request.dataset_index, request.epoch_number, num_samples_per_prompt)
            inference_results_Q.put(mock_result)

        # Accumulate results
        mock_args = self.create_mock_args(num_engines, num_samples_per_prompt)
        # Create a mock generation config with n
        mock_generation_config = Mock()
        mock_generation_config.n = num_samples_per_prompt

        mock_model_dims = self.create_mock_model_dims()
        combined_result, batch, prompt_lengths, response_lengths = grpo_fast.accumulate_inference_batches(
            inference_results_Q,
            pending_queries_map,
            mock_args,
            generation_config=mock_generation_config,
            num_prompts=num_prompts,
            model_dims=mock_model_dims,
        )

        # Verify results work correctly even with out-of-order processing
        self.assertEqual(len(batch.queries), num_prompts)
        self.assertEqual(len(combined_result.responses), num_prompts * num_samples_per_prompt)
        self.assertEqual(len(pending_queries_map), 0)

    def test_thread_safety_pending_queries_map(self):
        """Test concurrent access to pending_queries_map."""
        pending_queries_map = grpo_fast.PendingQueriesMap()
        errors = []
        num_threads = 4
        entries_per_thread = 50

        def add_and_remove_entries(thread_id):
            """Add and then remove entries from the map."""
            try:
                start_idx = thread_id * 100
                # Add entries
                for i in range(start_idx, start_idx + entries_per_thread):
                    pending_queries_map.insert(
                        i,
                        f"query_{thread_id}_{i}",
                        f"truth_{thread_id}_{i}",
                        f"dataset_{thread_id}_{i}",
                        f"query_{thread_id}_{i}",
                    )
                    time.sleep(0.0001)

                # Remove entries
                for i in range(start_idx, start_idx + entries_per_thread):
                    if i in pending_queries_map:
                        pending_queries_map.pop(i)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Run threads concurrently
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=add_and_remove_entries, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify no errors and map is empty
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(pending_queries_map), 0)

    def test_accumulate_waits_for_all_engines(self):
        """Test that accumulate_inference_batches waits for all engines."""
        num_engines = 4
        num_prompts = 16

        # Setup with results from only 3 engines
        # Queue size must be large enough for all results being put before accumulation starts
        expected_results = 3 * (num_prompts // num_engines)  # 3 engines * 4 results each = 12
        inference_results_Q = ray_queue.Queue(maxsize=max(expected_results, num_engines * 2))

        # Track queue for cleanup
        self._ray_queues.append(inference_results_Q)

        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Add entries to map
        for i in range(num_prompts):
            pending_queries_map.insert(i, f"q_{i}", f"t_{i}", f"d_{i}", f"q_{i}")

        # Add results from only 3 engines (missing one)
        # With individual prompts, we add individual results
        for engine_id in range(3):
            for i in range(engine_id * 4, (engine_id + 1) * 4):
                mock_result = self.create_mock_result(i, 1)
                inference_results_Q.put(mock_result)

        mock_args = self.create_mock_args(num_engines)

        completed = threading.Event()

        def run_accumulate():
            try:
                # Create a mock generation config with n=1 (default)
                mock_generation_config = Mock()
                mock_generation_config.n = 1

                mock_model_dims = self.create_mock_model_dims()
                grpo_fast.accumulate_inference_batches(
                    inference_results_Q,
                    pending_queries_map,
                    mock_args,
                    generation_config=mock_generation_config,
                    num_prompts=num_prompts,
                    model_dims=mock_model_dims,
                )
                completed.set()
            except Exception:
                completed.set()

        thread = threading.Thread(target=run_accumulate, daemon=True)
        thread.start()

        # Should timeout waiting for missing results
        self.assertFalse(completed.wait(timeout=1.0))
        self.assertTrue(thread.is_alive())

        # Queue should be empty after consuming 12 results
        self.assertEqual(inference_results_Q.qsize(), 0)
        # 12 entries should be removed from the map (4 still pending)
        self.assertEqual(len(pending_queries_map), 4)


class TestStreamingAccumulation(TestGrpoFastBase):
    """Test the new streaming accumulation functionality."""

    def test_more_engines_than_queries(self):
        """Test that split_and_insert_batch handles gracefully when engines > queries."""
        # More engines than queries - should handle gracefully with single-prompt batches
        num_engines = 8
        num_queries = 4

        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_queries)
        param_prompt_Q = ray_queue.Queue(maxsize=num_queries)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(param_prompt_Q)

        batch = model_utils.Batch(
            queries=queries, ground_truths=ground_truths, datasets=datasets, raw_queries=raw_queries, indices=indices
        )

        mock_generation_config = MagicMock()
        mock_generation_config.n = 1

        # Create mock args with inference_batch_size
        mock_args = MagicMock()
        mock_args.inference_batch_size = max(1, num_queries // num_engines)

        grpo_fast.split_and_insert_batch(
            batch,
            epoch_number=0,
            training_step=1,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
            generation_config=mock_generation_config,
            is_eval=False,
        )

        # Should have 4 batches (one for each query)
        self.assertEqual(
            param_prompt_Q.qsize(), num_queries, f"Should have {num_queries} batches for {num_queries} queries"
        )

        # Each request should have exactly 1 prompt
        prompt_count = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertIsNotNone(request.prompt, "Each request should have a prompt")
            prompt_count += 1

        # Should have exactly num_queries PromptRequests
        self.assertEqual(prompt_count, num_queries, f"Should have {num_queries} PromptRequests")
        # All queries should be in the pending map
        self.assertEqual(len(pending_queries_map), num_queries)

    def test_uneven_distribution_no_empty_batches(self):
        """Test that uneven query distribution doesn't create empty batches."""
        num_engines = 3
        num_queries = 7  # 7/3 = ceil(2.33) = 3, so distribution should be [3, 3, 1]

        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_queries)
        param_prompt_Q = ray_queue.Queue(maxsize=num_queries)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(param_prompt_Q)

        batch = model_utils.Batch(
            queries=queries, ground_truths=ground_truths, datasets=datasets, raw_queries=raw_queries, indices=indices
        )

        mock_generation_config = MagicMock()
        mock_generation_config.n = 1

        # Create mock args with inference_batch_size
        mock_args = MagicMock()
        mock_args.inference_batch_size = max(1, num_queries // num_engines + (1 if num_queries % num_engines else 0))

        grpo_fast.split_and_insert_batch(
            batch,
            epoch_number=0,
            training_step=1,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
            generation_config=mock_generation_config,
            is_eval=False,
        )

        # With single-prompt architecture, verify we have the right number of individual requests
        request_count = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertIsNotNone(request.prompt, "Each request should have a prompt")
            request_count += 1

        # Check that total requests equal total queries
        self.assertEqual(request_count, num_queries, "Total requests should match total queries")

        # With individual prompts, we should have exactly num_queries requests
        self.assertEqual(request_count, num_queries, f"Should have {num_queries} individual PromptRequests")

    def test_streaming_accumulation_basic(self):
        """Test basic streaming accumulation with in-order results."""
        num_prompts = 8

        # Create test data
        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_prompts)

        # Create queues and maps
        inference_results_Q = ray_queue.Queue(maxsize=num_prompts)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(inference_results_Q)

        # Insert data into pending_queries_map
        for i in range(num_prompts):
            pending_queries_map.insert(i, queries[i], ground_truths[i], datasets[i], raw_queries[i])

        # Create mock results - one per prompt
        for i in range(num_prompts):
            mock_result = self.create_mock_result(i, epoch_number=1)
            inference_results_Q.put(mock_result)

        # Simulate streaming accumulation logic
        results_list = []
        queries_list = []
        expected_results = num_prompts  # Now expecting one result per prompt

        while len(results_list) < expected_results:
            result = inference_results_Q.get()

            results_list.append(result)

            # Get query for this prompt
            dataset_index = result.dataset_index
            q, gt, d, raw_q = pending_queries_map.pop(dataset_index)
            queries_list.append((q, gt, d, raw_q))

        # Verify all results processed
        self.assertEqual(len(results_list), expected_results)
        self.assertEqual(len(pending_queries_map), 0)

        # Combine in order
        combined_queries = []
        for i in range(num_prompts):
            q, _, _, _ = queries_list[i]
            combined_queries.append(q)

        # Verify order is preserved
        self.assertEqual(combined_queries, queries)

    def test_streaming_with_multiple_samples(self):
        """Test streaming accumulation with multiple samples per prompt."""
        num_prompts = 4
        num_samples = 3

        # Create test data
        queries, ground_truths, datasets, raw_queries, indices = self.create_test_data(num_prompts)

        # Create queues and maps
        inference_results_Q = ray_queue.Queue(maxsize=num_prompts)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(inference_results_Q)

        # Insert data with reference counting for multiple samples
        for i in range(num_prompts):
            for _ in range(num_samples):
                pending_queries_map.insert(i, queries[i], ground_truths[i], datasets[i], raw_queries[i])

        # Create results - one per prompt with multiple samples
        for i in range(num_prompts):
            mock_result = self.create_mock_result(i, epoch_number=1, num_samples_per_prompt=num_samples)
            inference_results_Q.put(mock_result)

        # Process results
        total_responses = 0
        while not inference_results_Q.empty():
            result = inference_results_Q.get()

            # Verify number of responses matches num_samples for single prompt
            expected_responses = num_samples
            self.assertEqual(len(result.responses), expected_responses)
            total_responses += len(result.responses)

            # Pop multiple times to match the number of samples (reference counting)
            idx = result.dataset_index
            for _ in range(num_samples):
                pending_queries_map.pop(idx)

        # Verify total responses
        self.assertEqual(total_responses, num_prompts * num_samples)
        self.assertEqual(len(pending_queries_map), 0)


class TestShufflingIterator(unittest.TestCase):
    """Test ShufflingIterator state preservation functionality."""

    def test_basic_iteration(self):
        """Test basic iteration functionality."""

        data = np.arange(100)
        batch_size = 10
        iterator = grpo_fast.ShufflingIterator(data, batch_size, seed=42)

        # Get first batch
        batch1 = next(iterator)
        self.assertEqual(len(batch1), batch_size)
        self.assertTrue(all(isinstance(x, int) for x in batch1))

        # Get second batch
        batch2 = next(iterator)
        self.assertEqual(len(batch2), batch_size)
        # Batches should be different
        self.assertNotEqual(batch1, batch2)

    def test_state_preservation_and_restoration(self):
        """Test that state can be saved and restored correctly."""

        data = np.arange(100)
        batch_size = 10
        seed = 42

        # Create original iterator
        iter1 = grpo_fast.ShufflingIterator(data, batch_size, seed=seed)

        # Get a few batches
        _ = next(iter1)
        _ = next(iter1)
        _ = next(iter1)

        # Save state after 3 batches
        state = iter1.get_state()

        # Verify state contains expected keys
        self.assertIn("index", state)
        self.assertIn("data", state)
        self.assertIn("rng_state", state)
        self.assertEqual(state["index"], 30)  # 3 batches * 10 batch_size

        # Get next batches from original
        batch4_original = next(iter1)
        batch5_original = next(iter1)

        # Create new iterator with different seed and restore state
        iter2 = grpo_fast.ShufflingIterator(data, batch_size, seed=999)
        iter2.set_state(state)

        # Get batches from restored iterator
        batch4_restored = next(iter2)
        batch5_restored = next(iter2)

        # Batches should match exactly
        self.assertEqual(batch4_original, batch4_restored)
        self.assertEqual(batch5_original, batch5_restored)

    def test_epoch_boundary_state(self):
        """Test state preservation at epoch boundary."""

        data = np.arange(20)
        batch_size = 5

        # Create iterator and complete one epoch
        iterator = grpo_fast.ShufflingIterator(data, batch_size, seed=123)
        for _ in range(4):  # 20 / 5 = 4 batches per epoch
            next(iterator)

        # Save state at epoch boundary
        state = iterator.get_state()
        # After one complete epoch, index should reset
        self.assertEqual(state["index"], 20)

        # Create new iterator and restore state
        iter2 = grpo_fast.ShufflingIterator(data, batch_size, seed=456)
        iter2.set_state(state)

        # Next batches should match
        batch_original = next(iterator)
        batch_restored = next(iter2)
        self.assertEqual(batch_original, batch_restored)

    def test_rng_state_preservation(self):
        """Test that RNG state is properly preserved."""

        data = np.arange(1000)
        batch_size = 50

        # Create two iterators with same seed
        iter1 = grpo_fast.ShufflingIterator(data, batch_size, seed=42)
        _ = grpo_fast.ShufflingIterator(data, batch_size, seed=42)

        # Advance first iterator
        for _ in range(5):
            next(iter1)

        # Save state and create new iterator with different seed
        state = iter1.get_state()
        iter3 = grpo_fast.ShufflingIterator(data, batch_size, seed=999)

        # Restore state - this should override the different seed
        iter3.set_state(state)

        # Next 10 batches should match between iter1 and iter3
        for _ in range(10):
            batch1 = next(iter1)
            batch3 = next(iter3)
            self.assertEqual(batch1, batch3)


class TestDataPreparation(TestGrpoFastBase):
    """Test prepare_collated_data_for_workers function."""

    @parameterized.expand([
        (16, 4, 2, 10, False, 0),
        (32, 8, 4, 20, False, 0),
        (8, 2, 1, 5, False, 0),
        (17, 4, 2, 10, False, 0),
        (25, 8, 3, 15, False, 0),
        (4, 1, 4, 10, False, 0),
        (8, 2, 2, 10, True, 999),
    ])
    def test_distribution_and_structure(
        self, batch_size, world_size, per_device_train_batch_size, seq_length, variable_length, pad_token_id
    ):
        """Test data distribution, structure, micro-batch collation, and padding."""
        packed_sequences = self.create_mock_packed_sequences(batch_size, seq_length, variable_length)
        result = grpo_fast.prepare_collated_data_for_workers(
            packed_sequences, world_size, per_device_train_batch_size, pad_token_id
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), world_size)

        expected_keys = {
            "collated_query_responses",
            "collated_tool_masks",
            "collated_attention_masks",
            "collated_position_ids",
            "collated_advantages",
            "collated_response_masks",
            "collated_vllm_logprobs",
        }

        expected_samples_per_worker = batch_size // world_size
        samples_per_worker = batch_size // world_size
        expected_num_microbatches = (samples_per_worker + per_device_train_batch_size - 1) // per_device_train_batch_size

        for worker_data in result:
            self.assertIsInstance(worker_data, dict)
            self.assertEqual(set(worker_data.keys()), expected_keys)

            total_samples = sum(len(batch) for batch in worker_data["collated_query_responses"])
            self.assertEqual(total_samples, expected_samples_per_worker)

            num_microbatches = len(worker_data["collated_query_responses"])
            self.assertEqual(num_microbatches, expected_num_microbatches)

            for key, value in worker_data.items():
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

            for batch in worker_data["collated_query_responses"]:
                for row in batch:
                    padding_mask = row == pad_token_id
                    if not padding_mask.any():
                        continue
                    first_pad_idx = padding_mask.nonzero(as_tuple=True)[0][0].item()
                    self.assertTrue(torch.all(row[first_pad_idx:] == pad_token_id))


if __name__ == "__main__":
    unittest.main()
