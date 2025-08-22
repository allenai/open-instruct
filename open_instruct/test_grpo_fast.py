import gc
import os
import queue
import time
import unittest
from unittest.mock import Mock

import ray
import torch
from parameterized import parameterized
from ray.util import queue as ray_queue
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct import grpo_fast, model_utils, tool_utils, utils, vllm_utils3
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo


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

        utils.check_runtime_leaks()

        # Initialize Ray for this test
        ray.init(include_dashboard=False)

    def tearDown(self):
        """Check for leaks and shutdown Ray."""
        # Clean up Ray queues BEFORE shutting down Ray
        [rq.shutdown() for rq in self._ray_queues]

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

    def create_test_data(self, num_prompts, prefix="", start_idx=0):
        """Create test data with consistent naming."""
        indices = list(range(start_idx, start_idx + num_prompts))
        queries = [f"{prefix}query_{i}" for i in indices]
        ground_truths = [f"{prefix}truth_{i}" for i in indices]
        datasets = [f"{prefix}dataset_{i}" for i in indices]
        return queries, ground_truths, datasets, indices

    def create_mock_args(self, num_engines=4, num_samples=1, num_prompts=16):
        """Create mock args object."""
        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples
        mock_args.num_unique_prompts_rollout = num_prompts
        mock_args.verbose = False
        return mock_args

    def create_mock_result(self, dataset_index, training_step, num_samples_per_prompt=1):
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
        )

    def setup_and_split_batch(self, queries, ground_truths, datasets, indices, num_engines, training_step=1):
        """Setup queues and split batch - common pattern."""
        # Queue size should accommodate batches from all engines potentially multiple times
        queue_size = num_engines * len(queries)
        param_prompt_Q = ray_queue.Queue(maxsize=queue_size)
        inference_results_Q = ray_queue.Queue(maxsize=queue_size)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queues for cleanup
        self._ray_queues.extend([param_prompt_Q, inference_results_Q])

        batch = model_utils.Batch(queries=queries, ground_truths=ground_truths, datasets=datasets, indices=indices)

        # Create a mock generation_config for testing
        from unittest.mock import MagicMock

        mock_generation_config = MagicMock()
        mock_generation_config.n = 4

        grpo_fast.split_and_insert_batch(
            batch, training_step, num_engines, pending_queries_map, param_prompt_Q, mock_generation_config
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
        vllm_engines = vllm_utils3.create_vllm_engines(
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
            PromptRequest(prompts=[prompt_token_ids], dataset_index=0, sampling_params=generation_config)
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
        queries_next, ground_truths_next, datasets_next, dataset_indices = self.create_test_data(
            num_unique_prompts_rollout
        )

        # Setup and split batch
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries_next, ground_truths_next, datasets_next, dataset_indices, vllm_num_engines
        )

        # Verify that we have all prompts in the map
        self.assertEqual(len(pending_queries_map), num_unique_prompts_rollout)

        # Verify that we have individual prompts in the queue (changed from batches)
        self.assertEqual(param_prompt_Q.qsize(), num_unique_prompts_rollout)

        # Simulate vLLM processing - each prompt gets processed individually
        prompt_count = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertEqual(request.training_step, 1)
            self.assertIsInstance(request.dataset_index, int)  # Now individual prompts have single index

            # Create result for this individual prompt with n=4 samples
            mock_result = self.create_mock_result(
                request.dataset_index, request.training_step, num_samples_per_prompt=4
            )
            inference_results_Q.put(mock_result)
            prompt_count += 1

        # Verify we processed the right number of individual prompts
        self.assertEqual(prompt_count, num_unique_prompts_rollout)

        # Simulate accumulation
        combined_responses = []
        combined_queries = []
        combined_ground_truths = []
        combined_datasets = []

        # Process all results (we have num_unique_prompts_rollout individual results)
        for _ in range(num_unique_prompts_rollout):
            result = inference_results_Q.get()
            dataset_index = result.dataset_index

            # Get query for this index
            batch_queries = []
            batch_ground_truths = []
            batch_datasets = []
            q, gt, d = pending_queries_map.pop(dataset_index)
            batch_queries.append(q)
            batch_ground_truths.append(gt)
            batch_datasets.append(d)

            combined_responses.extend(result.responses)
            combined_queries.extend(batch_queries)
            combined_ground_truths.extend(batch_ground_truths)
            combined_datasets.extend(batch_datasets)

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
        # With n=4 samples per prompt, we expect 4x the number of responses
        self.assertEqual(len(combined_result.responses), len(queries_next) * 4)
        self.assertEqual(len(combined_result.finish_reasons), len(queries_next) * 4)
        self.assertEqual(len(combined_result.masks), len(queries_next) * 4)

        # Verify that the pending_queries_map is empty after accumulation
        self.assertEqual(len(pending_queries_map), 0)

        # Verify that the inference_results_Q is empty after accumulation
        self.assertEqual(inference_results_Q.qsize(), 0)

    def test_dataset_index_preservation_through_pipeline(self):
        """Test that dataset indices are correctly preserved through the pipeline."""
        vllm_num_engines = 4
        num_unique_prompts_rollout = 32

        # Create test data
        queries_next, ground_truths_next, datasets_next, dataset_indices = self.create_test_data(
            num_unique_prompts_rollout
        )

        # Setup and split batch
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries_next, ground_truths_next, datasets_next, dataset_indices, vllm_num_engines
        )

        # Simulate vLLM processing - processes individual prompts
        prompt_count = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            mock_result = self.create_mock_result(request.dataset_index, request.training_step)
            inference_results_Q.put(mock_result)
            prompt_count += 1

        # Simulate accumulation
        combined_queries = []
        combined_ground_truths = []
        combined_datasets = []

        # Process all individual results
        for _ in range(prompt_count):
            result = inference_results_Q.get()
            dataset_index = result.dataset_index

            q, gt, d = pending_queries_map.pop(dataset_index)
            combined_queries.append(q)
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
        queries_next, ground_truths_next, datasets_next, dataset_indices = self.create_test_data(
            num_unique_prompts_rollout
        )

        # Setup and split batch
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries_next, ground_truths_next, datasets_next, dataset_indices, vllm_num_engines
        )

        # Simulate vLLM processing with multiple samples - processes individual prompts
        prompt_count = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            mock_result = self.create_mock_result(request.dataset_index, request.training_step, num_samples_per_prompt)
            inference_results_Q.put(mock_result)
            prompt_count += 1

        # Simulate accumulation
        combined_responses = []
        combined_queries = []
        combined_ground_truths = []
        combined_datasets = []

        # Process all individual results
        for _ in range(prompt_count):
            result = inference_results_Q.get()
            dataset_index = result.dataset_index

            # Get query for this index
            batch_queries = []
            batch_ground_truths = []
            batch_datasets = []
            q, gt, d = pending_queries_map.pop(dataset_index)
            batch_queries.append(q)
            batch_ground_truths.append(gt)
            batch_datasets.append(d)

            combined_responses.extend(result.responses)
            combined_queries.extend(batch_queries)
            combined_ground_truths.extend(batch_ground_truths)
            combined_datasets.extend(batch_datasets)

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
        import random
        import time

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
        queries, ground_truths, datasets, indices = self.create_test_data(num_prompts)

        # Setup and split batch
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries, ground_truths, datasets, indices, num_engines
        )

        # Get all requests and process in reverse order
        requests = []
        while not param_prompt_Q.empty():
            requests.append(param_prompt_Q.get())

        # Put results back in REVERSE order to simulate out-of-order processing
        # Create one result per prompt with n responses
        for request in reversed(requests):
            mock_result = self.create_mock_result(request.dataset_index, request.training_step, num_samples_per_prompt)
            inference_results_Q.put(mock_result)

        # Accumulate results
        mock_args = self.create_mock_args(num_engines, num_samples_per_prompt, num_prompts)
        # Create a mock generation config with n
        mock_generation_config = Mock()
        mock_generation_config.n = num_samples_per_prompt

        combined_result, batch = grpo_fast.accumulate_inference_batches(
            inference_results_Q,
            pending_queries_map,
            mock_args,
            training_step=1,
            generation_config=mock_generation_config,
            num_prompts=num_prompts,
        )

        # Verify results work correctly even with out-of-order processing
        self.assertEqual(len(batch.queries), num_prompts)
        self.assertEqual(len(combined_result.responses), num_prompts * num_samples_per_prompt)
        self.assertEqual(len(pending_queries_map), 0)

    def test_thread_safety_pending_queries_map(self):
        """Test concurrent access to pending_queries_map."""
        import threading
        import time

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
                        i, f"query_{thread_id}_{i}", f"truth_{thread_id}_{i}", f"dataset_{thread_id}_{i}"
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

        # Setup with results from only 3 engines (missing one)
        # Queue size should accommodate results from engines
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * num_prompts)

        # Track queue for cleanup
        self._ray_queues.append(inference_results_Q)

        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Add entries to map
        for i in range(num_prompts):
            pending_queries_map.insert(i, f"q_{i}", f"t_{i}", f"d_{i}")

        # Add individual results (one per prompt) but missing some
        # accumulate_inference_batches now expects individual results (num_prompts * n)
        # Add results for only 12 prompts (missing 4)
        for i in range(12):  # Only 12 prompts, missing 4
            mock_result = self.create_mock_result(i, 1)
            inference_results_Q.put(mock_result)

        mock_args = self.create_mock_args(num_engines, num_prompts=num_prompts)

        # Test that accumulate blocks when missing results from the 4th engine
        import threading

        completed = threading.Event()

        def run_accumulate():
            try:
                # Create a mock generation config with n=1 (default)
                mock_generation_config = Mock()
                mock_generation_config.n = 1

                grpo_fast.accumulate_inference_batches(
                    inference_results_Q,
                    pending_queries_map,
                    mock_args,
                    training_step=1,
                    generation_config=mock_generation_config,
                    num_prompts=num_prompts,
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

        queries, ground_truths, datasets, indices = self.create_test_data(num_queries)
        param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(param_prompt_Q)

        batch = model_utils.Batch(queries=queries, ground_truths=ground_truths, datasets=datasets, indices=indices)

        # Create a mock generation_config
        from unittest.mock import MagicMock

        mock_generation_config = MagicMock()
        mock_generation_config.n = 1

        grpo_fast.split_and_insert_batch(
            batch,
            training_step=1,
            vllm_num_engines=num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
            generation_config=mock_generation_config,
        )

        # Should have 4 batches (one for each query)
        self.assertEqual(
            param_prompt_Q.qsize(), num_queries, f"Should have {num_queries} batches for {num_queries} queries"
        )

        # Each batch should have exactly 1 prompt
        batch_sizes = []
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertIsNotNone(request.prompt, "Each request should have a prompt")
            batch_sizes.append(1)  # Each PromptRequest contains exactly 1 prompt

        # All queries should be in the pending map
        self.assertEqual(len(pending_queries_map), num_queries)

    def test_uneven_distribution_no_empty_batches(self):
        """Test that split_and_insert_batch creates one PromptRequest per query."""
        num_engines = 3
        num_queries = 7

        queries, ground_truths, datasets, indices = self.create_test_data(num_queries)
        param_prompt_Q = ray_queue.Queue(maxsize=num_queries * 2)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(param_prompt_Q)

        batch = model_utils.Batch(queries=queries, ground_truths=ground_truths, datasets=datasets, indices=indices)

        # Create a mock generation_config
        from unittest.mock import MagicMock

        mock_generation_config = MagicMock()
        mock_generation_config.n = 1

        grpo_fast.split_and_insert_batch(
            batch,
            training_step=1,
            vllm_num_engines=num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
            generation_config=mock_generation_config,
        )

        # Verify we get one PromptRequest per query
        request_count = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            self.assertIsNotNone(request.prompt, "Each request should have a prompt")
            request_count += 1

        # Should have exactly num_queries PromptRequests
        self.assertEqual(request_count, num_queries, f"Should have {num_queries} PromptRequests")

    def test_streaming_accumulation_basic(self):
        """Test basic streaming accumulation with in-order results."""
        num_engines = 2
        num_prompts = 8

        # Create test data
        queries, ground_truths, datasets, indices = self.create_test_data(num_prompts)

        # Create queues and maps
        # Queue size should accommodate results from engines
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * num_prompts)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(inference_results_Q)

        # Insert data into pending_queries_map
        for i in range(num_prompts):
            pending_queries_map.insert(i, queries[i], ground_truths[i], datasets[i])

        # Create mock results - one per prompt
        for i in range(num_prompts):
            mock_result = self.create_mock_result(i, training_step=1)
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
            batch_queries = []
            batch_ground_truths = []
            batch_datasets = []
            q, gt, d = pending_queries_map.pop(dataset_index)
            batch_queries.append(q)
            batch_ground_truths.append(gt)
            batch_datasets.append(d)

            queries_list.append((batch_queries, batch_ground_truths, batch_datasets))

        # Verify all results processed
        self.assertEqual(len(results_list), expected_results)
        self.assertEqual(len(pending_queries_map), 0)

        # Combine in order
        combined_queries = []
        for i in range(num_prompts):
            q, _, _ = queries_list[i]
            combined_queries.extend(q)

        # Verify order is preserved
        self.assertEqual(combined_queries, queries)

    def test_streaming_with_multiple_samples(self):
        """Test streaming accumulation with multiple samples per prompt."""
        num_engines = 2
        num_prompts = 4
        num_samples = 3

        # Create test data
        queries, ground_truths, datasets, indices = self.create_test_data(num_prompts)

        # Create queues and maps
        # Queue size should accommodate results from engines
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * num_prompts)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(inference_results_Q)

        # Insert data with reference counting for multiple samples
        for i in range(num_prompts):
            for _ in range(num_samples):
                pending_queries_map.insert(i, queries[i], ground_truths[i], datasets[i])

        # Create results - one per prompt with multiple samples
        for i in range(num_prompts):
            mock_result = self.create_mock_result(i, training_step=1, num_samples_per_prompt=num_samples)
            inference_results_Q.put(mock_result)

        # Process results
        total_responses = 0
        while not inference_results_Q.empty():
            result = inference_results_Q.get()

            batch_prompts = 1  # Each result is for a single prompt now
            expected_responses = batch_prompts * num_samples
            self.assertEqual(len(result.responses), expected_responses)
            total_responses += len(result.responses)

            # Clean up pending_queries_map
            idx = result.dataset_index
            for _ in range(num_samples):
                if idx in pending_queries_map:
                    pending_queries_map.pop(idx)

        # Verify total responses
        self.assertEqual(total_responses, num_prompts * num_samples)
        self.assertEqual(len(pending_queries_map), 0)

    def test_vllm_tool_processing_completes(self):
        """Test that tool processing completes without hanging using actual tool settings."""
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, skipping test")

        # Create actual tools matching the script
        tools = {
            "</code>": tool_utils.CodeExecutionTool(
                start_str="<code>",
                end_str="</code>",
                api_endpoint="https://open-instruct-tool-server-10554368204.us-central1.run.app/execute",
            ),
            "</search>": tool_utils.SearchTool(
                start_str="<search>",
                end_str="</search>",
                api_endpoint="http://saturn-cs-aus-232.reviz.ai2.in:44177/search",
            ),
        }
        max_tool_calls = {"</code>": 5, "</search>": 5}

        # Create actual ActorManager via ray.remote
        actor_manager = ray.remote(vllm_utils3.ActorManager).remote(should_stop=False)

        # Create queues
        prompt_queue = queue.Queue()
        results_queue = queue.Queue()
        eval_results_queue = queue.Queue()

        # Create LLMRayActor
        model_name = "EleutherAI/pythia-14m"  # Small model for testing
        actor = vllm_utils3.LLMRayActor(
            model_name_or_path=model_name,
            actor_id=0,
            prompt_queue=prompt_queue,
            results_queue=results_queue,
            eval_results_queue=eval_results_queue,
            actor_manager=actor_manager,
            tools=tools,
            max_tool_calls=max_tool_calls,
            inference_batch_size=8,
            vllm_kwargs={"gpu_memory_utilization": 0.3, "max_model_len": 512, "enable_prefix_caching": True},
        )

        tokenizer = actor.llm_engine.tokenizer

        # Create test prompts that will trigger tools
        test_prompts = [
            "Write code to print hello: <code>print('hello')</code>",
            "Search for Python tutorials: <search>Python tutorial beginner</search>",
            "Calculate 2+2: <code>print(2+2)</code>",
            "Find vLLM documentation: <search>vLLM documentation</search>",
            "Write a simple function: <code>def greet(): return 'hi'</code>",
            "Search machine learning: <search>machine learning basics</search>",
            "Debug this code: <code>x = 1; print(x)</code>",
            "Look up PyTorch: <search>PyTorch tutorial</search>",
        ]

        # Create sampling params once
        sampling_params = SamplingParams(
            temperature=1.0, top_p=1.0, n=1, max_tokens=50, stop=["</code>", "</search>", "</answer>"]
        )

        # Add all requests to queue
        for i in range(16):
            prompt_text = test_prompts[i % len(test_prompts)]
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

            request = PromptRequest(
                prompt=prompt_ids, generation_config=sampling_params, training_step=0, dataset_index=i, is_eval=False
            )
            prompt_queue.put(request)

        # Set a timeout and process
        start_time = time.time()

        # Process requests - this is what we're testing!
        num_processed = actor.process_from_queue(timeout=30.0)

        elapsed = time.time() - start_time
        print(f"Processed {num_processed} requests in {elapsed:.2f} seconds")

        # Check results
        results_received = []
        while not results_queue.empty():
            try:
                result = results_queue.get_nowait()
                results_received.append(result)
            except queue.Empty:
                break

        print(f"Received {len(results_received)} results")

        # Verify we didn't hang
        self.assertLess(elapsed, 60, "Should complete in less than 60 seconds")

        # Verify we got results
        self.assertGreater(num_processed, 0, "Should have processed some requests")
        self.assertGreater(len(results_received), 0, "Should have received some results")

        # Verify tool processing worked
        for result in results_received:
            self.assertIsNotNone(result.responses)
            self.assertIsNotNone(result.request_info)
            self.assertIsNotNone(result.request_info.tool_calleds)
            # Check that at least some tools were called
            if any(result.request_info.tool_calleds):
                print(f"Tool was called for result with dataset_index {result.dataset_index}")


if __name__ == "__main__":
    unittest.main()
