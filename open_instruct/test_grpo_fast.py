import gc
import os
import unittest
from unittest.mock import Mock

import ray
import torch
from parameterized import parameterized
from ray.util import queue as ray_queue
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct import grpo_fast, model_utils, utils
from open_instruct.vllm_utils3 import GenerationResult, PromptRequest, RequestInfo, create_vllm_engines


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

        # Check for leaks after Ray init
        leak_report = utils.check_runtime_leaks()
        # After Ray init, we expect exactly one Ray head worker
        if len(leak_report.ray_workers) == 1:
            # Check if it's the head worker (worker ID all zeros or all f's)
            worker = leak_report.ray_workers[0]
            worker_id = worker.get("worker_id", "")
            if worker_id in [
                "01000000ffffffffffffffffffffffffffffffffffffffffffffffff",
                "00000000ffffffffffffffffffffffffffffffffffffffffffffffff",
            ]:
                # This is the expected Ray head worker, clear it
                leak_report.ray_workers = []

        if not leak_report.is_clean:
            self.fail(f"Leaks detected before test {self._testMethodName}:\n{leak_report.pretty()}")

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

        # Check for leaks before shutdown
        leak_report = utils.check_runtime_leaks()
        # We still expect the Ray head worker
        if len(leak_report.ray_workers) == 1:
            worker = leak_report.ray_workers[0]
            worker_id = worker.get("worker_id", "")
            if worker_id in [
                "01000000ffffffffffffffffffffffffffffffffffffffffffffffff",
                "00000000ffffffffffffffffffffffffffffffffffffffffffffffff",
            ]:
                # This is the expected Ray head worker, clear it
                leak_report.ray_workers = []

        if not leak_report.is_clean:
            self.fail(f"Leaks detected after test {self._testMethodName}:\n{leak_report.pretty()}")

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

    def create_mock_args(self, num_engines=4, num_samples=1):
        """Create mock args object."""
        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples
        return mock_args

    def create_mock_result(self, dataset_indices, training_step, num_samples_per_prompt=1):
        """Create a mock GenerationResult."""
        batch_size = len(dataset_indices)
        total_responses = batch_size * num_samples_per_prompt

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
            dataset_index=dataset_indices,
            training_step=training_step,
        )

    def setup_and_split_batch(self, queries, ground_truths, datasets, indices, num_engines, training_step=1):
        """Setup queues and split batch - common pattern."""
        param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)
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
        for engine in vllm_engines:
            engine.process_from_queue.remote(
                generation_config,
                generation_config,  # eval_sampling_params
                999,  # eval_freq (avoid evaluation)
                1,  # num_training_steps
                1,  # resume_training_step
            )

        # Put the test prompt in the queue using PromptRequest
        request = PromptRequest(prompts=[prompt_token_ids], dataset_index=0)
        param_prompt_Q.put(request)

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

        # Verify that we have individual prompts in the map (not batches)
        self.assertEqual(len(pending_queries_map), num_unique_prompts_rollout)

        # Verify that we have the expected number of items in the queue
        self.assertEqual(param_prompt_Q.qsize(), vllm_num_engines)

        # Simulate vLLM processing
        batch_idx = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertEqual(request.training_step, 1)
            self.assertIsInstance(request.dataset_index, list)

            mock_result = self.create_mock_result(request.dataset_index, request.training_step)
            inference_results_Q.put(mock_result)
            batch_idx += 1

        # Simulate streaming accumulation (simplified version for testing)
        combined_responses = []
        combined_queries = []
        combined_ground_truths = []
        combined_datasets = []

        for _ in range(vllm_num_engines):
            result = inference_results_Q.get()
            dataset_indices = result.dataset_index

            # Get queries from pending_queries_map
            batch_queries = []
            batch_ground_truths = []
            batch_datasets = []
            for idx in dataset_indices:
                q, gt, d = pending_queries_map.pop(idx)
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
            training_step=1,
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
        queries_next, ground_truths_next, datasets_next, dataset_indices = self.create_test_data(
            num_unique_prompts_rollout
        )

        # Setup and split batch
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries_next, ground_truths_next, datasets_next, dataset_indices, vllm_num_engines
        )

        # Simulate vLLM processing
        batch_idx = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            mock_result = self.create_mock_result(request.dataset_index, request.training_step)
            inference_results_Q.put(mock_result)
            batch_idx += 1

        # Simulate streaming accumulation
        combined_queries = []
        combined_ground_truths = []
        combined_datasets = []

        for _ in range(vllm_num_engines):
            result = inference_results_Q.get()
            dataset_indices = result.dataset_index

            for idx in dataset_indices:
                q, gt, d = pending_queries_map.pop(idx)
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

        # Simulate vLLM processing with multiple samples
        batch_idx = 0
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            mock_result = self.create_mock_result(request.dataset_index, request.training_step, num_samples_per_prompt)
            inference_results_Q.put(mock_result)
            batch_idx += 1

        # Simulate streaming accumulation
        combined_responses = []
        combined_queries = []
        combined_ground_truths = []
        combined_datasets = []

        for _ in range(vllm_num_engines):
            result = inference_results_Q.get()
            dataset_indices = result.dataset_index

            batch_queries = []
            batch_ground_truths = []
            batch_datasets = []
            for idx in dataset_indices:
                q, gt, d = pending_queries_map.pop(idx)
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
            training_step=1,
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
        for request in reversed(requests):
            mock_result = self.create_mock_result(request.dataset_index, request.training_step, num_samples_per_prompt)
            inference_results_Q.put(mock_result)

        # Accumulate results
        mock_args = self.create_mock_args(num_engines, num_samples_per_prompt)
        # Create a mock generation config with n
        mock_generation_config = Mock()
        mock_generation_config.n = num_samples_per_prompt

        combined_result, batch = grpo_fast.accumulate_inference_batches(
            inference_results_Q,
            pending_queries_map,
            mock_args,
            training_step=1,
            generation_config=mock_generation_config,
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

        # Setup with results from only 3 engines
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)

        # Track queue for cleanup
        self._ray_queues.append(inference_results_Q)

        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Add entries to map
        for i in range(num_prompts):
            pending_queries_map.insert(i, f"q_{i}", f"t_{i}", f"d_{i}")

        # Add results from only 3 engines (missing one)
        for engine_id in range(3):
            indices = list(range(engine_id * 4, (engine_id + 1) * 4))
            mock_result = self.create_mock_result(indices, 1)
            inference_results_Q.put(mock_result)

        mock_args = self.create_mock_args(num_engines)

        # Test that accumulate blocks when missing an engine
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
                )
                completed.set()
            except Exception:
                completed.set()

        thread = threading.Thread(target=run_accumulate, daemon=True)
        thread.start()

        # Should timeout waiting for 4th engine
        self.assertFalse(completed.wait(timeout=1.0))
        self.assertTrue(thread.is_alive())

        # Queue should be empty after consuming 3 results
        self.assertEqual(inference_results_Q.qsize(), 0)
        # Some entries should be removed
        self.assertLess(len(pending_queries_map), num_prompts)


class TestStreamingAccumulation(TestGrpoFastBase):
    """Test the new streaming accumulation functionality."""

    def test_streaming_accumulation_basic(self):
        """Test basic streaming accumulation with in-order results."""
        num_engines = 2
        num_prompts = 8

        # Create test data
        queries, ground_truths, datasets, indices = self.create_test_data(num_prompts)

        # Create queues and maps
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(inference_results_Q)

        # Insert data into pending_queries_map
        for i in range(num_prompts):
            pending_queries_map.insert(i, queries[i], ground_truths[i], datasets[i])

        # Create mock results with batch indices
        batch_size = num_prompts // num_engines
        for batch_idx in range(num_engines):
            start = batch_idx * batch_size
            end = start + batch_size
            mock_result = self.create_mock_result(list(range(start, end)), training_step=1)
            inference_results_Q.put(mock_result)

        # Simulate streaming accumulation logic
        results_list = []
        queries_list = []
        expected_batches = num_engines

        while len(results_list) < expected_batches:
            result = inference_results_Q.get()
            batch_idx = len(results_list)

            results_list.append(result)

            # Get queries for this batch
            dataset_indices = result.dataset_index
            batch_queries = []
            batch_ground_truths = []
            batch_datasets = []
            for idx in dataset_indices:
                q, gt, d = pending_queries_map.pop(idx)
                batch_queries.append(q)
                batch_ground_truths.append(gt)
                batch_datasets.append(d)

            queries_list.append((batch_queries, batch_ground_truths, batch_datasets))

        # Verify all batches processed
        self.assertEqual(len(results_list), expected_batches)
        self.assertEqual(len(pending_queries_map), 0)

        # Combine in order
        combined_queries = []
        for i in range(num_engines):
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
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Track queue for cleanup
        self._ray_queues.append(inference_results_Q)

        # Insert data with reference counting for multiple samples
        for i in range(num_prompts):
            for _ in range(num_samples):
                pending_queries_map.insert(i, queries[i], ground_truths[i], datasets[i])

        # Create results with multiple samples per prompt
        batch_size = num_prompts // num_engines
        for batch_idx in range(num_engines):
            start = batch_idx * batch_size
            end = start + batch_size
            dataset_indices = list(range(start, end))

            # Create result with num_samples responses per prompt
            mock_result = self.create_mock_result(dataset_indices, training_step=1, num_samples_per_prompt=num_samples)
            inference_results_Q.put(mock_result)

        # Process results
        total_responses = 0
        while not inference_results_Q.empty():
            result = inference_results_Q.get()

            # Verify number of responses matches num_samples * num_prompts_in_batch
            batch_prompts = len(result.dataset_index)
            expected_responses = batch_prompts * num_samples
            self.assertEqual(len(result.responses), expected_responses)
            total_responses += len(result.responses)

            # Clean up pending_queries_map
            for idx in result.dataset_index:
                for _ in range(num_samples):
                    if idx in pending_queries_map:
                        pending_queries_map.pop(idx)

        # Verify total responses
        self.assertEqual(total_responses, num_prompts * num_samples)
        self.assertEqual(len(pending_queries_map), 0)


if __name__ == "__main__":
    unittest.main()