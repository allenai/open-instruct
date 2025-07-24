import unittest
from unittest.mock import Mock

import ray
import torch
from parameterized import parameterized
from ray.util import queue as ray_queue
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct import grpo_fast
from open_instruct.vllm_utils3 import GenerationResult, PromptRequest, RequestInfo, create_vllm_engines


class TestGrpoFastBase(unittest.TestCase):
    """Base class with common test utilities."""

    @classmethod
    def setUpClass(cls):
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        # Shutdown Ray after test
        if ray.is_initialized():
            ray.shutdown()

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

        grpo_fast.split_and_insert_batch(
            queries, ground_truths, datasets, indices, training_step, num_engines, pending_queries_map, param_prompt_Q
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
                1,  # batch_size
            )

        # Put the test prompt in the queue using PromptRequest
        request = PromptRequest(prompt=prompt_token_ids, dataset_index=0)
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

        self.assertEqual(len(pending_queries_map), num_unique_prompts_rollout)

        self.assertEqual(param_prompt_Q.qsize(), num_unique_prompts_rollout)

        # Simulate vLLM processing
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertEqual(request.training_step, 1)
            self.assertIsInstance(request.dataset_index, list)

            mock_result = self.create_mock_result(request.dataset_index, request.training_step)
            inference_results_Q.put(mock_result)

        # Accumulate results
        mock_args = self.create_mock_args(vllm_num_engines)
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(inference_results_Q, pending_queries_map, mock_args, 1)
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

        # Verify that the test_pending_queries_map is empty after accumulation
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
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            mock_result = self.create_mock_result(request.dataset_index, request.training_step)
            inference_results_Q.put(mock_result)

        # Accumulate results
        mock_args = self.create_mock_args(vllm_num_engines)
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=1
            )
        )

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
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            mock_result = self.create_mock_result(request.dataset_index, request.training_step, num_samples_per_prompt)
            inference_results_Q.put(mock_result)

        # Accumulate results
        mock_args = self.create_mock_args(vllm_num_engines, num_samples_per_prompt)
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=1
            )
        )

        # Verify results - accumulate_inference_batches should NOT replicate
        self.assertEqual(combined_queries, queries_next)
        self.assertEqual(combined_ground_truths, ground_truths_next)
        self.assertEqual(combined_datasets, datasets_next)
        self.assertEqual(len(pending_queries_map), 0)

        # Verify correct number of responses
        expected_responses = num_unique_prompts_rollout * num_samples_per_prompt
        self.assertEqual(len(combined_result.responses), expected_responses)


class GrpoIntegrationTests(TestGrpoFastBase):
    """Integration tests for GRPO with parallel processing."""

    def test_training_step_isolation(self):
        """Test that different training steps are handled correctly."""
        num_engines = 2
        num_prompts = 16

        pending_queries_map = grpo_fast.PendingQueriesMap()
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 4)

        # Create data for two different training steps
        step1_data = self.create_test_data(num_prompts, "step1_", 0)
        step2_data = self.create_test_data(num_prompts, "step2_", 1000)

        # Process both steps
        for step, (queries, ground_truths, datasets, indices) in enumerate([step1_data, step2_data], 1):
            param_prompt_Q = ray_queue.Queue(maxsize=num_engines)

            grpo_fast.split_and_insert_batch(
                queries, ground_truths, datasets, indices, step, num_engines, pending_queries_map, param_prompt_Q
            )

            # Create results for this step
            while not param_prompt_Q.empty():
                request = param_prompt_Q.get()
                mock_result = self.create_mock_result(request.dataset_index, step)
                inference_results_Q.put(mock_result)

        # Process results from both steps
        mock_args = self.create_mock_args(num_engines)

        # First accumulation gets step 1 results
        result1, queries1, _, _ = grpo_fast.accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=2
        )
        self.assertTrue(all("step1_" in q for q in queries1))

        # Second accumulation gets step 2 results
        result2, queries2, _, _ = grpo_fast.accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=2
        )
        self.assertTrue(all("step2_" in q for q in queries2))

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

    def test_high_dataset_indices(self):
        """Test handling of high dataset indices (bug #25847)."""
        num_engines = 4
        num_prompts = 64
        num_samples_per_prompt = 16
        start_index = 25800

        # Create test data with high indices
        queries, ground_truths, datasets, indices = self.create_test_data(num_prompts, start_idx=start_index)

        # Setup and split batch
        param_prompt_Q, inference_results_Q, pending_queries_map = self.setup_and_split_batch(
            queries, ground_truths, datasets, indices, num_engines
        )

        # Verify the specific index from the bug is present
        self.assertIn(25847, pending_queries_map)

        # Process all requests
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            mock_result = self.create_mock_result(request.dataset_index, 1, num_samples_per_prompt)
            inference_results_Q.put(mock_result)

        # Accumulate results
        mock_args = self.create_mock_args(num_engines, num_samples_per_prompt)
        combined_result, _, _, _ = grpo_fast.accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=1
        )

        # Verify results
        self.assertEqual(len(combined_result.responses), num_prompts * num_samples_per_prompt)
        self.assertEqual(len(pending_queries_map), 0)

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
        combined_result, combined_queries, _, _ = grpo_fast.accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=1
        )

        # Verify results work correctly even with out-of-order processing
        self.assertEqual(len(combined_queries), num_prompts)
        self.assertEqual(len(combined_result.responses), num_prompts * num_samples_per_prompt)
        self.assertEqual(len(pending_queries_map), 0)

    def test_no_duplicate_indices_within_training_step(self):
        """Test that dataset indices are unique within a single training step."""
        num_engines = 4
        num_prompts = 64

        # Create test data
        queries, ground_truths, datasets, indices = self.create_test_data(num_prompts)

        # Setup and split batch
        param_prompt_Q, _, pending_queries_map = self.setup_and_split_batch(
            queries, ground_truths, datasets, indices, num_engines
        )

        # Check for duplicates
        self.assertEqual(len(pending_queries_map), num_prompts)
        self.assertEqual(set(pending_queries_map.keys()), set(indices))

        # Check each batch has unique indices
        all_indices_from_batches = []
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            batch_indices = request.dataset_index
            # Check no duplicates within this batch
            self.assertEqual(len(batch_indices), len(set(batch_indices)))
            all_indices_from_batches.extend(batch_indices)

        # Check total indices match
        self.assertEqual(sorted(all_indices_from_batches), sorted(indices))

    def test_overlapping_indices_between_steps(self):
        """Test handling of overlapping dataset indices between training steps."""
        num_engines = 2
        num_prompts = 16

        # Simulate overlapping indices
        step1_indices = list(range(0, num_prompts))  # [0, 1, ..., 15]
        step2_indices = list(range(10, num_prompts + 10))  # [10, 11, ..., 25]

        pending_queries_map = grpo_fast.PendingQueriesMap()
        param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)

        # Process step 1
        step1_data = self.create_test_data(num_prompts, "step1_")
        queries1, ground_truths1, datasets1, _ = step1_data

        grpo_fast.split_and_insert_batch(
            queries1, ground_truths1, datasets1, step1_indices, 1, num_engines, pending_queries_map, param_prompt_Q
        )

        # Clear queue and process step 2
        while not param_prompt_Q.empty():
            param_prompt_Q.get()

        step2_data = self.create_test_data(num_prompts, "step2_")
        queries2, ground_truths2, datasets2, _ = step2_data

        grpo_fast.split_and_insert_batch(
            queries2, ground_truths2, datasets2, step2_indices, 2, num_engines, pending_queries_map, param_prompt_Q
        )

        # Verify overlapping indices handling
        overlapping_indices = set(step1_indices) & set(step2_indices)

        for idx in overlapping_indices:
            query, ground_truth, dataset, count = pending_queries_map[idx]
            self.assertEqual(count, 2)  # Should have count 2
            # First insertion wins
            self.assertTrue(query.startswith("step1_"))
            self.assertTrue(ground_truth.startswith("step1_"))
            self.assertTrue(dataset.startswith("step1_"))

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

    @parameterized.expand(
        [
            (17, 4),  # 17 prompts, 4 engines
            (15, 4),  # 15 prompts, 4 engines
            (7, 3),  # 7 prompts, 3 engines
            (100, 7),  # 100 prompts, 7 engines
        ]
    )
    def test_uneven_batch_distribution(self, num_prompts, num_engines):
        """Test batch splitting when prompts don't divide evenly."""
        queries, ground_truths, datasets, indices = self.create_test_data(num_prompts)

        param_prompt_Q, _, pending_queries_map = self.setup_and_split_batch(
            queries, ground_truths, datasets, indices, num_engines
        )

        # Verify all indices are accounted for
        self.assertEqual(len(pending_queries_map), num_prompts)

        total_indices = []
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            total_indices.extend(request.dataset_index)

        self.assertEqual(sorted(total_indices), sorted(indices))

    def test_accumulate_waits_for_all_engines(self):
        """Test that accumulate_inference_batches waits for all engines."""
        num_engines = 4
        num_prompts = 16

        # Setup with results from only 3 engines
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)
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
                grpo_fast.accumulate_inference_batches(
                    inference_results_Q, pending_queries_map, mock_args, training_step=1
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


if __name__ == "__main__":
    unittest.main()
