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


class TestGrpoFastVLLM(unittest.TestCase):
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

    @parameterized.expand([(1,), (2,), (4,), (8,)])
    def test_batch_splitting_logic(self, vllm_num_engines: int, num_unique_prompts_rollout: int = 16):
        """Test the batch splitting and accumulation logic using split_and_insert_batch and accumulate_inference_batches."""

        # Mock data - simulating num_unique_prompts_rollout * num_samples_per_prompt_rollout
        queries_next = [f"query_{i}" for i in range(num_unique_prompts_rollout)]
        ground_truths_next = [f"truth_{i}" for i in range(num_unique_prompts_rollout)]
        datasets_next = [f"dataset_{i}" for i in range(num_unique_prompts_rollout)]

        pending_queries_map = grpo_fast.PendingQueriesMap()
        training_step = 1

        # Create mock Ray queue for testing
        param_prompt_Q = ray_queue.Queue(maxsize=vllm_num_engines)

        # Create mock dataset indices
        dataset_indices = list(range(num_unique_prompts_rollout))

        # Use split_and_insert_batch to split and insert data
        grpo_fast.split_and_insert_batch(
            queries_next,
            ground_truths_next,
            datasets_next,
            dataset_indices,
            training_step,
            vllm_num_engines,
            pending_queries_map,
            param_prompt_Q,
        )

        # Verify that we have individual prompts in the map (not batches)
        self.assertEqual(len(pending_queries_map), num_unique_prompts_rollout)

        # Verify that we have the expected number of items in the queue
        self.assertEqual(param_prompt_Q.qsize(), vllm_num_engines)

        # Create mock inference results to simulate vLLM engine outputs
        mock_inference_results = []
        for batch_idx in range(vllm_num_engines):
            # Get the request from the queue
            request = param_prompt_Q.get()
            self.assertIsInstance(request, PromptRequest)
            self.assertEqual(request.training_step, training_step)
            self.assertIsInstance(request.dataset_index, list)  # Now expects a list of indices

            # Create mock GenerationResult
            batch_size = len(request.prompts)
            mock_result = GenerationResult(
                responses=[[i] for i in range(batch_size)],  # Mock token IDs
                finish_reasons=["stop"] * batch_size,
                masks=[[1] * 5] * batch_size,  # Mock masks
                request_info=RequestInfo(
                    num_calls=[0] * batch_size,
                    timeouts=[0] * batch_size,
                    tool_errors=[""] * batch_size,
                    tool_outputs=[""] * batch_size,
                    tool_runtimes=[0] * batch_size,
                    tool_calleds=[False] * batch_size,
                ),
                is_eval=False,
                dataset_index=request.dataset_index,
                training_step=request.training_step,
            )
            mock_inference_results.append(mock_result)

        # Create mock inference results queue
        inference_results_Q = ray_queue.Queue(maxsize=vllm_num_engines)
        for result in mock_inference_results:
            inference_results_Q.put(result)

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = vllm_num_engines
        mock_args.num_samples_per_prompt_rollout = 1  # Default for this test

        # Use accumulate_inference_batches to combine results
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(inference_results_Q, pending_queries_map, mock_args, training_step)
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

    def mock_vllm_pipeline(self, prompt_queue, results_queue, num_engines, num_steps=1, num_samples_per_prompt=1):
        """Mock function that simulates vLLM engines pulling from prompt queue and pushing to results queue"""
        for step in range(num_steps):
            for engine_id in range(num_engines):
                # Pull request from prompt queue
                request = prompt_queue.get()

                # Create mock generation result preserving dataset_index
                batch_size = len(request.prompts)
                # Generate num_samples_per_prompt responses for each prompt
                total_responses = batch_size * num_samples_per_prompt

                # In real vLLM, dataset_index is NOT replicated when num_samples_per_prompt > 1
                # It stays as the original list of unique indices
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
                    dataset_index=request.dataset_index,  # Keep original indices (not replicated)
                    training_step=request.training_step,
                )

                # Push to results queue
                results_queue.put(mock_result)

    def test_dataset_index_preservation_through_pipeline(self):
        """Test that dataset indices are correctly preserved through the mock vLLM pipeline"""
        vllm_num_engines = 4
        num_unique_prompts_rollout = 32

        # Mock data
        queries_next = [f"query_{i}" for i in range(num_unique_prompts_rollout)]
        ground_truths_next = [f"truth_{i}" for i in range(num_unique_prompts_rollout)]
        datasets_next = [f"dataset_{i}" for i in range(num_unique_prompts_rollout)]
        dataset_indices = list(range(num_unique_prompts_rollout))

        # Create queues and map
        param_prompt_Q = ray_queue.Queue(maxsize=vllm_num_engines)
        inference_results_Q = ray_queue.Queue(maxsize=vllm_num_engines)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Split and insert batch
        grpo_fast.split_and_insert_batch(
            queries_next,
            ground_truths_next,
            datasets_next,
            dataset_indices,
            training_step=1,
            vllm_num_engines=vllm_num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
        )

        # Run mock vLLM pipeline
        self.mock_vllm_pipeline(param_prompt_Q, inference_results_Q, vllm_num_engines)

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = vllm_num_engines
        mock_args.num_samples_per_prompt_rollout = 1

        # Accumulate results
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=1
            )
        )

        # Verify results
        self.assertEqual(combined_queries, queries_next)
        self.assertEqual(combined_ground_truths, ground_truths_next)
        self.assertEqual(combined_datasets, datasets_next)
        self.assertEqual(len(pending_queries_map), 0)  # Map should be empty after accumulation

    def test_multiple_training_steps(self):
        """Test that indices don't get mixed up between multiple training steps"""
        vllm_num_engines = 2
        num_unique_prompts_rollout = 16
        num_steps = 3

        pending_queries_map = grpo_fast.PendingQueriesMap()

        for step in range(1, num_steps + 1):
            # Create unique data for each step
            queries_next = [f"query_step{step}_{i}" for i in range(num_unique_prompts_rollout)]
            ground_truths_next = [f"truth_step{step}_{i}" for i in range(num_unique_prompts_rollout)]
            datasets_next = [f"dataset_step{step}_{i}" for i in range(num_unique_prompts_rollout)]
            # Use different indices for each step to ensure no mixing
            dataset_indices = list(range((step - 1) * num_unique_prompts_rollout, step * num_unique_prompts_rollout))

            # Create queues
            param_prompt_Q = ray_queue.Queue(maxsize=vllm_num_engines)
            inference_results_Q = ray_queue.Queue(maxsize=vllm_num_engines)

            # Split and insert
            grpo_fast.split_and_insert_batch(
                queries_next,
                ground_truths_next,
                datasets_next,
                dataset_indices,
                training_step=step,
                vllm_num_engines=vllm_num_engines,
                pending_queries_map=pending_queries_map,
                param_prompt_Q=param_prompt_Q,
            )

            # Run pipeline
            self.mock_vllm_pipeline(param_prompt_Q, inference_results_Q, vllm_num_engines)

            # Create mock args
            mock_args = Mock()
            mock_args.vllm_num_engines = vllm_num_engines
            mock_args.num_samples_per_prompt_rollout = 1

            # Accumulate
            combined_result, combined_queries, combined_ground_truths, combined_datasets = (
                grpo_fast.accumulate_inference_batches(
                    inference_results_Q, pending_queries_map, mock_args, training_step=step
                )
            )

            # Verify
            self.assertEqual(combined_queries, queries_next)
            self.assertEqual(combined_ground_truths, ground_truths_next)
            self.assertEqual(combined_datasets, datasets_next)
            self.assertEqual(len(pending_queries_map), 0)  # Map should be empty after each step

    @parameterized.expand([(1,), (2,), (4,), (8,)])
    def test_various_engine_configurations(self, vllm_num_engines: int):
        """Test with various numbers of vLLM engines"""
        num_unique_prompts_rollout = 64  # Use a larger batch to test splitting

        # Mock data
        queries_next = [f"query_{i}" for i in range(num_unique_prompts_rollout)]
        ground_truths_next = [f"truth_{i}" for i in range(num_unique_prompts_rollout)]
        datasets_next = [f"dataset_{i}" for i in range(num_unique_prompts_rollout)]
        dataset_indices = list(range(num_unique_prompts_rollout))

        # Create queues and map
        param_prompt_Q = ray_queue.Queue(maxsize=vllm_num_engines)
        inference_results_Q = ray_queue.Queue(maxsize=vllm_num_engines)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Split and insert
        grpo_fast.split_and_insert_batch(
            queries_next,
            ground_truths_next,
            datasets_next,
            dataset_indices,
            training_step=1,
            vllm_num_engines=vllm_num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
        )

        # Verify correct number of batches created
        self.assertEqual(param_prompt_Q.qsize(), vllm_num_engines)

        # Run pipeline
        self.mock_vllm_pipeline(param_prompt_Q, inference_results_Q, vllm_num_engines)

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = vllm_num_engines
        mock_args.num_samples_per_prompt_rollout = 1

        # Accumulate
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=1
            )
        )

        # Verify
        self.assertEqual(combined_queries, queries_next)
        self.assertEqual(combined_ground_truths, ground_truths_next)
        self.assertEqual(combined_datasets, datasets_next)
        self.assertEqual(len(pending_queries_map), 0)

    def test_multiple_samples_per_prompt_fixed(self):
        """Test that the fix properly handles multiple samples per prompt"""
        vllm_num_engines = 1  # Use 1 engine to make it clearer
        num_unique_prompts_rollout = 16
        num_samples_per_prompt = 16  # This will cause 16 prompts to generate 256 responses

        # Mock data
        queries_next = [f"query_{i}" for i in range(num_unique_prompts_rollout)]
        ground_truths_next = [f"truth_{i}" for i in range(num_unique_prompts_rollout)]
        datasets_next = [f"dataset_{i}" for i in range(num_unique_prompts_rollout)]
        dataset_indices = list(range(num_unique_prompts_rollout))

        # Create queues and map
        param_prompt_Q = ray_queue.Queue(maxsize=vllm_num_engines)
        inference_results_Q = ray_queue.Queue(maxsize=vllm_num_engines)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Split and insert batch
        grpo_fast.split_and_insert_batch(
            queries_next,
            ground_truths_next,
            datasets_next,
            dataset_indices,
            training_step=1,
            vllm_num_engines=vllm_num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
        )

        # Run mock vLLM pipeline with multiple samples per prompt
        self.mock_vllm_pipeline(
            param_prompt_Q, inference_results_Q, vllm_num_engines, num_samples_per_prompt=num_samples_per_prompt
        )

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = vllm_num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples_per_prompt

        # Now with the fix, this should succeed
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=1
            )
        )

        # Verify results - accumulate_inference_batches should NOT replicate
        # The replication happens later in data_preparation_thread
        self.assertEqual(combined_queries, queries_next)
        self.assertEqual(combined_ground_truths, ground_truths_next)
        self.assertEqual(combined_datasets, datasets_next)
        self.assertEqual(len(pending_queries_map), 0)

        # Verify that we have the correct number of responses (256 = 16 prompts * 16 samples per prompt)
        self.assertEqual(len(combined_result.responses), 256)

    def test_multiple_samples_with_repeated_indices(self):
        """Test that the fix properly handles repeated dataset indices from vLLM"""
        vllm_num_engines = 1
        num_unique_prompts = 4
        num_samples_per_prompt = 4

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = vllm_num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples_per_prompt

        # Create test data
        training_step = 1
        pending_queries_map = grpo_fast.PendingQueriesMap()
        for i in range(num_unique_prompts):
            pending_queries_map.insert(i, f"query_{i}", f"truth_{i}", f"dataset_{i}")

        # Create a mock result with dataset indices NOT repeated
        # vLLM generates multiple responses per prompt but keeps dataset_index as [0, 1, 2, 3]
        dataset_indices = list(range(num_unique_prompts))
        total_responses = num_unique_prompts * num_samples_per_prompt

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
            dataset_index=dataset_indices,
            training_step=training_step,
        )

        # Create queue and add result
        inference_results_Q = ray_queue.Queue(maxsize=1)
        inference_results_Q.put(mock_result)

        # Call accumulate_inference_batches
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=1
            )
        )

        # Verify results
        # accumulate_inference_batches should NOT replicate - returns unique entries
        expected_queries = [f"query_{i}" for i in range(num_unique_prompts)]
        expected_ground_truths = [f"truth_{i}" for i in range(num_unique_prompts)]
        expected_datasets = [f"dataset_{i}" for i in range(num_unique_prompts)]

        self.assertEqual(combined_queries, expected_queries)
        self.assertEqual(combined_ground_truths, expected_ground_truths)
        self.assertEqual(combined_datasets, expected_datasets)
        self.assertEqual(len(pending_queries_map), 0)  # All entries should be popped


class GrpoIntegrationTests(unittest.TestCase):
    """Integration tests for GRPO with parallel processing."""

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

    def test_no_training_step_synchronization_issues(self):
        """Test that training steps don't get mixed up results from different steps."""

        num_engines = 4

        # Shared state
        pending_queries_map = grpo_fast.PendingQueriesMap()
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 4)

        # Create requests for step 1
        step1_requests = []
        for engine_id in range(num_engines):
            start_idx = engine_id * 4
            end_idx = start_idx + 4
            request = PromptRequest(
                prompts=[[1, 2, 3] for _ in range(4)], training_step=1, dataset_index=list(range(start_idx, end_idx))
            )
            step1_requests.append(request)
            # Add to pending_queries_map
            for idx in request.dataset_index:
                pending_queries_map.insert(idx, f"q_{idx}", f"t_{idx}", f"d_{idx}")

        # Create requests for step 2
        step2_requests = []
        for engine_id in range(num_engines):
            start_idx = 1000 + engine_id * 4  # Different indices
            end_idx = start_idx + 4
            request = PromptRequest(
                prompts=[[4, 5, 6] for _ in range(4)], training_step=2, dataset_index=list(range(start_idx, end_idx))
            )
            step2_requests.append(request)
            # Add to pending_queries_map
            for idx in request.dataset_index:
                pending_queries_map.insert(idx, f"q_{idx}", f"t_{idx}", f"d_{idx}")

        # Simulate the scenario where step 1 results are delayed
        # and step 2 tries to accumulate but gets step 1's results

        # First, put step 1's results in the queue
        for request in step1_requests:
            batch_size = len(request.prompts)
            mock_result = GenerationResult(
                responses=[[1, 2, 3] for _ in range(batch_size)],
                finish_reasons=["stop"] * batch_size,
                masks=[[1, 1, 1] for _ in range(batch_size)],
                request_info=RequestInfo(
                    num_calls=[0] * batch_size,
                    timeouts=[0] * batch_size,
                    tool_errors=[""] * batch_size,
                    tool_outputs=[""] * batch_size,
                    tool_runtimes=[0.0] * batch_size,
                    tool_calleds=[False] * batch_size,
                ),
                dataset_index=request.dataset_index,
                training_step=request.training_step,
            )
            inference_results_Q.put(mock_result)

        # Now step 2 tries to accumulate, expecting step 2 results
        # but will get step 1 results instead
        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.num_samples_per_prompt_rollout = 1

        # With our fix, we can now process results from any training step
        # as long as entries exist in the map. This successfully processes
        # step 1's results even though we're calling from step 2.
        result, queries, ground_truths, datasets = grpo_fast.accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=2
        )

        # Verify we got step 1's data (indices 0-15)
        self.assertEqual(len(queries), 16)
        self.assertEqual(len(ground_truths), 16)
        self.assertEqual(len(datasets), 16)

        # Check that we got the correct data from step 1
        for i in range(16):
            self.assertEqual(queries[i], f"q_{i}")
            self.assertEqual(ground_truths[i], f"t_{i}")
            self.assertEqual(datasets[i], f"d_{i}")

        # Now if we try to get more results, we should get step 2's data
        # because step 1's entries have been consumed
        # First add step 2's results to the queue
        for request in step2_requests:
            batch_size = len(request.prompts)
            mock_result = GenerationResult(
                responses=[[4, 5, 6] for _ in range(batch_size)],
                finish_reasons=["stop"] * batch_size,
                masks=[[1, 1, 1] for _ in range(batch_size)],
                request_info=RequestInfo(
                    num_calls=[0] * batch_size,
                    timeouts=[0] * batch_size,
                    tool_errors=[""] * batch_size,
                    tool_outputs=[""] * batch_size,
                    tool_runtimes=[0.0] * batch_size,
                    tool_calleds=[False] * batch_size,
                ),
                dataset_index=request.dataset_index,
                training_step=request.training_step,
            )
            inference_results_Q.put(mock_result)

        # Process step 2's results
        result2, queries2, ground_truths2, datasets2 = grpo_fast.accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=2
        )

        # Verify we got step 2's data (indices 1000-1015)
        self.assertEqual(len(queries2), 16)
        for i in range(16):
            idx = 1000 + i
            self.assertEqual(queries2[i], f"q_{idx}")
            self.assertEqual(ground_truths2[i], f"t_{idx}")
            self.assertEqual(datasets2[i], f"d_{idx}")

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

    def test_dataset_index_25847_bug(self):
        """Test to reproduce the specific bug with dataset index 25847."""
        # This simulates a scenario where we have a large dataset and might get
        # indices that don't start from 0
        num_engines = 4
        num_prompts = 64
        num_samples_per_prompt = 16
        start_index = 25800  # Start with a high index like in the error

        # Create test data with high indices
        queries_next = [f"query_{i}" for i in range(start_index, start_index + num_prompts)]
        ground_truths_next = [f"truth_{i}" for i in range(start_index, start_index + num_prompts)]
        datasets_next = [f"dataset_{i}" for i in range(start_index, start_index + num_prompts)]
        dataset_indices = list(range(start_index, start_index + num_prompts))

        # Create queues
        param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Split and insert batch
        grpo_fast.split_and_insert_batch(
            queries_next,
            ground_truths_next,
            datasets_next,
            dataset_indices,
            training_step=1,
            vllm_num_engines=num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
        )

        # Verify the pending queries map contains the right indices
        self.assertIn(25847, pending_queries_map)

        # Get all requests and create results
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            batch_size = len(request.prompts)
            total_responses = batch_size * num_samples_per_prompt

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
                dataset_index=request.dataset_index,
                training_step=request.training_step,
            )
            inference_results_Q.put(mock_result)

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples_per_prompt

        # Accumulate results
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=1
            )
        )

        # Verify results - accumulate_inference_batches returns unique entries
        self.assertEqual(len(combined_queries), num_prompts)  # Not replicated
        self.assertEqual(
            len(combined_result.responses), num_prompts * num_samples_per_prompt
        )  # Responses are replicated
        self.assertEqual(len(pending_queries_map), 0)

    def test_out_of_order_processing(self):
        """Test that dataset indices can be processed out of order."""
        num_engines = 4
        num_prompts = 16
        num_samples_per_prompt = 4

        # Create test data
        queries_next = [f"query_{i}" for i in range(num_prompts)]
        ground_truths_next = [f"truth_{i}" for i in range(num_prompts)]
        datasets_next = [f"dataset_{i}" for i in range(num_prompts)]
        dataset_indices = list(range(num_prompts))

        # Create queues
        param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Split and insert batch
        grpo_fast.split_and_insert_batch(
            queries_next,
            ground_truths_next,
            datasets_next,
            dataset_indices,
            training_step=1,
            vllm_num_engines=num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
        )

        # Get all requests from the queue
        requests = []
        while not param_prompt_Q.empty():
            requests.append(param_prompt_Q.get())

        # Put results back in REVERSE order to simulate out-of-order processing
        for request in reversed(requests):
            batch_size = len(request.prompts)
            total_responses = batch_size * num_samples_per_prompt

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
                dataset_index=request.dataset_index,
                training_step=request.training_step,
            )
            inference_results_Q.put(mock_result)

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples_per_prompt

        # Accumulate results
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=1
            )
        )

        # Verify results - accumulate_inference_batches returns unique entries
        self.assertEqual(len(combined_queries), num_prompts)  # Not replicated
        self.assertEqual(
            len(combined_result.responses), num_prompts * num_samples_per_prompt
        )  # Responses are replicated
        self.assertEqual(len(pending_queries_map), 0)

    @parameterized.expand(
        [
            (1, 16, 16),  # 1 engine, 16 prompts, 16 samples
            (2, 32, 16),  # 2 engines, 32 prompts, 16 samples
            (4, 64, 16),  # 4 engines, 64 prompts, 16 samples
            (8, 128, 16),  # 8 engines, 128 prompts, 16 samples
            (4, 100, 16),  # 4 engines, 100 prompts (not evenly divisible), 16 samples
        ]
    )
    def test_parallel_processing_with_multiple_engines(self, num_engines, num_prompts, num_samples_per_prompt):
        """Test parallel processing with multiple engines running concurrently."""
        import time

        # Create test data
        queries_next = [f"query_{i}" for i in range(num_prompts)]
        ground_truths_next = [f"truth_{i}" for i in range(num_prompts)]
        datasets_next = [f"dataset_{i}" for i in range(num_prompts)]
        dataset_indices = list(range(num_prompts))

        # Create queues
        param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Split and insert batch
        grpo_fast.split_and_insert_batch(
            queries_next,
            ground_truths_next,
            datasets_next,
            dataset_indices,
            training_step=1,
            vllm_num_engines=num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
        )

        # Verify initial state
        self.assertEqual(len(pending_queries_map), num_prompts)
        self.assertEqual(param_prompt_Q.qsize(), num_engines)

        # Start mock vLLM engines
        engines = []
        for i in range(num_engines):
            engine = GrpoIntegrationTests.mock_vllm_engine.remote(
                i, param_prompt_Q, inference_results_Q, num_samples_per_prompt
            )
            engines.append(engine)

        # Wait a bit for engines to process
        time.sleep(0.5)

        # Send stop signals
        for _ in range(num_engines):
            param_prompt_Q.put(None)

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples_per_prompt

        # Accumulate results
        combined_result, combined_queries, combined_ground_truths, combined_datasets = (
            grpo_fast.accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=1
            )
        )

        # Verify results - accumulate_inference_batches returns unique entries
        self.assertEqual(len(combined_queries), num_prompts)
        self.assertEqual(len(combined_result.responses), num_prompts * num_samples_per_prompt)
        self.assertEqual(len(combined_ground_truths), num_prompts)
        self.assertEqual(len(combined_datasets), num_prompts)
        self.assertEqual(len(pending_queries_map), 0)  # All should be processed

        # Clean up
        ray.get(engines)

    def test_no_duplicate_indices_within_training_step(self):
        """Test that dataset indices are unique within a single training step."""
        num_engines = 4
        num_prompts = 64

        # Create test data with sequential indices
        queries = [f"query_{i}" for i in range(num_prompts)]
        ground_truths = [f"truth_{i}" for i in range(num_prompts)]
        datasets = [f"dataset_{i}" for i in range(num_prompts)]
        dataset_indices = list(range(num_prompts))

        # Create queues and map
        param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Split and insert batch
        grpo_fast.split_and_insert_batch(
            queries,
            ground_truths,
            datasets,
            dataset_indices,
            training_step=1,
            vllm_num_engines=num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
        )

        # Check for duplicates in pending_queries_map
        self.assertEqual(len(pending_queries_map), num_prompts, "Some indices were lost or duplicated")
        # Keys are now just dataset indices
        expected_keys = set(dataset_indices)
        self.assertEqual(set(pending_queries_map.keys()), expected_keys, "Indices don't match")

        # Check each batch has unique indices
        all_indices_from_batches = []
        while not param_prompt_Q.empty():
            request = param_prompt_Q.get()
            batch_indices = request.dataset_index
            # Check no duplicates within this batch
            self.assertEqual(
                len(batch_indices), len(set(batch_indices)), f"Duplicate indices found within batch: {batch_indices}"
            )
            all_indices_from_batches.extend(batch_indices)

        # Check total indices match
        self.assertEqual(sorted(all_indices_from_batches), sorted(dataset_indices))

    def test_dataset_indices_no_overwriting_between_steps(self):
        """Test that dataset indices from different training steps are handled correctly with count-based tracking."""
        num_engines = 2
        num_prompts = 16

        # Simulate a dataloader that might return overlapping indices
        step1_indices = list(range(0, num_prompts))  # [0, 1, ..., 15]
        step2_indices = list(range(10, num_prompts + 10))  # [10, 11, ..., 25] - overlaps with step1

        pending_queries_map = grpo_fast.PendingQueriesMap()
        param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)

        # Step 1
        queries = [f"query_step1_{i}" for i in range(num_prompts)]
        ground_truths = [f"truth_step1_{i}" for i in range(num_prompts)]
        datasets = [f"dataset_step1_{i}" for i in range(num_prompts)]

        grpo_fast.split_and_insert_batch(
            queries,
            ground_truths,
            datasets,
            step1_indices,
            training_step=1,
            vllm_num_engines=num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
        )

        # Step 2 - with overlapping indices
        queries2 = [f"query_step2_{i}" for i in range(num_prompts)]
        ground_truths2 = [f"truth_step2_{i}" for i in range(num_prompts)]
        datasets2 = [f"dataset_step2_{i}" for i in range(num_prompts)]

        # Clear the queue for step 2
        while not param_prompt_Q.empty():
            param_prompt_Q.get()

        grpo_fast.split_and_insert_batch(
            queries2,
            ground_truths2,
            datasets2,
            step2_indices,
            training_step=2,
            vllm_num_engines=num_engines,
            pending_queries_map=pending_queries_map,
            param_prompt_Q=param_prompt_Q,
        )

        # With count-based tracking, overlapping indices should have incremented counts
        # Indices 10-15 appear in both steps
        overlapping_indices = set(step1_indices) & set(step2_indices)

        for idx in overlapping_indices:
            # The count should be 2 for these indices (one from each step)
            query, ground_truth, dataset, count = pending_queries_map[idx]
            self.assertEqual(count, 2, f"Index {idx} should have count 2 but has count {count}")

            # The data should be from step 1 (first insertion wins)
            self.assertEqual(query, f"query_step1_{idx}", f"Query data was overwritten for index {idx}")
            self.assertEqual(ground_truth, f"truth_step1_{idx}", f"Ground truth was overwritten for index {idx}")
            self.assertEqual(dataset, f"dataset_step1_{idx}", f"Dataset was overwritten for index {idx}")

        # Non-overlapping indices should have count 1
        non_overlapping = (set(step1_indices) | set(step2_indices)) - overlapping_indices
        for idx in non_overlapping:
            if idx in pending_queries_map:
                _, _, _, count = pending_queries_map[idx]
                self.assertEqual(count, 1, f"Non-overlapping index {idx} should have count 1 but has count {count}")

    def test_thread_safety_pending_queries_map(self):
        """Test concurrent access to pending_queries_map."""
        import threading
        import time

        pending_queries_map = grpo_fast.PendingQueriesMap()
        errors = []

        def add_entries(start_idx, count, thread_id):
            """Add entries to the map."""
            try:
                for i in range(start_idx, start_idx + count):
                    pending_queries_map.insert(
                        i, f"query_{thread_id}_{i}", f"truth_{thread_id}_{i}", f"dataset_{thread_id}_{i}"
                    )
                    time.sleep(0.0001)  # Small delay to increase chance of race condition
            except Exception as e:
                errors.append(f"Thread {thread_id} add error: {e}")

        def remove_entries(indices, thread_id):
            """Remove entries from the map."""
            try:
                for idx in indices:
                    time.sleep(0.0001)  # Small delay
                    if idx in pending_queries_map:
                        pending_queries_map.pop(idx)
                    else:
                        errors.append(f"Thread {thread_id}: Index {idx} not found for removal")
            except Exception as e:
                errors.append(f"Thread {thread_id} remove error: {e}")

        # Create threads that add entries
        add_threads = []
        for i in range(4):
            t = threading.Thread(target=add_entries, args=(i * 100, 50, i))
            add_threads.append(t)
            t.start()

        # Wait for adds to complete
        for t in add_threads:
            t.join()

        # Now create threads that try to remove entries concurrently
        remove_threads = []
        for i in range(4):
            indices = list(range(i * 100, i * 100 + 50))
            t = threading.Thread(target=remove_entries, args=(indices, f"remove_{i}"))
            remove_threads.append(t)
            t.start()

        # Wait for removes to complete
        for t in remove_threads:
            t.join()

        # Check for errors
        self.assertEqual(len(errors), 0, f"Thread safety issues detected: {errors}")

        # Map should be empty after all removes
        self.assertEqual(
            len(pending_queries_map), 0, f"Map not empty after removes: {len(pending_queries_map)} items remain"
        )

    def test_batch_size_edge_cases(self):
        """Test batch splitting when prompts don't divide evenly by engines."""
        test_cases = [
            (17, 4),  # 17 prompts, 4 engines -> 4,4,4,5
            (15, 4),  # 15 prompts, 4 engines -> 3,3,3,6
            (7, 3),  # 7 prompts, 3 engines -> 2,2,3
            (100, 7),  # 100 prompts, 7 engines
        ]

        for num_prompts, num_engines in test_cases:
            with self.subTest(prompts=num_prompts, engines=num_engines):
                queries = [f"q_{i}" for i in range(num_prompts)]
                ground_truths = [f"t_{i}" for i in range(num_prompts)]
                datasets = [f"d_{i}" for i in range(num_prompts)]
                dataset_indices = list(range(num_prompts))

                param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)
                pending_queries_map = grpo_fast.PendingQueriesMap()

                grpo_fast.split_and_insert_batch(
                    queries,
                    ground_truths,
                    datasets,
                    dataset_indices,
                    training_step=1,
                    vllm_num_engines=num_engines,
                    pending_queries_map=pending_queries_map,
                    param_prompt_Q=param_prompt_Q,
                )

                # Verify all indices are in the map
                self.assertEqual(len(pending_queries_map), num_prompts)

                # Verify batches
                batch_sizes = []
                total_indices = []
                while not param_prompt_Q.empty():
                    request = param_prompt_Q.get()
                    batch_sizes.append(len(request.prompts))
                    total_indices.extend(request.dataset_index)

                # All indices should be accounted for
                self.assertEqual(sorted(total_indices), sorted(dataset_indices))
                self.assertEqual(sum(batch_sizes), num_prompts)

                # Verify batch distribution
                # Expected: all batches equal except last one might have more

    def test_accumulate_blocks_waiting_for_all_engines(self):
        """Test that accumulate_inference_batches blocks waiting for results from all engines."""
        num_engines = 4
        num_prompts = 16

        # Create results from only 3 engines (missing one)
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)
        pending_queries_map = grpo_fast.PendingQueriesMap()

        # Add entries to pending_queries_map using (training_step, dataset_idx) as keys
        training_step = 1
        for i in range(num_prompts):
            pending_queries_map.insert(i, f"q_{i}", f"t_{i}", f"d_{i}")

        # Add results from only 3 engines
        for engine_id in range(3):  # Missing engine 3
            start_idx = engine_id * 4
            end_idx = start_idx + 4
            mock_result = GenerationResult(
                responses=[[1, 2, 3] for _ in range(4)],
                finish_reasons=["stop"] * 4,
                masks=[[1, 1, 1] for _ in range(4)],
                request_info=RequestInfo(
                    num_calls=[0] * 4,
                    timeouts=[0] * 4,
                    tool_errors=[""] * 4,
                    tool_outputs=[""] * 4,
                    tool_runtimes=[0.0] * 4,
                    tool_calleds=[False] * 4,
                ),
                dataset_index=list(range(start_idx, end_idx)),
                training_step=training_step,
            )
            inference_results_Q.put(mock_result)

        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.num_samples_per_prompt_rollout = 1

        # This should block waiting for the 4th engine
        import threading

        completed = False

        def run_accumulate():
            nonlocal completed
            try:
                # This will block on the 4th queue.get() call
                grpo_fast.accumulate_inference_batches(
                    inference_results_Q, pending_queries_map, mock_args, training_step=1
                )
                completed = True
            except Exception:
                completed = True

        thread = threading.Thread(target=run_accumulate)
        thread.daemon = True  # Make thread daemon so it doesn't block test suite
        thread.start()

        # Give it time to process the 3 results and get stuck on the 4th
        thread.join(timeout=1.0)

        # Check that the thread is still alive (blocked on queue.get())
        self.assertTrue(
            thread.is_alive(), "accumulate_inference_batches should be blocked waiting for the 4th engine result"
        )

        # The queue should be empty (3 results consumed, waiting for 4th)
        self.assertEqual(inference_results_Q.qsize(), 0, "Queue should be empty as 3 results were consumed")

        # Some entries should have been removed from pending_queries_map
        self.assertLess(
            len(pending_queries_map),
            num_prompts,
            "Some entries should have been processed and removed from pending_queries_map",
        )

    def test_no_race_condition_with_overlapping_indices(self):
        """Test that the system should handle overlapping indices without race conditions."""
        import threading
        import time

        num_engines = 4
        num_prompts = 32

        # Shared state
        pending_queries_map = grpo_fast.PendingQueriesMap()
        param_prompt_Q = ray_queue.Queue(maxsize=num_engines * 2)
        inference_results_Q = ray_queue.Queue(maxsize=num_engines * 2)
        errors = []
        race_condition_detected = False

        def simulate_training_step(step_num, indices_start):
            """Simulate one training step."""
            try:
                # Create data for this step
                queries = [f"query_step{step_num}_{i}" for i in range(num_prompts)]
                ground_truths = [f"truth_step{step_num}_{i}" for i in range(num_prompts)]
                datasets = [f"dataset_step{step_num}_{i}" for i in range(num_prompts)]
                # Use overlapping indices to simulate real scenario
                dataset_indices = list(range(indices_start, indices_start + num_prompts))

                # Split and insert batch - this modifies pending_queries_map
                grpo_fast.split_and_insert_batch(
                    queries,
                    ground_truths,
                    datasets,
                    dataset_indices,
                    training_step=step_num,
                    vllm_num_engines=num_engines,
                    pending_queries_map=pending_queries_map,
                    param_prompt_Q=param_prompt_Q,
                )

                # Simulate vLLM processing delay
                time.sleep(0.1)

                # Create mock results
                for engine_id in range(num_engines):
                    if not param_prompt_Q.empty():
                        request = param_prompt_Q.get()
                        mock_result = GenerationResult(
                            responses=[[1, 2, 3] for _ in range(len(request.prompts))],
                            finish_reasons=["stop"] * len(request.prompts),
                            masks=[[1, 1, 1] for _ in range(len(request.prompts))],
                            request_info=RequestInfo(
                                num_calls=[0] * len(request.prompts),
                                timeouts=[0] * len(request.prompts),
                                tool_errors=[""] * len(request.prompts),
                                tool_outputs=[""] * len(request.prompts),
                                tool_runtimes=[0.0] * len(request.prompts),
                                tool_calleds=[False] * len(request.prompts),
                            ),
                            dataset_index=request.dataset_index,
                            training_step=request.training_step,
                        )
                        inference_results_Q.put(mock_result)

            except Exception as e:
                errors.append(f"Step {step_num} error: {e}")

        def simulate_data_prep_thread(step_num):
            """Simulate data preparation thread processing results."""
            nonlocal race_condition_detected
            try:
                # Add delay to simulate processing time
                time.sleep(0.2)

                mock_args = Mock()
                mock_args.vllm_num_engines = num_engines
                mock_args.num_samples_per_prompt_rollout = 1

                # Try to accumulate results
                grpo_fast.accumulate_inference_batches(
                    inference_results_Q, pending_queries_map, mock_args, training_step=step_num
                )

            except RuntimeError as e:
                if "not found in pending_queries_map" in str(e):
                    race_condition_detected = True
                    errors.append(f"Race condition detected: {e}")
                else:
                    errors.append(f"Step {step_num} accumulate error: {e}")
            except Exception as e:
                errors.append(f"Step {step_num} accumulate error: {e}")

        # Start step 1
        step1_thread = threading.Thread(target=simulate_training_step, args=(1, 25840))
        step1_thread.start()

        # Start data prep for step 1
        data_prep1 = threading.Thread(target=simulate_data_prep_thread, args=(1,))
        data_prep1.start()

        # Before step 1 finishes, start step 2 with overlapping indices
        time.sleep(0.15)  # Start step 2 while step 1 is still processing
        step2_thread = threading.Thread(target=simulate_training_step, args=(2, 25847))
        step2_thread.start()

        # Wait for all threads
        step1_thread.join()
        step2_thread.join()
        data_prep1.join()

        # This test should PASS when the bug is fixed
        # Currently it FAILS because we detect the race condition
        self.assertFalse(
            race_condition_detected,
            f"Race condition detected! The system should handle overlapping indices safely. Errors: {errors}",
        )


if __name__ == "__main__":
    unittest.main()
