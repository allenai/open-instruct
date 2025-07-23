import unittest
from unittest.mock import Mock

import ray
import torch
from parameterized import parameterized
from ray.util import queue as ray_queue
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.grpo_fast import accumulate_inference_batches, split_and_insert_batch
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

        pending_queries_map = {}
        training_step = 1

        # Create mock Ray queue for testing
        param_prompt_Q = ray_queue.Queue(maxsize=vllm_num_engines)

        # Create mock dataset indices
        dataset_indices = list(range(num_unique_prompts_rollout))

        # Use split_and_insert_batch to split and insert data
        split_and_insert_batch(
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
        combined_result, combined_queries, combined_ground_truths, combined_datasets = accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step
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
        pending_queries_map = {}

        # Split and insert batch
        split_and_insert_batch(
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
        combined_result, combined_queries, combined_ground_truths, combined_datasets = accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=1
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

        pending_queries_map = {}

        for step in range(1, num_steps + 1):
            # Create unique data for each step
            queries_next = [f"query_step{step}_{i}" for i in range(num_unique_prompts_rollout)]
            ground_truths_next = [f"truth_step{step}_{i}" for i in range(num_unique_prompts_rollout)]
            datasets_next = [f"dataset_step{step}_{i}" for i in range(num_unique_prompts_rollout)]
            # Use different indices for each step to ensure no mixing
            dataset_indices = list(range((step-1) * num_unique_prompts_rollout, step * num_unique_prompts_rollout))

            # Create queues
            param_prompt_Q = ray_queue.Queue(maxsize=vllm_num_engines)
            inference_results_Q = ray_queue.Queue(maxsize=vllm_num_engines)

            # Split and insert
            split_and_insert_batch(
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
            combined_result, combined_queries, combined_ground_truths, combined_datasets = accumulate_inference_batches(
                inference_results_Q, pending_queries_map, mock_args, training_step=step
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
        pending_queries_map = {}

        # Split and insert
        split_and_insert_batch(
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
        combined_result, combined_queries, combined_ground_truths, combined_datasets = accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=1
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
        pending_queries_map = {}

        # Split and insert batch
        split_and_insert_batch(
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
        self.mock_vllm_pipeline(param_prompt_Q, inference_results_Q, vllm_num_engines,
                                num_samples_per_prompt=num_samples_per_prompt)

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = vllm_num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples_per_prompt

        # Now with the fix, this should succeed
        combined_result, combined_queries, combined_ground_truths, combined_datasets = accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=1
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
        pending_queries_map = {
            0: ("query_0", "truth_0", "dataset_0"),
            1: ("query_1", "truth_1", "dataset_1"),
            2: ("query_2", "truth_2", "dataset_2"),
            3: ("query_3", "truth_3", "dataset_3"),
        }

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
        )

        # Create queue and add result
        inference_results_Q = ray_queue.Queue(maxsize=1)
        inference_results_Q.put(mock_result)

        # Call accumulate_inference_batches
        combined_result, combined_queries, combined_ground_truths, combined_datasets = accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=1
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
        pending_queries_map = {}

        # Split and insert batch
        split_and_insert_batch(
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
            )
            inference_results_Q.put(mock_result)

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples_per_prompt

        # Accumulate results
        combined_result, combined_queries, combined_ground_truths, combined_datasets = accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=1
        )

        # Verify results - accumulate_inference_batches returns unique entries
        self.assertEqual(len(combined_queries), num_prompts)  # Not replicated
        self.assertEqual(len(combined_result.responses), num_prompts * num_samples_per_prompt)  # Responses are replicated
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
        pending_queries_map = {}

        # Split and insert batch
        split_and_insert_batch(
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
            )
            inference_results_Q.put(mock_result)

        # Create mock args
        mock_args = Mock()
        mock_args.vllm_num_engines = num_engines
        mock_args.num_samples_per_prompt_rollout = num_samples_per_prompt

        # Accumulate results
        combined_result, combined_queries, combined_ground_truths, combined_datasets = accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=1
        )

        # Verify results - accumulate_inference_batches returns unique entries
        self.assertEqual(len(combined_queries), num_prompts)  # Not replicated
        self.assertEqual(len(combined_result.responses), num_prompts * num_samples_per_prompt)  # Responses are replicated
        self.assertEqual(len(pending_queries_map), 0)

    @parameterized.expand([
        (1, 16, 16),   # 1 engine, 16 prompts, 16 samples
        (2, 32, 16),   # 2 engines, 32 prompts, 16 samples
        (4, 64, 16),   # 4 engines, 64 prompts, 16 samples
        (8, 128, 16),  # 8 engines, 128 prompts, 16 samples
        (4, 100, 16),  # 4 engines, 100 prompts (not evenly divisible), 16 samples
    ])
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
        pending_queries_map = {}

        # Split and insert batch
        split_and_insert_batch(
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
        combined_result, combined_queries, combined_ground_truths, combined_datasets = accumulate_inference_batches(
            inference_results_Q, pending_queries_map, mock_args, training_step=1
        )

        # Verify results - accumulate_inference_batches returns unique entries
        self.assertEqual(len(combined_queries), num_prompts)
        self.assertEqual(len(combined_result.responses), num_prompts * num_samples_per_prompt)
        self.assertEqual(len(combined_ground_truths), num_prompts)
        self.assertEqual(len(combined_datasets), num_prompts)
        self.assertEqual(len(pending_queries_map), 0)  # All should be processed

        # Clean up
        ray.get(engines)


if __name__ == "__main__":
    unittest.main()
