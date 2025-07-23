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

        # Verify results - note that queries/ground_truths/datasets are NOT replicated yet
        # They get replicated later in data_preparation_thread
        self.assertEqual(combined_queries, queries_next)
        self.assertEqual(combined_ground_truths, ground_truths_next)
        self.assertEqual(combined_datasets, datasets_next)
        self.assertEqual(len(pending_queries_map), 0)

        # Verify that we have the correct number of responses (256 = 16 prompts * 16 samples per prompt)
        self.assertEqual(len(combined_result.responses), 256)


if __name__ == "__main__":
    unittest.main()
