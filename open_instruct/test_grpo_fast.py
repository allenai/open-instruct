import unittest

import ray
import torch
from parameterized import parameterized
from ray.util import queue as ray_queue
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.vllm_utils3 import GenerationResult, PromptRequest, create_vllm_engines


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
        """Test the batch splitting logic used in sync_weights_and_prepare_prompts."""

        queries_next = [f"query_{i}" for i in range(num_unique_prompts_rollout)]
        ground_truths_next = [f"truth_{i}" for i in range(num_unique_prompts_rollout)]
        datasets_next = [f"dataset_{i}" for i in range(num_unique_prompts_rollout)]

        pending_queries_map = {}
        training_step = 1

        # Split the batch into multiple inference batches for vLLM engines
        inference_batch_size = len(queries_next) // vllm_num_engines

        all_queries = []
        all_ground_truths = []
        all_datasets = []

        for batch_idx in range(vllm_num_engines):
            start_idx = batch_idx * inference_batch_size
            end_idx = min(start_idx + inference_batch_size, len(queries_next))

            queries = queries_next[start_idx:end_idx]
            ground_truths = ground_truths_next[start_idx:end_idx]
            datasets = datasets_next[start_idx:end_idx]

            # Verify batch sizes
            if batch_idx < vllm_num_engines - 1:
                expected_size = inference_batch_size
            else:
                expected_size = len(queries_next) % inference_batch_size
            self.assertEqual(len(queries), expected_size)
            self.assertEqual(len(ground_truths), expected_size)
            self.assertEqual(len(datasets), expected_size)

            # Create unique dataset_index for this batch
            batch_dataset_index = f"{training_step}_{batch_idx}"
            pending_queries_map[batch_dataset_index] = (queries, ground_truths, datasets)

            # Collect all batches for verification
            all_queries.extend(queries)
            all_ground_truths.extend(ground_truths)
            all_datasets.extend(datasets)

        # Verify that all original data is preserved
        self.assertEqual(all_queries, queries_next)
        self.assertEqual(all_ground_truths, ground_truths_next)
        self.assertEqual(all_datasets, datasets_next)

        # Verify that we have the expected number of batches
        self.assertEqual(len(pending_queries_map), vllm_num_engines)

        # Verify that each batch has the correct dataset_index format
        for batch_idx in range(vllm_num_engines):
            batch_dataset_index = f"{training_step}_{batch_idx}"
            self.assertIn(batch_dataset_index, pending_queries_map)


if __name__ == "__main__":
    unittest.main()
