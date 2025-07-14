import unittest
import torch
import ray
import threading
import queue
from vllm import SamplingParams
from transformers import AutoTokenizer

from open_instruct.vllm_utils3 import create_vllm_engines
from open_instruct.grpo_fast import vllm_generate_thread


class TestGrpoFastVLLM(unittest.TestCase):
    def setUp(self):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, skipping test")

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def tearDown(self):
        # Shutdown Ray after test
        if ray.is_initialized():
            ray.shutdown()

    def test_vllm_generate_thread_single_prompt(self):
        """Test vllm_generate_thread with a single prompt 'What is the capital of France?'"""

        # Set up tokenizer
        tokenizer_name = "EleutherAI/pythia-14m"  # Using a small model for testing
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Tokenize the test prompt
        test_prompt = "What is the capital of France?"
        prompt_token_ids = tokenizer.encode(test_prompt, return_tensors="pt").tolist()[0]

        # Create vLLM engines
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
        )

        # Set up generation config
        generation_config = SamplingParams(
            temperature=0.0,  # Deterministic generation
            top_p=1.0,
            max_tokens=50,
            seed=42,
        )

        # Set up queues
        inference_results_Q = queue.Queue()
        param_prompt_Q = queue.Queue()
        evaluation_inference_results_Q = queue.Queue()

        # Put the test prompt in the queue
        param_prompt_Q.put((1, [[prompt_token_ids]], None))
        param_prompt_Q.put(None)  # Ensure thread exits after one prompt

        # Create and start the generation thread
        generate_thread = threading.Thread(
            target=vllm_generate_thread,
            args=(
                vllm_engines,
                generation_config,
                generation_config,  # Using same config for eval
                inference_results_Q,
                param_prompt_Q,
                1,  # num_training_steps
                None,  # eval_prompt_token_ids
                evaluation_inference_results_Q,
                1,  # eval_freq
                1,  # resume_training_step
                False,  # tool_use
            ),
        )
        generate_thread.start()

        try:
            result = inference_results_Q.get(timeout=30)
        except queue.Empty:
            self.fail("Timed out waiting for inference result")

        # The result should be a tuple with the structure we expect
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        response_ids, _, _, _ = result
        print(f"{response_ids=}")

        # Decode the response
        generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Print for debugging
        print(f"Generated text: {generated_text}")

        # The model should generate some response (we can't guarantee exact output with opt-125m)
        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), 0)

        generate_thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
