import gc
import unittest

import ray
import torch
from ray.util import queue as ray_queue
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct import utils, vllm_utils3
from open_instruct.queue_types import GenerationResult, PromptRequest
from open_instruct.vllm_utils3 import create_vllm_engines


class TestGrpoFastGPUBase(unittest.TestCase):
    """Base class with common test utilities for GPU tests."""

    def setUp(self):
        """Initialize Ray and check for pre-existing leaks."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, skipping test")

        ray.init(include_dashboard=False)

    def tearDown(self):
        ray.shutdown()

        gc.collect()

        utils.check_runtime_leaks()


class TestGrpoFastVLLMGPU(TestGrpoFastGPUBase):
    def test_vllm_queue_system_single_prompt(self):
        """Test the new queue-based vLLM system with a single prompt."""
        tokenizer_name = "EleutherAI/pythia-14m"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        test_prompt = "What is the capital of France?"
        prompt_token_ids = tokenizer.encode(test_prompt, return_tensors="pt").tolist()[0]
        param_prompt_Q = ray_queue.Queue(maxsize=1)
        inference_results_Q = ray_queue.Queue(maxsize=1)
        actor_manager = vllm_utils3.ActorManager.remote()
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
            vllm_gpu_memory_utilization=0.5,
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
            actor_manager=actor_manager,
        )

        generation_config = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=5, n=1)

        param_prompt_Q.put(
            PromptRequest(
                prompts=[prompt_token_ids], generation_config=generation_config, dataset_index=[0], training_step=0
            )
        )

        ray.get(vllm_engines[0].process_from_queue.remote(timeout=30))
        result = inference_results_Q.get_nowait()

        self.assertIsInstance(result, GenerationResult)
        self.assertIsNotNone(result.responses)
        self.assertEqual(len(result.responses), 1)
        self.assertEqual(result.dataset_index, [0])

        response_ids = result.responses[0]

        generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), 0)

        for queue in [param_prompt_Q, inference_results_Q]:
            queue.shutdown()
