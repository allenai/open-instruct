"""GPU tests for generation with tool invocation.

These tests require CUDA and will be skipped if not available.

To run:

    ./scripts/train/build_image_and_launch.sh scripts/train/debug/run_gpu_pytest.sh
"""

import json
import logging
import os
import pathlib
import subprocess
import time
import unittest

os.environ["VLLM_BATCH_INVARIANT"] = "1"

import datasets
import ray
import torch
from parameterized import parameterized
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from open_instruct.data_types import GenerationResult, PromptRequest
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.test_grpo_fast import TestGrpoFastBase
from open_instruct.tool_utils.tools import PythonCodeTool
from open_instruct.utils import maybe_update_beaker_description
from open_instruct.vllm_utils import SamplingConfig, create_vllm_engines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

maybe_update_beaker_description()

TEST_DATA_DIR = pathlib.Path(__file__).parent / "test_data"


class TestGeneration(TestGrpoFastBase):
    """Tests for tool invocation with vLLM."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.server_process = subprocess.Popen(
            ["uv", "run", "uvicorn", "tool_server:app", "--host", "0.0.0.0", "--port", "1212"],
            cwd="open_instruct/tool_utils",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        time.sleep(3)
        cls.tool_api_endpoint = "http://localhost:1212/execute"

    @classmethod
    def tearDownClass(cls):
        if cls.server_process:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
                cls.server_process.wait()
        super().tearDownClass()

    def _setup_engine_and_generate(self, tokenizer_name, prompt, tools=None, max_tool_calls=None, max_tokens=50):
        """Helper to create vLLM engine and run generation."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        param_prompt_Q = ray_queue.Queue(maxsize=100)
        inference_results_Q = ray_queue.Queue(maxsize=100)
        eval_results_Q = ray_queue.Queue(maxsize=100)
        self._ray_queues.extend([param_prompt_Q, inference_results_Q, eval_results_Q])

        prompt_token_ids = tokenizer.encode(prompt, return_tensors="pt").tolist()[0]
        stop = list(tools.keys()) if tools else None
        generation_config = SamplingConfig(
            temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=42, stop=stop, logprobs=1
        )
        request = PromptRequest(
            prompt=prompt_token_ids, dataset_index=0, prompt_id="test_0", generation_config=generation_config
        )

        pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
        ray.get(pg.ready())

        train_dataset = datasets.Dataset.from_dict({"ground_truth": [["4"]], "dataset": ["test"], "prompt": [prompt]})
        reward_config = RewardConfig()

        engines = create_vllm_engines(
            num_engines=1,
            tensor_parallel_size=1,
            enforce_eager=True,
            tokenizer_name_or_path=tokenizer_name,
            pretrain=tokenizer_name,
            revision="main",
            seed=42,
            enable_prefix_caching=False,
            max_model_len=2048,
            vllm_gpu_memory_utilization=0.5,
            single_gpu_mode=True,
            pg=pg,
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
            eval_results_queue=eval_results_Q,
            tools=tools,
            max_tool_calls=max_tool_calls,
            reward_config=reward_config,
            train_dataset=train_dataset,
        )

        ray.get(engines[0].ready.remote())
        param_prompt_Q.put(request)
        result = inference_results_Q.get(timeout=120)
        param_prompt_Q.put(None)

        return result

    TOOL_PROMPT = "Write 3 separate Python code blocks. Block 1 prints '1'. Block 2 prints '2'. Block 3 prints '3'. Surround each block with <code> and </code> tags. Execute each separately.\n\nBlock 1: <code>"
    NO_TOOL_PROMPT = "What is 2 + 2? Answer:"

    @parameterized.expand([("with_tools", TOOL_PROMPT, True, 1024), ("without_tools", NO_TOOL_PROMPT, False, 256)])
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_generation_deterministic(self, name: str, prompt: str, use_tools: bool, max_tokens: int):
        """Test generation produces expected output and tool invocation behavior."""
        test_data_filename = f"generation_{name}_expected.json"
        test_data_path = TEST_DATA_DIR / test_data_filename

        tokenizer_name = "Qwen/Qwen3-1.7B"
        tools = (
            {"</code>": PythonCodeTool(api_endpoint=self.tool_api_endpoint, start_str="<code>", end_str="</code>")}
            if use_tools
            else None
        )
        max_tool_calls = (5,) if use_tools else None

        result = self._setup_engine_and_generate(
            tokenizer_name=tokenizer_name,
            prompt=prompt,
            tools=tools,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
        )

        if use_tools:
            self.assertTrue(
                result.request_info.tool_calleds[0],
                "Tool should have been called when model generates text with stop string.",
            )

        if not test_data_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            test_data = {
                "model": tokenizer_name,
                "seed": 42,
                "temperature": 0.0,
                "prompt": prompt,
                "use_tools": use_tools,
                "expected_token_ids": result.responses[0],
                "expected_text": tokenizer.decode(result.responses[0]),
            }
            test_data_path.write_text(json.dumps(test_data, indent=2))
            self.fail(f"Test data generated at {test_data_path}. Re-run test to verify.")
            return

        expected = json.loads(test_data_path.read_text())
        self.assertEqual(result.responses[0], expected["expected_token_ids"])


class TestVLLMQueueSystem(TestGrpoFastBase):
    """Tests for the vLLM queue-based system."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_vllm_queue_system_single_prompt(self):
        """Test the new queue-based vLLM system with a single prompt 'What is the capital of France?'"""
        tokenizer_name = "EleutherAI/pythia-14m"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        test_prompt = "What is the capital of France?"
        prompt_token_ids = tokenizer.encode(test_prompt, return_tensors="pt").tolist()[0]

        param_prompt_Q = ray_queue.Queue(maxsize=1)
        inference_results_Q = ray_queue.Queue(maxsize=1)

        self._ray_queues.extend([param_prompt_Q, inference_results_Q])

        train_dataset = datasets.Dataset.from_dict(
            {"ground_truth": [["Paris"]], "dataset": ["test"], "prompt": [test_prompt]}
        )
        reward_config = RewardConfig()

        engines = create_vllm_engines(
            num_engines=1,
            tensor_parallel_size=1,
            enforce_eager=True,
            tokenizer_name_or_path=tokenizer_name,
            pretrain=tokenizer_name,
            revision="main",
            seed=42,
            enable_prefix_caching=False,
            max_model_len=2048,
            vllm_gpu_memory_utilization=0.5,
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
            reward_config=reward_config,
            train_dataset=train_dataset,
        )

        ray.get(engines[0].ready.remote())
        generation_config = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=5, seed=42)
        request = PromptRequest(
            prompt=prompt_token_ids, dataset_index=0, prompt_id="test_0", generation_config=generation_config
        )

        param_prompt_Q.put(request)
        result = inference_results_Q.get()

        self.assertIsInstance(result, GenerationResult)

        self.assertGreater(len(result.responses), 0)
        response_ids = result.responses[0]

        generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), 0)

        param_prompt_Q.put(None)


if __name__ == "__main__":
    unittest.main()
