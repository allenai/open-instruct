"""GPU tests for generation with tool invocation.

These tests require CUDA and will be skipped if not available.

To run on Beaker:

    BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
    BEAKER_IMAGE="${BEAKER_USER}/open-instruct-integration-test"

    uv run python mason.py \\
        --cluster ai2/saturn \\
        --image "$BEAKER_IMAGE" \\
        --description "GPU test: test_grpo_fast_gpu.py" \\
        --pure_docker_mode \\
        --workspace ai2/open-instruct-dev \\
        --priority urgent \\
        --num_nodes 1 \\
        --max_retries 0 \\
        --timeout 30m \\
        --budget ai2/oe-adapt \\
        --gpus 1 \\
        -- source configs/beaker_configs/ray_node_setup.sh \\&\\& pytest open_instruct/test_grpo_fast_gpu.py -xvs
"""

import json
import logging
import os
import pathlib
import unittest

import ray
import torch
from parameterized import parameterized
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.queue_types import PromptRequest
from open_instruct.test_grpo_fast import TestGrpoFastBase
from open_instruct.tool_utils.tools import PythonCodeTool
from open_instruct.vllm_utils import create_vllm_engines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_DATA_DIR = pathlib.Path(__file__).parent / "test_data"

DEFAULT_CODE_TOOL_ENDPOINT = "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute"


class TestGeneration(TestGrpoFastBase):
    """Tests for tool invocation with vLLM."""

    def _setup_engine_and_generate(self, tokenizer_name, prompt, tools=None, max_tool_calls=None, max_tokens=50):
        """Helper to create vLLM engine and run generation."""
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        param_prompt_Q = ray_queue.Queue(maxsize=100)
        inference_results_Q = ray_queue.Queue(maxsize=100)
        eval_results_Q = ray_queue.Queue(maxsize=100)
        self._ray_queues.extend([param_prompt_Q, inference_results_Q, eval_results_Q])

        print(f"[TEST] Created queues - prompt_queue id: {id(param_prompt_Q)}", flush=True)
        print(f"[TEST] prompt_queue actor: {param_prompt_Q.actor}", flush=True)

        prompt_token_ids = tokenizer.encode(prompt, return_tensors="pt").tolist()[0]
        stop = list(tools.keys()) if tools else None
        generation_config = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=42, stop=stop)
        request = PromptRequest(
            prompt=prompt_token_ids, dataset_index=0, prompt_id="test_0", generation_config=generation_config
        )

        print("[TEST] Starting create_vllm_engines...", flush=True)
        pg = placement_group([{"GPU": 1, "CPU": 1}], strategy="PACK")
        ray.get(pg.ready())

        engines = create_vllm_engines(
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
            single_gpu_mode=True,
            pg=pg,
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
            eval_results_queue=eval_results_Q,
            tools=tools,
            max_tool_calls=max_tool_calls,
        )

        print("[TEST] Engine created, submitting request directly...", flush=True)
        ray.get(engines[0].submit_request.remote(request))
        print("[TEST] Request submitted, waiting for result...", flush=True)
        result = inference_results_Q.get(timeout=120)
        print("[TEST] Got result!", flush=True)
        param_prompt_Q.put(None)

        return result

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_tool_triggered_on_stop_string(self):
        """Test that tools are properly triggered when model generates stop string."""
        code_endpoint = os.environ.get("CODE_TOOL_API_ENDPOINT", DEFAULT_CODE_TOOL_ENDPOINT)
        tools = {"</code>": PythonCodeTool(api_endpoint=code_endpoint, start_str="<code>", end_str="</code>")}
        prompt = "Write code to print hello world: <code>"

        result = self._setup_engine_and_generate(
            tokenizer_name="Qwen/Qwen3-1.7B", prompt=prompt, tools=tools, max_tool_calls=(5,), max_tokens=256
        )

        self.assertTrue(
            result.request_info.tool_calleds[0],
            "Tool should have been called when model generates text with stop string.",
        )

    @parameterized.expand([True, False])
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_generation_deterministic(self, use_tools):
        """Test generation produces expected output."""
        test_data_filename = f"generation_{'with' if use_tools else 'without'}_tools_expected.json"
        test_data_path = TEST_DATA_DIR / test_data_filename

        tokenizer_name = "Qwen/Qwen3-1.7B"
        code_endpoint = os.environ.get("CODE_TOOL_API_ENDPOINT", DEFAULT_CODE_TOOL_ENDPOINT)
        tools = (
            {"</code>": PythonCodeTool(api_endpoint=code_endpoint, start_str="<code>", end_str="</code>")}
            if use_tools
            else None
        )
        max_tool_calls = (5,) if use_tools else None
        prompt = "Write code to print hello world: <code>" if use_tools else "What is 2 + 2? Answer:"

        result = self._setup_engine_and_generate(
            tokenizer_name=tokenizer_name, prompt=prompt, tools=tools, max_tool_calls=max_tool_calls, max_tokens=256
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


if __name__ == "__main__":
    unittest.main()
