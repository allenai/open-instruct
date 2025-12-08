"""GPU tests for generation with tool invocation and checkpointing.

These tests require CUDA and will be skipped if not available.

To run:

    ./scripts/train/build_image_and_launch.sh scripts/train/debug/run_gpu_pytest.sh
"""

import gc
import json
import logging
import os
import pathlib
import random
import shutil
import subprocess
import tempfile
import time
import unittest

os.environ["VLLM_BATCH_INVARIANT"] = "1"

import datasets
import numpy as np
import ray
import torch
from parameterized import parameterized
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group, remove_placement_group
from transformers import AutoTokenizer

from open_instruct.data_types import GenerationResult, PromptRequest
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.grpo_fast import Args, ModelGroup, PolicyTrainerRayProcess
from open_instruct.model_utils import ModelConfig
from open_instruct.test_grpo_fast import TestGrpoFastBase
from open_instruct.tool_utils.tools import PythonCodeTool
from open_instruct.utils import (
    BeakerRuntimeConfig,
    calibrate_checkpoint_state_dir,
    clean_last_n_checkpoints_deepspeed,
    maybe_update_beaker_description,
    ray_get_with_progress,
)
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


class TestCheckpointing(TestGrpoFastBase):
    """GPU tests for checkpoint save/load with DeepSpeed."""

    def _create_test_args(
        self, output_dir: str, checkpoint_state_dir: str | None = None, deepspeed_stage: int = 2
    ) -> Args:
        return Args(
            output_dir=output_dir,
            checkpoint_state_dir=checkpoint_state_dir,
            deepspeed_stage=deepspeed_stage,
            deepspeed_offload_param=False,
            deepspeed_offload_optimizer=False,
            deepspeed_zpg=1,
            seed=42,
            backend_timeout=10,
            per_device_train_batch_size=1,
            learning_rate=1e-5,
            num_training_steps=10,
            num_epochs=1,
            num_mini_batches=1,
            warm_up_steps=0,
            warmup_ratio=0.0,
            lr_scheduler_type="constant",
            beta=0.0,
            load_ref_policy=False,
            keep_last_n_checkpoints=2,
            gs_checkpoint_state_dir=None,
            checkpoint_state_freq=1 if checkpoint_state_dir else -1,
            filter_zero_std_samples=False,
        )

    def _cleanup_model_group(self, model_group: ModelGroup, pg):
        for model in model_group.models:
            ray.kill(model)
        remove_placement_group(pg)
        gc.collect()
        time.sleep(1)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Need 2+ GPUs")
    def test_checkpoint_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._create_test_args(temp_dir)
            model_config = ModelConfig(model_name_or_path="EleutherAI/pythia-14m")
            beaker_config = BeakerRuntimeConfig(beaker_workload_id="test")
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

            pg = placement_group([{"GPU": 1, "CPU": 4}, {"GPU": 1, "CPU": 4}], strategy="PACK")
            ray.get(pg.ready())

            model_group = ModelGroup(pg, PolicyTrainerRayProcess, [1, 1], single_gpu_mode=False)

            ray_get_with_progress(
                [
                    m.from_pretrained.remote(args, model_config, beaker_config, None, tokenizer)
                    for m in model_group.models
                ],
                desc="Initializing models",
                timeout=600,
            )

            client_state = {"training_step": 5}
            ray_get_with_progress(
                [m.save_checkpoint_state.remote(temp_dir, client_state) for m in model_group.models],
                desc="Saving checkpoint",
                timeout=120,
            )

            checkpoint_dirs = [d for d in os.listdir(temp_dir) if d.startswith("global_step")]
            self.assertEqual(len(checkpoint_dirs), 1)
            checkpoint_path = os.path.join(temp_dir, checkpoint_dirs[0])

            self._cleanup_model_group(model_group, pg)

            args2 = self._create_test_args(temp_dir, checkpoint_state_dir=checkpoint_path)
            pg2 = placement_group([{"GPU": 1, "CPU": 4}, {"GPU": 1, "CPU": 4}], strategy="PACK")
            ray.get(pg2.ready())
            model_group2 = ModelGroup(pg2, PolicyTrainerRayProcess, [1, 1], single_gpu_mode=False)

            results = ray_get_with_progress(
                [
                    m.from_pretrained.remote(args2, model_config, beaker_config, None, tokenizer)
                    for m in model_group2.models
                ],
                desc="Loading from checkpoint",
                timeout=600,
            )

            self.assertEqual(results[0], 5)

            self._cleanup_model_group(model_group2, pg2)

    @parameterized.expand([(2,), (3,)])
    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Need 2+ GPUs")
    def test_checkpoint_deepspeed_stages(self, stage: int):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._create_test_args(temp_dir, deepspeed_stage=stage)
            model_config = ModelConfig(model_name_or_path="EleutherAI/pythia-14m")
            beaker_config = BeakerRuntimeConfig(beaker_workload_id="test")
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

            pg = placement_group([{"GPU": 1, "CPU": 4}, {"GPU": 1, "CPU": 4}], strategy="PACK")
            ray.get(pg.ready())

            model_group = ModelGroup(pg, PolicyTrainerRayProcess, [1, 1], single_gpu_mode=False)

            ray_get_with_progress(
                [
                    m.from_pretrained.remote(args, model_config, beaker_config, None, tokenizer)
                    for m in model_group.models
                ],
                desc=f"Initializing models (stage {stage})",
                timeout=600,
            )

            client_state = {"training_step": 3}
            ray_get_with_progress(
                [m.save_checkpoint_state.remote(temp_dir, client_state) for m in model_group.models],
                desc=f"Saving checkpoint (stage {stage})",
                timeout=120,
            )

            checkpoint_dirs = [d for d in os.listdir(temp_dir) if d.startswith("global_step")]
            self.assertEqual(len(checkpoint_dirs), 1)
            checkpoint_path = os.path.join(temp_dir, checkpoint_dirs[0])

            self._cleanup_model_group(model_group, pg)

            args2 = self._create_test_args(temp_dir, checkpoint_state_dir=checkpoint_path, deepspeed_stage=stage)
            pg2 = placement_group([{"GPU": 1, "CPU": 4}, {"GPU": 1, "CPU": 4}], strategy="PACK")
            ray.get(pg2.ready())
            model_group2 = ModelGroup(pg2, PolicyTrainerRayProcess, [1, 1], single_gpu_mode=False)

            results = ray_get_with_progress(
                [
                    m.from_pretrained.remote(args2, model_config, beaker_config, None, tokenizer)
                    for m in model_group2.models
                ],
                desc=f"Loading from checkpoint (stage {stage})",
                timeout=600,
            )

            self.assertEqual(results[0], 3)

            self._cleanup_model_group(model_group2, pg2)


class TestCheckpointingCPU(unittest.TestCase):
    """CPU tests for checkpoint utilities (no GPU required)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_rng_state_serialization_roundtrip(self):
        """Test that RNG states can be serialized and deserialized correctly."""
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        rng_states = {
            "torch_cpu_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        torch_before = torch.rand(10).tolist()
        numpy_before = np.random.rand(10).tolist()
        python_before = [random.random() for _ in range(10)]

        torch.set_rng_state(rng_states["torch_cpu_rng_state"])
        np.random.set_state(rng_states["numpy_rng_state"])
        random.setstate(rng_states["python_rng_state"])

        torch_after = torch.rand(10).tolist()
        numpy_after = np.random.rand(10).tolist()
        python_after = [random.random() for _ in range(10)]

        self.assertEqual(torch_before, torch_after)
        self.assertEqual(numpy_before, numpy_after)
        self.assertEqual(python_before, python_after)

    def test_clean_last_n_checkpoints_deepspeed(self):
        """Test checkpoint cleanup utility."""
        checkpoint_dir = self.temp_dir
        for step in [1, 2, 3, 4, 5]:
            step_dir = os.path.join(checkpoint_dir, f"global_step{step}")
            os.makedirs(step_dir)
            with open(os.path.join(step_dir, "dummy.pt"), "w") as f:
                f.write("dummy")

        clean_last_n_checkpoints_deepspeed(checkpoint_dir, keep_last_n_checkpoints=2)

        remaining = [d for d in os.listdir(checkpoint_dir) if d.startswith("global_step")]
        self.assertEqual(len(remaining), 2)
        self.assertIn("global_step5", remaining)
        self.assertIn("global_step4", remaining)

    def test_calibrate_checkpoint_state_dir_removes_incomplete(self):
        """Test that calibrate_checkpoint_state_dir removes incomplete checkpoints."""
        checkpoint_dir = self.temp_dir

        complete_dir = os.path.join(checkpoint_dir, "global_step1")
        os.makedirs(complete_dir)
        for i in range(5):
            with open(os.path.join(complete_dir, f"file{i}.pt"), "w") as f:
                f.write("data")

        incomplete_dir = os.path.join(checkpoint_dir, "global_step2")
        os.makedirs(incomplete_dir)
        with open(os.path.join(incomplete_dir, "file0.pt"), "w") as f:
            f.write("data")

        calibrate_checkpoint_state_dir(checkpoint_dir)

        self.assertTrue(os.path.exists(complete_dir))
        self.assertFalse(os.path.exists(incomplete_dir))


if __name__ == "__main__":
    unittest.main()
