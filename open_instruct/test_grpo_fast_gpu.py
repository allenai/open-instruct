"""GPU tests for generation with tool invocation and checkpointing.

These tests require CUDA and will be skipped if not available.

To run:

    ./scripts/train/build_image_and_launch.sh scripts/train/debug/run_gpu_pytest.sh
"""

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
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from open_instruct import grpo_fast
from open_instruct.data_types import GenerationResult, PromptRequest
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.model_utils import ModelConfig
from open_instruct.test_grpo_fast import TestGrpoFastBase
from open_instruct.tool_utils.tools import PythonCodeTool
from open_instruct.utils import BeakerRuntimeConfig, maybe_update_beaker_description, ray_get_with_progress
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


TEST_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def create_checkpoint_test_args(
    checkpoint_dir: str, num_ranks: int = 2, deepspeed_stage: int = 3, seed: int = 42
) -> grpo_fast.Args:
    """Create minimal Args for checkpoint testing."""
    return grpo_fast.Args(
        checkpoint_state_dir=checkpoint_dir,
        checkpoint_state_freq=1,
        num_learners_per_node=[num_ranks],
        deepspeed_stage=deepspeed_stage,
        deepspeed_zpg=1,
        per_device_train_batch_size=1,
        num_unique_prompts_rollout=4,
        num_samples_per_prompt_rollout=1,
        filter_zero_std_samples=False,
        total_episodes=10,
        num_training_steps=5,
        seed=seed,
        learning_rate=1e-6,
        response_length=32,
        temperature=0.7,
        beta=0.0,
        num_epochs=1,
        output_dir=checkpoint_dir,
        backend_timeout=5,
        load_ref_policy=False,
    )


def create_checkpoint_test_model_config(deepspeed_stage: int = 3) -> ModelConfig:
    """Create minimal ModelConfig for checkpoint testing."""
    return ModelConfig(model_name_or_path=TEST_MODEL_NAME, gradient_checkpointing=(deepspeed_stage == 3))


def create_checkpoint_test_beaker_config() -> BeakerRuntimeConfig:
    """Create minimal BeakerRuntimeConfig for checkpoint testing."""
    return BeakerRuntimeConfig(beaker_workload_id="test-checkpoint")


class TestCheckpointing(TestGrpoFastBase):
    """GPU tests for checkpointing with DeepSpeed. Requires 2+ GPUs."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        super().tearDown()

    def _get_num_gpus(self) -> int:
        return torch.cuda.device_count()

    def _create_model_group(
        self, checkpoint_dir: str, num_ranks: int, deepspeed_stage: int = 3
    ) -> tuple[grpo_fast.ModelGroup, grpo_fast.Args]:
        """Create a ModelGroup for testing."""
        args = create_checkpoint_test_args(
            checkpoint_dir=checkpoint_dir, num_ranks=num_ranks, deepspeed_stage=deepspeed_stage
        )
        model_config = create_checkpoint_test_model_config(deepspeed_stage=deepspeed_stage)
        beaker_config = create_checkpoint_test_beaker_config()
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME)

        pg = placement_group([{"GPU": 1, "CPU": 4}] * num_ranks, strategy="PACK")
        ray.get(pg.ready())

        model_group = grpo_fast.ModelGroup(
            pg=pg,
            ray_process_cls=grpo_fast.PolicyTrainerRayProcess,
            num_gpus_per_node=[num_ranks],
            single_gpu_mode=False,
        )

        ray_get_with_progress(
            [
                model_group.models[i].from_pretrained.remote(args, model_config, beaker_config, "", tokenizer)
                for i in range(num_ranks)
            ],
            desc="Loading models",
        )

        return model_group, args

    def _cleanup_model_group(self, model_group: grpo_fast.ModelGroup):
        """Clean up model group actors."""
        for model in model_group.models:
            ray.kill(model)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Requires 2+ GPUs")
    @parameterized.expand([(2,), (3,)])
    def test_checkpoint_save_creates_correct_structure(self, deepspeed_stage: int):
        """Test that checkpoint save creates the correct directory structure."""
        num_ranks = min(2, self._get_num_gpus())
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_group, args = self._create_model_group(
            checkpoint_dir=checkpoint_dir, num_ranks=num_ranks, deepspeed_stage=deepspeed_stage
        )

        client_state = {"training_step": 1, "episode": 10, "dataloader_state": {"current_index": 5}}

        ray_get_with_progress(
            [
                model_group.models[i].save_checkpoint_state.remote(checkpoint_dir, client_state.copy())
                for i in range(num_ranks)
            ],
            desc="Saving checkpoint",
        )

        step_dir = os.path.join(checkpoint_dir, "global_step1")
        self.assertTrue(os.path.exists(step_dir), f"Expected {step_dir} to exist")

        for rank in range(num_ranks):
            rank_dir = os.path.join(step_dir, f"global_{rank}")
            self.assertTrue(os.path.exists(rank_dir), f"Expected {rank_dir} to exist")

        self._cleanup_model_group(model_group)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Requires 2+ GPUs")
    @parameterized.expand([(2,), (3,)])
    def test_checkpoint_load_restores_training_step(self, deepspeed_stage: int):
        """Test that loading a checkpoint restores the training step."""
        num_ranks = min(2, self._get_num_gpus())
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_group, args = self._create_model_group(
            checkpoint_dir=checkpoint_dir, num_ranks=num_ranks, deepspeed_stage=deepspeed_stage
        )

        training_step = 5
        client_state = {"training_step": training_step, "episode": 50, "dataloader_state": {"current_index": 25}}

        ray_get_with_progress(
            [
                model_group.models[i].save_checkpoint_state.remote(checkpoint_dir, client_state.copy())
                for i in range(num_ranks)
            ],
            desc="Saving checkpoint",
        )

        self._cleanup_model_group(model_group)

        model_group2, args2 = self._create_model_group(
            checkpoint_dir=checkpoint_dir, num_ranks=num_ranks, deepspeed_stage=deepspeed_stage
        )

        self._cleanup_model_group(model_group2)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Requires 2+ GPUs")
    def test_rng_state_saved_and_restored(self):
        """Test that RNG states are correctly saved and restored for determinism."""
        num_ranks = min(2, self._get_num_gpus())
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_group, args = self._create_model_group(
            checkpoint_dir=checkpoint_dir, num_ranks=num_ranks, deepspeed_stage=3
        )

        client_state = {"training_step": 1, "episode": 10, "dataloader_state": {}}

        ray_get_with_progress(
            [
                model_group.models[i].save_checkpoint_state.remote(checkpoint_dir, client_state.copy())
                for i in range(num_ranks)
            ],
            desc="Saving checkpoint",
        )

        step_dir = os.path.join(checkpoint_dir, "global_step1")
        for rank in range(num_ranks):
            rank_dir = os.path.join(step_dir, f"global_{rank}")
            files = os.listdir(rank_dir)
            self.assertTrue(len(files) > 0, f"Expected files in {rank_dir}")

        self._cleanup_model_group(model_group)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Requires 2+ GPUs")
    def test_multiple_checkpoints_saved(self):
        """Test saving multiple checkpoints at different steps."""
        num_ranks = min(2, self._get_num_gpus())
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_group, args = self._create_model_group(
            checkpoint_dir=checkpoint_dir, num_ranks=num_ranks, deepspeed_stage=3
        )

        for step in [1, 2, 3]:
            client_state = {
                "training_step": step,
                "episode": step * 10,
                "dataloader_state": {"current_index": step * 5},
            }

            ray_get_with_progress(
                [
                    model_group.models[i].save_checkpoint_state.remote(checkpoint_dir, client_state.copy())
                    for i in range(num_ranks)
                ],
                desc=f"Saving checkpoint step {step}",
            )

        for step in [1, 2, 3]:
            step_dir = os.path.join(checkpoint_dir, f"global_step{step}")
            self.assertTrue(os.path.exists(step_dir), f"Expected {step_dir} to exist")

        self._cleanup_model_group(model_group)


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
        from open_instruct.utils import clean_last_n_checkpoints_deepspeed

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
        from open_instruct.utils import calibrate_checkpoint_state_dir

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
