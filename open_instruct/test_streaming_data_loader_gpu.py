"""GPU tests for StreamingDataLoader.

These tests require CUDA and Ray, and will be skipped if not available.

To run:
    ./scripts/train/build_image_and_launch.sh scripts/train/debug/run_gpu_pytest.sh
"""

import logging
import pathlib
import subprocess
import time
import unittest

import datasets
import ray
import torch
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from open_instruct import data_loader, data_types
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.test_grpo_fast import TestGrpoFastBase
from open_instruct.tool_utils.tools import PythonCodeTool
from open_instruct.utils import maybe_update_beaker_description
from open_instruct.vllm_utils import SamplingConfig, create_vllm_engines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

maybe_update_beaker_description()

TEST_DATA_DIR = pathlib.Path(__file__).parent / "test_data"


class TestStreamingDataLoaderGPU(TestGrpoFastBase):
    """Integration tests for StreamingDataLoader with real vLLM engines."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.server_process = subprocess.Popen(
            ["uv", "run", "uvicorn", "tool_server:app", "--host", "0.0.0.0", "--port", "1213"],
            cwd="open_instruct/tool_utils",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        time.sleep(3)
        cls.tool_api_endpoint = "http://localhost:1213/execute"

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

    def _create_test_dataset(self, tokenizer, prompts: list[str], ground_truths: list[list[str]]):
        data = {
            INPUT_IDS_PROMPT_KEY: [tokenizer.encode(p) for p in prompts],
            GROUND_TRUTHS_KEY: ground_truths,
            VERIFIER_SOURCE_KEY: ["test"] * len(prompts),
            RAW_PROMPT_KEY: prompts,
        }
        return datasets.Dataset.from_dict(data)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_streaming_dataloader_iteration_without_tools(self):
        tokenizer_name = "EleutherAI/pythia-14m"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?", "What is 5+5?"]
        ground_truths = [["4"], ["6"], ["8"], ["10"]]
        train_dataset = self._create_test_dataset(tokenizer, prompts, ground_truths)

        param_prompt_Q = ray_queue.Queue(maxsize=100)
        inference_results_Q = ray_queue.Queue(maxsize=100)
        eval_results_Q = ray_queue.Queue(maxsize=100)
        self._ray_queues.extend([param_prompt_Q, inference_results_Q, eval_results_Q])

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
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
            eval_results_queue=eval_results_Q,
            reward_config=RewardConfig(),
            train_dataset=train_dataset,
        )
        ray.get(engines[0].ready.remote())

        config = data_loader.StreamingDataLoaderConfig(
            max_prompt_token_length=64,
            response_length=32,
            async_steps=1,
            num_samples_per_prompt_rollout=2,
            filter_zero_std_samples=False,
            pack_length=128,
        )

        generation_config = SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=32, n=2)

        _actor = data_loader.DataPreparationActor.options(name="test_no_tools").remote(
            dataset=train_dataset,
            inference_results_Q=inference_results_Q,
            param_prompt_Q=param_prompt_Q,
            tokenizer=tokenizer,
            config=config,
            generation_config=generation_config,
            num_training_steps=2,
            seed=42,
            per_device_train_batch_size=2,
            global_batch_size=2,
            dp_world_size=1,
            max_possible_score=1.0,
            actor_manager=None,
            model_dims=self.create_llama7b_model_dims(),
            verbose=True,
            work_dir="/tmp",
        )

        loader = data_loader.StreamingDataLoader(
            data_prep_actor_name="test_no_tools",
            tokenizer=tokenizer,
            work_dir="/tmp",
            global_batch_size=2,
            num_training_steps=2,
            dp_world_size=1,
            dp_rank=0,
            fs_local_rank=0,
        )

        batches_received = 0
        for batch_data in loader:
            self.assertIn("batch", batch_data)
            self.assertIn("metrics", batch_data)
            batch = batch_data["batch"]
            self.assertIsInstance(batch, data_types.CollatedBatchData)
            batches_received += 1

        self.assertEqual(batches_received, 2)

        param_prompt_Q.put(None)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_streaming_dataloader_iteration_with_tools(self):
        tokenizer_name = "Qwen/Qwen3-1.7B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        prompts = [
            "Write Python code to print '1'. Surround it with <code> and </code> tags.\n\n<code>",
            "Write Python code to print '2'. Surround it with <code> and </code> tags.\n\n<code>",
            "Write Python code to print '3'. Surround it with <code> and </code> tags.\n\n<code>",
            "Write Python code to print '4'. Surround it with <code> and </code> tags.\n\n<code>",
        ]
        ground_truths = [["1"], ["2"], ["3"], ["4"]]
        train_dataset = self._create_test_dataset(tokenizer, prompts, ground_truths)

        param_prompt_Q = ray_queue.Queue(maxsize=100)
        inference_results_Q = ray_queue.Queue(maxsize=100)
        eval_results_Q = ray_queue.Queue(maxsize=100)
        self._ray_queues.extend([param_prompt_Q, inference_results_Q, eval_results_Q])

        tools = {"</code>": PythonCodeTool(api_endpoint=self.tool_api_endpoint, start_str="<code>", end_str="</code>")}

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
            max_model_len=1024,
            vllm_gpu_memory_utilization=0.5,
            single_gpu_mode=True,
            pg=pg,
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
            eval_results_queue=eval_results_Q,
            tools=tools,
            max_tool_calls=(3,),
            reward_config=RewardConfig(),
            train_dataset=train_dataset,
        )
        ray.get(engines[0].ready.remote())

        config = data_loader.StreamingDataLoaderConfig(
            max_prompt_token_length=128,
            response_length=128,
            async_steps=1,
            num_samples_per_prompt_rollout=2,
            filter_zero_std_samples=False,
            pack_length=512,
            mask_tool_use=True,
        )

        generation_config = SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=128, n=2, stop=list(tools.keys()))

        _actor = data_loader.DataPreparationActor.options(name="test_with_tools").remote(
            dataset=train_dataset,
            inference_results_Q=inference_results_Q,
            param_prompt_Q=param_prompt_Q,
            tokenizer=tokenizer,
            config=config,
            generation_config=generation_config,
            num_training_steps=2,
            seed=42,
            per_device_train_batch_size=2,
            global_batch_size=2,
            dp_world_size=1,
            max_possible_score=1.0,
            actor_manager=None,
            model_dims=self.create_llama7b_model_dims(),
            verbose=True,
            work_dir="/tmp",
        )

        loader = data_loader.StreamingDataLoader(
            data_prep_actor_name="test_with_tools",
            tokenizer=tokenizer,
            work_dir="/tmp",
            global_batch_size=2,
            num_training_steps=2,
            dp_world_size=1,
            dp_rank=0,
            fs_local_rank=0,
        )

        batches_received = 0
        for batch_data in loader:
            self.assertIn("batch", batch_data)
            self.assertIn("metrics", batch_data)
            batch = batch_data["batch"]
            self.assertIsInstance(batch, data_types.CollatedBatchData)
            batches_received += 1

        self.assertEqual(batches_received, 2)

        param_prompt_Q.put(None)


if __name__ == "__main__":
    unittest.main()
