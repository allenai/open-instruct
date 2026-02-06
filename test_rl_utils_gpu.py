"""GPU tests for rl_utils rollout saving functionality.

These tests require CUDA and Ray, and will be skipped if not available.

To run:
    ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh
"""

import json
import logging
import os
import tempfile
import time
import unittest
from dataclasses import fields

import datasets
import ray
import torch
from ray.util import queue as ray_queue
from transformers import AutoTokenizer

from open_instruct import data_loader, rl_utils
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.ground_truth_utils import RewardConfig
from open_instruct.test_grpo_fast import TestGrpoFastBase
from open_instruct.utils import maybe_update_beaker_description
from open_instruct.vllm_utils import SamplingConfig, create_vllm_engines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

maybe_update_beaker_description()


class TestRlUtilsGPU(TestGrpoFastBase):
    """Integration tests for rl_utils rollout saving with real vLLM engines."""

    def _create_test_dataset(self, tokenizer, prompts: list[str], ground_truths: list[list[str]]):
        data = {
            INPUT_IDS_PROMPT_KEY: [tokenizer.encode(p) for p in prompts],
            GROUND_TRUTHS_KEY: ground_truths,
            VERIFIER_SOURCE_KEY: ["test"] * len(prompts),
            RAW_PROMPT_KEY: prompts,
            "index": list(range(len(prompts))),
        }
        return datasets.Dataset.from_dict(data)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_streaming_dataloader_with_rollout_saving(self):
        """Test that rollout saving works during data preparation."""
        tokenizer_name = "Qwen/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        run_name = "test_rollout_save"

        with tempfile.TemporaryDirectory(prefix="test_rollouts_") as rollout_dir:
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
                save_traces=True,
                rollouts_save_path=rollout_dir,
            )

            generation_config = SamplingConfig(temperature=0.7, top_p=1.0, max_tokens=32, n=2)

            _actor = data_loader.DataPreparationActor.options(name="test_rollout_save").remote(
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
                tool_names=[],
                run_name=run_name,
                model_name=tokenizer_name,
            )

            loader = data_loader.StreamingDataLoader(
                data_prep_actor_name="test_rollout_save",
                tokenizer=tokenizer,
                work_dir="/tmp",
                global_batch_size=2,
                num_training_steps=2,
                dp_world_size=1,
                dp_rank=0,
                fs_local_rank=0,
            )

            for _batch_data in loader:
                pass

            param_prompt_Q.put(None)

            time.sleep(2)

            metadata_path = os.path.join(rollout_dir, f"{run_name}_metadata.jsonl")
            self.assertTrue(os.path.exists(metadata_path))

            metadata_fields = [f.name for f in fields(rl_utils.RolloutMetadata)]
            with open(metadata_path) as f:
                metadata = json.loads(f.readline())
            for field_name in metadata_fields:
                self.assertIn(field_name, metadata)
            self.assertEqual(metadata["run_name"], run_name)

            rollout_path = os.path.join(rollout_dir, f"{run_name}_rollouts_000000.jsonl")
            self.assertTrue(os.path.exists(rollout_path))

            rollout_fields = [f.name for f in fields(rl_utils.RolloutRecord)]
            record_count = 0
            with open(rollout_path) as f:
                for record in (json.loads(line) for line in f):
                    for field_name in rollout_fields:
                        self.assertIn(field_name, record)
                    record_count += 1

            # 4 prompts * 2 samples_per_prompt = 8 total records
            self.assertEqual(record_count, 8)


if __name__ == "__main__":
    unittest.main()
