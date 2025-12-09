import gc
import time
import unittest
from typing import Any
from unittest.mock import Mock

import ray
import torch
from datasets import Dataset
from parameterized import parameterized
from transformers import AutoTokenizer

from open_instruct import data_loader, rl_utils, utils
from open_instruct.data_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)


class TestGrpoFastBase(unittest.TestCase):
    """Base class with common test utilities."""

    def _get_resource_tracker_state(self):
        """Get current resource tracker state for debugging."""
        tracked_resources = {}
        try:
            from multiprocessing.resource_tracker import _resource_tracker

            if hasattr(_resource_tracker, "_cache"):
                for name, rtype in list(_resource_tracker._cache.items()):
                    if rtype not in tracked_resources:
                        tracked_resources[rtype] = []
                    tracked_resources[rtype].append(name)
        except Exception:
            try:
                import multiprocessing.resource_tracker as rt

                if hasattr(rt, "getfd"):
                    for attr_name in dir(rt):
                        attr = getattr(rt, attr_name)
                        if isinstance(attr, dict) and any("semaphore" in str(v) for v in attr.values()):
                            for k, v in attr.items():
                                if v not in tracked_resources:
                                    tracked_resources[v] = []
                                tracked_resources[v].append(k)
            except Exception:
                pass
        return tracked_resources

    def setUp(self):
        """Initialize Ray and check for pre-existing leaks."""
        self._initial_resources = self._get_resource_tracker_state()
        self._ray_queues = []
        utils.check_runtime_leaks()
        ray.init(include_dashboard=False)

    def _cleanup_ray_queues(self):
        """Clean up all Ray queues created during the test."""
        for queue in self._ray_queues:
            try:
                queue.shutdown()
            except Exception as e:
                print(f"Warning: Failed to shutdown Ray queue: {e}")
        self._ray_queues.clear()

    def tearDown(self):
        """Check for leaks and shutdown Ray."""
        self._cleanup_ray_queues()
        if ray.is_initialized():
            ray.shutdown()
        gc.collect()
        final_resources = self._get_resource_tracker_state()
        new_resources = {}
        for rtype, names in final_resources.items():
            initial_names = set(self._initial_resources.get(rtype, []))
            new_names = [n for n in names if n not in initial_names]
            if new_names:
                new_resources[rtype] = new_names
        utils.check_runtime_leaks()
        if new_resources:
            leak_msg = f"Resource leaks detected after test {self._testMethodName}:\n"
            for rtype, names in new_resources.items():
                leak_msg += f"  {rtype}: {names}\n"
            if "semaphore" in new_resources:
                self.fail(leak_msg)

    def create_test_data(self, num_prompts, prefix="", start_idx=0):
        """Create test data with consistent naming."""
        indices = list(range(start_idx, start_idx + num_prompts))
        queries = [f"{prefix}query_{i}" for i in indices]
        ground_truths = [f"{prefix}truth_{i}" for i in indices]
        datasets = [f"{prefix}dataset_{i}" for i in indices]
        raw_queries = [f"{prefix}rawquery_{i}" for i in indices]
        return queries, ground_truths, datasets, raw_queries, indices

    def create_mock_model_dims(self):
        """Create mock ModelDims object for tests."""
        mock_dims = Mock(spec=utils.ModelDims)
        mock_dims.num_layers = 32
        mock_dims.hidden_size = 4096
        mock_dims.intermediate_size = 11008
        mock_dims.vocab_size = 32000
        mock_dims.num_attn_heads = 32
        mock_dims.num_kv_heads = 32
        mock_dims.device_name = "h100"
        mock_dims.device_flops = 989.5e12
        mock_dims.device_memory_bandwidth = 3.35e12
        mock_dims.flops = Mock(return_value=1e12)
        mock_dims.memory_bytes = Mock(return_value=1e9)
        return mock_dims

    def create_mock_packed_sequences(self, batch_size: int, seq_length: int, variable_length: bool = False):
        """Create mock PackedSequences for testing."""
        lengths = [seq_length - (i % 3) if variable_length else seq_length for i in range(batch_size)]
        return rl_utils.PackedSequences(
            query_responses=[torch.full((length,), i, dtype=torch.long) for i, length in enumerate(lengths)],
            attention_masks=[torch.ones(length, dtype=torch.long) for length in lengths],
            response_masks=[torch.ones(length, dtype=torch.long) for length in lengths],
            original_responses=[[i] * seq_length for i in range(batch_size)],
            advantages=[torch.randn(length) for length in lengths],
            position_ids=[torch.arange(length, dtype=torch.long) for length in lengths],
            vllm_logprobs=[torch.randn(length) for length in lengths],
        )

    def create_mock_result_from_request(self, request: PromptRequest, num_samples_per_prompt=1):
        """Create a mock GenerationResult from a PromptRequest."""
        return self.create_mock_result(request.dataset_index, request.prompt_id, num_samples_per_prompt)

    def create_mock_result(self, dataset_index: int, prompt_id: str, num_samples_per_prompt=1, reward_scores=None):
        """Create a mock GenerationResult."""
        total_responses = num_samples_per_prompt
        if reward_scores is None:
            reward_scores = [i / max(total_responses, 1) for i in range(total_responses)]

        return GenerationResult(
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
            dataset_index=dataset_index,
            prompt_id=prompt_id,
            start_time=time.perf_counter(),
            token_statistics=TokenStatistics(
                num_prompt_tokens=10, num_response_tokens=3 * total_responses, generation_time=0.1
            ),
            logprobs=[[0.0, 0.0, 0.0] for _ in range(total_responses)],
            reward_scores=reward_scores,
            reward_metrics={"time/reward": 0.0},
        )

    def create_mock_tokenizer_and_reward_fn(self):
        tokenizer_name = "EleutherAI/pythia-14m"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        async def reward_fn(
            responses: list[torch.Tensor],
            decoded_responses: list[str],
            ground_truths: list[Any],
            datasets: list[str],
            finish_reasons: list[str],
            infos: list[list[int]],
            queries: list[str] | None = None,
        ) -> (list[float], dict[str, Any]):
            num_responses = len(responses)
            return [i / num_responses for i in range(num_responses)], {"time/reward": 0.0}

        return tokenizer, reward_fn

    def create_mock_dataset(self, queries, ground_truths, datasets, raw_queries):
        """Create a mock dataset from test data."""
        data = {
            INPUT_IDS_PROMPT_KEY: queries,
            GROUND_TRUTHS_KEY: ground_truths,
            VERIFIER_SOURCE_KEY: datasets,
            RAW_PROMPT_KEY: raw_queries,
        }
        return Dataset.from_dict(data)


class TestDataPreparation(TestGrpoFastBase):
    """Test prepare_collated_data_for_workers function."""

    @parameterized.expand(
        [
            (16, 4, 2, 10, False, 0),
            (32, 8, 4, 20, False, 0),
            (8, 2, 1, 5, False, 0),
            (16, 4, 2, 10, False, 0),
            (24, 8, 3, 15, False, 0),
            (4, 1, 4, 10, False, 0),
            (8, 2, 2, 10, True, 999),
        ]
    )
    def test_distribution_and_structure(
        self, batch_size, world_size, per_device_train_batch_size, seq_length, variable_length, pad_token_id
    ):
        """Test data distribution, structure, micro-batch collation, and padding."""
        packed_sequences = self.create_mock_packed_sequences(batch_size, seq_length, variable_length)
        result = data_loader.prepare_collated_data_for_workers(
            packed_sequences, world_size, per_device_train_batch_size, pad_token_id, pin_memory=False
        )

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), world_size)

        expected_fields = {
            "query_responses",
            "attention_masks",
            "position_ids",
            "advantages",
            "response_masks",
            "vllm_logprobs",
        }

        expected_samples_per_worker = batch_size // world_size
        samples_per_worker = batch_size // world_size
        expected_num_microbatches = (
            samples_per_worker + per_device_train_batch_size - 1
        ) // per_device_train_batch_size

        for worker_data in result:
            self.assertEqual(set(f.name for f in worker_data.__dataclass_fields__.values()), expected_fields)

            total_samples = sum(len(batch) for batch in worker_data.query_responses)
            self.assertEqual(total_samples, expected_samples_per_worker)

            num_microbatches = len(worker_data.query_responses)
            self.assertEqual(num_microbatches, expected_num_microbatches)

            for field_name in expected_fields:
                value = getattr(worker_data, field_name)
                self.assertIsInstance(value, list)
                self.assertEqual(len(value), expected_num_microbatches)
                for i, tensor in enumerate(value):
                    self.assertIsInstance(tensor, torch.Tensor)
                    if i < expected_num_microbatches - 1:
                        self.assertEqual(len(tensor), per_device_train_batch_size)
                    else:
                        self.assertLessEqual(len(tensor), per_device_train_batch_size)

            if not variable_length:
                continue

            for batch in worker_data.query_responses:
                for row in batch:
                    padding_mask = row == pad_token_id
                    if not padding_mask.any():
                        continue
                    first_pad_idx = padding_mask.nonzero(as_tuple=True)[0][0].item()
                    self.assertTrue(torch.all(row[first_pad_idx:] == pad_token_id))


if __name__ == "__main__":
    unittest.main()
