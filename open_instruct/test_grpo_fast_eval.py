import math
import unittest
from queue import Empty
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from datasets import Dataset

from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_utils
from open_instruct.data_types import EnvConfig
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.grpo_fast import create_generation_configs, maybe_evaluate


class _QueueWithSize:
    def __init__(self, size: int):
        self._size = size

    def qsize(self) -> int:
        return self._size


class TestCreateGenerationConfigs(unittest.TestCase):
    def test_eval_response_length_defaults_to_response_length(self):
        streaming_config = data_loader_lib.StreamingDataLoaderConfig(
            max_prompt_token_length=128, response_length=128, pack_length=512
        )
        self.assertEqual(streaming_config.eval_response_length, streaming_config.response_length)

    def test_eval_uses_pass_at_k_and_eval_response_length(self):
        args = grpo_utils.GRPOExperimentConfig(eval_pass_at_k=8)
        streaming_config = data_loader_lib.StreamingDataLoaderConfig(response_length=256, eval_response_length=512)
        vllm_config = data_loader_lib.VLLMConfig()

        configs = create_generation_configs(args, streaming_config, vllm_config)

        self.assertEqual(configs["train"].n, streaming_config.num_samples_per_prompt_rollout)
        self.assertEqual(configs["train"].max_tokens, 256)
        self.assertEqual(configs["eval"].n, 8)
        self.assertEqual(configs["eval"].max_tokens, 512)

    def test_vllm_max_model_len_uses_longest_response_length(self):
        streaming_config = data_loader_lib.StreamingDataLoaderConfig(
            max_prompt_token_length=1024, response_length=256, eval_response_length=512, pack_length=1536
        )
        max_model_len = streaming_config.max_prompt_token_length + max(
            streaming_config.response_length, streaming_config.eval_response_length
        )
        self.assertEqual(max_model_len, 1536)


class TestMaybeEvaluate(unittest.TestCase):
    def _build_eval_dataset(self, num_prompts: int) -> Dataset:
        return Dataset.from_dict(
            {
                INPUT_IDS_PROMPT_KEY: [[1, 2, 3] for _ in range(num_prompts)],
                GROUND_TRUTHS_KEY: ["42" for _ in range(num_prompts)],
                VERIFIER_SOURCE_KEY: ["unit_test" for _ in range(num_prompts)],
                RAW_PROMPT_KEY: ["prompt" for _ in range(num_prompts)],
                "index": list(range(num_prompts)),
            }
        )

    def test_non_final_step_defers_when_eval_results_incomplete(self):
        args = SimpleNamespace(num_training_steps=10, with_tracking=False)
        eval_dataset = self._build_eval_dataset(num_prompts=3)
        eval_queue = _QueueWithSize(size=2)
        eval_generation_config = SimpleNamespace(n=32)

        with patch("open_instruct.grpo_fast.accumulate_inference_batches") as mock_accumulate:
            maybe_evaluate(
                args=args,
                training_step=5,
                evaluation_inference_results_Q=eval_queue,
                tokenizer=Mock(),
                episode=0,
                eval_dataset=eval_dataset,
                eval_generation_config=eval_generation_config,
                model_dims=Mock(),
                base_env_config=EnvConfig(),
                max_possible_score=1.0,
            )

        mock_accumulate.assert_not_called()

    def test_final_step_calls_accumulate_even_when_queue_is_incomplete(self):
        args = SimpleNamespace(num_training_steps=10, with_tracking=False)
        eval_dataset = self._build_eval_dataset(num_prompts=3)
        eval_queue = _QueueWithSize(size=0)
        eval_generation_config = SimpleNamespace(n=32)

        with patch("open_instruct.grpo_fast.accumulate_inference_batches", side_effect=Empty) as mock_accumulate:
            maybe_evaluate(
                args=args,
                training_step=10,
                evaluation_inference_results_Q=eval_queue,
                tokenizer=Mock(),
                episode=0,
                eval_dataset=eval_dataset,
                eval_generation_config=eval_generation_config,
                model_dims=Mock(),
                base_env_config=EnvConfig(),
                max_possible_score=1.0,
            )

        mock_accumulate.assert_called_once()

    def test_records_pass_at_k_metrics(self):
        args = SimpleNamespace(num_training_steps=200, with_tracking=False)
        eval_dataset = self._build_eval_dataset(num_prompts=2)
        eval_queue = _QueueWithSize(size=2)
        eval_generation_config = SimpleNamespace(n=2)
        tokenizer = Mock()
        tokenizer.batch_decode.return_value = ["prompt"] * 4
        tokenizer.pad_token = "<pad>"

        eval_result = SimpleNamespace(
            responses=[[1], [2], [3], [4]],
            finish_reasons=["stop", "stop", "stop", "stop"],
            token_statistics=SimpleNamespace(num_prompt_tokens=10, num_response_tokens=4, generation_time=2.0),
        )
        eval_batch = SimpleNamespace(
            scores=[1.0, 0.0, 0.0, 1.0],
            queries=[[1, 2, 3]] * 4,
            decoded_responses=["resp_a", "resp_b", "resp_c", "resp_d"],
            ground_truths=["42"] * 4,
            active_tools=None,
        )

        with (
            patch(
                "open_instruct.grpo_fast.accumulate_inference_batches",
                return_value=(eval_result, eval_batch, {}, None),
            ),
            patch("open_instruct.grpo_fast.print_rich_single_line_metrics") as mock_print_metrics,
            patch("open_instruct.grpo_fast.print_rich_table"),
        ):
            maybe_evaluate(
                args=args,
                training_step=100,
                evaluation_inference_results_Q=eval_queue,
                tokenizer=tokenizer,
                episode=0,
                eval_dataset=eval_dataset,
                eval_generation_config=eval_generation_config,
                model_dims=Mock(),
                base_env_config=EnvConfig(),
                max_possible_score=1.0,
            )

        logged = mock_print_metrics.call_args.args[0]
        self.assertEqual(logged["eval/pass_at_1"], 0.5)
        self.assertEqual(logged["eval/pass_at_2"], 1.0)
        self.assertEqual(logged["eval/pass_at_1_unbiased"], 0.5)
        self.assertEqual(logged["eval/pass_at_2_unbiased"], 1.0)


class TestComputePassAtKMetrics(unittest.TestCase):
    def test_formula_matches_one_minus_comb_ratio_single_prompt(self):
        n, c, k = 8, 3, 4
        wrong = n - c
        expected = 1.0 - math.comb(wrong, k) / math.comb(n, k)
        correct = np.zeros((1, n), dtype=bool)
        correct[0, :c] = True
        m = grpo_utils.compute_pass_at_k_metrics(correct)
        self.assertAlmostEqual(m["eval/pass_at_4_unbiased"], expected)
        self.assertAlmostEqual(m["eval/pass_at_1"], c / n)

    def test_two_prompts_n2_matches_maybe_evaluate_mock(self):
        correct = np.array([[True, False], [False, True]])
        m = grpo_utils.compute_pass_at_k_metrics(correct)
        self.assertAlmostEqual(m["eval/pass_at_1"], 0.5)
        self.assertAlmostEqual(m["eval/pass_at_2"], 1.0)
        self.assertAlmostEqual(m["eval/pass_at_1_unbiased"], 0.5)
        self.assertAlmostEqual(m["eval/pass_at_2_unbiased"], 1.0)

    def test_all_correct(self):
        m = grpo_utils.compute_pass_at_k_metrics(np.ones((1, 4), dtype=bool))
        self.assertEqual(m["eval/pass_at_1"], 1.0)
        self.assertEqual(m["eval/pass_at_2_unbiased"], 1.0)

    def test_all_wrong_when_k_fits(self):
        m = grpo_utils.compute_pass_at_k_metrics(np.zeros((1, 4), dtype=bool))
        self.assertEqual(m["eval/pass_at_1"], 0.0)
        self.assertEqual(m["eval/pass_at_2_unbiased"], 0.0)

    def test_fewer_than_k_wrong_returns_one(self):
        """Any k-subset must include a correct completion (here k=2, only one wrong)."""
        m = grpo_utils.compute_pass_at_k_metrics(np.array([[True, True, True, False]]))
        self.assertEqual(m["eval/pass_at_1"], 0.75)
        self.assertEqual(m["eval/pass_at_2_unbiased"], 1.0)


if __name__ == "__main__":
    unittest.main()
