import unittest
from queue import Empty
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from open_instruct.grpo_fast import create_generation_configs, get_vllm_max_model_len, maybe_evaluate


class _QueueWithSize:
    def __init__(self, size: int):
        self._size = size

    def qsize(self) -> int:
        return self._size


class TestCreateGenerationConfigs(unittest.TestCase):
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

        self.assertEqual(get_vllm_max_model_len(streaming_config), 1536)


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

    def test_records_eval_model_step_range_and_deltas(self):
        args = SimpleNamespace(num_training_steps=200, with_tracking=False)
        eval_dataset = self._build_eval_dataset(num_prompts=1)
        eval_queue = _QueueWithSize(size=1)
        eval_generation_config = SimpleNamespace(n=2)
        tokenizer = Mock()
        tokenizer.batch_decode.return_value = ["prompt", "prompt"]
        tokenizer.pad_token = "<pad>"

        eval_result = SimpleNamespace(
            responses=[[1], [2]],
            finish_reasons=["stop", "stop"],
            token_statistics=SimpleNamespace(num_prompt_tokens=10, num_response_tokens=4, generation_time=2.0),
        )
        eval_batch = SimpleNamespace(
            scores=[1.0, 0.0],
            queries=[[1, 2, 3], [1, 2, 3]],
            decoded_responses=["resp_a", "resp_b"],
            ground_truths=["42", "42"],
            active_tools=None,
        )
        reward_metrics = {
            "model_step_min": 100.0,
            "model_step_max": 105.0,
            "model_step_mean": 103.0,
            "model_step_span": 5.0,
        }

        with (
            patch(
                "open_instruct.grpo_fast.accumulate_inference_batches",
                return_value=(eval_result, eval_batch, reward_metrics, None),
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
        self.assertEqual(logged["eval/model_step_diff_min"], 0.0)
        self.assertEqual(logged["eval/model_step_diff_max"], 5.0)
        self.assertEqual(logged["eval/model_step_diff_avg"], 3.0)
        self.assertEqual(logged["eval/model_step_diff_span"], 5.0)

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


if __name__ == "__main__":
    unittest.main()
