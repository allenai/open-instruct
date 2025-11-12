# Copyright 2024 AllenAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copied from https://github.com/huggingface/alignment-handbook/blob/main/tests/test_data.py
import json
import pathlib
import time
import unittest
from unittest import mock

import pytest
import torch
import vllm
from dateutil import parser
from parameterized import parameterized

from open_instruct import grpo_fast, utils
from open_instruct.finetune import FlatArguments


def _load_mbu_test_cases():
    test_data_path = pathlib.Path(__file__).parent / "test_data" / "mbu_reproduction_cases.json"
    with open(test_data_path) as f:
        data = json.load(f)
    return [(name, case_data) for name, case_data in data.items()]


MODEL_DIMS: dict[str, utils.ModelDims] = {
    "Qwen/Qwen2.5-7B": utils.ModelDims(
        num_layers=28,
        hidden_size=3584,
        intermediate_size=18944,
        vocab_size=152064,
        num_attn_heads=28,
        head_dim=128,
        num_kv_heads=4,
        device_name="h100",
    ),
    "Qwen/Qwen2.5-1.5B": utils.ModelDims(
        num_layers=28,
        hidden_size=1536,
        intermediate_size=8960,
        vocab_size=151936,
        num_attn_heads=12,
        head_dim=128,
        num_kv_heads=2,
        device_name="h100",
    ),
    "Qwen/Qwen3-1.7B": utils.ModelDims(
        num_layers=28,
        hidden_size=2048,
        intermediate_size=6144,
        vocab_size=151936,
        num_attn_heads=16,
        head_dim=128,
        num_kv_heads=8,
        device_name="h100",
    ),
}


class GetDatasetsTest(unittest.TestCase):
    """Each of these test datasets has 100 examples"""

    def test_loading_data_args(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 0.5,
            "HuggingFaceH4/testing_self_instruct_small": 0.3,
            "HuggingFaceH4/testing_codealpaca_small": 0.2,
        }
        datasets = utils.get_datasets(dataset_mixer, columns_to_keep=["prompt", "completion"])
        self.assertEqual(len(datasets["train"]), 100)
        self.assertEqual(len(datasets["test"]), 300)

    def test_loading_with_unit_fractions(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 1.0,
            "HuggingFaceH4/testing_self_instruct_small": 1.0,
            "HuggingFaceH4/testing_codealpaca_small": 1.0,
        }
        datasets = utils.get_datasets(dataset_mixer, columns_to_keep=["prompt", "completion"])
        self.assertEqual(len(datasets["train"]), 300)
        self.assertEqual(len(datasets["test"]), 300)

    def test_loading_with_fractions_greater_than_unity(self):
        dataset_mixer = {"HuggingFaceH4/testing_alpaca_small": 0.7, "HuggingFaceH4/testing_self_instruct_small": 0.4}
        datasets = utils.get_datasets(dataset_mixer, columns_to_keep=["prompt", "completion"])
        self.assertEqual(len(datasets["train"]), 70 + 40)
        self.assertEqual(len(datasets["test"]), 200)

    def test_loading_fails_with_negative_fractions(self):
        dataset_mixer = {"HuggingFaceH4/testing_alpaca_small": 0.7, "HuggingFaceH4/testing_self_instruct_small": -0.3}
        with pytest.raises(ValueError, match=r"Dataset fractions / lengths cannot be negative."):
            utils.get_datasets(dataset_mixer, columns_to_keep=["prompt", "completion"])

    def test_loading_single_split_with_unit_fractions(self):
        dataset_mixer = {"HuggingFaceH4/testing_alpaca_small": 1.0}
        datasets = utils.get_datasets(dataset_mixer, splits=["test"], columns_to_keep=["prompt", "completion"])
        self.assertEqual(len(datasets["test"]), 100)
        self.assertRaises(KeyError, lambda: datasets["train"])

    def test_loading_preference_data(self):
        dataset_mixer = {
            "ai2-adapt-dev/ultrafeedback-small": 1000,
            "ai2-adapt-dev/summarize_from_feedback_small": 1000,
        }
        pref_datasets = utils.get_datasets(dataset_mixer, splits=["train"], columns_to_keep=["chosen", "rejected"])
        self.assertEqual(len(pref_datasets["train"]), 2000)

    def test_time_parser_used_in_get_beaker_dataset_ids(self):
        # two special cases which beaker uses
        self.assertTrue(parser.parse("2024-09-16T19:03:02.31502Z"))
        self.assertTrue(parser.parse("0001-01-01T00:00:00Z"))


def _setup_beaker_mocks(mock_beaker_from_env, mock_is_beaker_job, initial_description):
    """Shared mock setup for beaker tests."""
    mock_is_beaker_job.return_value = True

    mock_client = mock.MagicMock()
    mock_beaker_from_env.return_value = mock_client

    # Mock the workload object
    mock_workload = mock.MagicMock()
    mock_client.workload.get.return_value = mock_workload

    # Mock the spec object returned by experiment.get_spec
    mock_spec = mock.MagicMock()
    mock_spec.description = initial_description
    mock_client.experiment.get_spec.return_value = mock_spec

    description_history = []

    def track_description(workload, description=None):
        if description is not None:
            description_history.append(description)

    mock_client.workload.update.side_effect = track_description

    return mock_client, mock_spec, description_history


class TestBeakerDescription(unittest.TestCase):
    """Test the beaker description update function."""

    @mock.patch("os.environ.get")
    @mock.patch("beaker.Beaker.from_env")
    @mock.patch("open_instruct.utils.is_beaker_job")
    def test_description_does_not_accumulate(self, mock_is_beaker_job, mock_beaker_from_env, mock_environ_get):
        """Test that the description doesn't accumulate git info and wandb URLs on repeated calls."""
        # Configure os.environ.get mock
        env_values = {"BEAKER_WORKLOAD_ID": "test-id-123", "GIT_COMMIT": "abc123", "GIT_BRANCH": "main"}
        mock_environ_get.side_effect = lambda key, default=None: env_values.get(key, default)

        mock_client, mock_spec, description_history = _setup_beaker_mocks(
            mock_beaker_from_env, mock_is_beaker_job, "Beaker-Mason job."
        )

        wandb_url = "https://wandb.ai/ai2-llm/open_instruct_internal/runs/1f3ow3oh"
        start_time = time.time()

        original_descriptions = {}

        for step in [10, 20, 30]:
            utils.maybe_update_beaker_description(
                current_step=step,
                total_steps=100,
                start_time=start_time,
                wandb_url=wandb_url,
                original_descriptions=original_descriptions,
            )
            if description_history:
                mock_spec.description = description_history[-1]

        self.assertEqual(len(description_history), 3)

        for i, desc in enumerate(description_history):
            git_commit_count = desc.count("git_commit:")
            git_branch_count = desc.count("git_branch:")
            wandb_count = desc.count(wandb_url)

            self.assertEqual(
                git_commit_count,
                1,
                f"Step {(i + 1) * 10}: git_commit should appear once, but appears {git_commit_count} times in: {desc}",
            )
            self.assertEqual(
                git_branch_count,
                1,
                f"Step {(i + 1) * 10}: git_branch should appear once, but appears {git_branch_count} times in: {desc}",
            )
            self.assertEqual(
                wandb_count,
                1,
                f"Step {(i + 1) * 10}: wandb URL should appear once, but appears {wandb_count} times in: {desc}",
            )

            self.assertIn("Beaker-Mason job.", desc)
            self.assertIn("git_commit: abc123", desc)
            self.assertIn("git_branch: main", desc)
            self.assertIn(wandb_url, desc)
            self.assertIn(f"% complete (step {(i + 1) * 10}/100)", desc)

    @mock.patch("os.environ.get")
    @mock.patch("beaker.Beaker.from_env")
    @mock.patch("open_instruct.utils.is_beaker_job")
    def test_description_without_progress(self, mock_is_beaker_job, mock_beaker_from_env, mock_environ_get):
        """Test description updates without progress information."""
        # Configure os.environ.get mock
        env_values = {"BEAKER_WORKLOAD_ID": "test-id-123", "GIT_COMMIT": "def456", "GIT_BRANCH": "dev"}
        mock_environ_get.side_effect = lambda key, default=None: env_values.get(key, default)

        mock_client, mock_spec, description_history = _setup_beaker_mocks(
            mock_beaker_from_env, mock_is_beaker_job, "Initial job description"
        )

        original_descriptions = {}

        utils.maybe_update_beaker_description(
            wandb_url="https://wandb.ai/team/project/runs/xyz789", original_descriptions=original_descriptions
        )

        self.assertEqual(len(description_history), 1)
        desc = description_history[0]

        self.assertIn("Initial job description", desc)
        self.assertIn("git_commit: def456", desc)
        self.assertIn("git_branch: dev", desc)
        self.assertIn("https://wandb.ai/team/project/runs/xyz789", desc)
        self.assertNotIn("% complete", desc)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in utils module."""

    @parameterized.expand(
        [
            ("basic_repeat", ["a", "b", "c"], 3, ["a", "a", "a", "b", "b", "b", "c", "c", "c"]),
            ("repeat_once", ["a", "b", "c"], 1, ["a", "b", "c"]),
            ("repeat_zero", ["a", "b", "c"], 0, []),
            ("empty_sequence", [], 3, []),
            ("integers", [1, 2, 3], 2, [1, 1, 2, 2, 3, 3]),
            ("mixed_types", ["a", 1, None, True], 2, ["a", "a", 1, 1, None, None, True, True]),
            ("single_element", ["x"], 5, ["x", "x", "x", "x", "x"]),
        ]
    )
    def test_repeat_each(self, name, sequence, k, expected):
        """Test the repeat_each function with various inputs."""
        result = utils.repeat_each(sequence, k)
        self.assertEqual(result, expected)

    def test_repeat_each_mutation_isolation(self):
        """Test that mutating a sequence item after repeat_each doesn't change the repeated versions."""
        original_list = [1, 2]
        sequence = [original_list, ["a", "b"], [True, False]]
        result = utils.repeat_each(sequence, 2)

        # Result should be: [[1, 2], [1, 2], ["a", "b"], ["a", "b"], [True, False], [True, False]]
        self.assertEqual(len(result), 6)

        # Mutate the original list
        original_list.append(3)

        # The repeated versions should all be affected since they are references to the same object
        self.assertEqual(result[0], [1, 2, 3])
        self.assertEqual(result[1], [1, 2, 3])

        # But the other lists should remain unchanged
        self.assertEqual(result[2], ["a", "b"])
        self.assertEqual(result[3], ["a", "b"])
        self.assertEqual(result[4], [True, False])
        self.assertEqual(result[5], [True, False])

    @parameterized.expand(
        [
            ("https://wandb.ai/org/project/runs/runid", "org/project/runid"),
            (
                "https://wandb.ai/ai2-llm/open_instruct_internal/runs/5nigq0mz",
                "ai2-llm/open_instruct_internal/5nigq0mz",
            ),
            (
                "https://wandb.ai/ai2-llm/open_instruct_internal/runs/vjyp36sp",
                "ai2-llm/open_instruct_internal/vjyp36sp",
            ),
        ]
    )
    def test_wandb_url_to_run_path(self, url: str, expected_run_path: str):
        self.assertEqual(utils.wandb_url_to_run_path(url), expected_run_path)

    @parameterized.expand(
        [
            ("NVIDIA H100 80GB HBM3", "h100"),
            ("NVIDIA L40S", "l40s"),
            ("NVIDIA RTX A6000", "a6000"),
            ("NVIDIA A100-SXM4-80GB", "a100"),
            ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "pro 6000"),
            ("NVIDIA RTX 6000 Ada Generation", "6000"),
            ("NVIDIA GeForce RTX 4090 Laptop GPU", "4090 laptop"),
        ]
    )
    def test_get_device_name(self, device_name: str, expected_name: str):
        result = utils.get_device_name(device_name)
        self.assertEqual(result, expected_name)

    @parameterized.expand(
        [
            ("NVIDIA H100 80GB HBM3", {"flops": 990e12, "memory_size": 80e9, "memory_bandwidth": 3.35e12}),
            ("NVIDIA RTX A6000", {"flops": 155e12, "memory_size": 48e9, "memory_bandwidth": 768e9}),
            (
                "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                {"flops": 503.8e12, "memory_size": 96e9, "memory_bandwidth": 1792e9},
            ),
            ("NVIDIA RTX 6000 Ada Generation", {"flops": 728.5e12, "memory_size": 48e9, "memory_bandwidth": 960e9}),
        ]
    )
    def test_get_device_name_returns_correct_specs(self, device_name: str, expected_specs: dict):
        device_key = utils.get_device_name(device_name)
        specs = utils.GPU_SPECS[device_key]
        self.assertEqual(specs["flops"], expected_specs["flops"])
        self.assertEqual(specs["memory_size"], expected_specs["memory_size"])
        self.assertEqual(specs["memory_bandwidth"], expected_specs["memory_bandwidth"])


class TestFlatArguments(unittest.TestCase):
    def test_additional_model_args(self) -> None:
        parser = utils.ArgumentParserPlus(FlatArguments)
        (args,) = parser.parse_args_into_dataclasses(
            ["--additional_model_arguments", '{"int": 1, "bool": true, "float": 0.0, "float2": 5e-7}']
        )
        self.assertIsInstance(args.additional_model_arguments, dict)
        self.assertIsInstance(args.additional_model_arguments["int"], int)
        self.assertIsInstance(args.additional_model_arguments["bool"], bool)
        self.assertIsInstance(args.additional_model_arguments["float"], float)
        self.assertIsInstance(args.additional_model_arguments["float2"], float)

    def test_no_additional_model_args(self) -> None:
        parser = utils.ArgumentParserPlus(FlatArguments)
        (args,) = parser.parse_args_into_dataclasses(["--exp_name", "test"])
        self.assertIsInstance(args.additional_model_arguments, dict)
        self.assertFalse(args.additional_model_arguments)


class TestModelDims(unittest.TestCase):
    def test_qwen25_7b_flops_calculation(self):
        sequence_length = 34048
        model_dims = MODEL_DIMS["Qwen/Qwen2.5-7B"]
        total_flops = model_dims.flops([sequence_length], [1])
        prefill_flops = model_dims.flops([sequence_length], None)
        decode_flops = total_flops - prefill_flops
        decode_flops_in_gflops = decode_flops / 1e9
        self.assertAlmostEqual(decode_flops_in_gflops, 27.92, delta=0.01)

    def test_qwen25_7b_memory_calculation(self):
        sequence_length = 34048
        batch_size = 16
        model_dims = MODEL_DIMS["Qwen/Qwen2.5-7B"]

        embedding_params = model_dims.vocab_size * model_dims.hidden_size
        weight_params = model_dims.num_params - embedding_params
        lm_head_bytes = model_dims.vocab_size * model_dims.hidden_size
        embedding_bytes = model_dims.hidden_size

        total_bytes = weight_params / batch_size
        total_bytes += lm_head_bytes + embedding_bytes
        total_bytes += 2 * model_dims.num_kv_heads * model_dims.head_dim * model_dims.num_layers * sequence_length
        total_bytes += 2 * model_dims.num_layers * model_dims.num_kv_heads * model_dims.head_dim
        total_bytes *= 2

        memory_in_gb = total_bytes / 1e9
        self.assertAlmostEqual(memory_in_gb, 3.926, delta=0.01)

    @parameterized.expand(_load_mbu_test_cases())
    def test_mbu_reproduction(self, name, case_data):
        metrics = grpo_fast.calculate_utilization_metrics(
            model_dims=MODEL_DIMS[case_data["model_name"]],
            prompt_lengths=case_data["prompt_lengths"],
            response_lengths=case_data["response_lengths"],
            total_generation_time=case_data["total_generation_time"],
            samples_per_prompt=case_data["samples_per_prompt"],
            num_engines=case_data["num_engines"],
            num_gpus_per_engine=case_data["num_gpus_per_engine"],
            training_time=case_data["training_time"],
            num_training_gpus=case_data["num_training_gpus"],
        )

        self.assertLessEqual(metrics["actor_mfu"], 100)
        self.assertLessEqual(metrics["actor_mbu"], 100)
        self.assertLessEqual(metrics["learner_mfu"], 100)

    @parameterized.expand(
        [
            ("two_engines_four_gpus_each", "Qwen/Qwen2.5-7B", 16, 2, 256, 256, 8, 2, 4, 4, 8.0, 4.0),
            ("four_engines_two_gpus_each", "Qwen/Qwen2.5-7B", 16, 2, 256, 256, 8, 4, 2, 4, 8.0, 4.0),
            ("single_engine_eight_gpus", "Qwen/Qwen2.5-7B", 16, 2, 256, 256, 8, 1, 8, 4, 8.0, 4.0),
        ]
    )
    def test_multi_engine_utilization(
        self,
        name,
        model_name,
        num_prompts,
        samples_per_prompt,
        prompt_len,
        response_len,
        num_inference_gpus,
        num_engines,
        num_gpus_per_engine,
        num_training_gpus,
        total_generation_time,
        training_time,
    ):
        prompt_lengths = [prompt_len] * num_prompts
        response_lengths = [int(response_len)] * (num_prompts * samples_per_prompt)

        metrics = grpo_fast.calculate_utilization_metrics(
            model_dims=MODEL_DIMS[model_name],
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            total_generation_time=total_generation_time,
            samples_per_prompt=samples_per_prompt,
            num_engines=num_engines,
            num_gpus_per_engine=num_gpus_per_engine,
            training_time=training_time,
            num_training_gpus=num_training_gpus,
        )

        self.assertLessEqual(
            metrics["actor_mfu"],
            100,
            f"{name}: Actor MFU {metrics['actor_mfu']:.2f}% exceeded 100% "
            f"(num_engines={num_engines}, num_gpus_per_engine={num_gpus_per_engine})",
        )
        self.assertLessEqual(
            metrics["actor_mbu"],
            100,
            f"{name}: Actor MBU {metrics['actor_mbu']:.2f}% exceeded 100% "
            f"(num_engines={num_engines}, num_gpus_per_engine={num_gpus_per_engine})",
        )
        self.assertLessEqual(metrics["learner_mfu"], 100)

    def test_model_dims_match_vllm_config(self):
        model_name = "Qwen/Qwen2.5-7B"
        expected_dims = MODEL_DIMS[model_name]

        mock_platform = mock.Mock()
        mock_platform.device_type = "cuda"
        mock_platform.is_cuda_alike.return_value = True
        mock_platform.supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        mock_platform.get_device_total_memory.return_value = 80 * 1024**3
        mock_platform.get_device_name.return_value = "NVIDIA H100 80GB HBM3"

        mock_model_cls = mock.Mock()
        mock_model_cls.supports_multimodal.return_value = False
        mock_model_cls.is_attention_free.return_value = False
        mock_model_cls.is_attention_free = False

        def mock_inspect_return(*args, **kwargs):
            return mock_model_cls, "Qwen2ForCausalLM"

        with (
            mock.patch("vllm.platforms.current_platform", mock_platform),
            mock.patch(
                "vllm.model_executor.models.registry.ModelRegistry.inspect_model_cls", side_effect=mock_inspect_return
            ),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
        ):
            engine_args = vllm.EngineArgs(model=model_name, load_format="dummy", max_model_len=512)
            vllm_config = engine_args.create_engine_config()
            vllm_dims = utils.ModelDims.from_vllm_config(vllm_config)
        vllm_dims.device_name = "h100"

        self.assertEqual(vllm_dims, expected_dims)


# useful for checking if public datasets are still available
# class CheckTuluDatasetsTest(unittest.TestCase):
#     """
#     Try to rebuild Tulu from public sources
#     """

#     def test_loading_tulu(self):
#         dataset_mixer = {
#             "natolambert/tulu-v2-sft-mixture-flan": 50000,
#             "natolambert/tulu-v2-sft-mixture-cot": 49747,
#             "allenai/openassistant-guanaco-reformatted": 7708,  # not exact subset
#             "Vtuber-plan/sharegpt-cleaned": 114046,
#             # original https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
#             "vicgalle/alpaca-gpt4": 20000,
#             "HuggingFaceH4/CodeAlpaca_20K": 18000,  # original uses https://github.com/sahil280114/codealpaca
#             "natolambert/tulu-v2-sft-mixture-lima": 1018,  # original has 1030
#             "WizardLMTeam/WizardLM_evol_instruct_V2_196k": 30000,
#             "Open-Orca/OpenOrca": 30000,
#             "natolambert/tulu-v2-sft-mixture-science": 7468,  # original data slightly different
#         }
#         _ = get_datasets(dataset_mixer, splits=["train"], columns_to_keep=["messages"])
