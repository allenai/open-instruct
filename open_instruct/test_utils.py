# coding=utf-8
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
import time
import unittest
from unittest import mock

import pytest
from dateutil import parser
from parameterized import parameterized

from open_instruct import utils
from open_instruct.finetune import FlatArguments


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


class TestModelDims(unittest.TestCase):
    def setUp(self):
        self.model_dims = utils.ModelDims(
            num_layers=32,
            hidden_size=4096,
            intermediate_size=11008,
            vocab_size=32000,
            num_attn_heads=32,
            num_kv_heads=8,
            device_name="a100",
        )

    def test_basic_properties(self):
        self.assertEqual(self.model_dims.head_dim, 128)
        self.assertEqual(self.model_dims.device_flops, 312e12)
        self.assertEqual(self.model_dims.device_memory_bandwidth, 2.0e12)

    def test_flops_are_positive(self):
        prompt_lengths = [100]
        response_lengths = [50]

        prefill_flops = self.model_dims.prefill_flops(prompt_lengths)
        decode_flops = self.model_dims.decode_flops(prompt_lengths, response_lengths)
        total_flops = self.model_dims.flops(prompt_lengths, response_lengths)

        self.assertGreater(prefill_flops, 0)
        self.assertGreater(decode_flops, 0)
        self.assertGreater(total_flops, 0)
        self.assertEqual(total_flops, prefill_flops + decode_flops)

    def test_memory_bytes_are_positive(self):
        prompt_lengths = [100]
        response_lengths = [50]

        prefill_memory = self.model_dims.prefill_memory_bytes(prompt_lengths)
        decode_memory = self.model_dims.decode_memory_bytes(prompt_lengths, response_lengths)
        total_memory = self.model_dims.memory_bytes(prompt_lengths, response_lengths)

        self.assertGreater(prefill_memory, 0)
        self.assertGreater(decode_memory, 0)
        self.assertGreater(total_memory, 0)
        self.assertEqual(total_memory, prefill_memory + decode_memory)

    def test_flops_scale_with_batch_size(self):
        single_batch_flops = self.model_dims.flops([100], [50])
        double_batch_flops = self.model_dims.flops([100, 100], [50, 50])

        self.assertAlmostEqual(double_batch_flops, 2 * single_batch_flops, delta=single_batch_flops * 0.01)

    def test_memory_bytes_scale_with_batch_size(self):
        single_batch_memory = self.model_dims.memory_bytes([100], [50])
        double_batch_memory = self.model_dims.memory_bytes([100, 100], [50, 50])

        self.assertGreater(double_batch_memory, single_batch_memory)

    @parameterized.expand(
        [
            ("single_sample", [100], [50], 1, 8823.30, 8),
            ("multiple_samples", [100], [50, 50], 2, 8823.30, 8),
            ("bug_case_256_completions", [100] * 64, [32000] * 256, 4, 8823.30, 8),
        ]
    )
    def test_mbu_never_exceeds_100_percent(
        self, name, prompt_lengths, response_lengths, samples_per_prompt, elapsed_time_seconds, num_gpus
    ):
        total_memory_bytes = self.model_dims.memory_bytes(
            prompt_lengths, response_lengths, samples_per_prompt=samples_per_prompt
        )
        bytes_per_second = total_memory_bytes / elapsed_time_seconds
        total_bandwidth = self.model_dims.device_memory_bandwidth * num_gpus
        mbu_percent = 100 * bytes_per_second / total_bandwidth

        self.assertLessEqual(
            mbu_percent,
            100,
            f"{name}: MBU should never exceed 100%, but got {mbu_percent:.2f}%. "
            f"Memory bytes: {total_memory_bytes / 1e9:.2f} GB, "
            f"Elapsed time: {elapsed_time_seconds:.2f}s, "
            f"Bytes/sec: {bytes_per_second / 1e12:.2f} TB/s, "
            f"Total bandwidth: {total_bandwidth / 1e12:.2f} TB/s",
        )

    @parameterized.expand(
        [
            ("single_sample", [100], [50], 1, 8823.30, 8),
            ("multiple_samples", [100], [50, 50], 2, 8823.30, 8),
            ("bug_case_256_completions", [100] * 64, [32000] * 256, 4, 8823.30, 8),
        ]
    )
    def test_mfu_never_exceeds_100_percent(
        self, name, prompt_lengths, response_lengths, samples_per_prompt, elapsed_time_seconds, num_gpus
    ):
        total_flops = self.model_dims.flops(prompt_lengths, response_lengths, samples_per_prompt=samples_per_prompt)
        flops_per_second = total_flops / elapsed_time_seconds
        total_device_flops = self.model_dims.device_flops * num_gpus
        mfu_percent = 100 * flops_per_second / total_device_flops

        self.assertLessEqual(
            mfu_percent,
            100,
            f"{name}: MFU should never exceed 100%, but got {mfu_percent:.2f}%. "
            f"FLOPs: {total_flops / 1e12:.2f} TFLOPs, "
            f"Elapsed time: {elapsed_time_seconds:.2f}s, "
            f"FLOPs/sec: {flops_per_second / 1e12:.2f} TFLOPs/s, "
            f"Total device FLOPs: {total_device_flops / 1e12:.2f} TFLOPs/s",
        )

    def test_bug_case_256_completions_32k_tokens(self):
        num_unique_prompts = 64
        samples_per_prompt = 4
        prompt_lengths = [100] * num_unique_prompts
        response_lengths = [32000] * (num_unique_prompts * samples_per_prompt)
        elapsed_time = 8823.30
        num_gpus = 8

        total_memory_bytes = self.model_dims.memory_bytes(
            prompt_lengths, response_lengths, samples_per_prompt=samples_per_prompt
        )
        bytes_per_second = total_memory_bytes / elapsed_time
        total_bandwidth = self.model_dims.device_memory_bandwidth * num_gpus
        mbu_percent = 100 * bytes_per_second / total_bandwidth

        total_flops = self.model_dims.flops(prompt_lengths, response_lengths, samples_per_prompt=samples_per_prompt)
        flops_per_second = total_flops / elapsed_time
        total_device_flops = self.model_dims.device_flops * num_gpus
        mfu_percent = 100 * flops_per_second / total_device_flops

        self.assertLessEqual(
            mbu_percent,
            100,
            f"Bug case: MBU={mbu_percent:.2f}% should be <= 100%. "
            f"Total tokens: {sum(response_lengths)}, "
            f"Memory bytes: {total_memory_bytes / 1e12:.2f} TB, "
            f"Bandwidth: {bytes_per_second / 1e12:.2f} TB/s, "
            f"Peak: {total_bandwidth / 1e12:.2f} TB/s",
        )

        self.assertLessEqual(
            mfu_percent,
            100,
            f"Bug case: MFU={mfu_percent:.2f}% should be <= 100%. "
            f"FLOPs: {total_flops / 1e15:.2f} PFLOPs, "
            f"FLOPs/s: {flops_per_second / 1e12:.2f} TFLOPs/s, "
            f"Peak: {total_device_flops / 1e12:.2f} TFLOPs/s",
        )

    def test_samples_per_prompt_scaling(self):
        prompt_lengths = [100]

        memory_1_sample = self.model_dims.memory_bytes(prompt_lengths, [50], samples_per_prompt=1)
        memory_4_samples = self.model_dims.memory_bytes(prompt_lengths, [50, 50, 50, 50], samples_per_prompt=4)

        self.assertGreater(memory_4_samples, memory_1_sample)

    def test_decode_flops_increase_with_response_length(self):
        prompt_lengths = [100]
        short_response = [10]
        long_response = [100]

        short_flops = self.model_dims.decode_flops(prompt_lengths, short_response)
        long_flops = self.model_dims.decode_flops(prompt_lengths, long_response)

        self.assertGreater(long_flops, short_flops)

    def test_kv_cache_read_bytes_with_multiple_samples(self):
        prompt_lengths = [100]
        response_lengths = [50, 50]
        samples_per_prompt = 2

        kv_read_bytes = self.model_dims.kv_cache_read_bytes(
            prompt_lengths, response_lengths, samples_per_prompt=samples_per_prompt
        )

        self.assertGreater(kv_read_bytes, 0)

    def test_training_flops_are_3x_inference(self):
        prompt_lengths = [100]
        response_lengths = [50]

        inference_flops = self.model_dims.flops(prompt_lengths, response_lengths, is_training=False)
        training_flops = self.model_dims.flops(prompt_lengths, response_lengths, is_training=True)

        self.assertEqual(training_flops, 3 * inference_flops)

    def test_compare_flops_with_palm_estimate(self):
        num_kv = (
            self.model_dims.num_kv_heads
            if self.model_dims.num_kv_heads is not None
            else self.model_dims.num_attn_heads
        )
        head_dim = self.model_dims.hidden_size // self.model_dims.num_attn_heads

        w_q = self.model_dims.hidden_size * self.model_dims.hidden_size
        w_k = self.model_dims.hidden_size * (num_kv * head_dim)
        w_v = self.model_dims.hidden_size * (num_kv * head_dim)
        w_o = self.model_dims.hidden_size * self.model_dims.hidden_size
        w_up = self.model_dims.hidden_size * self.model_dims.intermediate_size
        w_dn = self.model_dims.intermediate_size * self.model_dims.hidden_size

        nparams_per_layer = w_q + w_k + w_v + w_o + w_up + w_dn
        nparams_total_no_embed = self.model_dims.num_layers * nparams_per_layer
        nparams_embedding = self.model_dims.hidden_size * self.model_dims.vocab_size

        seq_len = 100
        palm_matmul_term = 6 * (nparams_total_no_embed + nparams_embedding)
        palm_attn_term = 12 * self.model_dims.num_layers * self.model_dims.num_attn_heads * head_dim * seq_len
        palm_estimate = palm_matmul_term + palm_attn_term

        our_prefill_flops = self.model_dims.prefill_flops([seq_len])

        attn_flops = self.model_dims.attn_flops(seq_len, seq_len)
        mlp_flops = self.model_dims.mlp_flops(seq_len)
        lm_head_flops = 2 * self.model_dims.hidden_size * self.model_dims.vocab_size

        print(f"\n--- FLOPs Comparison for seq_len={seq_len} ---")
        print("\nPaLM estimate:")
        print(f"  Matmul term (6*nparams): {palm_matmul_term / 1e12:.3f} TFLOPs")
        print(f"  Attention term (12*l*h*q*t): {palm_attn_term / 1e12:.3f} TFLOPs")
        print(f"  Total: {palm_estimate / 1e12:.3f} TFLOPs")
        print(f"  Per token: {palm_estimate / seq_len / 1e9:.2f} GFLOPs")
        print("\nOur implementation:")
        print(f"  Attn per layer: {attn_flops / 1e12:.3f} TFLOPs")
        print(f"  MLP per layer: {mlp_flops / 1e12:.3f} TFLOPs")
        print(f"  LM head: {lm_head_flops / 1e12:.3f} TFLOPs")
        print(f"  Total prefill: {our_prefill_flops / 1e12:.3f} TFLOPs")
        print(f"  Per token avg: {our_prefill_flops / seq_len / 1e9:.2f} GFLOPs")
        print(f"\nRatio (our/PaLM): {our_prefill_flops / palm_estimate:.3f}")
        print("Note: PaLM uses 6*nparams (assumes forward=2x, backward=4x for training)")
        print(f"Our implementation counts FLOPs more explicitly with FLOP_PER_MAC={utils.FLOP_PER_MAC}")

        self.assertGreater(our_prefill_flops, 0)
        self.assertGreater(palm_estimate, 0)

    def test_assertions_catch_expanded_prompt_lengths_bug(self):
        num_unique_prompts = 64
        samples_per_prompt = 4
        prompt_lengths_buggy = [100] * (num_unique_prompts * samples_per_prompt)
        response_lengths = [32000] * (num_unique_prompts * samples_per_prompt)

        with self.assertRaises(AssertionError) as context:
            self.model_dims.memory_bytes(prompt_lengths_buggy, response_lengths, samples_per_prompt=samples_per_prompt)

        self.assertIn("Expected 1024 response lengths, got 256", str(context.exception))

    def test_assertions_catch_wrong_samples_per_prompt(self):
        num_unique_prompts = 64
        samples_per_prompt_actual = 4
        prompt_lengths = [100] * num_unique_prompts
        response_lengths = [32000] * (num_unique_prompts * samples_per_prompt_actual)

        with self.assertRaises(AssertionError) as context:
            self.model_dims.memory_bytes(prompt_lengths, response_lengths, samples_per_prompt=1)

        self.assertIn("Expected 64 response lengths, got 256", str(context.exception))
