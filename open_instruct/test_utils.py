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
import os
import pathlib
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import pytest
import ray
import responses
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

    @mock.patch("os.environ.get")
    @mock.patch("beaker.Beaker.from_env")
    @mock.patch("open_instruct.utils.is_beaker_job")
    def test_description_does_not_duplicate_on_restart(
        self, mock_is_beaker_job, mock_beaker_from_env, mock_environ_get
    ):
        """Test that description doesn't duplicate when job restarts (fresh original_descriptions dict)."""
        env_values = {"BEAKER_WORKLOAD_ID": "test-id-123", "GIT_COMMIT": "abc123", "GIT_BRANCH": "main"}
        mock_environ_get.side_effect = lambda key, default=None: env_values.get(key, default)

        previous_run_description = (
            "Single GPU on Beaker with tool use test script. "
            "git_commit: e6df3c9c git_branch: finbarr/async-reward "
            "https://wandb.ai/ai2-llm/open_instruct_internal/runs/n53oxnzb "
            "[5.0% complete (step 1/20), eta 0m]"
        )
        mock_client, mock_spec, description_history = _setup_beaker_mocks(
            mock_beaker_from_env, mock_is_beaker_job, previous_run_description
        )

        wandb_url = "https://wandb.ai/ai2-llm/open_instruct_internal/runs/n53oxnzb"
        original_descriptions = {}

        utils.maybe_update_beaker_description(
            current_step=2,
            total_steps=20,
            start_time=time.time(),
            wandb_url=wandb_url,
            original_descriptions=original_descriptions,
        )

        self.assertEqual(len(description_history), 1)
        desc = description_history[0]

        git_commit_count = desc.count("git_commit:")
        git_branch_count = desc.count("git_branch:")
        wandb_count = desc.count("wandb.ai")

        self.assertEqual(
            git_commit_count, 1, f"git_commit should appear once, but appears {git_commit_count} times in: {desc}"
        )
        self.assertEqual(
            git_branch_count, 1, f"git_branch should appear once, but appears {git_branch_count} times in: {desc}"
        )
        self.assertEqual(wandb_count, 1, f"wandb URL should appear once, but appears {wandb_count} times in: {desc}")
        self.assertIn("Single GPU on Beaker with tool use test script.", desc)


class TestSlackMessage(unittest.TestCase):
    @responses.activate
    @mock.patch("open_instruct.utils.get_beaker_experiment_url")
    @mock.patch("os.environ.get")
    def test_send_slack_message_with_beaker_url(self, mock_environ_get, mock_get_beaker_url):
        webhook_url = "https://hooks.slack.com/services/test"
        mock_environ_get.return_value = webhook_url
        mock_get_beaker_url.return_value = "https://beaker.org/ex/test-456"

        responses.add(responses.POST, webhook_url, json={"ok": True}, status=200)

        utils.send_slack_message("<!here> Disk is nearly full.")

        self.assertEqual(len(responses.calls), 1)
        request_body = json.loads(responses.calls[0].request.body)
        self.assertIn("https://beaker.org/ex/test-456", request_body["text"])
        self.assertIn("Disk is nearly full", request_body["text"])


class TestWarnIfLowDiskSpace(unittest.TestCase):
    @parameterized.expand(
        [
            ("gcs", "gs://bucket/path"),
            ("s3", "s3://bucket/path"),
            ("azure", "az://container/path"),
            ("hdfs", "hdfs://cluster/path"),
            ("gcs localpath", "/filestore/path"),
        ]
    )
    def test_cloud_paths_skipped(self, name, path):
        with mock.patch("shutil.disk_usage") as mock_disk_usage:
            utils.warn_if_low_disk_space(path)
            mock_disk_usage.assert_not_called()

    @mock.patch("shutil.disk_usage")
    def test_no_warning_below_threshold(self, mock_disk_usage):
        mock_disk_usage.return_value = mock.Mock(total=100, used=50, free=50)
        with mock.patch.object(utils.logger, "warning") as mock_warning:
            utils.warn_if_low_disk_space("/tmp/test", threshold=0.85)
            mock_warning.assert_not_called()

    @mock.patch("shutil.disk_usage")
    def test_warning_above_threshold(self, mock_disk_usage):
        mock_disk_usage.return_value = mock.Mock(total=100 * 1024**3, used=90 * 1024**3, free=10 * 1024**3)
        with mock.patch.object(utils.logger, "warning") as mock_warning:
            utils.warn_if_low_disk_space("/tmp/test", threshold=0.85)
            mock_warning.assert_called_once()
            self.assertIn("90.0%", mock_warning.call_args[0][0])

    @responses.activate
    @mock.patch("shutil.disk_usage")
    @mock.patch("open_instruct.utils.get_beaker_experiment_url")
    @mock.patch("os.environ.get")
    def test_slack_alert_sent_when_enabled(self, mock_environ_get, mock_get_beaker_url, mock_disk_usage):
        webhook_url = "https://hooks.slack.com/services/test"
        mock_environ_get.return_value = webhook_url
        mock_get_beaker_url.return_value = None
        mock_disk_usage.return_value = mock.Mock(total=100 * 1024**3, used=90 * 1024**3, free=10 * 1024**3)
        responses.add(responses.POST, webhook_url, json={"ok": True}, status=200)

        utils.warn_if_low_disk_space("/tmp/test", send_slack_alerts=True)

        self.assertEqual(len(responses.calls), 1)
        request_body = json.loads(responses.calls[0].request.body)
        self.assertIn("Disk usage near capacity", request_body["text"])

    @mock.patch("shutil.disk_usage")
    def test_zero_total_disk_space_returns_early(self, mock_disk_usage):
        mock_disk_usage.return_value = mock.Mock(total=0, used=0, free=0)
        with mock.patch.object(utils.logger, "warning") as mock_warning:
            utils.warn_if_low_disk_space("/tmp/test")
            mock_warning.assert_not_called()

    def test_disk_usage_warns_for_failing_path(self):
        with mock.patch.object(utils.logger, "warning") as mock_warning:
            utils.warn_if_low_disk_space("/non/existant/path")
            mock_warning.assert_called()


class TestDownloadFromGsBucket(unittest.TestCase):
    def test_download_from_gs_bucket(self):
        src_paths = ["gs://bucket/data1", "gs://bucket/data2"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = pathlib.Path(tmp_dir) / "downloads"
            captured_cmd: dict[str, list[str]] = {}

            def mock_live_subprocess_output(cmd):
                captured_cmd["cmd"] = cmd

            with mock.patch.object(utils, "live_subprocess_output", side_effect=mock_live_subprocess_output):
                utils.download_from_gs_bucket(src_paths=src_paths, dest_path=str(dest_path))

            expected_cmd = [
                "gsutil",
                "-o",
                "GSUtil:parallel_thread_count=1",
                "-o",
                "GSUtil:sliced_object_download_threshold=150",
                "-m",
                "cp",
                "-r",
                *src_paths,
                str(dest_path),
            ]

            self.assertEqual(captured_cmd["cmd"], expected_cmd)
            self.assertTrue(dest_path.exists())


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
        expected_dims = MODEL_DIMS["Qwen/Qwen2.5-7B"]

        mock_hf_text_config = mock.Mock()
        mock_hf_text_config.intermediate_size = 18944
        mock_hf_text_config.sliding_window = None
        mock_hf_text_config.num_attention_heads = 28
        mock_hf_text_config.num_key_value_heads = 4

        mock_model_config = mock.Mock()
        mock_model_config.get_hidden_size.return_value = 3584
        mock_model_config.get_num_layers.return_value = 28
        mock_model_config.get_vocab_size.return_value = 152064
        mock_model_config.get_head_size.return_value = 128
        mock_model_config.hf_text_config = mock_hf_text_config

        mock_vllm_config = mock.Mock()
        mock_vllm_config.model_config = mock_model_config
        mock_vllm_config.parallel_config = mock.Mock()

        with (
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            vllm_dims = utils.ModelDims.from_vllm_config(mock_vllm_config)

        self.assertEqual(vllm_dims, expected_dims)


class TestModelDimsFromHFConfig(unittest.TestCase):
    def test_from_hf_config_with_sliding_window(self):
        config = SimpleNamespace(
            hidden_size=2048,
            intermediate_size=8192,
            sliding_window=4096,
            layer_types=["sliding_attention", "attention"],
            num_attention_heads=16,
            num_hidden_layers=24,
            vocab_size=32000,
            num_key_value_heads=8,
            head_dim=128,
        )

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=config) as mock_from_pretrained,
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            model_dims = utils.ModelDims.from_hf_config("test/model")

        mock_from_pretrained.assert_called_once_with("test/model", trust_remote_code=True)
        self.assertEqual(
            model_dims,
            utils.ModelDims(
                num_layers=24,
                hidden_size=2048,
                intermediate_size=8192,
                vocab_size=32000,
                num_attn_heads=16,
                head_dim=128,
                num_kv_heads=8,
                sliding_window=4096,
                num_sliding_window_layers=1,
                device_name="h100",
            ),
        )

    def test_from_hf_config_defaults(self):
        config = SimpleNamespace(hidden_size=1024, num_attention_heads=8, num_hidden_layers=12, vocab_size=64000)

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=config),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            model_dims = utils.ModelDims.from_hf_config("test/defaults")
        self.assertEqual(
            model_dims,
            utils.ModelDims(
                num_layers=12,
                hidden_size=1024,
                intermediate_size=4096,
                vocab_size=64000,
                num_attn_heads=8,
                head_dim=128,
                num_kv_heads=8,
                sliding_window=None,
                num_sliding_window_layers=0,
                device_name="h100",
            ),
        )

    def test_from_hf_config_sliding_window_no_layer_types(self):
        config = SimpleNamespace(
            hidden_size=2048,
            intermediate_size=8192,
            sliding_window=4096,
            num_attention_heads=16,
            num_hidden_layers=24,
            vocab_size=32000,
            num_key_value_heads=8,
            head_dim=128,
        )

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=config),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA H100 80GB HBM3"),
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            model_dims = utils.ModelDims.from_hf_config("test/model")

        self.assertEqual(model_dims.sliding_window, 4096)
        self.assertEqual(model_dims.num_sliding_window_layers, 24)

    def test_from_hf_config_cpu_only(self):
        config = SimpleNamespace(hidden_size=1024, num_attention_heads=8, num_hidden_layers=12, vocab_size=64000)

        with (
            mock.patch("transformers.AutoConfig.from_pretrained", return_value=config),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            model_dims = utils.ModelDims.from_hf_config("test/cpu")

        self.assertIsNone(model_dims.device_name)


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


class TestGetDenominator(unittest.TestCase):
    @parameterized.expand([("token", "token"), ("0.5", 0.5), (0.5, 0.5), (1, 1.0)])
    def test_valid_inputs(self, input_val, expected):
        self.assertEqual(utils.get_denominator(input_val), expected)

    @parameterized.expand(
        [
            ("invalid", "could not convert string to float"),
            ("-1", "loss_denominator must be greater than 0"),
            (0, "loss_denominator must be greater than 0"),
            ("0", "loss_denominator must be greater than 0"),
        ]
    )
    def test_invalid_inputs(self, input_val, error_msg):
        with self.assertRaisesRegex(ValueError, error_msg):
            utils.get_denominator(input_val)


class TestRayGetWithProgress(unittest.TestCase):
    def setUp(self):
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        ray.init(num_cpus=2, num_gpus=0)

    def tearDown(self):
        ray.shutdown()

    def test_timeout_error_includes_desc(self):
        @ray.remote
        def slow_task():
            time.sleep(10)
            return "done"

        refs = [slow_task.remote()]
        desc = "Test slow operation"

        with pytest.raises(TimeoutError, match=desc):
            utils.ray_get_with_progress(refs, desc=desc, enable=False, timeout=0.1)
