import unittest
from argparse import Namespace

import beaker
import parameterized

import mason


class TestBuildCommandWithoutArgs(unittest.TestCase):
    @parameterized.parameterized.expand([
        (
            "remove_arg_without_value",
            ["python", "script.py", "--with_tracking", "--output", "out.txt"],
            {"--with_tracking": False},
            ["python", "script.py", "--output", "out.txt"],
        ),
        (
            "remove_arg_with_value",
            ["python", "script.py", "--checkpoint_state_dir", "/path/to/dir", "--output", "out.txt"],
            {"--checkpoint_state_dir": True},
            ["python", "script.py", "--output", "out.txt"],
        ),
        (
            "remove_multiple_args",
            ["python", "script.py", "--with_tracking", "--checkpoint_state_dir", "/path", "--output", "out.txt"],
            {"--with_tracking": False, "--checkpoint_state_dir": True},
            ["python", "script.py", "--output", "out.txt"],
        ),
        (
            "arg_not_present",
            ["python", "script.py", "--output", "out.txt"],
            {"--nonexistent": True},
            ["python", "script.py", "--output", "out.txt"],
        ),
        (
            "empty_command",
            [],
            {"--with_tracking": False},
            [],
        ),
        (
            "empty_args_to_remove",
            ["python", "script.py", "--output", "out.txt"],
            {},
            ["python", "script.py", "--output", "out.txt"],
        ),
        (
            "remove_all_cache_excluded_args",
            [
                "python",
                "open_instruct/grpo_fast.py",
                "--with_tracking",
                "--checkpoint_state_freq",
                "200",
                "--checkpoint_state_dir",
                "/weka/path",
                "--gs_checkpoint_state_dir",
                "gs://bucket",
                "--output",
                "out.txt",
            ],
            mason.CACHE_EXCLUDED_ARGS,
            ["python", "open_instruct/grpo_fast.py", "--output", "out.txt"],
        ),
        (
            "arg_at_end_without_value",
            ["python", "script.py", "--output", "out.txt", "--with_tracking"],
            {"--with_tracking": False},
            ["python", "script.py", "--output", "out.txt"],
        ),
        (
            "arg_at_end_with_value",
            ["python", "script.py", "--output", "out.txt", "--checkpoint_dir", "/path"],
            {"--checkpoint_dir": True},
            ["python", "script.py", "--output", "out.txt"],
        ),
    ])
    def test_build_command_without_args(self, name, command, args_to_remove, expected):
        result = mason.build_command_without_args(command, args_to_remove)
        self.assertEqual(result, expected)


class TestExperimentSpec(unittest.TestCase):
    @parameterized.parameterized.expand([
        (
            "single_gpu",
            {
                "cluster": ["ai2/jupiter", "ai2/saturn", "ai2/ceres"],
                "image": "test-user/open-instruct-integration-test",
                "description": "Single GPU on Beaker test script.",
                "pure_docker_mode": True,
                "workspace": "ai2/open-instruct-dev",
                "priority": "urgent",
                "num_nodes": 1,
                "max_retries": 0,
                "timeout": "15m",
                "env": [{"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"}],
                "budget": "ai2/oe-adapt",
                "gpus": 1,
                "no_host_networking": False,
                "beaker_datasets": [],
                "secret": [],
                "shared_memory": "10.24gb",
                "task_name": "beaker_mason",
                "hostname": None,
                "preemptible": False,
            },
            900000000000,
        ),
        (
            "large_test",
            {
                "cluster": ["ai2/jupiter"],
                "image": "test-user/open-instruct-integration-test",
                "description": "Large (multi-node) test script.",
                "pure_docker_mode": True,
                "workspace": "ai2/open-instruct-dev",
                "priority": "urgent",
                "num_nodes": 2,
                "max_retries": 0,
                "timeout": "1h",
                "env": [{"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"}],
                "budget": "ai2/oe-adapt",
                "gpus": 8,
                "no_host_networking": False,
                "beaker_datasets": [],
                "secret": [],
                "shared_memory": "10.24gb",
                "task_name": "beaker_mason",
                "preemptible": True,
                "hostname": None,
            },
            3600000000000,
        ),
    ])
    def test_experiment_spec_timeout(self, name, args_dict, expected_timeout_ns):
        args = Namespace(**args_dict)
        full_command = "test command"
        beaker_secrets = "test-user"
        whoami = "test-user"
        resumable = False

        spec = mason.make_task_spec(args, full_command, 0, beaker_secrets, whoami, resumable)

        self.assertEqual(spec.timeout, expected_timeout_ns)

    @parameterized.parameterized.expand([
        (
            "single_gpu",
            {
                "cluster": ["ai2/jupiter", "ai2/saturn", "ai2/ceres"],
                "image": "test-user/open-instruct-integration-test",
                "description": "Single GPU on Beaker test script.",
                "pure_docker_mode": True,
                "workspace": "ai2/open-instruct-dev",
                "priority": "urgent",
                "num_nodes": 1,
                "max_retries": 0,
                "timeout": "15m",
                "env": [{"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"}],
                "budget": "ai2/oe-adapt",
                "gpus": 1,
                "no_host_networking": False,
                "beaker_datasets": [],
                "secret": [],
                "shared_memory": "10.24gb",
                "task_name": "beaker_mason",
                "hostname": None,
                "preemptible": False,
            },
        ),
    ])
    def test_experiment_spec_basic_fields(self, name, args_dict):
        args = Namespace(**args_dict)
        full_command = "test command"
        beaker_secrets = "test-user"
        whoami = "test-user"
        resumable = False

        spec = mason.make_task_spec(args, full_command, 0, beaker_secrets, whoami, resumable)

        self.assertEqual(spec.name, "beaker_mason__0")
        self.assertEqual(spec.resources.gpu_count, args.gpus)
        self.assertEqual(spec.replicas, args.num_nodes)
        self.assertEqual(spec.host_networking, True)
        self.assertEqual(spec.context.priority, beaker.BeakerJobPriority.urgent)

    @parameterized.parameterized.expand([
        ("15m", 900000000000),
        ("1h", 3600000000000),
        ("30m", 1800000000000),
        ("2h", 7200000000000),
        ("45m", 2700000000000),
    ])
    def test_timeout_conversion(self, timeout_str, expected_ns):
        from beaker.common import to_nanoseconds

        result = to_nanoseconds(timeout_str)
        self.assertEqual(result, expected_ns)


if __name__ == "__main__":
    unittest.main()
