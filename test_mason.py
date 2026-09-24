import io
import shlex
import subprocess
import unittest
from argparse import Namespace
from unittest import mock

import beaker
import parameterized

import mason


class TestBuildCommandWithoutArgs(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
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
            ("empty_command", [], {"--with_tracking": False}, []),
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
            (
                "value_arg_followed_by_other_flag",
                ["python", "script.py", "--checkpoint_dir", "--verbose"],
                {"--checkpoint_dir": True},
                ["python", "script.py", "--verbose"],
            ),
            ("adjacent_value_args", ["--checkpoint_dir", "--checkpoint_dir", "/path"], {"--checkpoint_dir": True}, []),
            (
                "remove_repeated_value_arg",
                ["python", "s.py", "--output_dir", "/tmp/y", "--foo", "bar", "--output_dir", "/tmp/z"],
                {"--output_dir": True},
                ["python", "s.py", "--foo", "bar"],
            ),
        ]
    )
    def test_build_command_without_args(self, name, command, args_to_remove, expected):
        result = mason.build_command_without_args(command, args_to_remove)
        self.assertEqual(result, expected)


class TestQuoteLiteralArgs(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("opening_tag", ["--reserved_slot_tokens", "<think>"], ["--reserved_slot_tokens", "'<think>'"]),
            ("closing_tag", ["--stop_strings", "</answer>"], ["--stop_strings", "'</answer>'"]),
            ("pipe_tag", ["--stop_strings", "<|im_end|>"], ["--stop_strings", "'<|im_end|>'"]),
            ("json", ["--dataset_mixer", '{"a": 1.0}'], ["--dataset_mixer", "'{\"a\": 1.0}'"]),
            ("json_with_tag", ['{"stop": "<think>"}'], ['\'{"stop": "<think>"}\'']),
            ("single_quote", ["<it's>"], ["'<it'\"'\"'s>'"]),
            ("tag_with_attributes", ['<tool name="search">'], ["'<tool name=\"search\">'"]),
            ("closing_tag_prefix", ["--stop_strings", "</tool_call"], ["--stop_strings", "'</tool_call'"]),
            (
                "redirections",
                ["echo", "hi", ">", "out", "2>&1", "<in.txt"],
                ["echo", "hi", ">", "out", "2>&1", "<in.txt"],
            ),
            (
                "shell_syntax",
                ["cd", "/stage", "&&", "echo", "$BEAKER_JOB_ID"],
                ["cd", "/stage", "&&", "echo", "$BEAKER_JOB_ID"],
            ),
        ]
    )
    def test_quote_literal_args(self, name, command, expected):
        self.assertEqual(mason.quote_literal_args(command), expected)

    def test_quoted_args_reach_bash_verbatim(self):
        args = ["<think>", "</think>", "<|im_end|>", '{"a": "b"}', "<it's>", '<tool name="search">']
        joined = "printf '%s\\n' " + " ".join(mason.quote_literal_args(args))
        result = subprocess.run(["/bin/bash", "-c", joined], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.splitlines(), args)


class TestMakeInternalCommandQuoting(unittest.TestCase):
    """Literal args must reach both the local cache run and the job exactly once, with no added quotes."""

    LITERALS = ["<think>", "</think>", '{"a": 1.0}', "</tool_call"]

    def test_cache_command_and_job_command_see_the_same_literals(self):
        command = [
            "python", "open_instruct/finetune.py",
            "--reserved_slot_tokens", "<think>", "</think>",
            "--dataset_mixer", '{"a": 1.0}',
            "--stop_strings", "</tool_call",
        ]  # fmt: skip
        args = Namespace(
            artifact_ttl=None,
            auto_checkpoint_state_dir="",
            auto_output_dir_path="/weka/oe-adapt-default/test",
            cluster=["ai2/jupiter"],
            no_auto_dataset_cache=False,
            num_nodes=1,
            pure_docker_mode=True,
        )
        cache_commands = []

        class FakeProcess:
            returncode = 0

            def __init__(self, cmd, **kwargs):
                cache_commands.append(cmd)
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("")

            def poll(self):
                return 0

        with (
            mock.patch.object(mason.subprocess, "Popen", FakeProcess),
            mock.patch.object(mason.select, "select", lambda r, w, x: (r, w, x)),
        ):
            job_command = mason.make_internal_command(list(command), args, "tester", is_external_user=True)

        self.assertEqual(len(cache_commands), 1)
        for shell_command in (cache_commands[0], job_command):
            words = shlex.split(shell_command)
            for literal in self.LITERALS:
                self.assertEqual(words.count(literal), 1, f"{literal!r} in {shell_command}")


class TestExperimentSpec(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
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
                    "min_runtime": "30m",
                    "mount_docker_socket": False,
                    "extra_weka_buckets": [],
                },
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
                    "min_runtime": None,
                    "hostname": None,
                    "mount_docker_socket": False,
                    "extra_weka_buckets": [],
                },
            ),
        ]
    )
    def test_experiment_spec(self, name, args_dict):
        args = Namespace(**args_dict)
        full_command = "test command"
        beaker_secrets = ["test-user"]
        whoami = "test-user"
        resumable = False

        actual_spec = mason.make_task_spec(args, full_command, 0, beaker_secrets, whoami, resumable)

        expected_spec = beaker.BeakerTaskSpec(
            name=f"{args.task_name}__0",
            image=beaker.BeakerImageSource(beaker=args.image),
            command=["/bin/bash", "-c"],
            arguments=[full_command],
            result=beaker.BeakerResultSpec(path="/output"),
            datasets=mason.get_datasets(args.beaker_datasets, args.cluster),
            context=beaker.BeakerTaskContext(
                priority=beaker.BeakerJobPriority[args.priority],
                min_runtime=args.min_runtime if args.min_runtime is not None else (0 if args.preemptible else None),
            ),
            constraints=beaker.BeakerConstraints(cluster=args.cluster)
            if args.hostname is None
            else beaker.BeakerConstraints(hostname=args.hostname),
            env_vars=mason.get_env_vars(
                args.pure_docker_mode,
                args.cluster,
                beaker_secrets,
                whoami,
                resumable,
                args.num_nodes,
                args.env,
                args.secret,
            ),
            resources=beaker.BeakerTaskResources(gpu_count=args.gpus, shared_memory=args.shared_memory),
            replicas=args.num_nodes,
            timeout=args.timeout,
        )
        if args.num_nodes > 1:
            expected_spec.leader_selection = True
            expected_spec.propagate_failure = True
            expected_spec.propagate_preemption = True
        if args.no_host_networking:
            expected_spec.host_networking = False
        else:
            expected_spec.host_networking = True

        self.assertEqual(actual_spec, expected_spec)
        self.assertEqual(
            actual_spec.context.to_json()["minRuntime"], 30 * 60 * 1_000_000_000 if args.min_runtime == "30m" else 0
        )
        self.assertNotIn("preemptible", actual_spec.context.to_json())


class TestGetDatasets(unittest.TestCase):
    def _buckets(self, mounts):
        return [(mount.mount_path, mount.source.weka) for mount in mounts]

    def test_weka_cluster_mounts_the_two_defaults(self):
        self.assertEqual(
            self._buckets(mason.get_datasets([], ["ai2/jupiter"])),
            [("/weka/oe-adapt-default", "oe-adapt-default"), ("/weka/oe-training-default", "oe-training-default")],
        )

    def test_extra_buckets_are_appended_and_deduplicated(self):
        mounts = mason.get_datasets(
            [], ["ai2/jupiter"], extra_weka_buckets=["olmo-3p5-checkpoints", "oe-adapt-default"]
        )
        self.assertEqual(
            self._buckets(mounts),
            [
                ("/weka/oe-adapt-default", "oe-adapt-default"),
                ("/weka/oe-training-default", "oe-training-default"),
                ("/weka/olmo-3p5-checkpoints", "olmo-3p5-checkpoints"),
            ],
        )

    def test_extra_buckets_are_ignored_off_weka(self):
        self.assertEqual(mason.get_datasets([], ["ai2/phobos"], extra_weka_buckets=["olmo-3p5-checkpoints"]), [])


if __name__ == "__main__":
    unittest.main()
