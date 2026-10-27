import argparse
import io
import shlex
import subprocess
import sys
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
                    "min_runtime": None,
                    "auto_resume": False,
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
                    "auto_resume": False,
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
                priority=beaker.BeakerJobPriority[args.priority], preemptible=args.preemptible
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


class TestMakeTaskContext(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("protected_default", False, None, False, {"priority": "normal", "preemptible": False}),
            ("preemptible", True, None, False, {"priority": "normal", "preemptible": True}),
            (
                "allocated",
                False,
                "2h",
                True,
                {"priority": "normal", "minRuntime": 2 * 60 * 60 * 1_000_000_000, "autoResume": True},
            ),
            ("unallocated", False, "0", False, {"priority": "normal", "minRuntime": 0, "autoResume": False}),
        ]
    )
    def test_context_json(self, name, preemptible, min_runtime, auto_resume, expected_json):
        args = Namespace(priority="normal", preemptible=preemptible, min_runtime=min_runtime, auto_resume=auto_resume)
        self.assertEqual(mason.make_task_context(args).to_json(), expected_json)


class TestParseMinRuntime(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("zero", "0"),
            ("lower_bound", "5m"),
            ("hours", "2h30m"),
            ("usual_upper_bound", "8h"),
            ("above_usual_upper_bound", "9h"),
            ("seconds_with_unit", "3600s"),
        ]
    )
    def test_valid(self, name, value):
        self.assertEqual(mason.parse_min_runtime(value), value)

    @parameterized.parameterized.expand([("plus_zero", "+0"), ("minus_zero", "-0"), ("zero_seconds", "0s")])
    def test_zero_spellings_normalize(self, name, value):
        self.assertEqual(mason.parse_min_runtime(value), "0")

    @parameterized.parameterized.expand(
        [
            ("below_lower_bound", "4m"),
            ("bare_number", "3600"),
            ("bare_decimal", "1.5"),
            ("decimal_zero", "0.0"),
            ("negative", "-5m"),
            ("not_a_duration", "soon"),
        ]
    )
    def test_invalid(self, name, value):
        with self.assertRaises(argparse.ArgumentTypeError):
            mason.parse_min_runtime(value)


class TestGetArgsMinRuntime(unittest.TestCase):
    def _get_args(self, *mason_args, command=("echo", "hello")):
        argv = ["mason.py", "--cluster", "ai2/jupiter", *mason_args, "--", *command]
        with mock.patch.object(sys, "argv", argv):
            return mason.get_args()

    def test_zero_spelling_reaches_task_context_as_zero(self):
        args, _ = self._get_args("--min_runtime", "0s", "--auto_resume")
        self.assertEqual(
            mason.make_task_context(args).to_json(), {"priority": "normal", "minRuntime": 0, "autoResume": True}
        )

    def _warnings(self, *mason_args, command=("echo", "hello")):
        with mock.patch.object(mason.console, "log") as log:
            self._get_args(*mason_args, command=command)
        return " ".join(str(call.args[0]) for call in log.call_args_list)

    def test_min_runtime_and_auto_resume(self):
        args, _ = self._get_args("--min_runtime", "30m", "--auto_resume")
        self.assertEqual(args.min_runtime, "30m")
        self.assertTrue(args.auto_resume)
        self.assertFalse(args.preemptible)

    @parameterized.parameterized.expand(
        [
            ("min_runtime_with_preemptible", ["--min_runtime", "30m", "--preemptible"]),
            ("auto_resume_without_min_runtime", ["--auto_resume"]),
            ("min_runtime_below_lower_bound", ["--min_runtime", "4m"]),
        ]
    )
    def test_rejects(self, name, mason_args):
        with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            self._get_args(*mason_args)

    def test_rejects_auto_resume_with_preemptible(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit):
            self._get_args("--preemptible", "--auto_resume")
        self.assertIn("--preemptible, which already auto-resumes", stderr.getvalue())

    @parameterized.parameterized.expand(
        [
            ("above_usual_upper_bound", ["--min_runtime", "9h", "--auto_resume"], "8h maximum"),
            ("zero_without_auto_resume", ["--min_runtime", "0"], "not requeued"),
            ("auto_resume_without_resumable_command", ["--min_runtime", "2h", "--auto_resume"], "restarts it"),
            ("allocated_without_auto_resume", ["--min_runtime", "1h"], "does not requeue"),
            ("timeout_beyond_min_runtime", ["--min_runtime", "1h", "--timeout", "6h"], "does not requeue"),
            ("zero_timeout_means_none", ["--min_runtime", "1h", "--timeout", "0"], "does not requeue"),
        ]
    )
    def test_warns(self, name, mason_args, expected):
        self.assertIn(expected, self._warnings(*mason_args))

    @parameterized.parameterized.expand(
        [
            ("allocated", ["--min_runtime", "2h", "--auto_resume"]),
            ("zero_with_auto_resume", ["--min_runtime", "0", "--auto_resume"]),
            ("timeout_within_min_runtime", ["--min_runtime", "2h", "--timeout", "1h"]),
            ("timeout_equal_to_min_runtime", ["--min_runtime", "2h", "--timeout", "2h"]),
            ("unparsable_timeout_left_to_beaker", ["--min_runtime", "2h", "--timeout", "soon"]),
        ]
    )
    def test_does_not_warn(self, name, mason_args):
        warnings = self._warnings(*mason_args)
        self.assertNotIn("8h maximum", warnings)
        self.assertNotIn("not requeued", warnings)
        self.assertNotIn("does not requeue", warnings)

    def test_auto_resume_with_non_resumable_flag_warns(self):
        command = ("python", "open_instruct/grpo_fast.py")
        warnings = self._warnings("--min_runtime", "2h", "--auto_resume", "--non_resumable", command=command)
        self.assertIn("restarts it", warnings)

    def test_auto_resume_replaces_generic_non_resumable_log(self):
        warnings = self._warnings("--min_runtime", "2h", "--auto_resume")
        self.assertNotIn("OPEN_INSTRUCT_RESUMABLES", warnings)

    def test_auto_resume_with_resumable_command_does_not_warn(self):
        command = ("python", "open_instruct/grpo_fast.py")
        warnings = self._warnings("--min_runtime", "2h", "--auto_resume", command=command)
        self.assertNotIn("restarts it", warnings)


MASON_DEFAULT_NAMES = [
    *mason.DEFAULT_ENV_VARS,
    "HF_TOKEN",
    "PATH",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "HF_HUB_CACHE",
    "CHECKPOINT_OUTPUT_DIR",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
]


class TestGetEnvVars(unittest.TestCase):
    def _env(self, env=(), secrets=()):
        env_vars = mason.get_env_vars(False, ["ai2/jupiter"], ["me_HF_TOKEN"], "me", True, 2, list(env), list(secrets))
        return [(env_var.name, env_var.value, env_var.secret) for env_var in env_vars]

    def _named(self, env_vars, name):
        return [env_var for env_var in env_vars if env_var[0] == name]

    def test_mason_defaults_are_set_once_without_overrides(self):
        env_vars = self._env()
        for name in MASON_DEFAULT_NAMES:
            self.assertEqual(len(self._named(env_vars, name)), 1, name)

    @parameterized.parameterized.expand([(name,) for name in MASON_DEFAULT_NAMES])
    def test_explicit_env_replaces_default(self, name):
        env_vars = self._env(env=[{"name": name, "value": "/explicit"}])
        self.assertEqual(self._named(env_vars, name), [(name, "/explicit", None)])

    @parameterized.parameterized.expand([(name,) for name in MASON_DEFAULT_NAMES])
    def test_explicit_secret_replaces_default(self, name):
        env_vars = self._env(secrets=[{"name": name, "value": "my_secret"}])
        self.assertEqual(self._named(env_vars, name), [(name, None, "my_secret")])

    def test_explicit_secret_is_kept_beside_unrelated_env(self):
        env_vars = self._env(env=[{"name": "FOO", "value": "1"}], secrets=[{"name": "BAR", "value": "bar_secret"}])
        self.assertEqual(self._named(env_vars, "FOO"), [("FOO", "1", None)])
        self.assertEqual(self._named(env_vars, "BAR"), [("BAR", None, "bar_secret")])

    @parameterized.parameterized.expand(
        [
            ("env_first", ["--env", "HF_TOKEN=plain", "--secret", "HF_TOKEN=x"]),
            ("secret_first", ["--secret", "HF_TOKEN=x", "--env", "HF_TOKEN=plain"]),
        ]
    )
    def test_same_name_as_env_and_secret_raises(self, _, flags):
        argv = ["mason.py", "--cluster", "ai2/jupiter", "--image", "img", *flags, "--", "echo", "hi"]
        with mock.patch("sys.argv", argv):
            args, _ = mason.get_args()
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            self._env(env=args.env, secrets=args.secret)

    def test_repeated_env_name_keeps_both_in_order(self):
        env_vars = self._env(env=[{"name": "HF_HOME", "value": "/a"}, {"name": "HF_HOME", "value": "/b"}])
        self.assertEqual(self._named(env_vars, "HF_HOME"), [("HF_HOME", "/a", None), ("HF_HOME", "/b", None)])

    def test_empty_env_value_replaces_default(self):
        env_vars = self._env(env=[mason.parse_env_var("HF_HOME=")])
        self.assertEqual(self._named(env_vars, "HF_HOME"), [("HF_HOME", "", None)])


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
