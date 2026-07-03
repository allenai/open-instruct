import unittest
from argparse import Namespace

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
                    "mount_docker_socket": False,
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
                    "hostname": None,
                    "mount_docker_socket": False,
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


class TestResolveIsExternalUser(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("explicit_local_wins", "local", True, True, True),
            ("explicit_beaker_wins", "beaker", False, False, False),
            ("auto_no_config_no_token_external", None, False, False, True),
            ("auto_config_present_internal", None, True, False, False),
            ("auto_token_present_internal", None, False, True, False),
        ]
    )
    def test_resolve(self, name, launcher, has_config, has_token, expected):
        self.assertEqual(mason.resolve_is_external_user(launcher, has_config, has_token), expected)


class TestValidateClusterForBeaker(unittest.TestCase):
    def test_beaker_without_cluster_raises(self):
        with self.assertRaises(ValueError):
            mason.validate_cluster_for_beaker(is_external_user=False, cluster=None)

    def test_beaker_with_empty_cluster_raises(self):
        with self.assertRaises(ValueError):
            mason.validate_cluster_for_beaker(is_external_user=False, cluster=[])

    def test_beaker_with_cluster_ok(self):
        mason.validate_cluster_for_beaker(is_external_user=False, cluster=["ai2/jupiter"])

    def test_local_without_cluster_ok(self):
        mason.validate_cluster_for_beaker(is_external_user=True, cluster=None)


class TestMakeInternalCommandLocal(unittest.TestCase):
    """The --launcher local path: no cluster, and no Ai2-org rewriting of the command."""

    def _args(self, cluster):
        return Namespace(
            cluster=cluster,
            num_nodes=1,
            pure_docker_mode=True,
            no_auto_dataset_cache=True,
            auto_output_dir_path="/weka/oe-adapt-default/deletable_checkpoint",
            auto_checkpoint_state_dir="/weka/oe-adapt-default/deletable_checkpoint_states",
            artifact_ttl=None,
        )

    def test_local_training_command_without_cluster_does_not_crash(self):
        # Regression: cluster=None on the external path must not TypeError in the
        # WEKA-cluster check (`any(... for c in args.cluster)` over None).
        command = ["python", "open_instruct/grpo_fast.py", "--model_name_or_path", "Qwen/Qwen3-0.6B"]
        result = mason.make_internal_command(command, self._args(cluster=None), "external_user", is_external_user=True)
        self.assertIsInstance(result, str)
        self.assertIn("open_instruct/grpo_fast.py", result)

    def test_local_training_command_omits_ai2_entities(self):
        # A local user's command must not be rewritten to push to Ai2's HF/W&B orgs.
        command = ["python", "open_instruct/grpo_fast.py", "--model_name_or_path", "Qwen/Qwen3-0.6B"]
        result = mason.make_internal_command(command, self._args(cluster=None), "external_user", is_external_user=True)
        self.assertNotIn("--hf_entity", result)
        self.assertNotIn("--wandb_entity", result)

    def test_ai2_training_command_keeps_ai2_entities(self):
        # Ai2 users: entity injection is unchanged (byte-for-byte behavior preserved).
        command = ["python", "open_instruct/grpo_fast.py", "--model_name_or_path", "Qwen/Qwen3-0.6B"]
        result = mason.make_internal_command(
            command, self._args(cluster=["ai2/jupiter"]), "testuser", is_external_user=False
        )
        self.assertIn("--hf_entity allenai", result)
        self.assertIn("--wandb_entity ai2-llm", result)


if __name__ == "__main__":
    unittest.main()
