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
                    "no_result": False,
                    "beaker_datasets": [],
                    "secret": [],
                    "shared_memory": "10.24gb",
                    "task_name": "beaker_mason",
                    "hostname": None,
                    "preemptible": False,
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
                    "no_result": False,
                    "beaker_datasets": [],
                    "secret": [],
                    "shared_memory": "10.24gb",
                    "task_name": "beaker_mason",
                    "preemptible": True,
                    "hostname": None,
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


class TestTorchrunMultiNodeTransformation(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "transform_c10d_to_static",
                "torchrun --nnodes 4 --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint $BEAKER_LEADER_REPLICA_HOSTNAME:29500 script.py",
                2,
                "--rdzv_backend=static",
            ),
            (
                "add_node_rank",
                "torchrun --nnodes 4 --nproc_per_node 8 script.py",
                2,
                "--node_rank $BEAKER_REPLICA_RANK",
            ),
            ("add_rdzv_id", "torchrun --nnodes 4 --nproc_per_node 8 script.py", 2, "--rdzv_id=12347"),
            ("add_rdzv_conf", "torchrun --nnodes 4 --nproc_per_node 8 script.py", 2, "--rdzv_conf='read_timeout=420'"),
            (
                "no_transform_single_node",
                "torchrun --nproc_per_node 8 --rdzv_backend c10d script.py",
                1,
                "--rdzv_backend c10d",
            ),
        ]
    )
    def test_torchrun_transformation(self, name, input_command, num_nodes, expected_substring):
        import re

        join_full_command = input_command
        if num_nodes > 1 and "torchrun" in join_full_command:
            join_full_command = re.sub(r"--rdzv_backend\s+c10d", "--rdzv_backend=static", join_full_command)
            if "--node_rank" not in join_full_command:
                join_full_command = re.sub(r"(torchrun\s+)", r"\1--node_rank $BEAKER_REPLICA_RANK ", join_full_command)
            if "--rdzv_id" not in join_full_command:
                join_full_command = re.sub(r"(torchrun\s+)", r"\1--rdzv_id=12347 ", join_full_command)
            if "--rdzv_conf" not in join_full_command:
                join_full_command = re.sub(r"(torchrun\s+)", r"\1--rdzv_conf='read_timeout=420' ", join_full_command)
            join_full_command = re.sub(
                r"--rdzv_endpoint\s+(\$BEAKER_LEADER_REPLICA_HOSTNAME):(\d+)",
                r"--rdzv_endpoint \1:29400",
                join_full_command,
            )

        self.assertIn(expected_substring, join_full_command)


if __name__ == "__main__":
    unittest.main()
