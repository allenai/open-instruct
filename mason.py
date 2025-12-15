import argparse
import hashlib
import os
import random
import re
import select
import sys
import time

import backoff
import beaker
import requests
from rich.console import Console
from rich.text import Text

from open_instruct.launch import generate_id, get_datasets, get_env_vars, parse_beaker_dataset, parse_env_var
from open_instruct.utils import GCP_CLUSTERS, INTERCONNECT_CLUSTERS, WEKA_CLUSTERS, download_from_gs_bucket

console = Console()


# ----------------------------------------------------------------------
# Open Instruct logic
OPEN_INSTRUCT_COMMANDS = [
    "open_instruct/finetune.py",
    "open_instruct/dpo_tune_cache.py",
    "open_instruct/grpo_fast.py",
    "open_instruct/ppo.py",
    "open_instruct/reward_modeling.py",
]

OPEN_INSTRUCT_RESUMABLES = ["open_instruct/grpo_fast.py"]

CACHE_EXCLUDED_ARGS = {
    "--with_tracking": False,
    "--checkpoint_state_freq": True,
    "--checkpoint_state_dir": True,
    "--gs_checkpoint_state_dir": True,
}


# ----------------------------------------------------------------------
# Mason logic
def build_command_without_args(command, args_to_remove):
    """Build new command list excluding specified arguments.

    Args:
        command: List of command arguments
        args_to_remove: Dict mapping argument names to boolean indicating if they have values
                       e.g., {"--with_tracking": False, "--checkpoint_state_dir": True}

    Returns:
        New command list with specified arguments removed
    """
    result = []
    skip_next = False

    for item in command:
        if skip_next:
            skip_next = False
            continue

        if item in args_to_remove:
            if args_to_remove[item]:
                skip_next = True
            continue

        result.append(item)

    return result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster", type=str, nargs="+", help="Beaker clusters on which the job could be run.", required=True
    )
    parser.add_argument(
        "--hostname", type=str, nargs="+", help="Beaker hostname on which the job could be run.", default=None
    )
    parser.add_argument("--max_retries", type=int, help="Number of retries", default=0)
    parser.add_argument("--budget", type=str, help="Budget to use.", required=True)
    parser.add_argument("--gpus", type=int, help="Number of gpus", default=0)
    parser.add_argument(
        "--shared_memory", type=str, help="Shared memory size (e.g., '10gb', '10.24gb')", default="10.24gb"
    )
    parser.add_argument("--num_nodes", type=int, help="Number of nodes", default=1)
    parser.add_argument(
        "--image",
        type=str,
        help="Beaker base image; usually fine to use AI2 base image.",
        default="ai2/cuda11.8-cudnn8-dev-ubuntu20.04",
    )
    parser.add_argument(
        "--workspace", type=str, help="The Beaker workspace to use. If not set, use your default.", default=None
    )
    parser.add_argument(
        "--beaker_datasets",
        nargs="*",
        help="""Beaker datasets to mount. You may give more than one, separated by
        spaces. Each dataset should be formatted like `[mount-point]:[beaker-dataset-id]`;
        for instance `/models:01HQXGAYGCS6D4ZK51K83CM49Y`.
        """,
        type=parse_beaker_dataset,
        default=[],
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Optionally, a description for this job in Beaker.",
        default="Beaker-Mason job.",
    )
    parser.add_argument("--task_name", type=str, help="Name for the Beaker task.", default="beaker_mason")
    parser.add_argument("--priority", type=str, help="Beaker job priority.", default="normal")
    parser.add_argument("--preemptible", action="store_true", help="If given, run as preemptible")
    parser.add_argument("--pure_docker_mode", action="store_true", help="If given, run in pure docker mode")
    parser.add_argument("--no_hf_cache_env", action="store_true", help="Getting deprecated; it does nothing")
    parser.add_argument("--no_mount_nfs", action="store_true", help="Getting deprecated; it does nothing")
    parser.add_argument("--non_resumable", action="store_true", help="If given, disable resumable mode")
    parser.add_argument(
        "--no_auto_dataset_cache", action="store_true", help="If given, don't cache the dataset automatically"
    )
    parser.add_argument(
        "--auto_output_dir_path",
        type=str,
        default="/weka/oe-adapt-default/allennlp/deletable_checkpoint",
        help="If given, automatically replace the `--output_dir` argument with this path, essentially using it as a prefix",
    )
    parser.add_argument(
        "--auto_checkpoint_state_dir",
        type=str,
        default="/weka/oe-adapt-default/allennlp/deletable_checkpoint_states",
        help="If given, automatically replace the `--checkpoint_state_dir` argument with this path, essentially using it as a prefix",
    )
    parser.add_argument(
        "--gs_model_name",
        type=str,
        default=None,
        help="If given, set as the name of the model uploaded to GS for Augusta",
    )
    parser.add_argument(
        "--env",
        type=parse_env_var,
        action="append",
        help="""Additional environment variables in the format 'name=value'.
        Can be specified multiple times. Example: --env MY_VAR=value1 --env OTHER_VAR=value2""",
        default=[],
    )
    parser.add_argument(
        "--secret",
        type=parse_env_var,
        action="append",
        help="""Additional secret env variables in the format 'name=value'.
        Can be specified multiple times. Example: --secret MY_VAR=value1 --secret OTHER_VAR=value2""",
        default=[],
    )
    parser.add_argument(
        "--no-host-networking",
        action="store_true",
        help="If set, don't use host networking in experiment. Note this will make multi-node jobs error.",
    )
    parser.add_argument(
        "--timeout",
        type=str,
        help="Timeout for the Beaker task as a duration string (e.g., '15m', '1h', '2h30m'). If not specified, no timeout is set.",
        default=None,
    )
    # Split up the mason args from the Python args.
    mason_args, command_args = parser.parse_known_args()
    commands = parse_commands(command_args)

    def _commands_include_resumable_target(cmds: list[list[str]]) -> bool:
        for cmd in cmds:
            for target in OPEN_INSTRUCT_RESUMABLES:
                if target in cmd:
                    return True
        return False

    # can resume if the command is in OPEN_INSTRUCT_RESUMABLES and --non_resumable is not set
    is_resumable = _commands_include_resumable_target(commands) and not mason_args.non_resumable
    if not is_resumable and not mason_args.non_resumable:
        console.log(
            "--non_resumable is not set, but the command is not in OPEN_INSTRUCT_RESUMABLES, so the job will not be resumable"
        )
    mason_args.resumable = is_resumable

    return mason_args, commands


global_wandb_id = generate_id()


def parse_commands(command_args: list[str]) -> list[list[str]]:
    """the inputs are ['--', 'which', 'python', '--', 'echo', 'hello'], and this function converts it into [['which', 'python'], ['echo', 'hello']]"""
    if command_args[0] != "--":
        msg = (
            "Please separate the Python command you want to run with ' -- ', like "
            "`mason [mason-args] -- python [python-args]`."
        )
        raise Exception(msg)

    commands = []
    command = []
    for item in command_args:
        if item == "--":
            if command:
                commands.append(command)
                command = []
        else:
            command.append(item)
    if command:
        commands.append(command)
    return commands


def make_internal_command(command: list[str], args: argparse.Namespace, whoami: str, is_external_user: bool) -> str:
    # pass through WANDB_ENTITY and WANDB_PROJECT
    if "WANDB_ENTITY" in os.environ:
        command = [f"WANDB_ENTITY={os.environ['WANDB_ENTITY']}"] + command
    if "WANDB_PROJECT" in os.environ:
        command = [f"WANDB_PROJECT={os.environ['WANDB_PROJECT']}"] + command
    if "WANDB_TAGS" in os.environ:
        command = [f"WANDB_TAGS={os.environ['WANDB_TAGS']}"] + command

    # escape the command (e.g., --stop_strings "</answer>")
    for i in range(len(command)):
        if "</" in command[i]:
            command[i] = f"'{command[i]}'"
    # breakpoint()

    is_open_instruct_training = any(cmd in command for cmd in OPEN_INSTRUCT_COMMANDS)
    if is_open_instruct_training:
        from open_instruct.dataset_transformation import get_commit_hash
        from open_instruct.utils import download_from_hf, gs_folder_exists, upload_to_gs_bucket

        # HACK: Cache dataset logic:
        # Here we basically try to run the tokenization full_command locally before running it on beaker
        # We could in theory submit a cpu only job to beaker to do this, but that requires setting up
        # dependency jobs somehow. Since tokenization is like ~5 minutes, we can just run it locally.
        # Once it's cached, we don't need to cache it again.

        # Add the whoami parts if not already present
        if not any("hf_entity" in c for c in command):
            command.append("--hf_entity")
            command.append("allenai")
        if not any("wandb_entity" in c for c in command):
            command.append("--wandb_entity")
            command.append("ai2-llm")

        dataset_cache_paths = []
        dataset_config_hashes = []
        if not args.no_auto_dataset_cache:
            for file in OPEN_INSTRUCT_COMMANDS:
                try:
                    idx = command.index(file)
                except ValueError:
                    continue

                filtered_command = build_command_without_args(command[idx:], CACHE_EXCLUDED_ARGS)
                filtered_command = maybe_download_tokenizer_from_gs_bucket(
                    filtered_command, args.auto_output_dir_path, whoami
                )
                caching_command = "python " + " ".join(filtered_command) + " --cache_dataset_only"
                console.log("📦📦📦 Running the caching command with `--cache_dataset_only`")
                import subprocess

                # Use Popen to get real-time output while also capturing it
                process = subprocess.Popen(
                    caching_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
                )

                stdout_data, stderr_data = [], []

                # Set up select to monitor both stdout and stderr
                assert process.stdout is not None and process.stderr is not None
                streams = [process.stdout, process.stderr]
                while True:
                    # Wait for output on either stream
                    reads = select.select(streams, [], [])[0]

                    done = True
                    for stream in reads:
                        line = stream.readline()
                        if line:
                            done = False
                            is_stdout = stream == process.stdout
                            print(line.rstrip(), file=sys.stdout if is_stdout else sys.stderr)
                            if is_stdout:
                                stdout_data.append(line)
                            else:
                                stderr_data.append(line)

                    if done and process.poll() is not None:
                        break

                result = type(
                    "SubprocessResult",
                    (),
                    {"returncode": process.returncode, "stdout": "".join(stdout_data), "stderr": "".join(stderr_data)},
                )()
                stdout = result.stdout
                # Extract the cached dataset path from stdout if it exists
                for line in stdout.splitlines():
                    if "✅ Found cached dataset at" in line:
                        dataset_cache_path = line.split("✅ Found cached dataset at")[1].strip()
                        dataset_config_hash = dataset_cache_path.split("/")[-1]
                        console.log(f"📦 Found cached dataset at: {dataset_cache_path}")
                        console.log(f"📦 Found cached dataset config hash: {dataset_config_hash}")
                        dataset_cache_paths.append(dataset_cache_path)
                        dataset_config_hashes.append(dataset_config_hash)
                return_code = result.returncode
                if return_code != 0:
                    raise Exception(f"Error code {return_code} when creating cached dataset")
                console.log("✅✅✅ Finished running the caching command")

                if file in OPEN_INSTRUCT_RESUMABLES and idx != -1 and len(args.auto_checkpoint_state_dir) > 0:
                    need_to_override_checkpoint_state_dir = True
                    default_checkpoint_state_freq = 200
                    for idx, cmd in enumerate(command):
                        if cmd == "--checkpoint_state_dir" and idx + 1 < len(command) and "/weka/" in command[idx + 1]:
                            need_to_override_checkpoint_state_dir = False
                        if cmd == "--checkpoint_state_freq" and idx + 1 < len(command):
                            default_checkpoint_state_freq = command[idx + 1]

                    if need_to_override_checkpoint_state_dir and is_open_instruct_training and not is_external_user:
                        new_checkpoint_state_dir = f"{args.auto_checkpoint_state_dir}/{whoami}/{int(time.time())}_{random.randint(0, 1000000)}"
                        console.log(
                            f"🔍🔍🔍 Automatically overriding the `--checkpoint_state_dir` argument to be in `{new_checkpoint_state_dir}`"
                        )
                        command.append("--checkpoint_state_dir")
                        command.append(new_checkpoint_state_dir)
                        command.append("--checkpoint_state_freq")
                        command.append(str(default_checkpoint_state_freq))

        # For Weka clusters, we need to override the output_dir parameter to make auto-evaluation work
        # If the output_dir is already set to a path in /weka/, we'll keep that path
        # Otherwise, we'll set a default path in the user's directory on Weka
        if any(c in WEKA_CLUSTERS for c in args.cluster):
            if len(args.auto_output_dir_path) > 0:
                need_to_override_output_dir = True
                for idx, cmd in enumerate(command):
                    if cmd == "--output_dir" and "/weka/" in command[idx + 1]:
                        need_to_override_output_dir = False
                        break
                if need_to_override_output_dir and is_open_instruct_training and not is_external_user:
                    new_output_dir = f"{args.auto_output_dir_path}/{whoami}/"
                    console.log(
                        f"🔍🔍🔍 Automatically overriding the `--output_dir` argument to be in `{new_output_dir}`"
                    )
                    command.append("--output_dir")
                    command.append(new_output_dir)
            else:
                no_eval_commands = [
                    ["--try_launch_beaker_eval_jobs", "False"],
                    ["--try_launch_beaker_eval_jobs_on_weka", "False"],
                    ["--no_try_launch_beaker_eval_jobs"],
                    ["--no_try_launch_beaker_eval_jobs_on_weka"],
                ]
                no_eval_concat_commands = [" ".join(cmd) for cmd in no_eval_commands]
                no_eval_concat_command_exists = any(cmd in command for cmd in no_eval_concat_commands)
                if not no_eval_concat_command_exists:
                    raise ValueError(
                        "To auto-evaluation is turned on by default, to make sure it works, you must:\n"
                        "1. run mason with`--auto_output_dir_path /weka/...`, or\n"
                        "2. in the training command, disable auto-evaluation with `--no_try_launch_beaker_eval_jobs`, or\n"
                        "3. in the training command, use a `--output_dir` that starts with `/weka/`"
                    )

        # For GCP clusters, since shared storage is slow, we optimize model loading by:
        if any(c in GCP_CLUSTERS for c in args.cluster):
            # 1. First downloading the model from HuggingFace to a local path
            # 2. Uploading it to a Google Storage bucket (if not already there)
            # 3. Then downloading it from the bucket to the compute node
            # 4. Finally, replacing the original --model_name_or_path argument with the local path
            model_name_or_path = None
            for idx, cmd in enumerate(command):
                if cmd == "--model_name_or_path":
                    model_name_or_path = command[idx + 1]
                    break
            model_revision = "main"
            for idx, cmd in enumerate(command):
                if cmd == "--model_revision":
                    model_revision = command[idx + 1]
                    break

            if model_name_or_path is None:
                raise ValueError("--model_name_or_path is required for GCP clusters")

            if model_name_or_path.startswith("gs://"):
                gs_saved_path = model_name_or_path
            else:
                commit_hash = get_commit_hash(model_name_or_path, model_revision, "config.json", "model")
                if os.path.exists(model_name_or_path):
                    path = model_name_or_path
                    assert args.gs_model_name is not None, (
                        "for local models to upload to gs, you must set --gs_model_name"
                    )
                    model_name_or_path = args.gs_model_name
                    # get the short commit hash (first 8 chars)
                    commit_hash = hashlib.md5(model_name_or_path.encode("utf-8")).hexdigest()[:8]
                    console.log(
                        f"Local model is already downloaded, using gs_model_name {model_name_or_path}, with hash of model path {commit_hash}"
                    )
                else:
                    download_from_hf(model_name_or_path, model_revision)  # first download the model
                    path = download_from_hf(model_name_or_path, model_revision)  # then get the path
                gs_saved_path = f"gs://ai2-llm/post-training/deletable_cache_models/{model_name_or_path}/{commit_hash}"
                gs_folder = gs_folder_exists(
                    gs_saved_path
                )  # race condition exists, but it's fine since we are launching mason sequentially
                if not gs_folder:
                    upload_to_gs_bucket(path, gs_saved_path)  # ty: ignore[invalid-argument-type]

            download_path = gs_saved_path.replace("gs://", "/gs/")
            download_path_without_last_folder = download_path.rsplit("/", 1)[0]
            gs_download_command = [
                "mkdir",
                "-p",
                download_path,
                "&&",
                "gsutil",
                "-o",
                "GSUtil:parallel_thread_count=1",
                "-o",
                "GSUtil:sliced_object_download_threshold=150",
                "-m",
                "cp",
                "-r",
                gs_saved_path,
                download_path_without_last_folder,
                "&&",
                "ls",
                download_path_without_last_folder,
                "&&",
                "ls",
                download_path,
                "&&",
            ]

            command.append("--gs_bucket_path")
            command.append("gs://ai2-llm/post-training/")

            # Replace the model_name_or_path with the downloaded path
            for idx, cmd in enumerate(command):
                if cmd == "--model_name_or_path":
                    command[idx + 1] = download_path
                    break
            for idx, cmd in enumerate(command):
                if cmd == "--model_revision":
                    command[idx + 1] = "main"
                    break

            # Save dataset to GCS
            if len(dataset_cache_paths) > 0:
                for cidx, (dataset_cache_path, dataset_config_hash) in enumerate(
                    zip(dataset_cache_paths, dataset_config_hashes)
                ):
                    gs_saved_path = f"gs://ai2-llm/post-training/deletable_cache_datasets/{dataset_cache_path}"
                    gs_folder = gs_folder_exists(
                        gs_saved_path
                    )  # race condition exists, but it's fine since we are launching mason sequentially
                    if not gs_folder:
                        upload_to_gs_bucket(dataset_cache_path, gs_saved_path)
                    dataset_cache_path_without_last_folder = dataset_cache_path.rsplit("/", 1)[0]
                    gs_download_command += [
                        "mkdir",
                        "-p",
                        dataset_cache_path_without_last_folder,
                        "&&",
                        "gsutil",
                        "cp",
                        "-r",
                        gs_saved_path,
                        dataset_cache_path_without_last_folder,
                        "&&",
                        "ls",
                        dataset_cache_path_without_last_folder,
                        "&&",
                        "ls",
                        dataset_cache_path,
                        "&&",
                    ]
                    if cidx == 0:
                        command.append("--dataset_config_hash")
                        command.append(dataset_config_hash)
                    elif cidx == 1:
                        command.append("--dataset_config_eval_hash")
                        command.append(dataset_config_hash)
            command = gs_download_command + command

    # special logic to deal with escape like
    # python mason.py ... -- python x.py --dataset_mixer '{"trl-internal-testing/sentiment-trl-style": 1.0}'
    # we need to wrap the json string with single quote
    for idx in range(len(command)):
        if "{" in command[idx]:
            command[idx] = "'" + command[idx] + "'"
    full_command = command
    setup_commands = ""
    if not args.pure_docker_mode:
        setup_commands = f"cd {os.getcwd()} && "

    join_full_command = " ".join(full_command)
    # override accelerate call
    if args.num_nodes > 1:
        if "--num_processes" not in join_full_command and "accelerate" in join_full_command:
            raise ValueError("num_processes must be specified in the command for accelerate-based multi-node jobs.")
        join_full_command = re.sub(
            r"--num_processes (\d+)",
            lambda m: (
                f"--num_processes {int(m.group(1)) * args.num_nodes} "
                f"--num_machines {args.num_nodes} "
                "--machine_rank $BEAKER_REPLICA_RANK "
                "--main_process_ip $BEAKER_LEADER_REPLICA_HOSTNAME "
                "--main_process_port 29400 "
            ),
            join_full_command,
        )
    full_command = setup_commands + join_full_command
    console.log("🔍🔍🔍 Full command")
    print(full_command)
    return full_command


def make_task_spec(args, full_command: str, i: int, beaker_secrets: list[str], whoami: str, resumable: bool):
    # Add a check to ensure that the user is using the correct clusters for multi-node jobs
    if args.num_nodes > 1 and not all(c in INTERCONNECT_CLUSTERS for c in args.cluster):
        confirmation = False
        while not confirmation:
            confirmation = input(
                "Interconnect clusters are required for multi-node jobs. Are you sure you want to continue? (y/n)"
            )
            if confirmation == "y":
                confirmation = True
            elif confirmation == "n":
                raise ValueError(
                    f"Interconnect clusters are required for multi-node jobs; please only use the following clusters: {INTERCONNECT_CLUSTERS}"
                )
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    if args.image == "ai2/cuda11.8-cudnn8-dev-ubuntu20.04" and any(c in GCP_CLUSTERS for c in args.cluster):
        raise ValueError("GCP clusters do not have the dev filesystem, please use a proper image")

    if args.hostname is not None:
        constraints = beaker.BeakerConstraints(hostname=args.hostname)
    else:
        constraints = beaker.BeakerConstraints(cluster=args.cluster)
    spec = beaker.BeakerTaskSpec(
        name=f"{args.task_name}__{i}",
        image=beaker.BeakerImageSource(beaker=args.image),
        command=["/bin/bash", "-c"],
        arguments=[full_command],
        result=beaker.BeakerResultSpec(path="/output"),
        datasets=get_datasets(args.beaker_datasets, args.cluster),
        context=beaker.BeakerTaskContext(
            priority=beaker.BeakerJobPriority[args.priority], preemptible=args.preemptible
        ),
        constraints=constraints,
        env_vars=get_env_vars(
            args.pure_docker_mode,
            args.cluster,
            beaker_secrets,
            whoami,
            resumable,
            args.num_nodes,
            args.env,
            args.secret,
            global_wandb_id,
        ),
        resources=beaker.BeakerTaskResources(gpu_count=args.gpus, shared_memory=args.shared_memory),
        replicas=args.num_nodes,
        timeout=args.timeout,
    )
    if args.num_nodes > 1:
        spec.leader_selection = True
        spec.propagate_failure = True
        spec.propagate_preemption = True
    if args.no_host_networking:
        spec.host_networking = False
    else:
        spec.host_networking = True

    return spec


def maybe_download_tokenizer_from_gs_bucket(filtered_command: str, auto_output_dir_path: str, whoami: str):
    """if model is only on gs, download tokenizer from gs to local cache folder for dataset preprocessing"""

    if "--model_name_or_path" not in filtered_command:
        return filtered_command

    model_arg_idx = filtered_command.index("--model_name_or_path")
    model_name_idx = model_arg_idx + 1
    model_name_or_path = filtered_command[model_name_idx].rstrip("/")

    if not model_name_or_path.startswith("gs://"):
        return filtered_command

    model_name_hash = hashlib.md5(model_name_or_path.encode("utf-8")).hexdigest()[:8]
    local_cache_folder = f"{auto_output_dir_path}/{whoami}/tokenizer_{model_name_hash}/"

    if not os.path.exists(local_cache_folder):
        download_from_gs_bucket(
            [
                f"{model_name_or_path}/tokenizer.json",
                f"{model_name_or_path}/tokenizer_config.json",
                f"{model_name_or_path}/config.json",
            ],
            local_cache_folder,
        )

    filtered_command[model_name_idx] = local_cache_folder

    return filtered_command


def main():
    args, commands = get_args()
    # If the user is not in Ai2, we run the command as is
    config_path = os.path.expanduser("~/.beaker/config.yml")
    is_external_user = not os.path.exists(config_path) and "BEAKER_TOKEN" not in os.environ
    if is_external_user:
        whoami = "external_user"
        beaker_secrets = []
    else:
        if args.workspace:
            beaker_client = beaker.Beaker.from_env(default_workspace=args.workspace)
        else:
            beaker_client = beaker.Beaker.from_env()
        beaker_secrets = [secret.name for secret in beaker_client.secret.list()]
        whoami = beaker_client.user.get().name

        # Increase timeout to 300s for large experiment specs.
        beaker.Beaker.TIMEOUT = 300

    full_commands = [make_internal_command(command, args, whoami, is_external_user) for command in commands]
    if is_external_user:
        console.rule("[bold red]Non-Ai2 User Detected[/bold red]")
        console.print(
            Text(
                (
                    "👋 Hi external user! The following command will be executed in our internal server; feel free to modify it to your needs. "
                    '(For example, you might need to replace `"$BEAKER_LEADER_REPLICA_HOSTNAME"` with your own hostname)'
                ),
                style="bold",
            )
        )
    for idx, full_command in enumerate(full_commands):
        console.rule(f"[bold blue]Command {idx + 1}[/bold blue]")
        console.print(Text(full_command))
    if is_external_user:
        return
    experiment_spec = beaker.BeakerExperimentSpec(
        description=args.description,
        tasks=[
            make_task_spec(args, full_command, i, beaker_secrets, whoami, args.resumable)
            for i, full_command in enumerate(full_commands)
        ],
        budget=args.budget,
        retry=beaker.BeakerRetrySpec(allowed_task_retries=args.max_retries),
    )

    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.Timeout,
        max_tries=5,
        # Factor here is the multiplier for the backoff delay, in seconds.
        factor=5,
    )
    def launch_experiment():
        exp = beaker_client.experiment.create(spec=experiment_spec)
        console.log(f"Kicked off Beaker job. https://beaker.org/ex/{exp.experiment.id}")
        return exp

    launch_experiment()


if __name__ == "__main__":
    main()
