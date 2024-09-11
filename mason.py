import argparse
from typing import List
import beaker
import os
import secrets
import string

def parse_beaker_dataset(dataset_str):
    splt = dataset_str.split(":")
    if len(splt) != 2:
        raise argparse.ArgumentError()

    return {"mount_path": splt[0], "beaker": splt[1]}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster",
        type=str,
        nargs="+",
        help="Beaker clusters on which the job could be run.",
        required=True,
    )
    parser.add_argument("--budget", type=str, help="Budget to use.", required=True)
    parser.add_argument("--gpus", type=int, help="Number of gpus", default=0)
    parser.add_argument(
        "--image",
        type=str,
        help="Beaker base image; usually fine to use AI2 base image.",
        default="ai2/cuda11.8-cudnn8-dev-ubuntu20.04",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="The Beaker workspace to use. If not set, use your default.",
        default=None,
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
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name for the Beaker task.",
        default="beaker_mason"
    )
    parser.add_argument(
        "--priority", type=str, help="Beaker job priority.", default="normal"
    )
    parser.add_argument(
        "--preemptible", action="store_true", help="If given, run as preemptible"
    )
    parser.add_argument(
        "--pure_docker_mode", action="store_true", help="If given, run in pure docker mode"
    )
    parser.add_argument(
        "--no_hf_cache_env", action="store_true", help="Getting deprecated; it does nothing"
    )
    parser.add_argument(
        "--no_mount_nfs", action="store_true", help="Getting deprecated; it does nothing"
    )
    parser.add_argument(
        "--resumable", action="store_true", help="If given, make the job resumable"
    )


    # Split up the mason args from the Python args.
    mason_args, command_args = parser.parse_known_args()
    commands = parse_commands(command_args)
    return mason_args, commands


def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


global_wandb_id = generate_id()


def parse_commands(command_args: List[str]) -> List[List[str]]:
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


def get_env_vars(pure_docker_mode, cluster: List[str], beaker_secrets, whoami, resumable):
    env_vars = []
    useful_secrets = [
        "HF_TOKEN",
        "WANDB_API_KEY",
        "BEAKER_TOKEN",
    ]
    for useful_secret in useful_secrets:
        if f"{whoami}_{useful_secret}" in beaker_secrets:
            env_vars.append(
                beaker.EnvVar(
                    name=useful_secret,
                    secret=f"{whoami}_{useful_secret}",
                )
            )
        elif useful_secret in beaker_secrets:
            env_vars.append(
                beaker.EnvVar(
                    name=useful_secret,
                    secret=useful_secret,
                )
            )

     # use the user's PATH; including the conda / python PATH
    if not pure_docker_mode:
        env_vars.extend([
            beaker.EnvVar(
                name="PATH",
                value=os.getenv("PATH"),
            ),
        ])
    
    # if we are not running on jupiter2, we try to mount the NFS
    if "ai2/jupiter-cirrascale-2" not in cluster:
        env_vars.extend([
            beaker.EnvVar(
                name="HF_DATASETS_CACHE",
                value="/net/nfs.cirrascale/allennlp/.cache/huggingface",
            ),
            beaker.EnvVar(
                name="HF_HUB_CACHE",
                value="/net/nfs.cirrascale/allennlp/.cache/hub",
            ),
            beaker.EnvVar(
                name="HF_ASSETS_CACHE",
                value="/net/nfs.cirrascale/allennlp/.cache/assets",
            ),
            beaker.EnvVar(
                name="CHECKPOINT_OUTPUT_DIR",
                value=f"/net/nfs.cirrascale/allennlp/deletable_checkpoint_states/{global_wandb_id}",
            ),
        ])
    # if we only run on jupiter 2, we try to mount weka
    elif len(cluster) == 1 and "ai2/jupiter-cirrascale-2" in cluster:
        env_vars.extend([
            beaker.EnvVar(
                name="HF_DATASETS_CACHE",
                value="/weka/allennlp/.cache/huggingface",
            ),
            beaker.EnvVar(
                name="HF_HUB_CACHE",
                value="/weka/allennlp/.cache/hub",
            ),
            beaker.EnvVar(
                name="CHECKPOINT_OUTPUT_DIR",
                value=f"/weka/allennlp/deletable_checkpoint_states/{global_wandb_id}",
            ),
        ])
    # don't mount anything; assume no cache
    else:
        pass

    if resumable:
        env_vars.extend([
            beaker.EnvVar(
                name="WANDB_RUN_ID",
                value=global_wandb_id,
            ),
            beaker.EnvVar(
                name="WANDB_RESUME",
                value="allow",
            ),
        ])

    return env_vars


def get_datasets(beaker_datasets, cluster: List[str]):
    """if pure docker mode we don't mount the NFS; so we can run it on jupiter2"""
    res = []
    # if we are not running on jupiter2, we try to mount the NFS
    if "ai2/jupiter-cirrascale-2" not in cluster:
        res = [
            beaker.DataMount(
                source=beaker.DataSource(host_path="/net/nfs.cirrascale"),
                mount_path="/net/nfs.cirrascale",
            ),
        ]
    # if we only run on jupiter 2, we try to mount weka
    elif len(cluster) == 1 and "ai2/jupiter-cirrascale-2" in cluster:
        res = [
            beaker.DataMount(
                source=beaker.DataSource(weka="oe-adapt-default"),
                mount_path="/weka",
            ),
        ]
    for beaker_dataset in beaker_datasets:
        to_append = beaker.DataMount(
            source=beaker.DataSource(beaker=beaker_dataset["beaker"]),
            mount_path=beaker_dataset["mount_path"],
        )
        res.append(to_append)

    return res


def make_task_spec(args, command, i, beaker_secrets, whoami, resumable: bool):
    # special logic to deal with escape like
    # python mason.py ... -- python x.py --dataset_mixer '{"trl-internal-testing/sentiment-trl-style": 1.0}'
    # we need to wrap the json string with single quote
    for idx in range(len(command)):
        if "{" in command[idx]:
            command[idx] = "'" + command[idx] + "'"
    full_command = command
    command = ['/bin/bash', '-c']
    setup_commands = (
        "git config --global safe.directory '*' && " # fix the permission issue with git
        "umask 000 && " # fix the permission issue with the cache folder
    )
    if not args.pure_docker_mode:
        setup_commands += f"cd {os.getcwd()} && "
    fully_command = setup_commands + " ".join(full_command)
    print(f"{full_command=}")


    spec = beaker.TaskSpec(
        name=f"{args.task_name}__{i}",
        image=beaker.ImageSource(beaker=args.image),
        command=command,
        arguments=[fully_command],
        result=beaker.ResultSpec(path="/output"),
        datasets=get_datasets(args.beaker_datasets, args.cluster),
        context=beaker.TaskContext(priority=beaker.Priority(args.priority),
                                   preemptible=args.preemptible),
        constraints=beaker.Constraints(cluster=args.cluster),
        env_vars=get_env_vars(args.pure_docker_mode, args.cluster, beaker_secrets, whoami, resumable),
        resources=beaker.TaskResources(gpu_count=args.gpus),
    )

    return spec


def main():
    args, commands = get_args()
    if args.workspace:
        beaker_client = beaker.Beaker.from_env(default_workspace=args.workspace)
    else:
        beaker_client = beaker.Beaker.from_env()

    beaker_secrets = [secret.name for secret in beaker_client.workspace.secrets()]
    whoami = beaker_client.account.whoami().name
    experiment_spec = beaker.ExperimentSpec(
        description=args.description,
        tasks=[make_task_spec(args, command, i, beaker_secrets, whoami, args.resumable) for i, command in enumerate(commands)],
        budget=args.budget,
    )

    exp = beaker_client.experiment.create(spec=experiment_spec)
    print(f"Kicked off Beaker job. https://beaker.org/ex/{exp.id}")


if __name__ == "__main__":
    main()