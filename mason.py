import argparse
import beaker
import os
from pathlib import Path
import sys


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

    # Split up the mason args from the Python args.
    mason_args, command_args = parser.parse_known_args()

    if command_args[0] != "--":
        msg = (
            "Please separate the Python command you want to run with ' -- ', like "
            "`mason [mason-args] -- python [python-args]`."
        )
        raise Exception(msg)

    return mason_args, command_args[1:]


def get_env_vars():
    # conda_exe = Path(os.getenv("CONDA_EXE"))
    # conda_root = conda_exe.parent.parent
    env_vars = [
        beaker.EnvVar(
            name="PATH",
            value=os.getenv("PATH"),
        ),
        beaker.EnvVar(
            name="HF_DATASETS_CACHE",
            value=os.getenv("HF_DATASETS_CACHE"),
        ),
        beaker.EnvVar(
            name="HF_HUB_CACHE",
            value=os.getenv("HF_HUB_CACHE"),
        ),
        beaker.EnvVar(
            name="HF_ASSETS_CACHE",
            value=os.getenv("HF_ASSETS_CACHE"),
        ),
        beaker.EnvVar(
            name="HF_TOKEN",
            secret="HF_TOKEN",
        ),
        beaker.EnvVar(
            name="WANDB_API_KEY",
            secret="WANDB_API_KEY",
        ),
    ]

    return env_vars


def get_datasets(beaker_datasets):
    res = [
        beaker.DataMount(
            source=beaker.DataSource(host_path="/net/nfs.cirrascale"),
            mount_path="/net/nfs.cirrascale",
        ),
    ]
    for beaker_dataset in beaker_datasets:
        to_append = beaker.DataMount(
            source=beaker.DataSource(beaker=beaker_dataset["beaker"]),
            mount_path=beaker_dataset["mount_path"],
        )
        res.append(to_append)

    return res


def make_task_spec(args, command):
    full_command = command
    command = ['/bin/bash', '-c']
    # make the following commmand more generalizable to different users
    # source /net/nfs.cirrascale/allennlp/costa/.bashrc && 
    fully_command = f"git config --global safe.directory '*' && cd {os.getcwd()} &&" + " ".join(full_command)
    print(f"{full_command=}")


    spec = beaker.TaskSpec(
        name=args.task_name,
        image=beaker.ImageSource(beaker=args.image),
        command=command,
        arguments=[fully_command],
        result=beaker.ResultSpec(path="/unused"),
        datasets=get_datasets(args.beaker_datasets),
        context=beaker.TaskContext(priority=beaker.Priority(args.priority)),
        constraints=beaker.Constraints(cluster=args.cluster),
        env_vars=get_env_vars(),
        resources=beaker.TaskResources(gpu_count=args.gpus),
    )

    return spec


def main():
    args, command = get_args()
    task_spec = make_task_spec(args, command)
    experiment_spec = beaker.ExperimentSpec(
        description=args.description, tasks=[task_spec], budget=args.budget
    )
    if args.workspace:
        beaker_client = beaker.Beaker.from_env(default_workspace=args.workspace)
    else:
        beaker_client = beaker.Beaker.from_env()

    exp = beaker_client.experiment.create(spec=experiment_spec)
    print(f"Kicked off Beaker job. https://beaker.org/ex/{exp.id}")


if __name__ == "__main__":
    main()