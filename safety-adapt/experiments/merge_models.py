"""
Merge together two model checkpoints. Useful in combination with continued finetuning.
"""

from argparse import ArgumentParser

import beaker
from beaker import Beaker, ExperimentSpec, TaskSpec


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_1_name", type=str, help="Name of first model.")
    parser.add_argument(
        "--model_1_dataset", type=str, help="Beaker dataset for first model."
    )
    parser.add_argument("--model_2_name", type=str, help="Name of second model.")
    parser.add_argument(
        "--model_2_dataset", type=str, help="Beaker dataset for second model."
    )
    parser.add_argument("--config_file", type=str, help="Yaml config file to use.")

    return parser.parse_args()


args = get_args()

conda_root = "/net/nfs.cirrascale/allennlp/davidw/miniconda3"

conda_command = [
    f"{conda_root}/bin/conda",
    "run",
    "--no-capture-output",
    "-n",
    "science-adapt",
]
merge_command = ["mergekit-yaml", args.config_file, "/output/"]

command = conda_command + merge_command

####################

# Beaker datasets

datasets = [  # Magic to get NFS mounted.
    beaker.DataMount(
        source=beaker.DataSource(host_path="/net/nfs.cirrascale"),
        mount_path="/net/nfs.cirrascale",
    ),
    beaker.DataMount(
        source=beaker.DataSource(beaker=args.model_1_dataset),
        mount_path="/model_1",
    ),
    beaker.DataMount(
        source=beaker.DataSource(beaker=args.model_2_dataset),
        mount_path="/model_2",
    ),
]


####################

# Environment variables

env_vars = [
    beaker.EnvVar(name="NFS_HOME", value="/net/nfs.cirrascale/allennlp/davidw"),
    beaker.EnvVar(
        name="HF_HOME",
        value="/net/nfs.cirrascale/allennlp/davidw/cache/huggingface",
    ),
    beaker.EnvVar(
        name="CONDA_ENVS_DIRS",
        value=f"{conda_root}/envs",
    ),
    beaker.EnvVar(
        name="CONDA_PKGS_DIRS",
        value=f"{conda_root}/pkgs",
    ),
]


####################

# Run the task.


tasks = [
    TaskSpec(
        name=f"merge-{args.model_1_name}-{args.model_2_name}",
        image=beaker.ImageSource(beaker="ai2/cuda11.8-cudnn8-dev-ubuntu20.04"),
        command=command,
        result=beaker.ResultSpec(path="/output"),
        datasets=datasets,
        context=beaker.TaskContext(priority=beaker.Priority("high")),  # Priority.
        constraints=beaker.Constraints(cluster=["ai2/s2-cirrascale-l40"]),
        env_vars=env_vars,
        resources=beaker.TaskResources(
            cpu_count=16
        ),
    ),
]

spec = ExperimentSpec(
    description=f"Merge models {args.model_1_name} and {args.model_2_name}",
    tasks=tasks,
)
workspace_name = "ai2/science-adapt"

# Make the experiment and run it.
beaker_client = Beaker.from_env(default_workspace=workspace_name)
beaker_client.experiment.create(
    spec=spec,
    workspace=workspace_name,
)
