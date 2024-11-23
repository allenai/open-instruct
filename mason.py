import argparse
import re
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

NFS_CLUSTERS = [
    "ai2/allennlp-cirrascale",
    "ai2/aristo-cirrascale",
    "ai2/climate-cirrascale",
    "ai2/general-cirrascale",
    "ai2/general-cirrascale-a5000",
    "ai2/mosaic-cirrascale",
    "ai2/mosaic-cirrascale-a100",
    "ai2/pluto-cirrascale",
    "ai2/prior-cirrascale",
    "ai2/s2-cirrascale",
    "ai2/s2-cirrascale-l40",
]

WEKA_CLUSTERS = [
    "ai2/jupiter-cirrascale-2",
    "ai2/saturn-cirrascale",
    "ai2/neptune-cirrascale",
    "ai2/allennlp-elara-cirrascale",

]
GCP_CLUSTERS = [
    "ai2/augusta-google-1"
]


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
    parser.add_argument("--num_nodes", type=int, help="Number of nodes", default=1)
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


def get_env_vars(pure_docker_mode: bool, cluster: List[str], beaker_secrets: List[str], whoami: str, resumable: bool, num_nodes: int):
    env_vars = []
    useful_secrets = [
        "HF_TOKEN",
        "WANDB_API_KEY",
        "BEAKER_TOKEN",
        "OPENAI_API_KEY",
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
    
    # if none of the cluster is in weka, we mount the NFS
    if all(c in NFS_CLUSTERS for c in cluster):
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
        if len(cluster) == 1 and "ai2/pluto-cirrascale" in cluster:
            env_vars.extend([
                beaker.EnvVar(
                    name="NCCL_IB_HCA",
                    value="^=mlx5_1,mlx5_2",
                ),
                beaker.EnvVar(
                    name="NCCL_DEBUG",
                    value="INFO",
                ),
            ])
    # if all cluster is in weka, we mount the weka
    elif all(c in WEKA_CLUSTERS for c in cluster):
        env_vars.extend([
            beaker.EnvVar(
                name="HF_HOME",
                value="/weka/oe-adapt-default/allennlp/.cache/huggingface",
            ),
            beaker.EnvVar(
                name="HF_DATASETS_CACHE",
                value="/weka/oe-adapt-default/allennlp/.cache/huggingface",
            ),
            beaker.EnvVar(
                name="HF_HUB_CACHE",
                value="/weka/oe-adapt-default/allennlp/.cache/hub",
            ),
            beaker.EnvVar(
                name="CHECKPOINT_OUTPUT_DIR",
                value=f"/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/{global_wandb_id}",
            ),
        ])
        if num_nodes > 1:
            env_vars.extend([
                beaker.EnvVar(
                    name="NCCL_SOCKET_IFNAME",
                    value="ib",
                ),
                beaker.EnvVar(
                    name="NCCL_IB_HCA",
                    value="^=mlx5_bond_0",
                ),
                beaker.EnvVar(
                    name="NCCL_DEBUG",
                    value="INFO",
                ),
            ])
    # if all cluster is in gcp we add the following env

    elif all(c in GCP_CLUSTERS for c in cluster):
        if num_nodes > 1:
            env_vars.extend([
                beaker.EnvVar(
                    name="LD_LIBRARY_PATH",
                    value=r"/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH}",
                ),
                beaker.EnvVar(
                    name="NCCL_CROSS_NIC",
                    value="0",
                ),
                beaker.EnvVar(
                    name="NCCL_ALGO",
                    value="Ring,Tree",
                ),
                beaker.EnvVar(
                    name="NCCL_PROTO",
                    value="Simple",
                ),
                beaker.EnvVar(
                    name="NCCL_MIN_NCHANNELS",
                    value="4",
                ),
                beaker.EnvVar(
                    name="NCCL_P2P_NET_CHUNKSIZE",
                    value="524288",
                ),
                beaker.EnvVar(
                    name="NCCL_P2P_PCI_CHUNKSIZE",
                    value="524288",
                ),
                beaker.EnvVar(
                    name="NCCL_P2P_NVL_CHUNKSIZE",
                    value="1048576",
                ),
                beaker.EnvVar(
                    name="NCCL_FASTRAK_NUM_FLOWS",
                    value="2",
                ),
                beaker.EnvVar(
                    name="NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL",
                    value="0",
                ),
                beaker.EnvVar(
                    name="NCCL_BUFFSIZE",
                    value="8388608",
                ),
                beaker.EnvVar(
                    name="NCCL_FASTRAK_USE_SNAP",
                    value="1",
                ),
                beaker.EnvVar(
                    name="CUDA_VISIBLE_DEVICES",
                    value="0,1,2,3,4,5,6,7",
                ),
                beaker.EnvVar(
                    name="NCCL_NET_GDR_LEVEL",
                    value="PIX",
                ),
                beaker.EnvVar(
                    name="NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING",
                    value="0",
                ),
                beaker.EnvVar(
                    name="NCCL_TUNER_PLUGIN",
                    value="libnccl-tuner.so",
                ),
                beaker.EnvVar(
                    name="NCCL_TUNER_CONFIG_PATH",
                    value="/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto",
                ),
                beaker.EnvVar(
                    name="NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE",
                    value="/var/lib/tcpxo/lib64/a3plus_guest_config.textproto",
                ),
                beaker.EnvVar(
                    name="NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS",
                    value="600000",
                ),
                beaker.EnvVar(
                    name="NCCL_NVLS_ENABLE",
                    value="0",
                ),
                beaker.EnvVar(
                    name="NCCL_DEBUG",
                    value="WARN",
                ),
                beaker.EnvVar(
                    name="NCCL_FASTRAK_CTRL_DEV",
                    value="enp0s12",
                ),
                beaker.EnvVar(
                    name="NCCL_FASTRAK_IFNAME",
                    value="enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0",
                ),
                beaker.EnvVar(
                    name="NCCL_SOCKET_IFNAME",
                    value="enp0s12",
                ),
                beaker.EnvVar(
                    name="NCCL_USE_SNAP",
                    value="1",
                ),
                beaker.EnvVar(
                    name="NCCL_FASTRAK_USE_LLCM",
                    value="1",
                ),
                beaker.EnvVar(
                    name="NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY",
                    value="/dev/aperture_devices",
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
    # if none of the cluster is in weka, we mount the NFS
    if all(c in NFS_CLUSTERS for c in cluster):
        res = [
            beaker.DataMount(
                source=beaker.DataSource(host_path="/net/nfs.cirrascale"),
                mount_path="/net/nfs.cirrascale",
            ),
        ]
    # if all cluster is in weka, we mount the weka
    elif all(c in WEKA_CLUSTERS for c in cluster):
        res = [
            beaker.DataMount(
                source=beaker.DataSource(weka="oe-adapt-default"),
                mount_path="/weka/oe-adapt-default",
            ),
            beaker.DataMount(
                source=beaker.DataSource(weka="oe-training-default"),
                mount_path="/weka/oe-training-default",
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
        "echo 'Running on host: $BEAKER_REPLICA_RANK' && "
        "echo 'Running on host: $BEAKER_LEADER_REPLICA_HOSTNAME' && "
        "git config --global safe.directory '*' && " # fix the permission issue with git
        "umask 000 && " # fix the permission issue with the cache folder
    )
    if not args.pure_docker_mode:
        setup_commands += f"cd {os.getcwd()} && "

    join_full_command = " ".join(full_command)
    # override accelerate call
    if args.num_nodes > 1:
        join_full_command = re.sub(
            r'--num_processes (\d+)',
            lambda m: (
                f'--num_processes {int(m.group(1)) * args.num_nodes} '
                f'--num_machines {args.num_nodes} '
                '--machine_rank $BEAKER_REPLICA_RANK '
                '--main_process_ip $BEAKER_LEADER_REPLICA_HOSTNAME '
                '--main_process_port 29400 '
            ),
            join_full_command
        )
    full_command = setup_commands + join_full_command
    print(f"{full_command=}")


    spec = beaker.TaskSpec(
        name=f"{args.task_name}__{i}",
        image=beaker.ImageSource(beaker=args.image),
        command=command,
        arguments=[full_command],
        result=beaker.ResultSpec(path="/output"),
        datasets=get_datasets(args.beaker_datasets, args.cluster),
        context=beaker.TaskContext(priority=beaker.Priority(args.priority),
                                   preemptible=args.preemptible),
        constraints=beaker.Constraints(cluster=args.cluster),
        env_vars=get_env_vars(args.pure_docker_mode, args.cluster, beaker_secrets, whoami, resumable, args.num_nodes),
        resources=beaker.TaskResources(gpu_count=args.gpus),
        replicas=args.num_nodes,
    )
    if args.num_nodes > 1:
        spec.leader_selection = True
        spec.host_networking = True
        spec.propagate_failure = True
        spec.propagate_preemption = True

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