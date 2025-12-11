import argparse
import base64
import dataclasses
import json
import os
import secrets
import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import backoff
import beaker
import requests
from rich.console import Console

from open_instruct.utils import GCP_CLUSTERS, INTERCONNECT_CLUSTERS, WEKA_CLUSTERS

if TYPE_CHECKING:
    from open_instruct.grpo_fast import ExperimentConfig

console = Console()

DEFAULT_ENV_VARS = {
    "RAY_CGRAPH_get_timeout": "300",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "NCCL_DEBUG": "ERROR",
    "VLLM_LOGGING_LEVEL": "WARNING",
    "VLLM_USE_V1": "1",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
}


@dataclass
class LaunchConfig:
    cluster: list[str]
    budget: str
    image: str = "ai2/cuda11.8-cudnn8-dev-ubuntu20.04"
    description: str = "Beaker-Mason job."
    workspace: str | None = None
    hostname: list[str] | None = None
    max_retries: int = 0
    gpus: int = 0
    shared_memory: str = "10.24gb"
    num_nodes: int = 1
    task_name: str = "beaker_mason"
    priority: str = "normal"
    preemptible: bool = False
    pure_docker_mode: bool = False
    non_resumable: bool = False
    no_auto_dataset_cache: bool = False
    auto_output_dir_path: str = "/weka/oe-adapt-default/allennlp/deletable_checkpoint"
    auto_checkpoint_state_dir: str = "/weka/oe-adapt-default/allennlp/deletable_checkpoint_states"
    gs_model_name: str | None = None
    beaker_datasets: list[dict[str, str]] = field(default_factory=list)
    env: list[dict[str, str]] = field(default_factory=list)
    secret: list[dict[str, str]] = field(default_factory=list)
    no_host_networking: bool = False
    timeout: str | None = None
    resumable: bool = True


def generate_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def parse_beaker_dataset(dataset_str: str) -> dict[str, str]:
    splt = dataset_str.split(":")
    if len(splt) != 2:
        raise argparse.ArgumentTypeError(f"Invalid dataset format: {dataset_str}. Expected 'mount_path:beaker_id'")
    return {"mount_path": splt[0], "beaker": splt[1]}


def parse_env_var(env_var_str: str) -> dict[str, str]:
    if "=" not in env_var_str:
        raise argparse.ArgumentTypeError(f"Environment variable must be in format 'name=value', got: {env_var_str}")
    name, value = env_var_str.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Environment variable name cannot be empty")
    return {"name": name, "value": value}


def get_env_vars(
    pure_docker_mode: bool,
    cluster: list[str],
    beaker_secrets: list[str],
    whoami: str,
    resumable: bool,
    num_nodes: int,
    additional_env_vars: list[dict[str, str]],
    additional_secrets: list[dict[str, str]],
    wandb_id: str,
):
    additional_env_var_names = {var["name"] for var in additional_env_vars}

    env_vars = [
        beaker.BeakerEnvVar(name=name, value=value)
        for name, value in DEFAULT_ENV_VARS.items()
        if name not in additional_env_var_names
    ]

    env_vars.extend(
        [beaker.BeakerEnvVar(name=env_var["name"], value=env_var["value"]) for env_var in additional_env_vars]
    )

    env_vars.extend(
        [beaker.BeakerEnvVar(name=secret["name"], secret=secret["value"]) for secret in additional_secrets]
    )

    useful_secrets = [
        "HF_TOKEN",
        "WANDB_API_KEY",
        "BEAKER_TOKEN",
        "OPENAI_API_KEY",
        "AZURE_API_KEY",
        "AZURE_API_BASE",
        "ANTHROPIC_API_KEY",
    ]
    for useful_secret in useful_secrets:
        if f"{whoami}_{useful_secret}" in beaker_secrets:
            env_vars.append(beaker.BeakerEnvVar(name=useful_secret, secret=f"{whoami}_{useful_secret}"))
        elif useful_secret in beaker_secrets:
            env_vars.append(beaker.BeakerEnvVar(name=useful_secret, secret=useful_secret))

    if not pure_docker_mode:
        env_vars.extend([beaker.BeakerEnvVar(name="PATH", value=os.getenv("PATH"))])

    if all(c in WEKA_CLUSTERS for c in cluster):
        env_vars.extend(
            [
                beaker.BeakerEnvVar(name="HF_HOME", value="/weka/oe-adapt-default/allennlp/.cache/huggingface"),
                beaker.BeakerEnvVar(
                    name="HF_DATASETS_CACHE", value="/weka/oe-adapt-default/allennlp/.cache/huggingface"
                ),
                beaker.BeakerEnvVar(name="HF_HUB_CACHE", value="/weka/oe-adapt-default/allennlp/.cache/hub"),
                beaker.BeakerEnvVar(
                    name="CHECKPOINT_OUTPUT_DIR",
                    value=f"/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/{wandb_id}",
                ),
            ]
        )
        if num_nodes > 1:
            env_vars.extend(
                [
                    beaker.BeakerEnvVar(name="NCCL_SOCKET_IFNAME", value="ib"),
                    beaker.BeakerEnvVar(name="NCCL_IB_HCA", value="^=mlx5_bond_0"),
                ]
            )

    elif all(c in GCP_CLUSTERS for c in cluster):
        env_vars.extend(
            [
                beaker.BeakerEnvVar(name="HF_HOME", value="/filestore/.cache/huggingface"),
                beaker.BeakerEnvVar(name="HF_DATASETS_CACHE", value="/filestore/.cache/huggingface"),
                beaker.BeakerEnvVar(name="HF_HUB_CACHE", value="/filestore/.cache/hub"),
                beaker.BeakerEnvVar(name="HF_HUB_ENABLE_HF_TRANSFER", value="0"),
            ]
        )
        if num_nodes > 1:
            env_vars.extend(
                [
                    beaker.BeakerEnvVar(name="NCCL_CROSS_NIC", value="0"),
                    beaker.BeakerEnvVar(name="NCCL_PROTO", value="Simple,LL128"),
                    beaker.BeakerEnvVar(name="NCCL_MIN_NCHANNELS", value="4"),
                    beaker.BeakerEnvVar(name="NCCL_P2P_NET_CHUNKSIZE", value="524288"),
                    beaker.BeakerEnvVar(name="NCCL_P2P_PCI_CHUNKSIZE", value="524288"),
                    beaker.BeakerEnvVar(name="NCCL_P2P_NVL_CHUNKSIZE", value="1048576"),
                    beaker.BeakerEnvVar(name="NCCL_NVLSTREE_MAX_CHUNKSIZE", value="131072"),
                    beaker.BeakerEnvVar(name="NCCL_FASTRAK_NUM_FLOWS", value="2"),
                    beaker.BeakerEnvVar(name="NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL", value="0"),
                    beaker.BeakerEnvVar(name="NCCL_BUFFSIZE", value="8388608"),
                    beaker.BeakerEnvVar(name="NCCL_FASTRAK_USE_SNAP", value="1"),
                    beaker.BeakerEnvVar(name="CUDA_VISIBLE_DEVICES", value="0,1,2,3,4,5,6,7"),
                    beaker.BeakerEnvVar(name="NCCL_NET_GDR_LEVEL", value="PIX"),
                    beaker.BeakerEnvVar(name="NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING", value="0"),
                    beaker.BeakerEnvVar(name="NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS", value="600000"),
                    beaker.BeakerEnvVar(name="NCCL_USE_SNAP", value="1"),
                    beaker.BeakerEnvVar(name="NCCL_FASTRAK_USE_LLCM", value="1"),
                    beaker.BeakerEnvVar(name="NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY", value="/dev/aperture_devices"),
                    beaker.BeakerEnvVar(name="NCCL_TUNER_PLUGIN", value="libnccl-tuner.so"),
                    beaker.BeakerEnvVar(
                        name="NCCL_TUNER_CONFIG_PATH", value="/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto"
                    ),
                    beaker.BeakerEnvVar(
                        name="NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE",
                        value="/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto",
                    ),
                    beaker.BeakerEnvVar(name="NCCL_FASTRAK_CTRL_DEV", value="enp0s12"),
                    beaker.BeakerEnvVar(
                        name="NCCL_FASTRAK_IFNAME",
                        value="enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0",
                    ),
                    beaker.BeakerEnvVar(name="NCCL_SOCKET_IFNAME", value="enp0s12"),
                    beaker.BeakerEnvVar(name="NCCL_DEBUG_SUBSYS", value="INIT,NET"),
                ]
            )

    if resumable:
        env_vars.extend(
            [
                beaker.BeakerEnvVar(name="WANDB_RUN_ID", value=wandb_id),
                beaker.BeakerEnvVar(name="WANDB_RESUME", value="allow"),
            ]
        )

    return env_vars


def get_datasets(beaker_datasets: list[dict[str, str]], cluster: list[str]) -> list[beaker.BeakerDataMount]:
    res: list[beaker.BeakerDataMount] = []
    if all(c in WEKA_CLUSTERS for c in cluster):
        res = [
            beaker.BeakerDataMount(
                source=beaker.BeakerDataSource(weka="oe-adapt-default"), mount_path="/weka/oe-adapt-default"
            ),
            beaker.BeakerDataMount(
                source=beaker.BeakerDataSource(weka="oe-training-default"), mount_path="/weka/oe-training-default"
            ),
        ]
    elif all(c in GCP_CLUSTERS for c in cluster):
        res = [
            beaker.BeakerDataMount(
                source=beaker.BeakerDataSource(host_path="/mnt/filestore_1"), mount_path="/filestore"
            )
        ]
    for beaker_dataset in beaker_datasets:
        to_append = beaker.BeakerDataMount(
            source=beaker.BeakerDataSource(beaker=beaker_dataset["beaker"]), mount_path=beaker_dataset["mount_path"]
        )
        res.append(to_append)

    return res


def make_task_spec(
    config: "LaunchConfig", full_command: str, task_index: int, beaker_secrets: list[str], whoami: str, wandb_id: str
) -> beaker.BeakerTaskSpec:
    if config.num_nodes > 1 and not all(c in INTERCONNECT_CLUSTERS for c in config.cluster):
        raise ValueError(
            f"Interconnect clusters are required for multi-node jobs; please use: {INTERCONNECT_CLUSTERS}"
        )
    if config.image == "ai2/cuda11.8-cudnn8-dev-ubuntu20.04" and any(c in GCP_CLUSTERS for c in config.cluster):
        raise ValueError("GCP clusters do not have the dev filesystem, please use a proper image")

    if config.hostname is not None:
        constraints = beaker.BeakerConstraints(hostname=config.hostname)
    else:
        constraints = beaker.BeakerConstraints(cluster=config.cluster)

    spec = beaker.BeakerTaskSpec(
        name=f"{config.task_name}__{task_index}",
        image=beaker.BeakerImageSource(beaker=config.image),
        command=["/bin/bash", "-c"],
        arguments=[full_command],
        result=beaker.BeakerResultSpec(path="/output"),
        datasets=get_datasets(config.beaker_datasets, config.cluster),
        context=beaker.BeakerTaskContext(
            priority=beaker.BeakerJobPriority[config.priority], preemptible=config.preemptible
        ),
        constraints=constraints,
        env_vars=get_env_vars(
            config.pure_docker_mode,
            config.cluster,
            beaker_secrets,
            whoami,
            config.resumable,
            config.num_nodes,
            config.env,
            config.secret,
            wandb_id,
        ),
        resources=beaker.BeakerTaskResources(gpu_count=config.gpus, shared_memory=config.shared_memory),
        replicas=config.num_nodes,
        timeout=config.timeout,
    )
    if config.num_nodes > 1:
        spec.leader_selection = True
        spec.propagate_failure = True
        spec.propagate_preemption = True
    if config.no_host_networking:
        spec.host_networking = False
    else:
        spec.host_networking = True

    return spec


def serialize_experiment_config(config: "ExperimentConfig") -> dict:
    def serialize_dataclass(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            result = {}
            for f in dataclasses.fields(obj):
                value = getattr(obj, f.name)
                result[f.name] = serialize_dataclass(value)
            return result
        elif isinstance(obj, list):
            return [serialize_dataclass(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: serialize_dataclass(v) for k, v in obj.items()}
        else:
            return obj

    return {
        "args": serialize_dataclass(config.args),
        "tokenizer_config": serialize_dataclass(config.tokenizer_config),
        "model_config": serialize_dataclass(config.model_config),
    }


def launch_on_beaker(
    experiment_config: "ExperimentConfig",
    launch_config: LaunchConfig,
    setup_command: str = "source configs/beaker_configs/ray_node_setup.sh",
    script_path: str = "open_instruct/grpo_fast.py",
) -> str:
    wandb_id = generate_id()

    config_dict = serialize_experiment_config(experiment_config)
    config_json = json.dumps(config_dict)
    config_base64 = base64.b64encode(config_json.encode()).decode()

    config_write_command = f"echo {config_base64} | base64 -d > /tmp/experiment_config.json"

    full_command = (
        f"{setup_command} && {config_write_command} && python {script_path} --config_file /tmp/experiment_config.json"
    )

    if launch_config.workspace:
        beaker_client = beaker.Beaker.from_env(default_workspace=launch_config.workspace)
    else:
        beaker_client = beaker.Beaker.from_env()

    beaker_secrets = [secret.name for secret in beaker_client.secret.list()]
    whoami = beaker_client.user.get().name

    beaker.Beaker.TIMEOUT = 300

    task_spec = make_task_spec(
        launch_config, full_command, task_index=0, beaker_secrets=beaker_secrets, whoami=whoami, wandb_id=wandb_id
    )

    experiment_spec = beaker.BeakerExperimentSpec(
        description=launch_config.description,
        tasks=[task_spec],
        budget=launch_config.budget,
        retry=beaker.BeakerRetrySpec(allowed_task_retries=launch_config.max_retries),
    )

    @backoff.on_exception(backoff.expo, requests.exceptions.Timeout, max_tries=5, factor=5)
    def _launch():
        exp = beaker_client.experiment.create(spec=experiment_spec)
        url = f"https://beaker.org/ex/{exp.experiment.id}"
        console.log(f"Kicked off Beaker job. {url}")
        return url

    return _launch()
