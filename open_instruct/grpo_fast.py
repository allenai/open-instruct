# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------
# Part of the code is adapted from https://github.com/OpenRLHF/OpenRLHF
# which has the following license:
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
import logging
import os
import random
import shutil
import socket
import subprocess
import threading
import time
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple

from open_instruct.dataset_transformation import (
    TokenizerConfig,
    get_cached_dataset_rlvr,
)
from open_instruct.ground_truth_utils import soft_format_reward_func
from open_instruct.rl_utils2 import pack_sequences
from peft import PeftModel, get_peft_model_state_dict

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA

import deepspeed
import numpy as np
import pandas as pd
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from datasets import Dataset
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from huggingface_hub import HfApi
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from transformers.integrations import HfDeepSpeedConfig
from vllm import SamplingParams, LLM

from open_instruct.dataset_processor import (
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    DatasetConfig,
    visualize_token,
)
from open_instruct.model_utils import (
    ModelConfig,
    apply_verifiable_reward_fast,
    disable_dropout_in_model,
    exact_div,
    log_softmax_and_gather,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
)
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    upload_metadata_to_hf,
)
from open_instruct.vllm_utils2 import create_vllm_engines, init_process_group

api = HfApi()
INVALID_LOGPROB = 1.0


@dataclass
class Args:
    # required dataset args
    dataset_mixer_list: List[str] = None
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_eval_list: List[str] = None
    """A list of datasets (local or HF) to sample from for evaluation."""
    dataset_mixer_list_splits: List[str] = None
    """The dataset splits to use for training"""
    dataset_mixer_eval_list_splits: Optional[List[str]] = None
    """The dataset splits to use for evaluation"""

    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""

    # optimizer args
    eps: float = 1e-5
    """The epsilon value for the optimizer"""
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    warmup_ratio: float = 0.0
    """Ratio of warmup steps to total steps (takes precedence over `warm_up_steps`)"""

    # various batch sizes
    gradient_accumulation_steps: Optional[int] = None
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: Optional[int] = 1
    """The forward batch size per device (local_micro_batch_size)"""
    per_device_eval_batch_size: Optional[int] = 1
    """The forward batch size per device for evaluation (local_micro_batch_size)"""
    total_episodes: Optional[int] = 100000
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_rollout_batch_size: int = 64
    """The number of rollout episodes per iteration per device"""
    local_total_prompts: Optional[int] = None
    """The total number of prompts per device"""
    rollout_batch_size: Optional[int] = None
    """The number of rollout episodes per iteration"""
    num_training_steps: Optional[int] = None
    """The number of training_steps to train"""
    num_evals: int = 4
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """The frequency of evaluation steps"""
    local_dataloader_batch_size: Optional[int] = None
    """The batch size per GPU for the dataloader"""
    save_freq: int = -1
    """How many train steps to save the model"""

    # online settings
    num_epochs: int = 4
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: int = 64
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_rollout_forward_batch_size: int = 1
    """per rank no grad forward pass in the rollout phase"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""
    reward_model_revision: Optional[str] = None
    """the revision of the reward model"""

    # generation config
    response_length: int = 53
    """the length of the response"""
    stop_token: Optional[Literal["eos", "period"]] = None
    """the stop token"""
    stop_token_id: Optional[int] = None
    """the truncation token id"""
    min_response_length: int = 0
    """stop only after this many tokens"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: float = -1.0
    """the reward value for responses that do not contain `stop_token_id`"""
    non_stop_penalty: bool = False
    """whether to penalize responses that do not contain `stop_token_id`"""
    number_samples_per_prompt: int = 1
    """the number of samples to generate per prompt, useful for easy-star"""
    stop_strings: List[str] = None
    """List of strings that stop the generation when they are generated.
    The returned output will not contain the stop strings."""
    eval_max_length: int = 4096  # max generation length for evaluation

    # online PPO specific args
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    cliprange: float = 0.2
    """the clip range"""
    gamma: float = 1
    """the discount factor"""
    kl_estimator: Literal["kl1", "kl2", "kl3", "kl4"] = "kl3"
    """the KL estimator to use"""
    apply_verifiable_reward: bool = False
    """whether to apply verifiable reward"""
    reward_model_multiplier: float = 1.0
    """the reward model multiplier, for down/upscaling the reward model output"""
    verification_reward: float = 10.0
    """the reward value for verifiable responses"""
    add_r1_style_format_reward: bool = False
    """whether to add the R1 style format reward"""
    r1_style_format_reward: float = 1.0
    """the reward value for R1 style format reward"""
    pack_length: int = 2048
    """the length of the pack (you should prob set to the max length of the model)"""

    # async setting
    async_mode: bool = True
    """Whether to run the generation in async mode which learns from the second latest policy like Cleanba (https://arxiv.org/abs/2310.00036)"""

    # ray
    actor_num_gpus_per_node: List[int] = field(default_factory=lambda: [1])
    """number of gpus per node for actor"""
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node (mostly for debugging purposes)"""
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine for multi-GPU inference"""
    vllm_enforce_eager: bool = False
    """whether to enforce eager mode for vLLM -- slow inference but needed for multi-node"""
    vllm_sync_backend: str = "nccl"
    """DeepSpeed -> vLLM weight sync backend"""
    vllm_gpu_memory_utilization: float = 0.9
    """vLLM GPU memory utilization"""
    enable_prefix_caching: bool = False
    """whether to enable prefix caching"""
    deepspeed_stage: int = 0
    """the deepspeed stage"""
    gather_whole_model: bool = True
    """whether to gather the whole model to boardcast (not doable for 70B but can be faster for 8B)"""

    # wandb and HF tracking configs
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: Optional[str] = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: Optional[str] = None
    """Where to save the model"""
    checkpoint_output_dir: Optional[str] = None
    """Where to save the model checkpoints in case of preemption"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    oe_eval_tasks: Optional[List[str]] = None
    """The beaker evaluation tasks to launch"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""

    def __post_init__(self):
        assert self.number_samples_per_prompt > 1, "Number of samples per prompt must be greater than 1 for GRPO!"


def process_dataset_mixer(value) -> Tuple[Optional[dict], Optional[str]]:
    # if passed through cli: convert the dataset mixers to dictionaries
    if isinstance(value, str):
        return json.loads(value), value
    # if passed through yaml: convert the dataset mixers to strings
    elif isinstance(value, dict):
        return value, json.dumps(value)
    else:
        raise ValueError("Input must be either a string or a dictionary")


def calculate_runtime_args(args: Args, model_config: ModelConfig):
    """calculate (in-place) runtime args such as the effective batch size, word size, etc."""
    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.gradient_accumulation_steps = exact_div(
        args.local_mini_batch_size,
        args.per_device_train_batch_size,
        "`local_mini_batch_size` must be a multiple of `per_device_train_batch_size`",
    )
    args.world_size = sum(args.actor_num_gpus_per_node)
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.local_total_prompts = args.local_rollout_batch_size * args.number_samples_per_prompt
    args.rollout_batch_size = int(args.local_rollout_batch_size * args.world_size)
    args.mini_batch_size = int(args.local_mini_batch_size * args.world_size)
    args.num_mini_batches = exact_div((args.rollout_batch_size * args.number_samples_per_prompt), args.mini_batch_size)
    args.num_training_steps = args.total_episodes // (args.rollout_batch_size * args.number_samples_per_prompt)
    args.eval_freq = max(1, args.num_training_steps // args.num_evals)
    # PPO logic: do checks and set up dataloader batch size
    if args.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    args.local_dataloader_batch_size = args.rollout_batch_size
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    if args.with_tracking:
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()


def get_train_ds_config(
    offload,
    adam_offload=False,
    stage=0,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=True,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # # ZeRO++
        # "zero_hpz_partition_size": zpg,
        # "zero_quantized_weights": False,
        # "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
    }


def get_eval_ds_config(
    offload,
    stage=0,
    bf16=True,
):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "cpu" if offload else "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": bf16,
        },
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def remove_padding(sequences, pad_token_id):
    return [[inneritem for inneritem in item if inneritem != pad_token_id] for item in sequences]


class MetricsTracker:
    """A simple class to prellocate all metrics in an array
    so we can do only one allreduce operation to get the metrics mean"""

    def __init__(self, max_metrics: int = 32, device: torch.device = torch.device("cuda")):
        self.metrics = torch.zeros(max_metrics, device=device)
        self.names2idx = {}
        self.current_idx = 0
        self.max_metrics = max_metrics

    def add(self, name: str, value: torch.tensor):
        if name not in self.names2idx:
            if self.current_idx >= self.max_metrics:
                raise ValueError(f"Exceeded maximum number of metrics ({self.max_metrics})")
            self.names2idx[name] = self.current_idx
            self.current_idx += 1

        self.metrics[self.names2idx[name]] = value
        return self

    def get_reduced_metrics(self) -> dict[str, float]:
        self.metrics /= dist.get_world_size()
        dist.all_reduce(self.metrics, op=dist.ReduceOp.SUM)
        # convert to list so to avoid multiple .item() torch calls
        reduced_metrics = self.metrics.tolist()
        return {name: reduced_metrics[idx] for name, idx in self.names2idx.items()}

def collate_fn(tensors_list: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(tensors_list, batch_first=True, padding_value=pad_token_id)

def to_device_inplace(tensors_list: List[torch.Tensor], device: torch.device):
    for i in range(len(tensors_list)):
        tensors_list[i] = tensors_list[i].to(device, non_blocking=True)

class Timer:
    """A context manager for timing code blocks"""

    def __init__(self, description: str, noop: int = 0):
        self.description = description
        self.noop = noop

    def __enter__(self):
        if self.noop:
            return
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if self.noop:
            return
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"{self.description}: {self.duration:.2f} seconds")


class ShufflingIterator:
    def __init__(self, data: np.ndarray, batch_size: int, seed: Optional[int] = None):
        self.data = data.copy()
        self.batch_size = batch_size
        self.index = 0
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.data)

        # Ensure the effective dataset size is divisible by batch_size
        self.effective_size = len(self.data) - (len(self.data) % batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        return self

    def __next__(self) -> List[int]:
        if self.index >= self.effective_size:
            self.index = 0
            self.rng.shuffle(self.data)

        end_index = self.index + self.batch_size
        batch = self.data[self.index : end_index].tolist()
        self.index = end_index

        return batch


class RayProcess:
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.master_addr = master_addr if master_addr else self.get_current_node_ip()
        self.master_port = master_port if master_port else self.get_free_port()
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = "0"
        random.seed(self.rank)
        np.random.seed(self.rank)
        torch.manual_seed(self.rank)

    @staticmethod
    def get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self.master_addr, self.master_port

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def from_pretrained(
        self, args: Args, model_config: ModelConfig, beaker_config: BeakerRuntimeConfig, wandb_url: str
    ):
        self.args = args
        self.model_config = model_config
        self.beaker_config = beaker_config
        self.wandb_url = wandb_url
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(self.local_rank)
        deepspeed.init_distributed()

        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=False,
            stage=args.deepspeed_stage,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        # Costa: MAGIC: it's actually needed to initialize this `dschf`, so
        # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
        # next line instructs transformers to partition the model directly over multiple gpus using
        # deepspeed.zero.Init when model's `from_pretrained` method is called.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        print(f"{dschf=}")

        self.original_tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path, revision=model_config.model_revision
        )
        self.policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        disable_dropout_in_model(self.policy)
        self.policy.gradient_checkpointing_enable()
        # AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        # AdamOptimizer = FusedAdam
        # weight_decay = 0.0
        # optim_params = get_optimizer_grouped_parameters(self.policy, weight_decay)
        # self.optimizer = AdamOptimizer(optim_params, lr=args.learning_rate)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=args.learning_rate)
        num_scheduler_steps = args.num_training_steps * args.num_epochs * args.num_mini_batches
        warm_up_steps = args.warm_up_steps
        if args.warmup_ratio >= 0.0:
            warm_up_steps = int(num_scheduler_steps * args.warmup_ratio)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_scheduler_steps,
        )
        print(ds_config)
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.policy,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=True,
        )
        self.model.train()

        # reference model
        ds_config = get_eval_ds_config(
            offload=False,
            # inference model only has stage 3 (sharding) or stage 0 (no sharding)
            # stage 2 is optimizer sharding which doesn't apply to inference
            stage=args.deepspeed_stage if args.deepspeed_stage == 3 else 0,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        # Costa: MAGIC: it's actually needed to initialize this `dschf`, so
        # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
        # next line instructs transformers to partition the model directly over multiple gpus using
        # deepspeed.zero.Init when model's `from_pretrained` method is called.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        print(f"{dschf=}")

        self.ref_policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        disable_dropout_in_model(self.ref_policy)
        self.ref_policy, *_ = deepspeed.initialize(model=self.ref_policy, config=ds_config)
        self.ref_policy.eval()

        # reward model
        if args.reward_model_multiplier:
            self.reward_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
                args.reward_model_path,
                revision=args.reward_model_revision,
                num_labels=1,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                use_cache=False,
            )
            disable_dropout_in_model(self.reward_model)
            ds_config = get_eval_ds_config(
                offload=False,
                # inference model only has stage 3 (sharding) or stage 0 (no sharding)
                # stage 2 is optimizer sharding which doesn't apply to inference
                stage=args.deepspeed_stage if args.deepspeed_stage == 3 else 0,
                bf16=True,
            )
            ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
            ds_config["gradient_accumulation_steps"] = 1
            self.reward_model, *_ = deepspeed.initialize(model=self.reward_model, config=ds_config)
            self.reward_model.eval()

        assert (
            args.reward_model_multiplier or args.apply_verifiable_reward
        ), "Either `reward_model_multiplier` must be non-zero or `apply_verifiable_reward` must be True."
        
        self.local_metrics = MetricsTracker(max_metrics=32, device=self.device)

    def forward(
        self,
        model: PreTrainedModel,
        query_response: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_token_id: int,
        temperature: float,
    ) -> torch.Tensor:
        # Replace pad tokens with 0s so that we don't run into index out of bounds errors
        input_ids = torch.masked_fill(query_response, ~(query_response != pad_token_id), 0)
        # NOTE: the [:-1] and [1:] are because the logits and generated tokens are off by 1 in index
        output = model(
            input_ids=input_ids[:, :-1],
            # @vwxyzjn: without clamp, we get index out of bounds errors; TODO: investigate
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits
        logits /= temperature + 1e-7
        logprob = log_softmax_and_gather(logits, input_ids[:, 1:])
        return logprob


    def setup_model_update_group(self, vllm_engines):
        self.vllm_engines = vllm_engines
        if self.rank == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            vllm_num_engines, vllm_tensor_parallel_size = (
                self.args.vllm_num_engines,
                self.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            backend = self.args.vllm_sync_backend
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    "openrlhf",
                    backend=backend,
                )
                for i, engine in enumerate(vllm_engines)
            ]
            self.model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
            )
            ray.get(refs)
        torch.distributed.barrier()

    def broadcast_to_vllm(self):
        # avoid OOM
        torch.cuda.empty_cache()
        model = self.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        refss = []
        if self.args.gather_whole_model:
            with deepspeed.zero.GatheredParameters(model.parameters(), enabled=self.args.deepspeed_stage == 3):
                for name, param in model.named_parameters():
                    count += 1  # empty_cache at last param
                    # Fire all vllm engines for broadcast
                    if torch.distributed.get_rank() == 0:
                        shape = param.shape if self.args.deepspeed_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight.remote(
                                name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                            )
                            for engine in self.vllm_engines
                        ]
                        refss.extend(refs)
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
        else:  # broadcast each parameter independently
            for name, param in model.named_parameters():
                count += 1
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.args.deepspeed_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                        )
                        for engine in self.vllm_engines
                    ]
                    refss.extend(refs)
                with deepspeed.zero.GatheredParameters([param], enabled=self.args.deepspeed_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
        if torch.distributed.get_rank() == 0:
            ray.get(refss)


    def train(
        self,
        packed_query_responses: torch.Tensor,
        packed_attention_masks: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_advantages: torch.Tensor,
        packed_response_masks: torch.Tensor,
        pad_token_id: int,
        accumulation_steps: int,
    ):
        args = self.args
        # Shuffle the batch and collate the data
        b_inds = np.random.permutation(len(packed_query_responses))
        collated_query_responses = []
        collated_attention_masks = []
        collated_position_ids = []
        collated_response_masks = []
        collated_advantages = []
        for i in range(0, len(packed_query_responses), args.per_device_train_batch_size):
            micro_range = b_inds[i:i+args.per_device_train_batch_size]
            collated_query_responses.append(collate_fn([packed_query_responses[idx] for idx in micro_range], pad_token_id))
            collated_attention_masks.append(collate_fn([packed_attention_masks[idx] for idx in micro_range], 0))
            collated_position_ids.append(collate_fn([packed_position_ids[idx] for idx in micro_range], 0))
            collated_response_masks.append(collate_fn([packed_response_masks[idx] for idx in micro_range], 0))
            collated_advantages.append(collate_fn([packed_advantages[idx] for idx in micro_range], 0))
        to_device_inplace(collated_query_responses, self.device)
        to_device_inplace(collated_attention_masks, self.device)
        to_device_inplace(collated_position_ids, self.device)
        to_device_inplace(collated_advantages, self.device)
        to_device_inplace(collated_response_masks, self.device)

        # Calculate the logprob of the reference policy
        collated_ref_logprobs = []
        with Timer("Inference Calculation", noop = self.rank != 0 ):
            with torch.no_grad():
                for i in range(len(collated_query_responses)):
                    query_response = collated_query_responses[i]
                    attention_mask = collated_attention_masks[i]
                    position_id = collated_position_ids[i]
                    response_mask = collated_response_masks[i]
                    ref_logprob = self.forward(
                        self.ref_policy,
                        query_response,
                        attention_mask,
                        position_id,
                        pad_token_id,
                        args.temperature,
                    )
                    ref_logprob = torch.masked_fill(ref_logprob, ~response_mask[:, 1:].bool(), INVALID_LOGPROB)
                    collated_ref_logprobs.append(ref_logprob)
                    torch.cuda.empty_cache()
        local_step = 0
        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        with Timer("[Training Processes] Loss calculation", noop = self.rank != 0 ):
            old_logprobs = [None for _ in range(len(collated_query_responses))]
            kl1_stats = torch.zeros(len(collated_query_responses))
            kl2_stats = torch.zeros(len(collated_query_responses))
            kl3_stats = torch.zeros(len(collated_query_responses))
            kl4_stats = torch.zeros(len(collated_query_responses))
            pg_clipfrac_stats = torch.zeros(len(collated_query_responses))
            pg_loss_stats = torch.zeros(len(collated_query_responses))
            loss_stats = torch.zeros(len(collated_query_responses))
            ratio_stats = torch.zeros(len(collated_query_responses))
            for epoch_idx in range(args.num_epochs):
                for i in range(len(collated_query_responses)):
                    mb_ref_logprob = collated_ref_logprobs[i]
                    mb_query_responses = collated_query_responses[i]
                    mb_advantages = collated_advantages[i]
                    mb_response_masks = collated_response_masks[i]
                    mb_attention_mask = collated_attention_masks[i]
                    mb_position_id = collated_position_ids[i]
                    mb_new_logprobs = self.forward(
                        self.model, mb_query_responses, mb_attention_mask, mb_position_id, pad_token_id, args.temperature
                    )
                    mb_new_logprobs = torch.masked_fill(
                        mb_new_logprobs, ~mb_response_masks[:, 1:].bool(), INVALID_LOGPROB
                    )

                    # Cache the old logprobs
                    with torch.no_grad():
                        if epoch_idx == 0:
                            old_logprobs[i] = mb_new_logprobs

                    # Calculate the policy's loss
                    mb_old_logprobs = old_logprobs[i]
                    logprobs_diff = mb_new_logprobs - mb_old_logprobs
                    ratio = torch.exp(logprobs_diff)
                    pg_losses = -mb_advantages[:, 1:] * ratio
                    pg_losses2 = -mb_advantages[:, 1:] * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    # Here we recalculate kl: we want the KL loss to backpropagate through the model
                    # We also clamp the KL loss to avoid numerical instability
                    # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
                    ref_logprobs_diff = (mb_new_logprobs - mb_ref_logprob).clamp(-40.0, 40.0)
                    kl1 = ref_logprobs_diff
                    kl2 = (ref_logprobs_diff) ** 2 / 2
                    kl3 = torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff  # this is more numerically stable
                    kl4 = ratio * ref_logprobs_diff
                    # TODO: the kl is packed... may need to unpack it for calculating
                    if args.kl_estimator == "kl1":
                        kl = kl1
                    elif args.kl_estimator == "kl2":
                        kl = kl2
                    elif args.kl_estimator == "kl3":
                        kl = kl3
                    elif args.kl_estimator == "kl4":
                        kl = kl4
                    # grpo change: directly subtract KL in loss (add)
                    loss = masked_mean(pg_loss_max + (args.beta * kl), mb_response_masks[:, 1:].bool())
                    loss = loss / accumulation_steps
                    self.model.backward(loss)
                    if (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                    local_step += 1
                    with torch.no_grad():
                        kl1_stats[i] = kl1.mean().float()
                        kl2_stats[i] = kl2.mean().float()
                        kl3_stats[i] = kl3.mean().float()
                        kl4_stats[i] = kl4.mean().float()
                        pg_clipfrac_stats[i] = masked_mean(
                            (pg_losses2 > pg_losses).float(), mb_response_masks[:, 1:].bool()
                        )
                        pg_loss_stats[i] = masked_mean(pg_loss_max, mb_response_masks[:, 1:].bool())
                        loss_stats[i] = loss
                        ratio_stats[i] = ratio.mean()

            with torch.no_grad():
                # breakpoint()
                self.local_metrics.add("objective/kl_avg", kl1_stats.mean())
                self.local_metrics.add("objective/kl2_avg", kl2_stats.mean())
                self.local_metrics.add("objective/kl3_avg", kl3_stats.mean())
                self.local_metrics.add("objective/kl4_avg", kl4_stats.mean())
                self.local_metrics.add("loss/policy_avg", pg_loss_stats.mean())
                self.local_metrics.add("loss/policy_avg", loss_stats.mean())
                self.local_metrics.add("policy/clipfrac_avg", pg_clipfrac_stats.mean())
                self.local_metrics.add("val/ratio", ratio_stats.mean())
                self.local_metrics.add("val/ratio_var", ratio_stats.var())
                reduced_metrics = self.local_metrics.get_reduced_metrics()
                return reduced_metrics

    def save_model(self, model_to_save: PreTrainedModel, output_dir: str) -> None:
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # save model weights for ZeRO2/3
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.rank == 0:
                    output_state_dict[k] = vv

        if self.rank == 0:
            state_dict = model_to_save.state_dict()

            # copy named_buffers with `persistent=True`
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
            else:
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict)

            # save tokenizer
            self.original_tokenizer.save_pretrained(output_dir)

    def launch_ai2_evals_on_weka(self, step_dir: str, training_step: Optional[int] = None) -> None:
        """auto eval the metrics as `f"{args.exp_name}_step_{training_step}"` in our leaderboard"""
        args = self.args
        beaker_config = self.beaker_config
        model_config = self.model_config
        wandb_url = self.wandb_url
        # Ai2 specific logic
        if is_beaker_job() and self.rank == 0:
            if training_step is not None:
                leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
            else:
                leaderboard_name = args.hf_repo_revision
            if args.hf_metadata_dataset:
                dataset_list = args.dataset_mixer_list
                # mainly just focussing here on what would be useful for the leaderboard.
                # wandb will have even more useful information.
                metadata_blob = {
                    "model_name": args.exp_name,
                    "model_type": "ppo",
                    "datasets": dataset_list,
                    "base_model": model_config.model_name_or_path,
                    "wandb_path": wandb_url,
                    "beaker_experiment": beaker_config.beaker_experiment_url,
                    "beaker_datasets": beaker_config.beaker_dataset_id_urls,
                }
                upload_metadata_to_hf(
                    metadata_blob,
                    "metadata.json",
                    args.hf_metadata_dataset,
                    "results/" + leaderboard_name,  # to match what the auto-evals name as.
                )

            command = f"""\
python scripts/submit_eval_jobs.py \
    --model_name {leaderboard_name} \
    --location {step_dir} \
    --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
    --is_tuned \
    --workspace "tulu-3-results" \
    --priority high \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image "nathanl/open_instruct_auto" \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id {wandb_url} \
    --oe_eval_max_length {args.eval_max_length} \
    --skip_oi_evals"""
            if training_step is not None:
                command += f" --step {training_step}"
            if args.oe_eval_tasks is not None:
                command += f" --oe_eval_tasks {','.join(args.oe_eval_tasks)}"
            print(f"Launching eval jobs with command: {command}")
            process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(f"Submit jobs after model training is finished - Stdout:\n{stdout.decode()}")
            print(f"Submit jobs after model training is finished - Stderr:\n{stderr.decode()}")
            print(f"Submit jobs after model training is finished - process return code: {process.returncode}")


def kill_ray_cluster_if_a_worker_dies(object_refs: List[Any], stop_event: threading.Event):
    while True:
        if stop_event.is_set():
            break
        for ref in object_refs:
            try:
                ray.get(ref, timeout=0.01)
            except ray.exceptions.GetTimeoutError:
                pass
            except Exception as e:
                print(e)
                print(f"Actor {ref} died")
                time.sleep(120)
                ray.shutdown()
                os._exit(1)  # Force shutdown the process

        time.sleep(30)


class ModelGroup:
    def __init__(
        self,
        pg: PlacementGroup,
        ray_process_cls: RayProcess,
        num_gpus_per_node: List[int],
        single_gpu_mode: bool,
    ):
        self.pg = pg
        self.ray_process_cls = ray_process_cls
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 0.48 if single_gpu_mode else 1
        self.num_cpus_per_actor = 4
        self.models = []
        world_size = sum(self.num_gpus_per_node)
        master_policy = ray_process_cls.options(
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=0
            ),
        ).remote(world_size, 0, 0, None, None)

        self.models.append(master_policy)
        master_addr, master_port = ray.get(master_policy.get_master_addr_port.remote())

        def get_bundle_index(rank, num_gpus_per_node):
            """given a rank and a list of num_gpus_per_node, return the index of the bundle that the rank belongs to"""
            bundle_idx = 0
            while rank >= num_gpus_per_node[bundle_idx]:
                rank -= num_gpus_per_node[bundle_idx]
                bundle_idx += 1
            return bundle_idx

        assert get_bundle_index(0, [7, 8, 4]) == 0
        assert get_bundle_index(1, [7, 8, 4]) == 0
        assert get_bundle_index(7, [7, 8, 4]) == 1
        assert get_bundle_index(8, [7, 8, 4]) == 1
        assert get_bundle_index(9, [7, 8, 4]) == 1
        assert get_bundle_index(16, [7, 8, 4]) == 2

        # Setup worker models
        for rank in range(1, world_size):
            print(f"{rank=}, {world_size=}, {rank=}, {master_addr=}, {master_port=}")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.pg,
                placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node),
            )
            worker_policy = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(world_size, rank, 0, master_addr, master_port)
            self.models.append(worker_policy)


def main(args: Args, dataset_config: DatasetConfig, model_config: ModelConfig):
    calculate_runtime_args(args, model_config)

    # Setup experiment tracking and seeds
    all_configs = {}
    beaker_config = None
    if is_beaker_job():
        args.checkpoint_output_dir = os.environ.get("CHECKPOINT_OUTPUT_DIR", None)
        beaker_config = maybe_get_beaker_config()
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(dataset_config), **asdict(model_config))
    if args.with_tracking:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=all_configs,
            name=args.run_name,
            save_code=True,
            tags=[args.exp_name] + get_wandb_tags(),
        )
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Setup tokenizer and get datasets
    tokenizer_revision = (
        model_config.model_revision if model_config.tokenizer_revision is None else model_config.tokenizer_revision
    )
    tokenizer_name = (
        model_config.tokenizer_name if model_config.tokenizer_name is not None else model_config.model_name_or_path
    )
    if tokenizer_revision != model_config.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{model_config.model_revision}`."""
        print(warning)
    tc = TokenizerConfig(
        model_name_or_path=tokenizer_name,
        revision=model_config.model_revision,
        use_fast=not model_config.use_slow_tokenizer,
        chat_template_name=model_config.chat_template_name,
        add_bos=model_config.add_bos,
    )
    tokenizer = tc.tokenizer
    train_dataset = get_cached_dataset_rlvr(
        args.dataset_mixer_list,
        args.dataset_mixer_list_splits,
        tc,
        dataset_config.max_prompt_token_length,
        dataset_config.max_token_length,
        args.hf_entity,
    )
    train_dataset = train_dataset.shuffle(seed=args.seed)
    eval_dataset = None
    if args.dataset_mixer_eval_list is not None:
        eval_dataset = get_cached_dataset_rlvr(
            args.dataset_mixer_eval_list,
            args.dataset_mixer_eval_list_splits,
            tc,
            dataset_config.max_prompt_token_length,
            dataset_config.max_token_length,
            args.hf_entity,
        )
        eval_dataset = eval_dataset.shuffle(seed=args.seed)

    if args.cache_dataset_only:
        return

    # Runtime setups and quick logging
    if args.stop_token:
        if args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        if args.stop_token == "period":
            args.stop_token_id = tokenizer.encode(".")[0]
    pprint([args, dataset_config, model_config])
    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)

    # Create the model and optimizer
    pg = None
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.actor_num_gpus_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    inits = []
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.actor_num_gpus_per_node,
        args.single_gpu_mode,
    )
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits.extend(
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url) for model in policy_group.models
    )
    max_len = dataset_config.max_prompt_token_length + args.response_length
    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.vllm_enforce_eager,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        args.enable_prefix_caching,
        max_len,
        args.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
    )
    ray.get(inits)
    print("======== ✅ all models and vLLM engines initialized =========")

    ray.get([m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models])
    print("======== ✅ model update group setup successfully =========")


    # Setup training
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        n=args.number_samples_per_prompt,
        stop=args.stop_strings,
    )
    eval_generation_config = SamplingParams(
        temperature=0.001,
        top_p=1.0,
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        n=1,  # since we are doing greedy sampling, don't need to generate more
        stop=args.stop_strings,
    )
    train_dataset_idxs = np.arange(len(train_dataset))
    iter_dataloader = ShufflingIterator(train_dataset_idxs, args.rollout_batch_size, seed=args.seed)

    response_ids_Q = Queue(maxsize=1)
    param_prompt_Q = Queue(maxsize=1)
    evaluation_Q = Queue(maxsize=1)
    num_eval_samples = 32
    def vllm_generate(
        generation_config: SamplingParams,
        response_ids_Q: Queue,
        param_prompt_Q: Queue,
        num_training_steps: int,
        eval_prompt_token_ids: Optional[List[int]],
        evaluation_Q: Queue,
        eval_freq: int,
        resume_training_step: int = 1,
    ):
        def generate_with_engines(prompts: List[List[int]], sampling_params: SamplingParams):
            # Split queries between engines
            queries_per_engine = (len(prompts) + len(vllm_engines) - 1) // len(vllm_engines)
            split_queries = [
                prompts[i : i + queries_per_engine] for i in range(0, len(prompts), queries_per_engine)
            ]
            # Generate responses in parallel across engines
            futures = [
                vllm_engine.generate.remote(
                    sampling_params=sampling_params, prompt_token_ids=queries, use_tqdm=False
                )
                for vllm_engine, queries in zip(vllm_engines, split_queries)
            ]
            # Gather all responses
            all_outputs = ray.get(futures)
            response_ids = []
            for outputs in all_outputs:
                response_ids.extend([list(out.token_ids) for output in outputs for out in output.outputs])
            return response_ids

        for training_step in range(resume_training_step, num_training_steps + 1):
            items = param_prompt_Q.get()
            if items is None:
                break
            _, g_queries_list = items

            with Timer(f"🔥 Generation time"):
                response_ids = generate_with_engines(g_queries_list, generation_config)
            response_ids_Q.put(response_ids)

            # Evaluate the model
            if eval_prompt_token_ids is not None and (training_step - 1) % eval_freq == 0:
                response_ids = generate_with_engines(
                    eval_prompt_token_ids, eval_generation_config
                )
                evaluation_Q.put(response_ids)
    
    eval_prompt_token_ids = None
    if eval_dataset is not None:
        eval_prompt_token_ids = eval_dataset[:num_eval_samples][INPUT_IDS_PROMPT_KEY]
    resume_training_step = 1
    thread = threading.Thread(
        target=vllm_generate,
        args=(
            generation_config,
            response_ids_Q,
            param_prompt_Q,
            args.num_training_steps,
            eval_prompt_token_ids,
            evaluation_Q,
            args.eval_freq,
            resume_training_step,
        ),
    )
    thread.start()
    print("======== ✅ vllm generate thread starts =========")

    data_next = train_dataset[next(iter_dataloader)]
    queries_next = data_next[INPUT_IDS_PROMPT_KEY]
    ground_truths_next = data_next[GROUND_TRUTHS_KEY]
    datasets_next = data_next[DATASET_SOURCE_KEY]
    param_prompt_Q.put((None, queries_next))

    episode = 0
    total_tokens = 0
    start_time = time.time()
    for training_step in range(resume_training_step, args.num_training_steps + 1):
        print("-"*100)
        episode += args.rollout_batch_size * args.number_samples_per_prompt  # each sample is an episode
        queries = queries_next
        ground_truths = ground_truths_next
        datasets = datasets_next

        # ------------------------------------------------------------------------------------------------
        # Optionally evaluate the model
        try:
            evaluation_responses = evaluation_Q.get(timeout=0.01)
            print("[Main Thread] 📊 Evaluation responses received")
            table = {}
            table["prompt"] = tokenizer.batch_decode(eval_prompt_token_ids)
            table["response"] = tokenizer.batch_decode(evaluation_responses)
            table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
            df = pd.DataFrame(table)
            if args.with_tracking:
                wandb.log({"sample_completions": wandb.Table(dataframe=df)})
            else:
                print_rich_table(df.iloc[:1])
            del table
        except Empty:
            print("[Main Thread] 🙈 Evaluation responses not received")

        # ------------------------------------------------------------------------------------------------
        # Sync weights and send the next batch of prompts to vLLM
        if args.async_mode:
            if training_step != 1:
                data_next = train_dataset[next(iter_dataloader)]
                queries_next = data_next[INPUT_IDS_PROMPT_KEY]
                ground_truths_next = data_next[GROUND_TRUTHS_KEY]
                datasets_next = data_next[DATASET_SOURCE_KEY]
                # torch.cuda.synchronize() # this could make the measurement more accurate
                with Timer(f"🔄 Loading weights using shared memory"):
                    ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
            param_prompt_Q.put((None, queries_next))
        else:
            if training_step != 1:
                # NOTE: important: the indent here is different for sync mode
                # we also set to use `queries = queries_next` immediately
                data_next = train_dataset[next(iter_dataloader)]
                queries_next = data_next[INPUT_IDS_PROMPT_KEY]
                ground_truths_next = data_next[GROUND_TRUTHS_KEY]
                datasets_next = data_next[DATASET_SOURCE_KEY]
                with Timer(f"🔄 Loading weights using shared memory"):
                    ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
                param_prompt_Q.put((None, queries_next))
                queries = queries_next
                ground_truths = ground_truths_next
                datasets = datasets_next

        # ------------------------------------------------------------------------------------------------
        # Pack sequences
        if args.number_samples_per_prompt > 1:
            queries = [item for item in queries for _ in range(args.number_samples_per_prompt)]
            ground_truths = [item for item in ground_truths for _ in range(args.number_samples_per_prompt)]
            datasets = [item for item in datasets for _ in range(args.number_samples_per_prompt)]
        with Timer(f"🚀 Getting response ids"):
            responses = response_ids_Q.get()
        with Timer(f"📦 Packing sequences"):
            packed_sequences = pack_sequences(
                queries=queries,
                responses=responses,
                pack_length=args.pack_length,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        B = len(packed_sequences.query_responses) // args.world_size # essentially doing `drop_last=True`, which is fine.
        print(f"Number of training examples per device: {B=}, packed sequence fraction of original sequences: {len(packed_sequences.query_responses) / len(responses)}")
        if B == 0:
            print("🤡 After packing, there is not enough data to train")
            continue

        total_tokens += sum(len(seq) for seq in packed_sequences.query_responses)

        with Timer(f"🔥 Decoding responses", noop=True):
            global_decoded_responses = tokenizer.batch_decode(
                responses, skip_special_tokens=True
            )

        scores = [0] * len(responses)
        with Timer(f"🧮 Calculating scores"):
            # calculate the scores
            if args.add_r1_style_format_reward:
                format_scores = soft_format_reward_func(
                    global_decoded_responses, args.r1_style_format_reward
                )
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]

        with Timer(f"🏆 Applying verifiable reward"):
            if args.apply_verifiable_reward:
                verifiable_rewards = apply_verifiable_reward_fast(
                    global_decoded_responses,
                    ground_truths,
                    datasets,
                    verify_reward=args.verification_reward,
                )
                if len(verifiable_rewards) != len(scores):
                    raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")
                for i in range(len(verifiable_rewards)):
                    scores[i] = verifiable_rewards[i] + scores[i]

        with Timer("🎆 Calculating advantages"):
            scores = np.array(scores)
            print(f"{len(scores)=}")
            scores_per_prompt = scores.reshape(-1, args.number_samples_per_prompt)
            global_mean_grouped_rewards = scores_per_prompt.mean(axis=-1)
            global_mean_grouped_rewards = np.repeat(
                global_mean_grouped_rewards, args.number_samples_per_prompt, axis=0
            )
            global_std_grouped_rewards = scores_per_prompt.std(axis=-1)
            global_std_grouped_rewards = np.repeat(
                global_std_grouped_rewards, args.number_samples_per_prompt, axis=0
            )
            global_advantages = (scores - global_mean_grouped_rewards) / (
                global_std_grouped_rewards + 1e-8
            )
            global_advantages_lst = global_advantages.tolist()
            packed_advantages = []
            for i in range(len(packed_sequences.response_masks)):
                packed_response_mask = packed_sequences.response_masks[i]
                packed_advantage = torch.zeros_like(packed_response_mask, dtype=torch.float32)
                for j in range(len(packed_advantage)):
                    if packed_response_mask[j] >= 1: # note that response masks are 1-indexed
                        packed_advantage[j] = global_advantages_lst[packed_response_mask[j] - 1]
                packed_advantages.append(packed_advantage)
            packed_sequences.advantages = packed_advantages

        # ------------------------------------------------------------------------------------------------
        # Train the model
        with Timer(f"🗡️ Training"):
            print(f"Accumulation steps: {B // args.num_mini_batches=}")
            reduced_metricss = ray.get([
                policy_group.models[i].train.remote(
                    packed_query_responses=packed_sequences.query_responses[B * i : B * (i + 1)],
                    packed_attention_masks=packed_sequences.attention_masks[B * i : B * (i + 1)],
                    packed_position_ids=packed_sequences.position_ids[B * i : B * (i + 1)],
                    packed_advantages=packed_sequences.advantages[B * i : B * (i + 1)],
                    packed_response_masks=packed_sequences.response_masks[B * i : B * (i + 1)],
                    pad_token_id=tokenizer.pad_token_id,
                    accumulation_steps=B // args.num_mini_batches,
                ) for i in range(args.world_size)
            ])

            reduced_metrics = reduced_metricss[0] # it's the same for all workers
            sequence_lengths = np.array([len(response) for response in responses])
            metrics = {
                "episode": episode,
                "training_step": training_step,
                "objective/scores": np.array(scores).mean(),
                "val/sequence_lengths": sequence_lengths.mean(),
                "val/sequence_lengths_min": sequence_lengths.min(),
                "val/sequence_lengths_max": sequence_lengths.max(),
                # "lr": self.scheduler.get_last_lr()[0],
                "val/total_tokens": total_tokens,
                "epoch": episode / len(train_dataset),
                "tokens_per_second": total_tokens / (time.time() - start_time),
                **reduced_metrics,
            }
            if args.apply_verifiable_reward:
                np_verifiable_rewards = np.array(verifiable_rewards)
                metrics["objective/verifiable_reward"] = np_verifiable_rewards.mean()
                metrics["objective/verifiable_correct_rate"] = (np_verifiable_rewards > 0.0).mean()
            if args.add_r1_style_format_reward:
                metrics["val/format_scores"] = np.array(format_scores).mean()
            print_rich_single_line_metrics(metrics)
            for key, value in metrics.items():
                writer.add_scalar(key, value, episode)

    ray.shutdown()

        #     del (
        #         queries,
        #         packed_responses,
        #         packed_query_responses,
        #         packed_ref_logprobs,
        #         packed_advantages,
        #         packed_response_masks,
        #         reduced_metrics,
        #     )
        #     gc.collect()
        #     torch.cuda.empty_cache()

        #     # save steps
        #     if args.save_freq > 0 and training_step % args.save_freq == 0:
        #         checkpoint_dir = f"{args.output_dir}_checkpoints"
        #         os.makedirs(checkpoint_dir, exist_ok=True)
        #         step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
        #         os.makedirs(step_dir, exist_ok=True)
        #         print(f"Saving model at step {training_step} to {step_dir}")
        #         self.save_model(self.model, step_dir)
        #         if args.try_launch_beaker_eval_jobs_on_weka:
        #             self.launch_ai2_evals_on_weka(step_dir, training_step)
        # print(f"Saving final model at step {training_step} to {args.output_dir}")
        # self.save_model(self.model, args.output_dir)
        # if args.try_launch_beaker_eval_jobs_on_weka:
        #     self.launch_ai2_evals_on_weka(args.output_dir)

        # # Ai2 logic: we use /output to store the artifacts of the job, so we
        # # make a copy of the model to `/output` in the end.
        # if (
        #     args.try_auto_save_to_beaker
        #     and self.rank == 0
        #     and is_beaker_job()
        #     and len(self.beaker_config.beaker_dataset_id_urls) > 0
        #     and args.output_dir.rstrip("/") != "/output"
        # ):
        #     shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
        # print("finished training")
    # metrics_queue = RayQueue()
    # refs = []
    # for i, policy_model in enumerate(policy_group.models):
    #     refs.append(
    #         policy_model.train.remote(
    #             train_dataset=train_dataset,
    #             eval_dataset=eval_dataset,
    #             tokenizer=tokenizer,
    #             vllm_engines=vllm_engines,
    #             metrics_queue=metrics_queue,
    #         )
    #     )

    # # somtimes a worker dies due to CUDA issues, but the rest of the cluster would just hang
    # # so we need kill the ray cluster when this happens.
    # stop_event = threading.Event()
    # threading.Thread(target=kill_ray_cluster_if_a_worker_dies, args=(refs, stop_event)).start()

    # # train and gather metrics
    # resume_training_step = 1
    # for training_step in range(resume_training_step, args.num_training_steps + 1):
    #     result = metrics_queue.get()
    #     metrics, episode, df = result
    #     for key, value in metrics.items():
    #         writer.add_scalar(key, value, episode)

    #     if df is not None:
    #         if args.with_tracking:
    #             wandb.log({"sample_completions": wandb.Table(dataframe=df)})
    #         else:
    #             print_rich_table(df.iloc[:1])
    # ray.get(refs)

    # # save model
    
    # stop_event.set()

    # accelerator = Namespace()
    # accelerator.is_main_process = True  # hack
    # if args.push_to_hub:
    #     print("Pushing model to hub")
    #     push_folder_to_hub(
    #         accelerator,
    #         args.output_dir,
    #         args.hf_repo_id,
    #         args.hf_repo_revision,
    #     )


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, DatasetConfig, ModelConfig))
    main(*parser.parse())
