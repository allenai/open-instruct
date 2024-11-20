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

import deepspeed
import numpy as np
import pandas as pd
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import vllm
from datasets import Dataset, DatasetDict
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from huggingface_hub import HfApi
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from transformers.integrations import HfDeepSpeedConfig
from vllm import SamplingParams

from open_instruct.dataset_processor import (
    CHAT_TEMPLATES,
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    DatasetConfig,
    SFTGroundTruthDatasetProcessor,
    SimpleGenerateCollatorWithGroundTruth,
    visualize_token,
)
from open_instruct.model_utils import (
    ModelConfig,
    apply_verifiable_reward,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
    truncate_response,
)
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    combine_dataset,
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
    dataset_mixer: str = None
    """A dictionary of datasets (local or HF) to sample from."""
    dataset_train_splits: List[str] = None
    """The dataset splits to use for training"""
    dataset_eval_mixer: Optional[str] = None
    """A dictionary of datasets (local or HF) to sample from for evaluation"""
    dataset_eval_splits: Optional[List[str]] = None
    """The dataset splits to use for evaluation"""
    dataset_mixer_dict: Optional[dict] = None
    """The dataset mixer as a dictionary"""
    dataset_eval_mixer_dict: Optional[dict] = None
    """The dataset eval mixer as a dictionary"""

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
    num_train_epochs: int = 1
    """Number of epochs to train"""
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
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""
    reward_model_path: str = "EleutherAI/pythia-160m"
    """the path to the reward model"""
    reward_model_revision: Optional[str] = None
    """the revision of the reward model"""
    init_value_from_scratch: bool = False
    """whether to initialize the value model from scratch"""

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

    # online PPO specific args
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    cliprange: float = 0.2
    """the clip range"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
    kl_estimator: Literal["kl1", "kl2", "kl3"] = "kl1"
    """the KL estimator to use"""
    apply_verifiable_reward: bool = False
    """whether to apply verifiable reward"""
    reward_model_multiplier: float = 1.0
    """the reward model multiplier, for down/upscaling the reward model output"""
    answer_extraction_model: str = None

    # async setting
    async_mode: bool = True
    """Whether to run the generation in async mode which learns from the second latest policy like Cleanba (https://arxiv.org/abs/2310.00036)"""

    # ray
    actor_num_gpus_per_node: List[int] = field(default_factory=lambda: [1])
    """number of gpus per node for actor"""
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine for multi-GPU inference"""
    vllm_sync_backend: str = "nccl"
    """DeepSpeed -> vLLM weight sync backend"""
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

    # Ai2 specific settings
    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    oe_eval_tasks: Optional[List[str]] = None
    """The beaker evaluation tasks to launch"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""

    def __post_init__(self):
        self.dataset_mixer_dict, self.dataset_mixer = process_dataset_mixer(self.dataset_mixer)
        if self.dataset_eval_mixer is not None:
            self.dataset_eval_mixer_dict, self.dataset_eval_mixer = process_dataset_mixer(self.dataset_eval_mixer)


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
    # args.world_size = accelerator.num_processes
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.gradient_accumulation_steps = exact_div(
        args.local_mini_batch_size,
        args.per_device_train_batch_size,
        "`local_mini_batch_size` must be a multiple of `per_device_train_batch_size`",
    )
    args.world_size = sum(args.actor_num_gpus_per_node)
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.rollout_batch_size = int(args.local_rollout_batch_size * args.world_size)
    args.mini_batch_size = int(args.local_mini_batch_size * args.world_size)
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
        deepspeed.init_distributed()

        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=False,
            stage=args.deepspeed_stage,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["train_batch_size"] = args.mini_batch_size
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
        num_training_steps = args.num_training_steps * args.num_train_epochs * args.num_epochs
        warm_up_steps = args.warm_up_steps
        if args.warmup_ratio >= 0.0:
            warm_up_steps = int(num_training_steps * args.warmup_ratio)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_training_steps,
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

        # value model
        self.value_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model_path,
            revision=args.reward_model_revision,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        if args.init_value_from_scratch:
            self.value_model.init_weights()  # re-initialize the value model from scratch
        disable_dropout_in_model(self.value_model)
        self.value_model.gradient_checkpointing_enable()
        # AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        # AdamOptimizer = FusedAdam
        # weight_decay = 0.0
        # optim_params = get_optimizer_grouped_parameters(self.value_model, weight_decay)
        # self.optimizer = AdamOptimizer(optim_params, lr=args.learning_rate)
        self.optimizer = torch.optim.AdamW(self.value_model.parameters(), lr=args.learning_rate)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_training_steps,
        )
        self.value_model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.value_model,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=True,
        )
        self.value_model.train()

        # reference model
        ds_config = get_eval_ds_config(
            offload=False,
            stage=args.deepspeed_stage,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["train_batch_size"] = args.mini_batch_size
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
            stage=args.deepspeed_stage,
            bf16=True,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["train_batch_size"] = args.mini_batch_size
        self.reward_model, *_ = deepspeed.initialize(model=self.reward_model, config=ds_config)
        self.reward_model.eval()

    def get_vocab_size(self):
        return self.policy.config.vocab_size

    def forward(
        self,
        query_response: torch.LongTensor,
        response: torch.LongTensor,
        pad_token_id: int,
        context_length: int,
        temperature: float,
    ) -> torch.Tensor:
        output = forward(self.model, query_response, pad_token_id)
        logits = output.logits[:, context_length - 1 : -1]
        logits /= temperature + 1e-7
        all_logprob = F.log_softmax(logits, dim=-1)
        logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
        return logprob

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        vllm_engines: List[ray.actor.ActorHandle],
        metrics_queue: RayQueue,
        data_collator: Callable,
    ):
        torch.set_printoptions(precision=4, sci_mode=False)

        args = self.args

        accelerator = Namespace()
        accelerator.process_index = self.rank
        accelerator.num_processes = self.world_size
        accelerator.is_main_process = self.rank == 0
        torch.distributed.barrier()
        if self.rank == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            vllm_num_engines, vllm_tensor_parallel_size = (
                args.vllm_num_engines,
                args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            backend = args.vllm_sync_backend
            # https://github.com/OpenRLHF/OpenRLHF/issues/313
            if vllm.__version__ > "0.4.2" and os.getenv("NCCL_P2P_DISABLE", "0") == "0":
                backend = "gloo"
                print(
                    "Warning: using --vllm_sync_backend=gloo for vLLM version > 0.4.2 (or export NCCL_P2P_DISABLE=1)"
                )
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

        def broadcast_to_vllm():
            # avoid OOM
            torch.cuda.empty_cache()
            model = self.model.module
            count, num_params = 0, len(list(model.named_parameters()))
            refss = []
            if args.gather_whole_model:
                with deepspeed.zero.GatheredParameters(model.parameters(), enabled=args.deepspeed_stage == 3):
                    for name, param in model.named_parameters():
                        count += 1  # empty_cache at last param
                        # Fire all vllm engines for broadcast
                        if torch.distributed.get_rank() == 0:
                            shape = param.shape if args.deepspeed_stage != 3 else param.ds_shape
                            refs = [
                                engine.update_weight.remote(
                                    name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                                )
                                for engine in vllm_engines
                            ]
                            refss.extend(refs)
                        if torch.distributed.get_rank() == 0:
                            torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
            else:  # broadcast each parameter independently
                for name, param in model.named_parameters():
                    count += 1
                    if torch.distributed.get_rank() == 0:
                        shape = param.shape if args.deepspeed_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight.remote(
                                name, dtype=param.dtype, shape=shape, empty_cache=count == num_params
                            )
                            for engine in vllm_engines
                        ]
                        refss.extend(refs)
                    with deepspeed.zero.GatheredParameters([param], enabled=args.deepspeed_stage == 3):
                        if torch.distributed.get_rank() == 0:
                            torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
            if torch.distributed.get_rank() == 0:
                ray.get(refss)

        # broadcast_to_vllm()
        if args.stop_token:
            if args.stop_token == "eos":
                args.stop_token_id = tokenizer.eos_token_id
            if args.stop_token == "period":
                args.stop_token_id = tokenizer.encode(".")[0]
        # data_collator = SimpleGenerateCollator(pad_token_id=tokenizer.pad_token_id)
        train_dataset_idxs = np.arange(len(train_dataset))
        shuffling_iter = ShufflingIterator(train_dataset_idxs, args.rollout_batch_size, seed=args.seed)

        # hack to left pad
        def repeat_generator():
            while True:
                batch_idxs = next(shuffling_iter)
                yield [train_dataset[i] for i in batch_idxs]

        iter_dataloader = iter(repeat_generator())
        generation_config = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.response_length,
            include_stop_str_in_output=True,
            n=args.number_samples_per_prompt,
        )
        # print("setup async queues")
        param_prompt_Q = None
        response_ids_Q = None
        evaluation_Q = None
        response_ids_Q = Queue(maxsize=1)
        param_prompt_Q = Queue(maxsize=1)
        evaluation_Q = Queue(maxsize=1)
        num_eval_samples = 32
        sample_evaluation_prompt_token_ids = None
        if eval_dataset is not None:
            sample_evaluation_prompt_token_ids = eval_dataset[:num_eval_samples][INPUT_IDS_PROMPT_KEY]

        def vllm_generate(
            generation_config: SamplingParams,
            response_ids_Q: Queue,
            param_prompt_Q: Queue,
            num_training_steps: int,
            sample_evaluation_prompt_token_ids: Optional[List[int]],
            evaluation_Q: Queue,
            eval_freq: int,
            resume_training_step: int,
        ):
            llm = vllm_engines[0]
            for training_step in range(resume_training_step, num_training_steps + 1):
                items = param_prompt_Q.get()
                if items is None:
                    break
                unwrapped_model, g_queries_list = items
                # if unwrapped_model is not None:
                generation_start_time = time.time()

                outputs = ray.get(
                    llm.generate.remote(
                        sampling_params=generation_config, prompt_token_ids=g_queries_list, use_tqdm=False
                    )
                )
                response_ids = [list(out.token_ids) for output in outputs for out in output.outputs]
                print(f"🔥🔥🔥 Generation time: {time.time() - generation_start_time:.2f} seconds")
                response_ids_Q.put(response_ids)

                if sample_evaluation_prompt_token_ids is not None and (training_step - 1) % eval_freq == 0:
                    outputs = ray.get(
                        llm.generate.remote(
                            prompt_token_ids=sample_evaluation_prompt_token_ids,
                            sampling_params=generation_config,
                            use_tqdm=False,
                        )
                    )
                    # for evaluation, even if we have multiple outputs, we only look at one of them for simplicity
                    response_ids = [list(output.outputs[0].token_ids) for output in outputs]
                    evaluation_Q.put(response_ids)

        resume_training_step = 1
        if accelerator.is_main_process:
            thread = threading.Thread(
                target=vllm_generate,
                args=(
                    generation_config,
                    response_ids_Q,
                    param_prompt_Q,
                    args.num_training_steps,
                    sample_evaluation_prompt_token_ids,
                    evaluation_Q,
                    args.eval_freq,
                    resume_training_step,
                ),
            )
            thread.start()
            print("vllm generate thread starts")

        # set up the metrics and initial states
        device = torch.device(self.local_rank)
        g_vllm_responses = torch.zeros(
            (args.rollout_batch_size * args.number_samples_per_prompt, args.response_length),
            device=device,
            dtype=torch.long,
        )
        stats_shape = (
            args.num_epochs,
            args.num_mini_batches * args.number_samples_per_prompt,
            args.gradient_accumulation_steps,
        )
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        local_metrics = torch.zeros((20,), device=device)
        episode = args.rollout_batch_size * (resume_training_step - 1)

        # training loop
        start_time = time.time()
        global_data = next(iter_dataloader)
        data = data_collator(
            global_data[self.rank * args.local_rollout_batch_size : (self.rank + 1) * args.local_rollout_batch_size]
        )
        global_queries = data_collator(global_data)[
            INPUT_IDS_PROMPT_KEY
        ].tolist()  # can be simplified since we `remove_padding` later anyway
        queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)
        ground_truths_next = data[GROUND_TRUTHS_KEY]
        datasets_next = data[DATASET_SOURCE_KEY]
        if accelerator.is_main_process:
            param_prompt_Q.put((None, remove_padding(global_queries, tokenizer.pad_token_id)))

        answer_extraction_model = None
        answer_extraction_tokenizer = None
        # for _ in range(1, resume_training_step):  # we didn't store scheduler state
        #     scheduler.step()

        for training_step in range(resume_training_step, args.num_training_steps + 1):
            episode += args.rollout_batch_size * args.number_samples_per_prompt  # each sample is an episode
            queries = queries_next
            ground_truths = ground_truths_next
            datasets = datasets_next

            if accelerator.is_main_process:
                df = None
                try:
                    evaluation_responses = evaluation_Q.get(timeout=0.01)
                    print("🔥🔥🔥 Evaluation responses received")
                    table = {}
                    table["prompt"] = tokenizer.batch_decode(sample_evaluation_prompt_token_ids)
                    table["response"] = tokenizer.batch_decode(evaluation_responses)
                    table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
                    df = pd.DataFrame(table)
                    del table
                except Empty:
                    print("🙈 Evaluation responses not received")

            # (optionally) evaluate the model
            if args.async_mode:
                if training_step != 1:
                    global_data = next(iter_dataloader)
                    data = data_collator(
                        global_data[
                            self.rank * args.local_rollout_batch_size : (self.rank + 1) * args.local_rollout_batch_size
                        ]
                    )
                    global_queries = data_collator(global_data)[INPUT_IDS_PROMPT_KEY].tolist()
                    queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)
                    ground_truths_next = data[GROUND_TRUTHS_KEY]
                    datasets_next = data[DATASET_SOURCE_KEY]

                start_time = time.time()
                broadcast_to_vllm()
                print(
                    f"🔥🔥🔥 Loading weights using shared memory; Time to load weights: {time.time() - start_time:.2f} seconds"
                )
                if accelerator.is_main_process:
                    param_prompt_Q.put((None, remove_padding(global_queries, tokenizer.pad_token_id)))
            else:
                if training_step != 1:
                    # NOTE: important: the indent here is different for sync mode
                    # we also set to use `queries = queries_next` immediately
                    global_data = next(iter_dataloader)
                    data = data_collator(
                        global_data[
                            self.rank * args.local_rollout_batch_size : (self.rank + 1) * args.local_rollout_batch_size
                        ]
                    )
                    global_queries = data_collator(global_data)[INPUT_IDS_PROMPT_KEY].tolist()
                    queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)
                    ground_truths_next = data[GROUND_TRUTHS_KEY]
                    datasets_next = data[DATASET_SOURCE_KEY]
                    start_time = time.time()
                    broadcast_to_vllm()
                    print(
                        f"🔥🔥🔥 Loading weights using shared memory; Time to load weights: {time.time() - start_time:.2f} seconds"
                    )
                    if accelerator.is_main_process:
                        param_prompt_Q.put((None, remove_padding(global_queries, tokenizer.pad_token_id)))
                    queries = queries_next
                    ground_truths = ground_truths_next
                    datasets = datasets_next

            torch.cuda.empty_cache()
            # print('get reward stuff starts')
            # if we generate multiple samples per prompt, we need to repeat the queries and ground truths
            # to match the vllm outputs.
            if args.number_samples_per_prompt > 1:
                queries = queries.repeat_interleave(args.number_samples_per_prompt, dim=0)
                ground_truths = [gt for gt in ground_truths for _ in range(args.number_samples_per_prompt)]
                datasets = [ds for ds in datasets for _ in range(args.number_samples_per_prompt)]

            training_time_start = time.time()
            with torch.no_grad():
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                verifiable_counts = []
                sequence_lengths = []
                values = []
                if accelerator.is_main_process:
                    g_response_token_ids = response_ids_Q.get()
                    DUMMY_PAD_TOKEN = 0  # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
                    g_padded_response_ids = [
                        response + [DUMMY_PAD_TOKEN] * (args.response_length - len(response))
                        for response in g_response_token_ids
                    ]
                    g_padded_response_ids = torch.tensor(g_padded_response_ids, device=device)
                    g_vllm_responses[:] = g_padded_response_ids
                dist.broadcast(g_vllm_responses, src=0)
                local_vllm_responses = g_vllm_responses[
                    accelerator.process_index * queries.shape[0] : (accelerator.process_index + 1) * queries.shape[0]
                ]
                # print(f"{local_vllm_responses.shape=}, {local_vllm_responses=}")
                query_responses = torch.cat((queries, local_vllm_responses), 1)
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    # print(f"get reward stuff starts {i=}")
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]

                    logprob = self.forward(
                        query_response, response, tokenizer.pad_token_id, context_length, args.temperature
                    )
                    torch.cuda.empty_cache()

                    ref_output = forward(self.ref_policy, query_response, tokenizer.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )
                    # print("get reward stuff starts 2")
                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                    )
                    if args.reward_model_multiplier != 1.0:
                        score *= args.reward_model_multiplier
                    # also apply verifiable reward
                    if args.apply_verifiable_reward:
                        # we need to batch the gt to match query.
                        ground_truth = ground_truths[i : i + args.local_rollout_forward_batch_size]
                        dataset = datasets[i : i + args.local_rollout_forward_batch_size]
                        verifiable_reward, verifiable_count = apply_verifiable_reward(
                            postprocessed_query_response,
                            tokenizer,
                            ground_truth,
                            dataset,
                            verify_reward=10,
                            answer_extraction_model=answer_extraction_model,
                            answer_extraction_tokenizer=answer_extraction_tokenizer,
                        )
                        score += verifiable_reward
                    else:
                        verifiable_count = torch.tensor([0.0], device=device).float()
                    full_value, _, _ = get_reward(
                        self.value_model, query_response, tokenizer.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                    verifiable_counts.append(verifiable_count)
                    # print(f"get reward stuff starts 5")

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                verifiable_counts = torch.cat(verifiable_counts, 0)
                verifiable_correct_rate = verifiable_counts.sum() / queries.shape[0]
                values = torch.cat(values, 0)
                # print(f"get reward stuff finished")
                del (logprob, ref_logprob, full_value, value, score)
                gc.collect()
                torch.cuda.empty_cache()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_stop_token = torch.any(postprocessed_responses == args.stop_token_id, dim=-1)
                # NOTE: only apply the stop token filter if the response is long enough
                # otherwise the model could learn to generate the first token as the stop token
                contain_stop_token = contain_stop_token & (sequence_lengths >= args.min_response_length)
                if args.non_stop_penalty:
                    scores = torch.where(
                        contain_stop_token, scores, torch.full_like(scores, args.penalty_reward_value)
                    )

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)
                # print(f"get reward stuff finished 2")

                # 4. compute rewards
                kl1 = logprobs - ref_logprobs
                kl2 = (kl1) ** 2 / 2
                kl3 = (-kl1).exp() - 1 + kl1
                if args.kl_estimator == "kl1":
                    kl = kl1
                elif args.kl_estimator == "kl2":
                    kl = kl2
                elif args.kl_estimator == "kl3":
                    kl = kl3
                # if self.rank==0:
                #     print(f"{logprobs[0][:40]=}, {ref_logprobs[0][:40]=}, {kl.sum(1)=}")
                non_score_reward = -args.beta * kl
                non_score_reward_sum = non_score_reward.sum(1)
                rlhf_reward = scores + non_score_reward_sum
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores
                # print(f"get reward stuff finished 3")

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # print('gae')
                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # print('training starts')
            # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
            for epoch_idx in range(args.num_epochs):
                b_inds = np.random.permutation(args.local_rollout_batch_size * args.number_samples_per_prompt)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0, args.local_rollout_batch_size * args.number_samples_per_prompt, args.local_mini_batch_size
                ):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        # print("micro batch start", micro_batch_start, self.rank)
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]
                        mb_return = returns[micro_batch_inds]
                        mb_values = values[micro_batch_inds]
                        mb_padding_mask_p1 = padding_mask_p1[micro_batch_inds]

                        vpred_temp = get_reward(
                            self.value_model, mb_query_responses, tokenizer.pad_token_id, context_length
                        )
                        vpred_temp = vpred_temp[0]
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        vpred = torch.masked_fill(vpred, mb_padding_mask_p1, 0)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.cliprange_value,
                            mb_values + args.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(vf_loss_max, ~mb_padding_mask_p1)
                        self.value_model.backward(vf_loss * args.vf_coef)
                        self.value_model.step()

                        new_logprobs = self.forward(
                            mb_query_responses, mb_responses, tokenizer.pad_token_id, context_length, args.temperature
                        )
                        # if self.rank==0:
                        #     print(f"{new_logprobs[0][:40]=}, {mb_logprobs[0][:40]=}")
                        new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                        loss = pg_loss
                        self.model.backward(loss)
                        # print("backward loss", self.rank, "micro batch start", micro_batch_start)
                        # print("trying to step", self.rank, "micro batch start", micro_batch_start)
                        self.model.step()
                        # print("step", self.rank, "micro batch start", micro_batch_start)
                        with torch.no_grad():
                            # print("waiting for value model step", self.rank, "micro batch start", micro_batch_start)
                            # vf_loss, vf_clipfrac = ray.get(value_model_step_future)
                            vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~mb_padding_mask_p1)
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                            )
                            # print("value model stepped", self.rank, "micro batch start", micro_batch_start)
                            # prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            # entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            # entropy_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # fmt: off
                    del mb_advantage, mb_responses, mb_query_responses, mb_logprobs, mb_return, mb_values, mb_padding_mask_p1
                    del new_logprobs, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max, pg_loss, loss
                    # del vpred_temp, vpred, vpredclipped, vf_losses1, vf_losses2, vf_loss_max
                    # del vf_loss, vf_clipfrac, pg_clipfrac, approxkl
                    # fmt: on
                    # del everything and empty cache
                    torch.cuda.empty_cache()
                del b_inds, mini_batch_inds
            # print("start metrics")
            with torch.no_grad():
                local_metrics[0] = sequence_lengths.float().mean()
                local_metrics[1] = (responses == args.stop_token_id).sum().float().mean()
                local_metrics[2] = kl.sum(1).mean()
                local_metrics[3] = (-logprobs).sum(1).mean()
                local_metrics[4] = non_score_reward_sum.mean()
                local_metrics[5] = rlhf_reward.mean()
                local_metrics[6] = scores.mean()
                local_metrics[7] = approxkl_stats.mean()
                local_metrics[8] = pg_clipfrac_stats.mean()
                local_metrics[9] = pg_loss_stats.mean()
                local_metrics[10] = vf_loss_stats.mean()
                local_metrics[11] = vf_clipfrac_stats.mean()
                local_metrics[12] = entropy_stats.mean()
                local_metrics[13] = ratio_stats.mean()
                local_metrics[14] = ratio_stats.var()
                local_metrics[15] = ((kl) ** 2 / 2).sum(1).mean()
                local_metrics[16] = ((-kl).exp() - 1 + kl).sum(1).mean()
                local_metrics[17] = verifiable_correct_rate
                local_metrics[18] = contain_stop_token.float().mean()
                # global_metrics = accelerator.reduce(local_metrics, reduction="mean").tolist()
                local_metrics /= dist.get_world_size()
                dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
                global_metrics = local_metrics.tolist()
                metrics = {
                    "episode": episode,
                    "training_step": training_step,
                    "lr": self.scheduler.get_last_lr()[0],
                    "epoch": episode / len(train_dataset),
                    "time/from_scratch": time.time() - start_time,
                    "time/training": time.time() - training_time_start,
                    "val/sequence_lengths": global_metrics[0],
                    "val/num_stop_token_ids": global_metrics[1],
                    "objective/kl": global_metrics[2],
                    "objective/kl2": global_metrics[15],
                    "objective/kl3": global_metrics[16],
                    "objective/entropy": global_metrics[3],
                    "objective/non_score_reward": global_metrics[4],
                    "objective/rlhf_reward": global_metrics[5],
                    "objective/scores": global_metrics[6],
                    "policy/approxkl_avg": global_metrics[7],
                    "policy/clipfrac_avg": global_metrics[8],
                    "loss/policy_avg": global_metrics[9],
                    "loss/value_avg": global_metrics[10],
                    "val/clipfrac_avg": global_metrics[11],
                    "policy/entropy_avg": global_metrics[12],
                    "val/ratio": global_metrics[13],
                    "val/ratio_var": global_metrics[14],
                    "objective/verifiable_correct_rate": global_metrics[17],
                    "val/stop_token_rate": global_metrics[18],
                }
                if accelerator.is_main_process:
                    print_rich_single_line_metrics(metrics)
                    metrics_queue.put((metrics, episode, df))
            del (queries, responses, postprocessed_responses, logprobs, ref_logprobs, sequence_lengths, scores, values)
            del (global_metrics, metrics, kl, non_score_reward, non_score_reward_sum, rlhf_reward)
            gc.collect()
            torch.cuda.empty_cache()
            # print(f"finished training {training_step}")

            # save steps
            if args.save_freq > 0 and training_step % args.save_freq == 0:
                checkpoint_dir = f"{args.output_dir}_checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
                os.makedirs(step_dir, exist_ok=True)
                print(f"Saving model at step {training_step} to {step_dir}")
                self.save_model(step_dir)
                if args.try_launch_beaker_eval_jobs_on_weka:
                    self.launch_ai2_evals_on_weka(step_dir, training_step)
        print(f"Saving final model at step {training_step} to {args.output_dir}")
        self.save_model(args.output_dir)
        if args.try_launch_beaker_eval_jobs_on_weka:
            self.launch_ai2_evals_on_weka(args.output_dir)

        # Ai2 logic: we use /output to store the artifacts of the job, so we
        # make a copy of the model to `/output` in the end.
        if self.rank == 0 and len(self.beaker_config.beaker_dataset_id_urls) > 0:
            shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
        print("finished training")

    def save_model(self, output_dir: str) -> None:
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # save model weights for ZeRO2/3
        model_to_save = self.model
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

            # # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            # if isinstance(model_to_save, PeftModel):
            #     model_to_save.save_pretrained(output_dir, **kwargs)
            #     if self.stage == 3:
            #         torch.save(
            #             get_peft_model_state_dict(model_to_save, output_state_dict),
            #             os.path.join(output_dir, "adapter_model.bin"),
            #         )
            # else:
            # save model
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
                dataset_list = list(args.dataset_mixer_dict.keys())
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
    --upload_to_hf allenai/tulu-3-evals \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_safety_evaluations \
    --skip_oi_evals"""
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
    ):
        self.pg = pg
        self.ray_process_cls = ray_process_cls
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 1
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

    # set up experiment tracking and seeds
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

    # create a tokenizer (pad from right)
    config = AutoConfig.from_pretrained(model_config.model_name_or_path, revision=model_config.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, revision=model_config.model_revision, padding_side="right"
    )
    if config.architectures == "LlamaForCausalLM" and config.bos_token_id == 128000:
        tokenizer.pad_token_id = 128002  # <|reserved_special_token_0|>
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding
    tokenizer.chat_template = CHAT_TEMPLATES[dataset_config.chat_template]

    # create the dataset
    dataset_dict = DatasetDict()
    dataset_processor = SFTGroundTruthDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    if len(args.dataset_train_splits) != len(args.dataset_mixer_dict) and len(args.dataset_train_splits) == 1:
        args.dataset_train_splits = [args.dataset_train_splits[0]] * len(args.dataset_mixer_dict)
        print(
            f"Dataset splits not provided for all datasets. Using the same {args.dataset_train_splits[0]} split for all datasets."
        )
    if len(args.dataset_eval_splits) != len(args.dataset_eval_mixer_dict) and len(args.dataset_eval_splits) == 1:
        args.dataset_eval_splits = [args.dataset_eval_splits[0]] * len(args.dataset_eval_mixer_dict)
        print(
            f"Dataset splits not provided for all datasets. Using the same {args.dataset_eval_splits[0]} split for all datasets."
        )
    train_dataset = combine_dataset(
        args.dataset_mixer_dict,
        splits=args.dataset_train_splits,
        columns_to_keep=[
            dataset_config.sft_messages_key,
            dataset_config.ground_truths_key,
            dataset_config.dataset_source_key,
        ],
    )
    if dataset_config.sanity_check:
        train_dataset = train_dataset.select(
            range(0, min(len(train_dataset), dataset_config.sanity_check_max_samples))
        )
    train_dataset = dataset_processor.tokenize(train_dataset)
    train_dataset = dataset_processor.filter(train_dataset, need_contain_labels=False)
    dataset_dict["train"] = train_dataset
    eval_dataset = None
    if args.dataset_eval_mixer is not None:
        eval_dataset = combine_dataset(
            args.dataset_eval_mixer_dict,
            splits=args.dataset_eval_splits,
            columns_to_keep=[
                dataset_config.sft_messages_key,
                dataset_config.ground_truths_key,
                dataset_config.dataset_source_key,
            ],
        )
        if dataset_config.sanity_check:
            eval_dataset = eval_dataset.select(
                range(0, min(len(eval_dataset), dataset_config.sanity_check_max_samples))
            )
        eval_dataset = dataset_processor.tokenize(eval_dataset)
        eval_dataset = dataset_processor.filter(eval_dataset, need_contain_labels=False)
        dataset_dict["eval"] = eval_dataset
    data_collator = SimpleGenerateCollatorWithGroundTruth(pad_token_id=tokenizer.pad_token_id)

    # some more runtime logging
    pprint([args, dataset_config, model_config])
    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)
    if args.with_tracking:
        # upload the visualized token length
        dataset_processor.get_token_length_visualization(
            dataset_dict, save_path=f"runs/{args.run_name}/token_length.png"
        )
        wandb.log({"token_length": wandb.Image(f"runs/{args.run_name}/token_length.png")})

    # create the model and optimizer
    pg = None
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.actor_num_gpus_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    inits = []
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.actor_num_gpus_per_node,
    )
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits.extend(
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url) for model in policy_group.models
    )
    max_len = dataset_config.max_prompt_token_length + args.response_length
    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        args.enable_prefix_caching,
        max_len,
    )

    metrics_queue = RayQueue()
    ray.get(inits)
    print("======== all models initialized =========")
    ray.get(policy_group.models[0].get_vocab_size.remote())
    # print(f"{policy_vocab_size=}, {reward_vocab_size=}")
    # if policy_vocab_size != reward_vocab_size:
    #     ray.shutdown()  # shutdown here so this error message is not buried in the logs
    #     raise ValueError(
    #         "Policy and reward model must have the same vocab size. "
    #         f"Policy: {policy_vocab_size}, Reward: {reward_vocab_size}. "
    #         "If they don't have the same vocab size, the policy could generate tokens which "
    #         "is going to cause index out of bound error in the reward model."
    #     )

    refs = []
    for i, policy_model in enumerate(policy_group.models):
        refs.append(
            policy_model.train.remote(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                vllm_engines=vllm_engines,
                metrics_queue=metrics_queue,
                data_collator=data_collator,
            )
        )

    # somtimes a worker dies due to CUDA issues, but the rest of the cluster would just hang
    # so we need kill the ray cluster when this happens.
    stop_event = threading.Event()
    threading.Thread(target=kill_ray_cluster_if_a_worker_dies, args=(refs, stop_event)).start()

    # train and gather metrics
    resume_training_step = 1
    for training_step in range(resume_training_step, args.num_training_steps + 1):
        result = metrics_queue.get()
        metrics, episode, df = result
        for key, value in metrics.items():
            writer.add_scalar(key, value, episode)

        if df is not None:
            if args.with_tracking:
                wandb.log({"sample_completions": wandb.Table(dataframe=df)})
            else:
                print_rich_table(df.iloc[:1])
    ray.get(refs)

    # save model
    ray.shutdown()
    stop_event.set()

    # Ai2 specific logic
    if is_beaker_job():
        if args.hf_metadata_dataset:
            dataset_list = list(args.dataset_mixer_dict.keys())
            # mainly just focussing here on what would be useful for the leaderboard.
            # wandb will have even more useful information.
            metadata_blob = {
                "model_name": args.exp_name,
                "model_type": "sft",
                "datasets": dataset_list,
                "base_model": model_config.model_name_or_path,
                "wandb_path": wandb.run.get_url(),
                "beaker_experiment": beaker_config.beaker_experiment_url,
                "beaker_datasets": beaker_config.beaker_dataset_id_urls,
            }
            upload_metadata_to_hf(
                metadata_blob,
                "metadata.json",
                args.hf_metadata_dataset,
                "results/" + args.hf_repo_revision,  # to match what the auto-evals name as.
            )

        if args.try_launch_beaker_eval_jobs and len(beaker_config.beaker_dataset_id_urls) > 0:
            command = f"""\
            python mason.py  \
                --cluster ai2/allennlp-cirrascale ai2/general-cirrascale-a5000 ai2/general-cirrascale-a5000 ai2/s2-cirrascale ai2/general-cirrascale \
                --priority low \
                --preemptible \
                --budget ai2/allennlp \
                --workspace ai2/tulu-2-improvements \
                --image nathanl/open_instruct_auto \
                --pure_docker_mode \
                --gpus 0 -- python scripts/wait_beaker_dataset_model_upload_then_evaluate_model.py \
                --beaker_workload_id {beaker_config.beaker_workload_id} \
                --model_name {args.hf_repo_revision}
            """
            process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(f"Submit jobs after model training is finished - Stdout:\n{stdout.decode()}")
            print(f"Submit jobs after model training is finished - Stderr:\n{stderr.decode()}")
            print(f"Submit jobs after model training is finished - process return code: {process.returncode}")

    accelerator = Namespace()
    accelerator.is_main_process = True  # hack
    if args.push_to_hub:
        print("Pushing model to hub")
        push_folder_to_hub(
            accelerator,
            args.output_dir,
            args.hf_repo_id,
            args.hf_repo_revision,
        )

    # The `checkpoint_output_dir` is only used in case of preemption and should be deleted if the run was successful.
    # We use `--save_freq` to save intermediate checkpoints in the output folder instead.
    if args.checkpoint_output_dir is not None and os.path.exists(args.checkpoint_output_dir):
        shutil.rmtree(args.checkpoint_output_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, DatasetConfig, ModelConfig))
    main(*parser.parse())
