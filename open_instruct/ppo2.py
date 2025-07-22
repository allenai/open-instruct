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
"""
ppo2.py is a modified version of ppo_vllm_thread_ray_gtrl.py. The main difference is that
it uses async model by default and add KL to the loss directly instead of using the
KL penalty term in the rewards.
"""

# isort: off
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
try:
    import deepspeed
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    # @vwxyzjn: when importing on CPU-only machines, we get the following error:
    # RuntimeError: 0 active drivers ([]). There should only be one.
    # so we need to catch the exception and do nothing
    # https://github.com/deepspeedai/DeepSpeed/issues/7028
except Exception:
    pass
# isort: on

import gc
import json
import logging
import math
import random
import shutil
import socket
import subprocess
import threading
import time
from argparse import Namespace
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import ray
import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
from datasets import Dataset
from huggingface_hub import HfApi
from peft import PeftModel, get_peft_model_state_dict
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from transformers.integrations import HfDeepSpeedConfig
from vllm import SamplingParams

from open_instruct.dataset_processor import SimpleGenerateCollatorWithGroundTruth
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.ground_truth_utils import soft_format_reward_func
from open_instruct.model_utils import (
    ModelConfig,
    apply_verifiable_reward,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    get_reward,
    log_softmax_and_gather,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
    truncate_response,
)
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    upload_metadata_to_hf,
)
from open_instruct.vllm_utils3 import create_vllm_engines, init_process_group

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
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_filter_v1"])
    """The list of transform functions to apply to the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: Optional[str] = None
    """The hash of the dataset configuration."""
    dataset_config_eval_hash: Optional[str] = None
    """The hash of the dataset configuration for evaluation."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    max_token_length: int = 512
    """The maximum token length to use for the dataset"""
    max_prompt_token_length: int = 256
    """The maximum prompt token length to use for the dataset"""

    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """A unique name of this run"""
    saved_tokenizer_type: Literal["original", "tokenizer_config"] = "tokenizer_config"
    """The type of tokenizer to save (original means the unmodified tokenizer directly loaded from .from_pretrained(),
    tokenizer_config means the tokenizer obtained via `TokenizerConfig`"""

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
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 1.0
    """the lambda value for GAE (default to 1 per https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf)"""
    kl_estimator: Literal["kl1", "kl2", "kl3"] = "kl1"
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
    output_dir: str = "output"
    """Where to save the model"""
    checkpoint_output_dir: Optional[str] = None
    """Where to save the model checkpoints in case of preemption"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    save_value_model: bool = False
    """Whether to save the value model along with the policy model"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    oe_eval_tasks: Optional[List[str]] = None
    """The beaker evaluation tasks to launch"""
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    """the priority of auto-launched evaluation jobs"""
    hf_metadata_dataset: Optional[str] = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""


def process_dataset_mixer(value) -> Tuple[Optional[dict], Optional[str]]:
    # if passed through cli: convert the dataset mixers to dictionaries
    if isinstance(value, str):
        return json.loads(value), value
    # if passed through yaml: convert the dataset mixers to strings
    elif isinstance(value, dict):
        return value, json.dumps(value)
    else:
        raise ValueError("Input must be either a string or a dictionary")


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
        "offload_optimizer": {"device": "cpu" if adam_offload else "none", "pin_memory": True},
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
        "bf16": {"enabled": bf16},
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
    }


def get_eval_ds_config(offload, stage=0, bf16=True):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {"device": "cpu" if offload else "none", "pin_memory": True},
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": bf16},
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
            format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
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

        ds_config = get_train_ds_config(offload=False, adam_offload=False, stage=args.deepspeed_stage, bf16=True)
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

        self.policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        self.policy_vocab_size = self.policy.config.vocab_size
        disable_dropout_in_model(self.policy)
        self.policy.gradient_checkpointing_enable()
        # from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
        # AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        # AdamOptimizer = FusedAdam
        # weight_decay = 0.0
        # optim_params = get_optimizer_grouped_parameters(self.policy, weight_decay)
        # self.optimizer = AdamOptimizer(optim_params, lr=args.learning_rate)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=args.learning_rate)
        num_scheduler_steps = args.num_training_steps * args.num_epochs * args.num_mini_batches
        warm_up_steps = args.warm_up_steps
        if args.warmup_ratio > 0.0:
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

        # value model
        from open_instruct.olmo_adapter import (
            Olmo2Config,
            Olmo2ForSequenceClassification,
            OlmoeConfig,
            OlmoeForSequenceClassification,
        )

        AutoModelForSequenceClassification.register(Olmo2Config, Olmo2ForSequenceClassification)
        AutoModelForSequenceClassification.register(OlmoeConfig, OlmoeForSequenceClassification)
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
            num_training_steps=num_scheduler_steps,
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
            # inference model only has stage 3 (sharding) or stage 0 (no sharding)
            # stage 2 is optimizer sharding which doesn't apply to inference
            stage=args.deepspeed_stage if args.deepspeed_stage == 3 else 0,
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
        if args.reward_model_multiplier:
            self.reward_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
                args.reward_model_path,
                revision=args.reward_model_revision,
                num_labels=1,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                use_cache=False,
            )
            self.reward_model_vocab_size = self.reward_model.config.vocab_size
            if self.policy_vocab_size != self.reward_model_vocab_size:
                raise ValueError(
                    "Policy and reward model must have the same vocab size. "
                    f"Policy: {self.policy_vocab_size}, Reward: {self.reward_model_vocab_size}. "
                    "If they don't have the same vocab size, the policy could generate tokens which "
                    "is going to cause index out of bound error in the reward model."
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
            ds_config["train_batch_size"] = args.mini_batch_size
            self.reward_model, *_ = deepspeed.initialize(model=self.reward_model, config=ds_config)
            self.reward_model.eval()

        assert args.reward_model_multiplier or args.apply_verifiable_reward, (
            "Either `reward_model_multiplier` must be non-zero or `apply_verifiable_reward` must be True."
        )

    def forward(
        self,
        model: PreTrainedModel,
        query_response: torch.LongTensor,
        response: torch.LongTensor,
        pad_token_id: int,
        context_length: int,
        temperature: float,
    ) -> torch.Tensor:
        attention_mask = query_response != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_response, ~attention_mask, 0)
        output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=True)
        logits = output.logits[:, context_length - 1 : -1]
        logits /= temperature + 1e-7
        logprob = log_softmax_and_gather(logits, response)
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
        self.tokenizer = tokenizer

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
            vllm_num_engines, vllm_tensor_parallel_size = (args.vllm_num_engines, args.vllm_tensor_parallel_size)
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            backend = args.vllm_sync_backend
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
            stop=args.stop_strings,
        )
        evaluation_generation_config = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.response_length,
            include_stop_str_in_output=True,
            n=1,  # since we are doing greedy sampling, don't need to generate more
            stop=args.stop_strings,
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
            def generate_with_engines(prompts: List[List[int]], sampling_params: SamplingParams):
                # Split queries between engines
                queries_per_engine = math.ceil(len(prompts) / len(vllm_engines))
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

                with Timer("🔥🔥🔥 Generation time", noop=self.rank != 0):
                    response_ids = generate_with_engines(g_queries_list, generation_config)
                response_ids_Q.put(response_ids)

                # Evaluate the model
                if sample_evaluation_prompt_token_ids is not None and (training_step - 1) % eval_freq == 0:
                    response_ids = generate_with_engines(
                        sample_evaluation_prompt_token_ids, evaluation_generation_config
                    )
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
        stats_shape = (args.num_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        non_score_reward_sum_stats = torch.zeros(stats_shape, device=device)
        kl1_stats = torch.zeros((args.local_total_prompts), device=device)
        kl2_stats = torch.zeros((args.local_total_prompts), device=device)
        kl3_stats = torch.zeros((args.local_total_prompts), device=device)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        local_metrics = MetricsTracker(max_metrics=32, device=device)
        episode = args.rollout_batch_size * (resume_training_step - 1)

        # training loop
        start_time = time.time()
        eval_futures = deque([])
        global_data = next(iter_dataloader)
        data = data_collator(
            global_data[self.rank * args.local_rollout_batch_size : (self.rank + 1) * args.local_rollout_batch_size]
        )
        global_queries = data_collator(global_data)[
            INPUT_IDS_PROMPT_KEY
        ].tolist()  # can be simplified since we `remove_padding` later anyway
        queries_next = data[INPUT_IDS_PROMPT_KEY].to(device)
        ground_truths_next = data[GROUND_TRUTHS_KEY]
        datasets_next = data[VERIFIER_SOURCE_KEY]
        if self.rank == 0:
            param_prompt_Q.put((None, remove_padding(global_queries, tokenizer.pad_token_id)))

        # for _ in range(1, resume_training_step):  # we didn't store scheduler state
        #     scheduler.step()

        for training_step in range(resume_training_step, args.num_training_steps + 1):
            episode += args.rollout_batch_size * args.number_samples_per_prompt  # each sample is an episode
            queries = queries_next
            ground_truths = ground_truths_next
            datasets = datasets_next

            if self.rank == 0:
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

            # async mode implementation
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
                datasets_next = data[VERIFIER_SOURCE_KEY]
                with Timer("🔥🔥🔥 Loading weights using shared memory", noop=self.rank != 0):
                    broadcast_to_vllm()
            if self.rank == 0:
                param_prompt_Q.put((None, remove_padding(global_queries, tokenizer.pad_token_id)))

            torch.cuda.empty_cache()
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
                ref_logprobs = []
                scores = []
                verifiable_counts = []
                sequence_lengths = []
                values = []
                per_func_rewards = defaultdict(list)
                if self.rank == 0:
                    g_response_token_ids = response_ids_Q.get()
                    DUMMY_PAD_TOKEN = args.stop_token_id  # we can't use tokenizer.pad_token_id because it's outside vocab and `torch.gather(all_logprob, 2, response.unsqueeze(-1))` will error out
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
                query_responses = torch.cat((queries, local_vllm_responses), 1)
                decoded_response = tokenizer.batch_decode(local_vllm_responses)

                if args.add_r1_style_format_reward:
                    format_scores = torch.tensor(
                        soft_format_reward_func(decoded_response, args.r1_style_format_reward), device=device
                    )
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]

                    # Get reference model logprob
                    ref_logprob = self.forward(
                        self.ref_policy,
                        query_response,
                        response,
                        tokenizer.pad_token_id,
                        context_length,
                        args.temperature,
                    )
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )
                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                    score = torch.zeros(query.shape[0], device=query.device)
                    if args.reward_model_multiplier:
                        _, score, _ = get_reward(
                            self.reward_model, postprocessed_query_response, tokenizer.pad_token_id, context_length
                        )
                        score *= args.reward_model_multiplier
                    if args.apply_verifiable_reward:
                        # we need to batch the gt to match query.
                        ground_truth = ground_truths[i : i + args.local_rollout_forward_batch_size]
                        dataset = datasets[i : i + args.local_rollout_forward_batch_size]
                        verifiable_reward, per_func_reward = apply_verifiable_reward(
                            responses=postprocessed_response,
                            decoded_responses=decoded_response,
                            ground_truths=ground_truth,
                            datasets=dataset,
                            reward_mult=args.verification_reward,
                        )
                        verifiable_reward = torch.tensor(verifiable_reward, device=score.device)
                        verifiable_count = verifiable_reward > 0
                        score += verifiable_reward
                        # For each sample, aggregate each per-function reward into a single dict.
                        for reward_dict in per_func_reward:
                            for key, value in reward_dict.items():
                                per_func_rewards[key].append(value)
                    else:
                        verifiable_count = torch.tensor([0.0], device=device).float()

                    if args.add_r1_style_format_reward:
                        score += format_scores[i : i + args.local_rollout_forward_batch_size]

                    full_value, _, _ = get_reward(
                        self.value_model, query_response, tokenizer.pad_token_id, context_length
                    )
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    values.append(value)
                    verifiable_counts.append(verifiable_count)

                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                verifiable_counts = torch.cat(verifiable_counts, 0)
                verifiable_correct_rate = verifiable_counts.sum() / queries.shape[0]
                values = torch.cat(values, 0)
                del (ref_logprob, full_value, value, score)
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
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                rewards = torch.zeros_like(ref_logprobs)
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

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

            # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
            old_logprobs = torch.zeros_like(ref_logprobs)
            for epoch_idx in range(args.num_epochs):
                b_inds = np.random.permutation(args.local_total_prompts)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_total_prompts, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    # NOTE: deepspeed handles gradient accumulation automatically; see https://github.com/microsoft/DeepSpeed/issues/758#issuecomment-801580724
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        # print("micro batch start", micro_batch_start, self.rank)
                        micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_reflogprobs = ref_logprobs[micro_batch_inds]
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
                            vpred, mb_values - args.cliprange_value, mb_values + args.cliprange_value
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(vf_loss_max, ~mb_padding_mask_p1)
                        self.value_model.backward(vf_loss * args.vf_coef)
                        self.value_model.step()

                        new_logprobs = self.forward(
                            self.model,
                            mb_query_responses,
                            mb_responses,
                            tokenizer.pad_token_id,
                            context_length,
                            args.temperature,
                        )
                        new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                        if epoch_idx == 0:
                            # .detach() is important here. See the following blog post for more details:
                            # https://costa.sh/blog-understanding-why-there-isn't-a-log-probability-in-trpo-and-ppo's-objective
                            old_logprobs[micro_batch_inds] = new_logprobs.detach()
                        mb_logprobs = old_logprobs[micro_batch_inds]
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)

                        # Here we recalculate kl: we want the KL loss to backpropagate through the model
                        # We also clamp the KL loss to avoid numerical instability
                        # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
                        ref_logprobs_diff = (new_logprobs - mb_reflogprobs).clamp(-40.0, 40.0)
                        kl1 = ref_logprobs_diff
                        kl2 = (ref_logprobs_diff) ** 2 / 2
                        kl3 = torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff  # this is more numerically stable
                        kl4 = ratio * ref_logprobs_diff
                        if args.kl_estimator == "kl1":
                            kl = kl1
                        elif args.kl_estimator == "kl2":
                            kl = kl2
                        elif args.kl_estimator == "kl3":
                            kl = kl3
                        elif args.kl_estimator == "kl4":
                            kl = kl4

                        # directly apply the KL loss (like GRPO)
                        loss = masked_mean(pg_loss_max + (args.beta * kl), ~padding_mask[micro_batch_inds])
                        self.model.backward(loss)
                        self.model.step()
                        with torch.no_grad():
                            if epoch_idx == 0:
                                kl1_stats[micro_batch_inds] = kl1.sum(1).float()
                                kl2_stats[micro_batch_inds] = kl2.sum(1).float()
                                kl3_stats[micro_batch_inds] = kl3.sum(1).float()
                                # don't need to keep track of kl4 because it's the same as kl1 numerically
                            vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~mb_padding_mask_p1)
                            pg_clipfrac = masked_mean(
                                (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                            )
                            non_score_reward = -args.beta * kl
                            non_score_reward_sum_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                non_score_reward.sum(1).mean()
                            )
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = masked_mean(
                                pg_loss_max, ~padding_mask[micro_batch_inds]
                            )
                            vf_loss_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            ratio_stats[epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # fmt: off
                    del mb_advantage, mb_responses, mb_query_responses, mb_logprobs, mb_return, mb_values, mb_padding_mask_p1
                    del new_logprobs, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max, loss
                    # fmt: on
                    # del everything and empty cache
                    torch.cuda.empty_cache()
                del b_inds, mini_batch_inds
            with torch.no_grad():
                local_metrics.add("val/sequence_lengths", sequence_lengths.float().mean())
                local_metrics.add("val/sequence_lengths_min", sequence_lengths.float().min())
                local_metrics.add("val/sequence_lengths_max", sequence_lengths.float().max())
                local_metrics.add("val/num_stop_token_ids", (responses == args.stop_token_id).sum().float().mean())
                local_metrics.add("objective/kl", kl1_stats.mean())
                local_metrics.add("objective/kl2", kl2_stats.mean())
                local_metrics.add("objective/kl3", kl3_stats.mean())
                local_metrics.add("objective/entropy", (-old_logprobs).sum(1).mean())
                local_metrics.add("objective/non_score_reward", non_score_reward_sum_stats.mean())
                local_metrics.add("objective/rlhf_reward", scores.mean() + non_score_reward_sum_stats.mean())
                local_metrics.add("objective/scores", scores.mean())
                local_metrics.add("objective/verifiable_correct_rate", verifiable_correct_rate)
                local_metrics.add("loss/policy_avg", pg_loss_stats.mean())
                local_metrics.add("loss/value_avg", vf_loss_stats.mean())
                local_metrics.add("policy/approxkl_avg", approxkl_stats.mean())
                local_metrics.add("policy/clipfrac_avg", pg_clipfrac_stats.mean())
                local_metrics.add("val/clipfrac_avg", vf_clipfrac_stats.mean())
                local_metrics.add("policy/entropy_avg", entropy_stats.mean())
                local_metrics.add("val/ratio", ratio_stats.mean())
                local_metrics.add("val/ratio_var", ratio_stats.var())
                local_metrics.add("val/stop_token_rate", contain_stop_token.float().mean())
                if args.add_r1_style_format_reward:
                    local_metrics.add("val/format_scores", format_scores.float().mean())
                for key, reward_list in per_func_rewards.items():
                    rewards_tensor = torch.tensor(reward_list, device=device)
                    # Mean per-function reward
                    local_metrics.add(f"objective/{key}_reward", rewards_tensor.mean())
                    # Correct rate: fraction of samples with positive reward
                    local_metrics.add(f"objective/{key}_correct_rate", (rewards_tensor > 0).float().mean())

                metrics = {
                    "episode": episode,
                    "training_step": training_step,
                    "lr": self.scheduler.get_last_lr()[0],
                    "epoch": episode / len(train_dataset),
                    "time/from_scratch": time.time() - start_time,
                    "time/training": time.time() - training_time_start,
                    **local_metrics.get_reduced_metrics(),
                }
                if self.rank == 0:
                    print_rich_single_line_metrics(metrics)
                    metrics_queue.put((metrics, episode, df))
            del (queries, responses, postprocessed_responses, ref_logprobs, sequence_lengths, scores, values)
            del (metrics, kl, non_score_reward)
            gc.collect()
            torch.cuda.empty_cache()

            # save steps
            if args.save_freq > 0 and training_step % args.save_freq == 0:
                checkpoint_dir = f"{args.output_dir}_checkpoints"
                step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
                print(f"Saving model at step {training_step} to {step_dir}")
                self.save_model(self.model, step_dir)
                if args.try_launch_beaker_eval_jobs_on_weka:
                    leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
                    if self.rank == 0 and is_beaker_job():
                        eval_futures.append(
                            ray.remote(launch_ai2_evals_on_weka)
                            .options(num_cpus=1)
                            .remote(
                                step_dir,
                                leaderboard_name,
                                args.oe_eval_max_length,
                                self.wandb_url,
                                training_step,
                                args.oe_eval_tasks,
                                args.stop_strings,
                                args.gs_bucket_path,
                                args.eval_priority,
                            )
                        )
                        # if a future is done, remove it from the deque
                        if len(eval_futures) > 0:
                            is_ready = len(ray.wait([eval_futures[0]], timeout=0.001)[0]) > 0
                            if is_ready:
                                print(f"Eval future {eval_futures[0]} is done")
                                eval_futures.popleft()
        print(f"Saving final model at step {training_step} to {args.output_dir}")
        self.save_model(self.model, args.output_dir)
        if args.try_launch_beaker_eval_jobs_on_weka:
            leaderboard_name = args.hf_repo_revision
            if self.rank == 0 and is_beaker_job():
                eval_futures.append(
                    ray.remote(launch_ai2_evals_on_weka)
                    .options(num_cpus=1)
                    .remote(
                        args.output_dir,
                        leaderboard_name,
                        args.oe_eval_max_length,
                        self.wandb_url,
                        training_step,
                        args.oe_eval_tasks,
                        args.stop_strings,
                        args.gs_bucket_path,
                        args.eval_priority,
                    )
                )
                ray.get(list(eval_futures))
                print("======== ✅ Evaluation jobs finished =========")

        # Ai2 logic: we use /output to store the artifacts of the job, so we
        # make a copy of the model to `/output` in the end.
        if (
            args.try_auto_save_to_beaker
            and self.rank == 0
            and is_beaker_job()
            and len(self.beaker_config.beaker_dataset_id_urls) > 0
            and args.output_dir.rstrip("/") != "/output"
        ):
            shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
        print("finished training")

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

            assert state_dict_keys.issubset(output_state_dict_keys), (
                f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"
            )

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
            self.tokenizer.save_pretrained(output_dir)


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
        self, pg: PlacementGroup, ray_process_cls: RayProcess, num_gpus_per_node: List[int], single_gpu_mode: bool
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
                placement_group=self.pg, placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node)
            )
            worker_policy = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(world_size, rank, 0, master_addr, master_port)
            self.models.append(worker_policy)


def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig):
    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if (
        tc.tokenizer_revision != model_config.model_revision
        and tc.tokenizer_name_or_path != model_config.model_name_or_path
    ):
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{model_config.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{model_config.model_name_or_path=}`."""
        print(warning)
    tokenizer = tc.tokenizer

    # ------------------------------------------------------------
    # Set up runtime variables
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
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
        assert args.local_mini_batch_size >= 8, (
            f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        )
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

    # ------------------------------------------------------------
    # Setup experiment tracking and seeds
    all_configs = {}
    beaker_config = None
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(tc), **asdict(model_config))
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

    # ------------------------------------------------------------
    # Set up datasets
    transform_fn_args = [
        {},
        {"max_token_length": args.max_token_length, "max_prompt_token_length": args.max_prompt_token_length},
    ]
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
    )
    train_dataset = train_dataset.shuffle(seed=args.seed)
    eval_dataset = None
    if len(args.dataset_mixer_eval_list) > 0:
        eval_dataset = get_cached_dataset_tulu(
            args.dataset_mixer_eval_list,
            args.dataset_mixer_eval_list_splits,
            tc,
            args.dataset_transform_fn,
            transform_fn_args,
            hf_entity=args.hf_entity,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_eval_hash,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )
        eval_dataset = eval_dataset.shuffle(seed=args.seed)
    if args.cache_dataset_only:
        return

    data_collator = SimpleGenerateCollatorWithGroundTruth(pad_token_id=tokenizer.pad_token_id)

    # some more runtime logging
    pprint([args, tc, model_config])
    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)

    # create the model and optimizer
    pg = None
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.actor_num_gpus_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    inits = []
    policy_group = ModelGroup(pg, PolicyTrainerRayProcess, args.actor_num_gpus_per_node, args.single_gpu_mode)
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits.extend(
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url) for model in policy_group.models
    )
    max_len = args.max_prompt_token_length + args.response_length
    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.vllm_enforce_eager,
        tc.tokenizer_name_or_path,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        args.enable_prefix_caching,
        max_len,
        args.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
    )

    metrics_queue = RayQueue()
    ray.get(inits)
    print("======== all models initialized =========")

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
            dataset_list = args.dataset_mixer_list
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
                --upload_to_hf {args.hf_metadata_dataset} \
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
        push_folder_to_hub(accelerator, args.output_dir, args.hf_repo_id, args.hf_repo_revision)


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    main(*parser.parse())
