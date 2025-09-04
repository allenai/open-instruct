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
# isort: off
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
try:
    import deepspeed

    # @vwxyzjn: when importing on CPU-only machines, we get the following error:
    # RuntimeError: 0 active drivers ([]). There should only be one.
    # so we need to catch the exception and do nothing
    # https://github.com/deepspeedai/DeepSpeed/issues/7028
except Exception:
    pass
# isort: on

import asyncio
import atexit
import json
import logging
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import Callable, Dict, Iterator, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import ray
import torch
import torch.utils
import torch.utils.data
from huggingface_hub import HfApi
from peft import PeftModel, get_peft_model_state_dict
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from transformers.integrations import HfDeepSpeedConfig
from vllm import SamplingParams

from open_instruct.dataset_transformation import (
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.ground_truth_utils import (
    build_all_verifiers,
    cleanup_all_llm_judge_clients,
    soft_format_reward_func,
)
from open_instruct.model_utils import (
    ModelConfig,
    apply_verifiable_reward,
    apply_finegrained_reward,
    disable_dropout_in_model,
    log_softmax_and_gather,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
)
from open_instruct.rl_utils2 import Timer, pack_sequences
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    RayProcess,
    _z3_params_to_fetch,
    calibrate_checkpoint_state_dir,
    clean_last_n_checkpoints_deepspeed,
    download_latest_checkpoint_from_gs,
    extract_user_query,
    get_beaker_whoami,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    sync_gs_bucket,
)
from open_instruct.vllm_utils3 import create_vllm_engines, init_process_group

api = HfApi()
INVALID_LOGPROB = 1.0


@dataclass
class Args:
    # Dataset
    dataset_mixer_list: List[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_eval_list: List[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    """A list of datasets (local or HF) to sample from for evaluation."""
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_mixer_eval_list_splits: List[str] = field(default_factory=lambda: ["test"])
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
    shuffle_eval_dataset: bool = False
    """Whether to shuffle the evaluation dataset."""
    max_token_length: int = 512
    """The maximum token length to use for the dataset"""
    max_prompt_token_length: int = 256
    """The maximum prompt token length to use for the dataset"""

    # Experiment
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: Optional[str] = None
    """RUNTIME VALUE: A unique name of this run"""

    # Optimizer
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
    weight_decay: float = 0.0
    """Weight decay for AdamW if we apply some."""
    set_weight_decay_on_bias_and_norm: bool = True
    """Whether to set weight decay on bias and norm layers"""
    fused_optimizer: bool = False
    """Whether to use fused optimizer"""

    # Batch sizes
    per_device_train_batch_size: int = 1
    """The forward batch size per device (local_micro_batch_size)"""
    total_episodes: int = 100000
    """The total number of episodes in the dataset"""
    world_size: Optional[int] = None
    """RUNTIME VALUE: The number of processes (GPUs) to use"""
    num_training_steps: Optional[int] = None
    """RUNTIME VALUE: The number of training_steps to train"""
    num_evals: int = 10
    """The number of evaluations to run throughout training"""
    eval_freq: Optional[int] = None
    """RUNTIME VALUE: The frequency of evaluation steps"""
    save_freq: int = -1
    """How many train steps to save the model"""
    allow_world_padding: bool = False
    """Whether to allow world padding. This is useful for model sweeps, but wastes compute."""

    # Generation
    response_length: int = 256
    """the length of the response"""
    temperature: float = 0.7
    """the sampling temperature"""
    num_unique_prompts_rollout: int = 16
    """The number of unique prompts during rollout"""
    num_samples_per_prompt_rollout: int = 4
    """the number of samples to generate per prompt during rollout, useful for easy-star"""
    stop_strings: Optional[List[str]] = None
    """List of strings that stop the generation when they are generated.
    The returned output will not contain the stop strings."""

    # Algorithm
    async_mode: bool = True
    """Whether to run the generation in async mode which learns from the second latest policy like Cleanba (https://arxiv.org/abs/2310.00036)"""
    async_steps: int = 1
    """Number of steps ahead to generate responses. Only used when async_mode is True."""
    num_epochs: int = 1
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    clip_lower: float = 0.2
    """the lower clip range"""
    clip_higher: float = 0.2
    """the higher clip range. Sometimes we want this to be higher, see DAPO (https://arxiv.org/abs/2503.14476)"""
    kl_estimator: Literal["kl1", "kl2", "kl3", "kl4"] = "kl3"
    """the KL estimator to use"""
    pack_length: int = 512
    """the length of the pack (you should prob set to the max length of the model)"""
    masked_mean_axis: Optional[int] = None
    """the axis to compute the mean of the masked values"""
    alpha: float = 0.6
    """The alpha value for doing polyak updates (ref_param = alpha * param + (1 - alpha) * ref_param)
    reference: [TR-DPO](https://huggingface.co/papers/2404.09656), but it's actually pretty commonly
    used. E.g., [TD3](https://arxiv.org/abs/1802.09477) uses https://github.com/vwxyzjn/cleanrl/blob/dcc289fc6f0bda492fa7360a155262cf826b12a5/cleanrl/td3_continuous_action.py#L269
    """
    ref_policy_update_freq: Optional[int] = None
    """How many training steps to take before updating the reference policy."""
    advantage_normalization_type: Literal["standard", "centered", "finegrained"] = "finegrained"
    """The type of advantage normalization to use. Standard normalization is the default: it subtracts the mean and
    divides by the standard deviation. Centered normalization is the same but subtracts the mean only (e.g., used in
    DR.GRPO https://arxiv.org/pdf/2503.20783)."""
    mask_truncated_completions: bool = False
    """Whether to mask out truncated completions. Also called overlong filtering, from DAPO (https://arxiv.org/abs/2503.14476)."""
    
    # Advantage visualization settings
    log_advantage_visualization: bool = False
    """Whether to log advantage visualization examples during training."""
    advantage_vis_frequency: int = 1
    """Log advantage visualization every N steps (1 = every step, 10 = every 10 steps)."""
    advantage_vis_num_examples: int = 5
    """Number of examples to show in advantage visualization."""
    advantage_vis_show_token_details: bool = False
    """Whether to show detailed token breakdown in advantage visualization."""

    # Training rollout logging
    log_training_rollouts: bool = False
    """Whether to log training rollouts to wandb"""
    log_training_rollouts_freq: int = 10
    """How often to log training rollouts (every N training steps)"""
    num_training_rollouts_to_log: int = 16
    """Number of training rollouts to log to wandb"""

    # Reward
    # -- r1 style format reward
    apply_r1_style_format_reward: bool = False
    """whether to add the R1 style format reward"""
    r1_style_format_reward: float = 1.0
    """the reward value for R1 style format reward"""
    additive_format_reward: bool = False
    """whether to add the format reward to the final reward"""

    # -- verifiable reward
    apply_verifiable_reward: bool = False
    """whether to apply verifiable reward"""
    apply_finegrained_reward: bool = True
    """whether to apply finegrained reward"""
    finegrained_reward: float = 10.0
    """the reward value for finegrained reward"""
    verification_reward: float = 10.0
    """the reward value for verifiable responses"""

    # -- llm verifiers
    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    """the model to use for the llm judge"""
    llm_judge_max_tokens: int = 2048
    """the max tokens to use for the llm judge"""
    llm_judge_temperature: float = 1.0
    """the temperature to use for the llm judge"""
    llm_judge_timeout: int = 60
    """the timeout to use for the llm judge"""
    llm_judge_max_context_length: int = 2048
    """the max context length to use for the llm judge"""

    # -- code verifier
    code_api_url: str = os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program"
    """the api url to use for the code verifier"""
    code_max_execution_time: float = 1.0
    """the max execution time to use for the code verifier"""

    # -- non stop penalty
    non_stop_penalty: bool = False
    """whether to penalize responses which did not finish generation"""
    non_stop_penalty_value: float = 0.0
    """the reward value for responses which did not finish generation"""

    # Ray
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node (mostly for debugging purposes)"""
    num_learners_per_node: List[int] = field(default_factory=lambda: [1])
    """number of GPU deepspeed learners per node (e.g., --num_learners_per_node 2 4 means 2 learner processes
    on the first node and 4 learner processes on the second node; each process will have 1 GPU)"""
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
    vllm_enable_prefix_caching: bool = False
    """whether to enable prefix caching"""
    vllm_top_p: float = 1.0
    """vLLM top p for nucleus sampling"""
    deepspeed_stage: int = 0
    """the deepspeed stage"""
    gather_whole_model: bool = True
    """whether to gather the whole model to boardcast (not doable for 70B but can be faster for 8B)"""

    # Experiment tracking
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
    save_traces: bool = False
    """Whether to save learning data traces"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    keep_last_n_checkpoints: int = 3
    """How many checkpoints to keep in the output directory. -1 for all."""
    checkpoint_state_freq: int = -1
    """How often to save the model checkpoint, optimizer states, and lr scheduler states (in steps)"""
    checkpoint_state_dir: Optional[str] = None
    """Where to save the model checkpoint (if applicable)"""
    gs_checkpoint_state_dir: Optional[str] = None
    """The actual `checkpoint_state_dir` to use (handling the case where gs_bucket_path is provided)"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: Optional[str] = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: Optional[List[str]] = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    """the priority of auto-launched evaluation jobs"""

    # Tool settings
    tools: Optional[List[str]] = None
    """If set, use the tool mapped to the string. Currently only supports `search` and `code`"""
    max_tool_calls: List[int] = field(default_factory=lambda: [5])
    """Maximum number of tool calls allowed. If a list is provided, it must have length 1 (applies to all tools) or same length as tools (per-tool limit)."""
    mask_tool_use: bool = True
    """Whether to mask the tool output. By default on."""
    only_reward_good_outputs: bool = False
    """Whether to only reward good outputs. By default off. Useful to force the model to use the tool(s)."""
    use_mcp_tools: bool = False
    """Whether to use MCP tools. For now if you use the MCP tools, you need to run an MCP server on the background."""
    mcp_tool_name: Optional[str] = "s2"
    """The name of the MCP tool to use. For now only supports `s2` and `serper`."""
    mcp_server_command: Optional[str] = None
    """Command to run MCP server subprocess when use_mcp_tools is enabled. Example: 'fastmcp run rag_mcp/main.py:mcp --transport streamable-http --port 8000'"""


    # rl-rag specific settngs
    number_documents_to_search: int = 3
    """The maximum number of documents to retrieve for each query."""
    search_api_endpoint: Optional[str] = None
    """The API endpoint for the search engine."""
    use_massive_ds: bool = False
    """Whether to use massive ds for search."""

    # code-tool specific settings
    code_tool_api_endpoint: Optional[str] = None

    # Reward function override
    overwrite_reward_fn_tag: Optional[str] = None
    """If set, force all datasets to use this specific reward function type instead of dataset-based selection"""

    def __post_init__(self):
        assert self.num_samples_per_prompt_rollout > 0, "Number of samples per prompt must be greater than 0!"
        if self.num_samples_per_prompt_rollout == 1:
            print("WARNING: num_samples_per_prompt_rollout is 1. This reduces GRPO to REINFORCE. ")
        assert (
            self.apply_verifiable_reward or self.apply_r1_style_format_reward or self.non_stop_penalty or self.apply_finegrained_reward
        ), "At least one reward must be applied!"
        assert (
            self.pack_length >= self.max_prompt_token_length + self.response_length
        ), "The `pack_length` needs to be greater than the sum of `max_prompt_token_length` and `response_length`!"
        if self.checkpoint_state_freq > 0 and self.checkpoint_state_dir is None:
            raise ValueError("`checkpoint_state_dir` must be provided if `checkpoint_state_freq` is greater than 0!")
        if self.checkpoint_state_dir is not None and self.checkpoint_state_freq == -1:
            raise ValueError("`checkpoint_state_freq` must be greater than 0 if `checkpoint_state_dir` is provided!")
        if self.gs_bucket_path is not None and self.gs_checkpoint_state_dir is None:
            beaker_users = get_beaker_whoami()
            if beaker_users is not None:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{beaker_users}/{self.checkpoint_state_dir}"
            else:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{self.checkpoint_state_dir}"
        if self.gs_checkpoint_state_dir is not None:
            download_latest_checkpoint_from_gs(self.gs_checkpoint_state_dir, self.checkpoint_state_dir)
        if self.checkpoint_state_dir is not None:
            calibrate_checkpoint_state_dir(self.checkpoint_state_dir)
        if self.tools is not None and len(self.tools) > 0:
            for tool in self.tools:
                if tool not in ["search", "code"]:
                    raise ValueError(f"Tool {tool} is not supported. Supported tools are: search, code")
            assert len(self.tools) == len(set(self.tools)), "Duplicate tools are not allowed"

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return ((values * mask).sum(axis=axis) / mask.sum(axis=axis)).mean()
    else:
        return (values * mask).sum() / mask.sum()


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

    def get_metrics_list(self) -> dict[str, float]:
        metrics_list = self.metrics.tolist()
        return {name: metrics_list[idx] for name, idx in self.names2idx.items()}


def collate_fn(tensors_list: List[torch.Tensor], pad_token_id: int, pin_memory: bool = True) -> torch.Tensor:
    padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors_list, batch_first=True, padding_value=pad_token_id)
    if pin_memory:
        padded_tensor = padded_tensor.pin_memory()
    return padded_tensor


def to_device_inplace(tensors_list: List[torch.Tensor], device: torch.device):
    for i in range(len(tensors_list)):
        tensors_list[i] = tensors_list[i].to(device, non_blocking=True)


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


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def from_pretrained(
        self,
        args: Args,
        model_config: ModelConfig,
        beaker_config: BeakerRuntimeConfig,
        wandb_url: str,
        tokenizer: PreTrainedTokenizer,
    ):
        # ------------------------------------------------------------
        # Monkey patch to load checkpoints with `weights_only=False`
        # otherwise it errors out with:
        # `_pickle.UnpicklingError: Weights only load failed. ` with pytorch 2.6.0
        from deepspeed.runtime.checkpoint_engine import torch_checkpoint_engine
        from deepspeed.utils import logger

        def load(self, path: str, map_location=None):
            logger.info(f"[Torch] Loading checkpoint from {path}...")
            partition = torch.load(path, map_location=map_location, weights_only=False)
            logger.info(f"[Torch] Loaded checkpoint from {path}.")
            return partition

        torch_checkpoint_engine.TorchCheckpointEngine.load = load

        # ------------------------------------------------------------
        self.args = args
        self.tokenizer = tokenizer
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
        # @vwxyzjn: MAGIC: it's actually needed to initialize this `dschf`, so
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
        disable_dropout_in_model(self.policy)
        self.policy.gradient_checkpointing_enable()
        # AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        # AdamOptimizer = FusedAdam
        if args.set_weight_decay_on_bias_and_norm:
            optim_params = get_optimizer_grouped_parameters(self.policy, args.weight_decay)
        else:
            optim_params = self.policy.parameters()
        # self.optimizer = AdamOptimizer(optim_params, lr=args.learning_rate)
        self.optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer)
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
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.policy,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=True,
        )
        optimization_steps_done = 0
        if args.checkpoint_state_dir:
            # check if the dir exists
            if not os.path.exists(args.checkpoint_state_dir):
                print(f"Skipping loading checkpoint state from {args.checkpoint_state_dir} because it does not exist!")
            else:
                path, states = self.model.load_checkpoint(
                    args.checkpoint_state_dir,
                    load_module_strict=True,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                    load_module_only=False,
                )
                if path is None:
                    raise ValueError(f"Failed to load checkpoint from {args.checkpoint_state_dir}")
                optimization_steps_done = states["training_step"]
                print(
                    f"{self.rank=}: Loaded checkpoint from {args.checkpoint_state_dir} with {optimization_steps_done=}"
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
        self.local_metrics = MetricsTracker(max_metrics=32, device=self.device)
        return optimization_steps_done

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
        padding_mask = query_response != pad_token_id
        input_ids = torch.masked_fill(query_response, ~padding_mask, 0)
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
        # clear vllm cache if we need to
        cache_reset_refs = []
        if self.args.vllm_enable_prefix_caching and torch.distributed.get_rank() == 0:
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

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
        if args.vllm_enable_prefix_caching and torch.distributed.get_rank() == 0:
            ray.get(cache_reset_refs)

    def update_ref_policy(self):
        for ref_param, param in zip(self.ref_policy.parameters(), self.model.parameters()):
            if self.args.deepspeed_stage == 3:
                with deepspeed.zero.GatheredParameters(
                    [param, ref_param],
                    modifier_rank=0,
                ):
                    if deepspeed.comm.get_rank() == 0:
                        ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)
            else:
                ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)

    def train(
        self,
        collated_query_responses,
        collated_tool_masks,
        collated_attention_masks,
        collated_position_ids,
        collated_advantages,
        collated_response_masks,
        pad_token_id: int,
        num_mini_batches: int,
    ):
        args = self.args
        to_device_inplace(collated_query_responses, self.device)
        to_device_inplace(collated_tool_masks, self.device)
        to_device_inplace(collated_attention_masks, self.device)
        to_device_inplace(collated_position_ids, self.device)
        to_device_inplace(collated_advantages, self.device)
        to_device_inplace(collated_response_masks, self.device)
        accumulation_steps = math.ceil(len(collated_query_responses) / num_mini_batches - 0.5)
        leftover = len(collated_query_responses) % accumulation_steps
        if leftover > 0:
            collated_query_responses = collated_query_responses[0:-leftover]
            collated_tool_masks = collated_tool_masks[0:-leftover]
            collated_attention_masks = collated_attention_masks[0:-leftover]
            collated_position_ids = collated_position_ids[0:-leftover]
            collated_advantages = collated_advantages[0:-leftover]
            collated_response_masks = collated_response_masks[0:-leftover]

        # Calculate the logprob of the reference policy
        collated_ref_logprobs = []
        with Timer("Inference Calculation", noop=self.rank != 0):
            with torch.no_grad():
                for i in range(len(collated_query_responses)):
                    query_response = collated_query_responses[i]
                    tool_mask = collated_tool_masks[i]
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
                    if args.mask_tool_use and args.tool_use:
                        # mask logprobs for tool tokens
                        response_mask = response_mask.bool() & tool_mask.bool()
                    else:
                        response_mask = response_mask.bool()
                    ref_logprob = torch.masked_fill(ref_logprob, ~response_mask[:, 1:], INVALID_LOGPROB)
                    collated_ref_logprobs.append(ref_logprob)
                    torch.cuda.empty_cache()
        local_step = 0
        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        with Timer("[Training Processes] Loss calculation", noop=self.rank != 0):
            old_logprobs = [None for _ in range(len(collated_query_responses))]
            kl1_stats = torch.zeros(len(collated_query_responses))
            kl2_stats = torch.zeros(len(collated_query_responses))
            kl3_stats = torch.zeros(len(collated_query_responses))
            kl4_stats = torch.zeros(len(collated_query_responses))
            kl_loss_stats = torch.zeros(len(collated_query_responses))
            pg_clipfrac_stats = torch.zeros(len(collated_query_responses))
            pg_loss_stats = torch.zeros(len(collated_query_responses))
            loss_stats = torch.zeros(len(collated_query_responses))
            ratio_stats = torch.zeros(len(collated_query_responses))
            for epoch_idx in range(args.num_epochs):
                for i in range(len(collated_query_responses)):
                    mb_ref_logprob = collated_ref_logprobs[i]
                    mb_query_responses = collated_query_responses[i]
                    mb_tool_mask = collated_tool_masks[i]
                    # todo: make sure the advantages return these stats
                    mb_advantages_list = [collated_advantages[i]]  # Wrap in list for compatibility
                    mb_response_masks = collated_response_masks[i]
                    mb_response_masks_bool = mb_response_masks[:, 1:].bool()
                    # if masking snippets, do it here.
                    if args.mask_tool_use and args.tool_use:
                        mb_response_masks_bool = mb_response_masks[:, 1:].bool() & mb_tool_mask[:, 1:].bool()
                    mb_attention_mask = collated_attention_masks[i]
                    mb_position_id = collated_position_ids[i]
                    mb_new_logprobs = self.forward(
                        self.model,
                        mb_query_responses,
                        mb_attention_mask,
                        mb_position_id,
                        pad_token_id,
                        args.temperature,
                    )
                    # For fgrpo, we don't need to mask logprobs since zero advantages naturally contribute zero gradient
                    mb_new_logprobs_list = [mb_new_logprobs for j in range(len(mb_advantages_list))]

                    # Cache the old logprobs
                    with torch.no_grad():
                        if epoch_idx == 0:
                            old_logprobs[i] = mb_new_logprobs_list
                        mb_old_logprobs_list = [old_logprobs[i][j].detach() for j in range(len(mb_advantages_list))]

                    # Calculate the policy's loss
                    logprobs_diff_list = [mb_new_logprobs_list[j] - mb_old_logprobs_list[j] for j in range(len(mb_advantages_list))]
                    ratio_list = [torch.exp(logprobs_diff_list[j]) for j in range(len(mb_advantages_list))]
                    
                    pg_losses_list = [-mb_advantages_list[j][:, 1:] * ratio_list[j] for j in range(len(mb_advantages_list))]
                    pg_losses2_list = [-mb_advantages_list[j][:, 1:] * torch.clamp(
                        ratio_list[j], 1.0 - args.clip_lower, 1.0 + args.clip_higher
                    ) for j in range(len(mb_advantages_list))]
                    pg_loss_max_list = [torch.max(pg_losses_list[j], pg_losses2_list[j]) for j in range(len(mb_advantages_list))]

                    # Here we recalculate kl: we want the KL loss to backpropagate through the model
                    # We also clamp the KL loss to avoid numerical instability
                    # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
                    ref_logprobs_diff_list = [(mb_new_logprobs_list[j] - mb_ref_logprob).clamp(-40.0, 40.0) for j in range(len(mb_advantages_list))]
                    kl1_list = [ref_logprobs_diff_list[j] for j in range(len(mb_advantages_list))]
                    kl2_list = [(ref_logprobs_diff_list[j]) ** 2 / 2 for j in range(len(mb_advantages_list))]
                    kl3_list = [torch.expm1(-ref_logprobs_diff_list[j]) + ref_logprobs_diff_list[j] for j in range(len(mb_advantages_list))] # this is more numerically stable
                    kl4_list = [ratio_list[j] * ref_logprobs_diff_list[j] for j in range(len(mb_advantages_list))]
                    if args.kl_estimator == "kl1":
                        kl_list = kl1_list
                    elif args.kl_estimator == "kl2":
                        kl_list = kl2_list
                    elif args.kl_estimator == "kl3":
                        kl_list = kl3_list
                    elif args.kl_estimator == "kl4":
                        kl_list = kl4_list

                    # grpo change: directly subtract KL in loss (add)
                    loss_list = [masked_mean(pg_loss_max_list[j] + (args.beta * kl_list[j]), mb_response_masks_bool, args.masked_mean_axis) for j in range(len(mb_advantages_list))]
                    loss = torch.mean(torch.stack(loss_list))
                    loss = loss / accumulation_steps
                    self.model.backward(loss)
                    if (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                    local_step += 1
                    with torch.no_grad():
                        # NOTE: in packed implementation, kl calculation are averages over response tokens
                        kl1_stats = torch.mean(torch.stack([masked_mean(kl1_list[j], mb_response_masks_bool, args.masked_mean_axis).float() for j in range(len(mb_advantages_list))]))
                        kl2_stats = torch.mean(torch.stack([masked_mean(kl2_list[j], mb_response_masks_bool, args.masked_mean_axis).float() for j in range(len(mb_advantages_list))]))
                        kl3_stats = torch.mean(torch.stack([masked_mean(kl3_list[j], mb_response_masks_bool, args.masked_mean_axis).float() for j in range(len(mb_advantages_list))]))
                        kl4_stats = torch.mean(torch.stack([masked_mean(kl4_list[j], mb_response_masks_bool, args.masked_mean_axis).float() for j in range(len(mb_advantages_list))]))
                        kl_loss_stats = torch.mean(torch.stack([kl1_stats * args.beta, kl2_stats * args.beta, kl3_stats * args.beta, kl4_stats * args.beta]))
                        pg_clipfrac_stats = torch.mean(torch.stack([masked_mean(
                            (pg_losses2_list[j] > pg_losses_list[j]).float(), mb_response_masks_bool, args.masked_mean_axis) for j in range(len(mb_advantages_list))]))
                        pg_loss_stats = torch.mean(torch.stack([masked_mean(pg_loss_max_list[j], mb_response_masks_bool, args.masked_mean_axis) for j in range(len(mb_advantages_list))]))
                        loss_stats = torch.mean(torch.stack([loss_list[j] for j in range(len(mb_advantages_list))]))
                        ratio_stats = torch.mean(torch.stack([masked_mean(ratio_list[j], mb_response_masks_bool, args.masked_mean_axis) for j in range(len(mb_advantages_list))]))

            with torch.no_grad():
                self.local_metrics.add("objective/kl_avg", kl1_stats.mean())
                self.local_metrics.add("objective/kl2_avg", kl2_stats.mean())
                self.local_metrics.add("objective/kl3_avg", kl3_stats.mean())
                self.local_metrics.add("objective/kl4_avg", kl4_stats.mean())
                self.local_metrics.add("loss/policy_avg", pg_loss_stats.mean())
                self.local_metrics.add("loss/kl_avg", kl_loss_stats.mean())
                self.local_metrics.add("loss/total_avg", loss_stats.mean())
                self.local_metrics.add("policy/clipfrac_avg", pg_clipfrac_stats.mean())
                self.local_metrics.add("val/ratio", ratio_stats.mean())
                self.local_metrics.add("val/ratio_var", ratio_stats.var())
                self.local_metrics.add("lr", self.scheduler.get_last_lr()[0])
                return self.local_metrics.get_metrics_list()

    def save_checkpoint_state(self, checkpoint_state_dir: str, client_state: Dict[str, str]) -> None:
        args = self.args
        self.model.save_checkpoint(checkpoint_state_dir, client_state=client_state)
        # `save_checkpoint` needs to be called on all ranks, only rank 0 will have all the states
        if self.rank == 0:
            if args.keep_last_n_checkpoints >= 0:
                clean_last_n_checkpoints_deepspeed(checkpoint_state_dir, args.keep_last_n_checkpoints)

            if args.gs_bucket_path is not None:
                ray.remote(sync_gs_bucket).options(num_cpus=1).remote(
                    checkpoint_state_dir,
                    args.gs_checkpoint_state_dir,
                )

    def save_model(self, output_dir: str) -> None:
        model_to_save = self.model
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
            self.tokenizer.save_pretrained(output_dir)

    # we need this because we don't know which node is rank 0 is on
    def launch_ai2_evals_on_weka_wrapper(self, step_dir, leaderboard_name, wandb_url, training_step):
        args = self.args
        if self.rank == 0:
            ray.remote(launch_ai2_evals_on_weka).options(num_cpus=1).remote(
                step_dir,
                leaderboard_name,
                args.oe_eval_max_length,
                wandb_url,
                training_step,
                args.oe_eval_tasks,
                args.stop_strings,
                args.gs_bucket_path,
                args.eval_priority,
            )


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


def vllm_generate_thread(
    vllm_engines: List[ray.actor.ActorHandle],
    generation_config: SamplingParams,
    eval_generation_config: SamplingParams,
    inference_results_Q: Queue,
    param_prompt_Q: Queue,
    num_training_steps: int,
    eval_prompt_token_ids: Optional[List[int]],
    evaluation_inference_results_Q: Queue,
    eval_freq: int,
    resume_training_step: int = 1,
    tool_use: bool = False,
):
    def generate_with_engines(prompts: List[List[int]], sampling_params: SamplingParams):
        # Split queries between engines
        queries_per_engine = (len(prompts) + len(vllm_engines) - 1) // len(vllm_engines)
        split_queries = [prompts[i : i + queries_per_engine] for i in range(0, len(prompts), queries_per_engine)]
        # Generate responses in parallel across engines
        futures = [
            vllm_engine.generate.remote(sampling_params=sampling_params, prompt_token_ids=queries, use_tqdm=False)
            for vllm_engine, queries in zip(vllm_engines, split_queries)
        ]
        # Gather all responses
        all_outputs = ray.get(futures)
        response_ids = []
        finish_reasons = []  # either "stop" or "length"
        masks = []
        num_calls = []
        timeouts = []
        tool_errors = []
        tool_outputs = []
        tool_runtimes = []
        tool_calleds = []
        for outputs in all_outputs:
            response_ids.extend([list(out.token_ids) for output in outputs for out in output.outputs])
            finish_reasons.extend([out.finish_reason for output in outputs for out in output.outputs])
            if tool_use:
                masks.extend([out.mask for output in outputs for out in output.outputs])
                num_calls.extend([out.num_calls for output in outputs for out in output.outputs])
                timeouts.extend([out.timeout for output in outputs for out in output.outputs])
                tool_errors.extend([out.tool_error for output in outputs for out in output.outputs])
                tool_outputs.extend([out.tool_output for output in outputs for out in output.outputs])
                tool_runtimes.extend([out.tool_runtime for output in outputs for out in output.outputs])
                tool_calleds.extend([out.tool_called for output in outputs for out in output.outputs])
        # if not using the tool, mask is all 1s
        if not tool_use:
            masks = [[1] * len(response_ids[i]) for i in range(len(response_ids))]
            num_calls = [0] * len(response_ids)
            timeouts = [0] * len(response_ids)
            tool_errors = [""] * len(response_ids)
            tool_outputs = [""] * len(response_ids)
            tool_runtimes = [0] * len(response_ids)
            tool_calleds = [False] * len(response_ids)
        return (
            response_ids,
            finish_reasons,
            masks,
            (num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds),
        )

    for training_step in range(resume_training_step, num_training_steps + 1):
        items = param_prompt_Q.get()
        if items is None:
            break
        _, g_queries_list = items

        with Timer("🔥 Generation time"):
            response_ids, finish_reasons, masks, info = generate_with_engines(g_queries_list, generation_config)
        inference_results_Q.put((response_ids, finish_reasons, masks, info))

        # Evaluate the model
        if eval_prompt_token_ids is not None and (training_step - 1) % eval_freq == 0:
            response_ids, finish_reasons, masks, info = generate_with_engines(
                eval_prompt_token_ids, eval_generation_config
            )
            evaluation_inference_results_Q.put((response_ids, finish_reasons, masks, info))


def data_preparation_thread(
    reward_fn: Callable,
    inference_results_Q: Queue,
    packed_sequences_Q: Queue,
    queries_prompt_Q: Queue,
    args: Args,
    tokenizer: PreTrainedTokenizer,
    num_training_steps: int,
):
    for training_step in range(1, num_training_steps + 1):
        # Get next batch of prompts and responses
        items = queries_prompt_Q.get()
        queries, ground_truths, datasets = items

        # ------------------------------------------------------------------------------------------------
        # Pack sequences
        if args.num_samples_per_prompt_rollout > 1:
            queries = [item for item in queries for _ in range(args.num_samples_per_prompt_rollout)]
            ground_truths = [item for item in ground_truths for _ in range(args.num_samples_per_prompt_rollout)]
            datasets = [item for item in datasets for _ in range(args.num_samples_per_prompt_rollout)]
        with Timer("🚀 [Data Preparation Thread] Getting response ids"):
            responses, finish_reasons, masks, infos = inference_results_Q.get()
            num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = infos
            good_outputs = [
                len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
                for i in range(len(tool_outputs))
            ]
            for i in range(len(finish_reasons)):
                # edge case: sometimes it outputs eos immediately, and we get an empty response
                # in that case, we need to add the eos token to the response
                # note that this also adds eos to the end of reponses that stopped for other reasons.
                if finish_reasons[i] == "stop" and (
                    len(responses[i]) == 0 or responses[i][-1] != tokenizer.eos_token_id
                ):
                    responses[i].append(tokenizer.eos_token_id)
                    masks[i].append(1)  # never mask the eos token for now?

        with Timer("🔥 [Data Preparation Thread] Decoding responses", noop=True):
            decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
            decoded_queries = tokenizer.batch_decode(queries, skip_special_tokens=True)
            decoded_queries = [extract_user_query(query) for query in decoded_queries]
            stop_rate = sum(int(finish_reason == "stop") for finish_reason in finish_reasons) / len(finish_reasons)

        with Timer("💰 [Data Preparation Thread] Calculating rewards and advantages"):
            # fine-grained rewards are a list of FinegrainedScore objects of length num responses
            # each finegrained_score is a FinegrainedScore object including attributes: score, effective_spans, reward_group_id, query_idx, advantage
            all_finegrained_rewards, reward_metrics = asyncio.run(
                reward_fn(
                    responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, decoded_queries
                )
            )
            print("Number of finegrained score outputs: ", len(all_finegrained_rewards))  # should be number of responses
            print("Number of responses: ", len(responses)) 
            
        with Timer("🔄 [Data Preparation Thread] Converting string spans to token spans"):
            def convert_string_span_to_token_span(effective_spans, decoded_resp, token_resp, tokenizer):
                """
                Convert character spans to token spans and create a mask for training.
                
                Args:
                    effective_spans: List of tuples (start_char, end_char) in decoded response
                    decoded_resp: Decoded string response
                    token_resp: Tokenized response (list of token IDs)
                    tokenizer: Tokenizer used to decode the responses
                
                Returns:
                    Tuple of (mask, span_mapping_stats) where:
                    - mask: List of integers where 1 means the token should be trained on, 0 means masked
                    - span_mapping_stats: Dict with mapping quality statistics
                """
                logger = logging.getLogger(__name__)
                
                # Initialize statistics tracking
                span_stats = {
                    "total_spans": len(effective_spans) if effective_spans else 0,
                    "valid_start_mappings": 0,
                    "valid_end_mappings": 0,
                    "fallback_start_mappings": 0,
                    "fallback_end_mappings": 0,
                }
                
                if not effective_spans or len(token_resp) == 0:
                    # If no effective spans or no tokens, mask everything except last EOS
                    mask = [0] * len(token_resp)
                    if len(token_resp) > 0 and token_resp[-1] == tokenizer.eos_token_id:
                        mask[-1] = 1
                    return mask, span_stats
                
                # Build character-to-token mapping by decoding each token
                char_to_token = {}
                current_pos = 0
                
                for token_idx, token_id in enumerate(token_resp):
                    # Decode individual token
                    token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    
                    if token_text:
                        # Find where this token appears in the decoded response
                        token_start = decoded_resp.find(token_text, current_pos)
                        if token_start != -1:
                            token_end = token_start + len(token_text)
                            # Map all characters in this token to the token index
                            for char_idx in range(token_start, token_end):
                                if char_idx < len(decoded_resp):
                                    char_to_token[char_idx] = token_idx
                            current_pos = token_end
                        else:
                            # Token not found at expected position, map current position
                            if current_pos < len(decoded_resp):
                                char_to_token[current_pos] = token_idx
                                current_pos += 1
                    else:
                        # Empty token text, map current position if valid
                        if current_pos < len(decoded_resp):
                            char_to_token[current_pos] = token_idx
                            current_pos += 1
                
                # Fill any remaining unmapped characters with the last token
                if len(token_resp) > 0:
                    for char_idx in range(current_pos, len(decoded_resp)):
                        char_to_token[char_idx] = len(token_resp) - 1
                
                # Initialize mask - everything masked by default
                mask = [0] * len(token_resp)
                
                # Unmask tokens that fall within effective spans
                for span_idx, (start_char, end_char) in enumerate(effective_spans):
                    original_start, original_end = start_char, end_char
                    start_char = max(0, min(start_char, len(decoded_resp)))
                    end_char = max(start_char, min(end_char, len(decoded_resp)))
                    
                    if start_char < len(decoded_resp) and start_char in char_to_token:
                        token_start = char_to_token[start_char]
                        span_stats["valid_start_mappings"] += 1
                    else:
                        logger.warning(f"Span {span_idx} [{original_start}:{original_end}]: start_char {start_char} not in char_to_token, using fallback token_start=0")
                        token_start = 0
                        span_stats["fallback_start_mappings"] += 1
                    
                    if end_char > 0 and (end_char - 1) in char_to_token:
                        token_end = char_to_token[end_char - 1] + 1
                        span_stats["valid_end_mappings"] += 1
                    else:
                        logger.warning(f"Span {span_idx} [{original_start}:{original_end}]: end_char-1 {end_char-1} not in char_to_token, using fallback token_end=len(token_resp) ({len(token_resp)})")
                        token_end = len(token_resp)
                        span_stats["fallback_end_mappings"] += 1
                    
                    # Clamp to valid token range
                    token_start = max(0, min(token_start, len(token_resp)))
                    token_end = max(token_start, min(token_end, len(token_resp)))
                    
                    # Unmask tokens in this span
                    for token_idx in range(token_start, token_end):
                        mask[token_idx] = 1
                
                # Always keep the last EOS token unmasked if it exists
                if len(token_resp) > 0 and token_resp[-1] == tokenizer.eos_token_id:
                    mask[-1] = 1
                
                return mask, span_stats
            
            # Calculate per-token advantages by averaging advantages from all effective spans that cover each token
            advantages = []
            # Collect span mapping statistics across all responses
            total_span_stats = {
                "total_spans": 0,
                "valid_start_mappings": 0,
                "valid_end_mappings": 0,
                "fallback_start_mappings": 0,
                "fallback_end_mappings": 0,
            }
            
            for i, (token_resp, decoded_resp, finegrained_rewards) in enumerate(zip(responses, decoded_responses, all_finegrained_rewards)):                
                # Initialize per-token advantage array
                token_advantage = np.zeros(len(token_resp), dtype=np.float32)
                token_advantage_count = np.zeros(len(token_resp), dtype=np.int32)
                
                for score_obj in finegrained_rewards.finegrained_scores:
                    # Get token mask for this finegrained score's spans
                    span_mask, span_stats = convert_string_span_to_token_span(
                        score_obj.effective_spans, decoded_resp, token_resp, tokenizer
                    )
                    
                    # Accumulate statistics
                    for key in total_span_stats:
                        total_span_stats[key] += span_stats[key]
                    
                    # Add this advantage to all tokens covered by the spans
                    for token_idx, is_covered in enumerate(span_mask):
                        if is_covered == 1:
                            token_advantage[token_idx] += score_obj.advantage
                            token_advantage_count[token_idx] += 1
                    
                # Calculate average advantages
                for token_idx in range(len(token_resp)):
                    if token_advantage_count[token_idx] > 0:
                        token_advantage[token_idx] /= token_advantage_count[token_idx]
                    
                # print(f"🔫 Token response decoded: {tokenizer.batch_decode(token_resp, skip_special_tokens=True)}")
                # print(f"🔫 Token advantage: {token_advantage}")
                # print(f"🔫 Token advantage count: {token_advantage_count}")
                advantages.append(token_advantage)
            # Note: advantages is a list of variable-length arrays, not converted to numpy array
            
            # Log span mapping statistics
            if total_span_stats["total_spans"] > 0:
                valid_start_ratio = total_span_stats["valid_start_mappings"] / total_span_stats["total_spans"]
                valid_end_ratio = total_span_stats["valid_end_mappings"] / total_span_stats["total_spans"]
                fallback_start_ratio = total_span_stats["fallback_start_mappings"] / total_span_stats["total_spans"]
                fallback_end_ratio = total_span_stats["fallback_end_mappings"] / total_span_stats["total_spans"]
                
                print(f"📊 [Span Mapping Stats] Total spans: {total_span_stats['total_spans']}")
                print(f"📊 [Span Mapping Stats] Valid start mappings: {total_span_stats['valid_start_mappings']} ({valid_start_ratio:.1%})")
                print(f"📊 [Span Mapping Stats] Valid end mappings: {total_span_stats['valid_end_mappings']} ({valid_end_ratio:.1%})")
                print(f"📊 [Span Mapping Stats] Fallback start mappings: {total_span_stats['fallback_start_mappings']} ({fallback_start_ratio:.1%})")
                print(f"📊 [Span Mapping Stats] Fallback end mappings: {total_span_stats['fallback_end_mappings']} ({fallback_end_ratio:.1%})")
            
        # Log advantage visualization examples for monitoring
        if args.log_advantage_visualization and training_step % args.advantage_vis_frequency == 0:
            try:
                print(f"🔫 Logging advantage visualization for step {training_step}")
                from open_instruct.search_rewards.utils.advantage_visualization import log_advantage_examples, log_advantage_statistics
                
                # Log examples based on user settings
                if len(all_finegrained_rewards) > 0:
                    log_advantage_examples(
                        responses=responses,
                        decoded_responses=decoded_responses,
                        all_finegrained_rewards=all_finegrained_rewards,
                        tokenizer=tokenizer,
                        step=training_step,
                        num_examples=args.advantage_vis_num_examples,
                        show_token_details=args.advantage_vis_show_token_details
                    )
                    
                    log_advantage_statistics(
                        all_finegrained_rewards=all_finegrained_rewards,
                        step=training_step
                    )
                print(f"🔫 Logged advantage visualization for step {training_step}")
            except Exception as e:
                print(f"Warning: Failed to log advantage visualization: {e}")

        # Log training rollouts if enabled
        training_rollouts_data = None
        if args.log_training_rollouts and training_step % args.log_training_rollouts_freq == 0 and args.with_tracking:
            print(f"[Data Preparation Thread] 📊 Preparing training rollouts for logging at step {training_step}")
            
            # Select a subset of samples for logging
            num_to_log = min(args.num_training_rollouts_to_log, len(responses))
            if num_to_log > 0:
                indices_to_log = np.random.choice(len(responses), size=num_to_log, replace=False)
                
                # Prepare detailed training rollout data with token-level information
                rollout_entries = []
                for idx in indices_to_log:
                    i = int(idx)  # Convert numpy int to Python int
                    token_resp = responses[i]
                    decoded_resp = decoded_responses[i]
                    finegrained_rewards = all_finegrained_rewards[i]
                    token_advantages = advantages[i]
                    
                    # Decode tokens for visualization
                    token_texts = []
                    for token_id in token_resp:
                        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                        token_texts.append(token_text)
                    
                    # Collect all spans and their rewards for this response
                    spans_info = []
                    total_score = 0
                    for score_obj in finegrained_rewards.finegrained_scores:
                        spans_info.append({
                            "spans": score_obj.effective_spans,
                            "score": score_obj.score,
                            "advantage": score_obj.advantage,
                            "reward_group_id": score_obj.reward_group_id,
                        })
                        total_score += score_obj.score
                    
                    # Create token-level details
                    token_details = []
                    for token_idx, (token_id, token_text, token_adv) in enumerate(zip(token_resp, token_texts, token_advantages)):
                        token_details.append({
                            "token_idx": token_idx,
                            "token_id": int(token_id),
                            "token_text": token_text,
                            "advantage": float(token_adv),
                        })
                    
                    rollout_entry = {
                        "prompt": extract_user_query(tokenizer.decode(queries[i], skip_special_tokens=True)) or tokenizer.decode(queries[i], skip_special_tokens=True),
                        "response": decoded_resp,
                        "total_score": float(total_score),
                        "ground_truth": str(ground_truths[i]),
                        "finish_reason": finish_reasons[i],
                        "dataset": datasets[i],
                        "training_step": training_step,
                        "spans_info": spans_info,
                        "token_details": token_details,
                        "num_tokens": len(token_resp),
                        "avg_advantage": float(np.mean(token_advantages)) if len(token_advantages) > 0 else 0.0,
                        "advantage_std": float(np.std(token_advantages)) if len(token_advantages) > 0 else 0.0,
                    }
                    rollout_entries.append(rollout_entry)
                
                training_rollouts_data = {
                    "rollout_entries": rollout_entries,
                    "training_step": training_step,
                    "num_samples": len(rollout_entries),
                }

        with Timer("📦 [Data Preparation Thread] Filtering sequences"):
            # For finegrained rewards, we simply filter out responses where all advantages are zero
            # This is different from GRPO where we check std=0 across prompt groups
            # Here we just need to remove rollouts with all-zero advantages
            non_zero_advantage_mask = np.array([(adv != 0).any() for adv in advantages])
            non_zero_gradient_index = np.where(non_zero_advantage_mask)[0]
            
            real_batch_size_ratio = len(non_zero_gradient_index) / len(advantages)
            advantages = [advantages[i] for i in non_zero_gradient_index]
            responses = [responses[i] for i in non_zero_gradient_index]
            masks = [masks[i] for i in non_zero_gradient_index]
            queries = [queries[i] for i in non_zero_gradient_index]
            ground_truths = [ground_truths[i] for i in non_zero_gradient_index]
            datasets = [datasets[i] for i in non_zero_gradient_index]
            finish_reasons = [finish_reasons[i] for i in non_zero_gradient_index]
            if args.mask_truncated_completions:
                stop_idxes = torch.tensor([i for i in range(len(finish_reasons)) if finish_reasons[i] == "stop"])
                scores = scores[stop_idxes]
                advantages = [advantages[i] for i in stop_idxes]
                responses = [responses[i] for i in stop_idxes]
                masks = [masks[i] for i in stop_idxes]
                queries = [queries[i] for i in stop_idxes]
                ground_truths = [ground_truths[i] for i in stop_idxes]
                datasets = [datasets[i] for i in stop_idxes]
                finish_reasons = [finish_reasons[i] for i in stop_idxes]

        with Timer("📦 [Data Preparation Thread] Packing sequences"):
            packed_sequences = pack_sequences(
                queries=queries,
                responses=responses,
                masks=masks,
                pack_length=args.pack_length,
                pad_token_id=tokenizer.pad_token_id,
            )
            num_new_tokens = sum(len(seq) for seq in packed_sequences.query_responses)
            # For finegrained rewards, we need to map per-token advantages to packed sequences
            # Each response has variable-length per-token advantages
            packed_advantages = []
            for packed_mask in packed_sequences.response_masks:
                packed_adv = np.zeros(len(packed_mask), dtype=np.float32)
                # Optimize: track position counters for each response_id to avoid O(n²) complexity
                response_positions = {}
                for token_idx, response_id in enumerate(packed_mask):
                    if response_id > 0:  # response_id is 1-indexed, 0 means query token
                        response_idx = response_id - 1  # convert to 0-indexed
                        if response_idx < len(advantages):
                            # Initialize position counter for this response_id if not seen
                            if response_id not in response_positions:
                                response_positions[response_id] = 0
                            
                            response_token_pos = response_positions[response_id]
                            if response_token_pos < len(advantages[response_idx]):
                                packed_adv[token_idx] = advantages[response_idx][response_token_pos]
                            
                            # Increment position counter for this response_id
                            response_positions[response_id] += 1
                packed_advantages.append(torch.tensor(packed_adv, dtype=torch.float32))
            packed_sequences.advantages = packed_advantages

        # if we have less batches than world size, we need to pad out so each world is fine
        # ideally, you should avoid this since its wasting computation.
        if args.allow_world_padding:
            with Timer("🤺 [Data Preparation Thread] Padding sequences for world size"):
                shortfall = args.world_size - len(packed_sequences.query_responses)
                if shortfall > 0:
                    print(
                        f"Padding {shortfall} sequences for world size. In future, you should adjust your compute this."
                    )
                    # construct "dummy" sequences for padding out the world size
                    dummy_qr = torch.tensor([tokenizer.pad_token_id, tokenizer.eos_token_id], dtype=torch.long)
                    dummy_tool_mask = torch.zeros_like(dummy_qr)
                    dummy_attention = torch.tensor([1, 1], dtype=torch.long)
                    dummy_position_ids = torch.arange(len(dummy_qr), dtype=torch.long)
                    dummy_response_mask = torch.zeros_like(dummy_qr)
                    dummy_advantage = torch.zeros_like(dummy_qr, dtype=torch.float)
                    # pad out the world size
                    for _ in range(shortfall):
                        packed_sequences.query_responses.append(dummy_qr)
                        packed_sequences.tool_masks.append(dummy_tool_mask)
                        packed_sequences.attention_masks.append(dummy_attention)
                        packed_sequences.position_ids.append(dummy_position_ids)
                        packed_sequences.response_masks.append(dummy_response_mask)
                        packed_sequences.advantages.append(dummy_advantage)

        with Timer("🔄 [Data Preparation Thread] Prepare collated data for each worker"):
            B = (
                len(packed_sequences.query_responses) // args.world_size
            )  # essentially doing `drop_last=True`, which is fine.
            collated_data = []
            
            # Handle B=0 case like GRPO does
            if B == 0:
                # Create empty collated data for each worker
                for i in range(args.world_size):
                    collated_data.append({
                        "collated_query_responses": [],
                        "collated_tool_masks": [],
                        "collated_attention_masks": [],
                        "collated_position_ids": [],
                        "collated_advantages": [],
                        "collated_response_masks": [],
                    })
            else:
                for i in range(args.world_size):
                    per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
                    per_device_packed_tool_masks = packed_sequences.tool_masks[B * i : B * (i + 1)]
                    per_device_packed_attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
                    per_device_packed_position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
                    per_device_packed_advantages = packed_sequences.advantages[B * i : B * (i + 1)]
                    per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]

                    # Shuffle the batch and collate the data
                    b_inds = np.random.permutation(len(per_device_packed_query_responses))
                    collated_query_responses = []
                    collated_tool_masks = []
                    collated_attention_masks = []
                    collated_position_ids = []
                    collated_response_masks = []
                    collated_advantages = []
                    for j in range(0, len(per_device_packed_query_responses), args.per_device_train_batch_size):
                        micro_range = b_inds[j : j + args.per_device_train_batch_size]
                        collated_query_responses.append(
                            collate_fn(
                                [per_device_packed_query_responses[idx] for idx in micro_range], tokenizer.pad_token_id
                            )
                        )
                        collated_tool_masks.append(
                            collate_fn([per_device_packed_tool_masks[idx] for idx in micro_range], 0)
                        )
                        collated_attention_masks.append(
                            collate_fn([per_device_packed_attention_masks[idx] for idx in micro_range], 0)
                        )
                        collated_position_ids.append(
                            collate_fn([per_device_packed_position_ids[idx] for idx in micro_range], 0)
                        )
                        collated_response_masks.append(
                            collate_fn([per_device_packed_response_masks[idx] for idx in micro_range], 0)
                        )
                        # For fgrpo, just use the advantages directly - no need for masking
                        collated_advs = collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0)
                        collated_advantages.append(collated_advs)
                    collated_data.append(
                        {
                            "collated_query_responses": collated_query_responses,
                            "collated_tool_masks": collated_tool_masks,
                            "collated_attention_masks": collated_attention_masks,
                            "collated_position_ids": collated_position_ids,
                            "collated_advantages": collated_advantages,
                            "collated_response_masks": collated_response_masks,
                        }
                    )

        # Create a result package with metrics and data
        sequence_lengths = np.array([len(response) for response in responses])
        if len(responses) > 0:
            # Calculate span mapping ratios for wandb logging
            span_mapping_metrics = {}
            if total_span_stats["total_spans"] > 0:
                span_mapping_metrics = {
                    "span_mapping/total_spans": total_span_stats["total_spans"],
                    "span_mapping/valid_start_ratio": total_span_stats["valid_start_mappings"] / total_span_stats["total_spans"],
                    "span_mapping/valid_end_ratio": total_span_stats["valid_end_mappings"] / total_span_stats["total_spans"],
                    "span_mapping/fallback_start_ratio": total_span_stats["fallback_start_mappings"] / total_span_stats["total_spans"],
                    "span_mapping/fallback_end_ratio": total_span_stats["fallback_end_mappings"] / total_span_stats["total_spans"],
                }
            
            metrics = {
                "real_batch_size_ratio": real_batch_size_ratio,
                "packed_ratio": len(packed_sequences.query_responses) / len(responses),
                "val/sequence_lengths": sequence_lengths.mean(),
                "val/sequence_lengths_min": sequence_lengths.min(),
                "val/sequence_lengths_max": sequence_lengths.max(),
                "val/stop_rate": stop_rate,
                "val/advantages_mean": np.concatenate(advantages).mean() if len(advantages) > 0 else 0.0,
                "val/advantages_min": np.concatenate(advantages).min() if len(advantages) > 0 else 0.0,
                "val/advantages_max": np.concatenate(advantages).max() if len(advantages) > 0 else 0.0,
                "val/advantages_hist": np.concatenate(advantages) if len(advantages) > 0 else np.array([]),
                **reward_metrics,
                **span_mapping_metrics,
            }
        else:
            print("No responses to evaluate. Not logging any metrics. We will end up skipping this step.")
            metrics = {}

        if args.save_traces:
            traces = {
                "advantages": [adv.tolist() for adv in advantages],
                "finish_reasons": finish_reasons,
                "responses": responses,
                "queries": queries,
                "ground_truths": ground_truths,
                "datasets": datasets,
                "training_step": training_step,
                **reward_metrics,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            with open(f"{args.output_dir}/traces_{args.run_name}.jsonl", "a") as f:
                json.dump(traces, f)
                f.write("\n")

        # Put the packed sequences and metrics into the output queue
        packed_sequences_Q.put(
            {
                "packed_sequences": packed_sequences,  # for debugging purposes
                "collated_data": collated_data,
                "metrics": metrics,
                "responses_count": len(responses),
                "num_new_tokens": num_new_tokens,
                "B": B,
                "training_rollouts_data": training_rollouts_data,  # Add training rollouts data
            }
        )


def launch_mcp_subprocess(use_mcp_tools: bool, run_mcp_command: str, output_dir: str) -> Optional[subprocess.Popen]:
    """
    Launch MCP server subprocess if use_mcp_tools is enabled.
    
    Args:
        use_mcp_tools: Whether to launch MCP server
        run_mcp_command: Command to run MCP server
        output_dir: Base output directory for logs
        
    Returns:
        Popen object if launched, None otherwise
    """
    if not use_mcp_tools:
        return None
        
    print(f"🚀 Launching MCP server subprocess: {run_mcp_command}")
    
    # Debug: Check if fastmcp command exists
    try:
        import shutil
        fastmcp_path = shutil.which("fastmcp")
        if fastmcp_path:
            print(f"✅ Found fastmcp at: {fastmcp_path}")
        else:
            print("⚠️  Warning: fastmcp command not found in PATH")
    except Exception as e:
        print(f"⚠️  Warning: Could not check for fastmcp: {e}")
    
    # Debug: Show current working directory
    print(f"📁 Current working directory: {os.getcwd()}")
    
    # Create log files for MCP server output
    mcp_logs_dir = os.path.join(output_dir, "mcp_logs")
    os.makedirs(mcp_logs_dir, exist_ok=True)
    mcp_stdout = open(os.path.join(mcp_logs_dir, "mcp_server_stdout.log"), "w")
    mcp_stderr = open(os.path.join(mcp_logs_dir, "mcp_server_stderr.log"), "w")
    
    mcp_process = subprocess.Popen(
        [run_mcp_command],
        shell=True,
        stdout=mcp_stdout,
        stderr=mcp_stderr,
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
    )
    try:
        # Launch subprocess in new process group to avoid signal interference
        
        # Give the server time to start
        time.sleep(3)
        
        # Check if process is still running
        if mcp_process.poll() is None:
            print(f"✅ MCP server started successfully (PID: {mcp_process.pid})")
            print(f"📋 MCP server logs: {os.path.relpath(mcp_logs_dir)}/mcp_server_stdout.log, {os.path.relpath(mcp_logs_dir)}/mcp_server_stderr.log")
            
            # Register cleanup function
            def cleanup_mcp():
                if mcp_process.poll() is None:
                    print("🧹 Cleaning up MCP server...")
                    try:
                        # Kill the entire process group
                        os.killpg(os.getpgid(mcp_process.pid), signal.SIGTERM)
                        time.sleep(2)
                        if mcp_process.poll() is None:
                            os.killpg(os.getpgid(mcp_process.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
                mcp_stdout.close()
                mcp_stderr.close()
            
            atexit.register(cleanup_mcp)
            return mcp_process
            
        else:
            print(f"❌ MCP server failed to start (exit code: {mcp_process.returncode})")
            # Read any error output
            mcp_stderr.close()
            mcp_stdout.close()
            try:
                with open(os.path.join(mcp_logs_dir, "mcp_server_stderr.log"), "r") as f:
                    stderr_content = f.read().strip()
                    if stderr_content:
                        print(f"📋 MCP server stderr: {stderr_content}")
                with open(os.path.join(mcp_logs_dir, "mcp_server_stdout.log"), "r") as f:
                    stdout_content = f.read().strip()
                    if stdout_content:
                        print(f"📋 MCP server stdout: {stdout_content}")
            except Exception as e:
                print(f"⚠️  Could not read MCP server logs: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Error launching MCP server: {e}")
        mcp_stdout.close()
        mcp_stderr.close()
        return None


def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig, reward_fn: Callable):
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
    args.world_size = sum(args.num_learners_per_node)
    args.num_training_steps = args.total_episodes // (
        args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    )
    args.eval_freq = max(1, args.num_training_steps // args.num_evals)
    args.try_launch_beaker_eval_jobs_on_weka = args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job()
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
    args.tool_use = args.tools is not None and len(args.tools) > 0

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
        {
            "max_token_length": args.max_token_length,
            "max_prompt_token_length": args.max_prompt_token_length,
        },
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
        if args.shuffle_eval_dataset:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)
    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)
    if args.cache_dataset_only:
        return

    # ------------------------------------------------------------
    # Runtime setups and quick logging
    pprint([args, model_config])

    # ------------------------------------------------------------
    # Launch MCP subprocess if needed
    mcp_process = None
    if args.use_mcp_tools:
        if args.mcp_server_command is None:
            print("mcp_server_command is not provided when use_mcp_tools is True; please make sure to launch the MCP server manually.")
        else:
            mcp_process = launch_mcp_subprocess(args.use_mcp_tools, args.mcp_server_command, args.output_dir)
            if mcp_process is None:
                raise RuntimeError("Failed to launch MCP server subprocess")

    # ------------------------------------------------------------
    # Create the model and optimizer
    try:
        ray.init(dashboard_host="0.0.0.0")  # enable debugging from a different machine (e.g., phobos)
    except:
        ray_temp_dir = os.path.join("/tmp", f"ray-{os.getpid()}")
        ray.init(dashboard_host="0.0.0.0", _temp_dir=ray_temp_dir)  # enable debugging from a different machine (e.g., phobos)
    pg = None
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    inits = []
    policy_group = ModelGroup(
        pg,
        PolicyTrainerRayProcess,
        args.num_learners_per_node,
        args.single_gpu_mode,
    )
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits.extend(
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
        for model in policy_group.models
    )
    max_len = args.max_prompt_token_length + args.response_length
    # make tool list
    tool_objects = {}
    if args.tools:
        for tool in args.tools:
            if tool.lower() == "search":
                if args.use_mcp_tools:
                    from open_instruct.tool_utils.tool_mcp import (
                        SemanticScholarSnippetSearchTool,
                        SerperSearchTool,
                    )

                    if args.mcp_tool_name == "s2":
                        tool = SemanticScholarSnippetSearchTool(
                            start_str="<search>",
                            end_str="</search>",
                        )
                    elif args.mcp_tool_name == "serper":
                        tool = SerperSearchTool(
                            start_str="<search>",
                            end_str="</search>",
                        )
                    else:
                        raise ValueError(f"Unknown MCP tool: {args.mcp_tool_name}")
                else:
                    from open_instruct.search_utils.search_tool import SearchTool

                    tool = SearchTool(
                        start_str="<search>",
                        end_str="</search>",
                        use_massive_ds=args.use_massive_ds,
                        api_endpoint=args.search_api_endpoint,
                        number_documents_to_search=args.number_documents_to_search,
                    )
                
                tool_objects[tool.end_str] = tool
                
            elif tool.lower() == "code":
                from open_instruct.tool_utils.tool_vllm import PythonCodeTool

                tool = PythonCodeTool(
                    start_str="<code>",
                    end_str="</code>",
                    api_endpoint=args.code_tool_api_endpoint,
                )
                tool_objects[tool.end_str] = tool
            else:
                raise ValueError(f"Unknown tool: {tool}")

    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.vllm_enforce_eager,
        tc.tokenizer_name_or_path,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        args.vllm_enable_prefix_caching,
        max_len,
        args.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
        tools=tool_objects,
        max_tool_calls=args.max_tool_calls,
    )
    resume_training_step = ray.get(inits)[0] + 1
    episode = (resume_training_step - 1) * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    print("======== ✅ all models and vLLM engines initialized =========")

    ray.get([m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models])
    print("======== ✅ model update group setup successfully =========")
    if resume_training_step > 1:
        print(f"Resuming training from step {resume_training_step}... Broadcasting weights to vLLM engines.")
        with Timer("[Main Thread] 🔄 Loading weights using shared memory"):
            ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])

    # Setup training
    stop_strings = [] if args.stop_strings is None else args.stop_strings
    if args.tool_use:
        stop_strings += list(tool_objects.keys())
    generation_config = SamplingParams(
        temperature=args.temperature,
        top_p=args.vllm_top_p,  # prevent rare out-of-vocab tokens with qwen
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        n=args.num_samples_per_prompt_rollout,
        stop=stop_strings,
    )
    eval_generation_config = SamplingParams(
        temperature=0.6,
        top_p=args.vllm_top_p,  # prevent rare out-of-vocab tokens with qwen
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        n=1,  # since we are doing greedy sampling, don't need to generate more
        stop=stop_strings,
    )
    train_dataset_idxs = np.arange(len(train_dataset))
    iter_dataloader = ShufflingIterator(train_dataset_idxs, args.num_unique_prompts_rollout, seed=args.seed)

    inference_results_Q = Queue(maxsize=args.async_steps)
    param_prompt_Q = Queue(maxsize=args.async_steps)
    evaluation_inference_results_Q = Queue(maxsize=1)
    packed_sequences_Q = Queue(maxsize=args.async_steps)
    queries_prompt_Q = Queue(maxsize=args.async_steps)
    num_eval_samples = 32

    eval_prompt_token_ids = None
    eval_ground_truths = None
    if eval_dataset is not None:
        eval_prompt_token_ids = eval_dataset[:num_eval_samples][INPUT_IDS_PROMPT_KEY]
        eval_ground_truths = eval_dataset[:num_eval_samples][GROUND_TRUTHS_KEY]
        eval_dataset_names = eval_dataset[:num_eval_samples][DATASET_SOURCE_KEY]
    thread = threading.Thread(
        target=vllm_generate_thread,
        args=(
            vllm_engines,
            generation_config,
            eval_generation_config,
            inference_results_Q,
            param_prompt_Q,
            args.num_training_steps,
            eval_prompt_token_ids,
            evaluation_inference_results_Q,
            args.eval_freq,
            resume_training_step,
            args.tool_use,
        ),
    )
    thread.start()
    print("======== ✅ vllm generate thread starts =========")

    packing_thread = threading.Thread(
        target=data_preparation_thread,
        args=(
            reward_fn,
            inference_results_Q,
            packed_sequences_Q,
            queries_prompt_Q,
            args,
            tokenizer,
            args.num_training_steps,
        ),
    )
    packing_thread.start()
    print("======== ✅ data preparation thread starts =========")

    # Send initial data to both threads
    data_next = train_dataset[next(iter_dataloader)]
    queries_next = data_next[INPUT_IDS_PROMPT_KEY]
    ground_truths_next = data_next[GROUND_TRUTHS_KEY]
    datasets_next = data_next[DATASET_SOURCE_KEY]
    queries_prompt_Q.put((queries_next, ground_truths_next, datasets_next))
    param_prompt_Q.put((None, queries_next))

    num_total_tokens = 0
    start_time = time.time()
    try:
        for training_step in range(resume_training_step, args.num_training_steps + 1):
            print("-" * 100)
            episode += (
                args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
            )  # each sample is an episode

            # ------------------------------------------------------------------------------------------------
            # Sync weights and send the next batch of prompts to vLLM
            if args.async_mode:
                if training_step != 1:
                    data_next = train_dataset[next(iter_dataloader)]
                    queries_next = data_next[INPUT_IDS_PROMPT_KEY]
                    ground_truths_next = data_next[GROUND_TRUTHS_KEY]
                    datasets_next = data_next[DATASET_SOURCE_KEY]
                    with Timer("[Main Thread] 🔄 Loading weights using shared memory"):
                        ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
                queries_prompt_Q.put((queries_next, ground_truths_next, datasets_next))
                param_prompt_Q.put((None, queries_next))
            else:
                if training_step != 1:
                    # NOTE: important: the indent here is different for sync mode
                    # we also set to use `queries = queries_next` immediately
                    data_next = train_dataset[next(iter_dataloader)]
                    queries_next = data_next[INPUT_IDS_PROMPT_KEY]
                    ground_truths_next = data_next[GROUND_TRUTHS_KEY]
                    datasets_next = data_next[DATASET_SOURCE_KEY]
                    with Timer("🔄 Loading weights using shared memory"):
                        ray.get([m.broadcast_to_vllm.remote() for m in policy_group.models])
                    queries_prompt_Q.put((queries_next, ground_truths_next, datasets_next))
                    param_prompt_Q.put((None, queries_next))

            # ------------------------------------------------------------------------------------------------
            # Get the packed sequences with advantages from the packing thread
            with Timer("[Main Thread] 📦 Getting packed sequences from thread"):
                packed_data = packed_sequences_Q.get()
                data_thread_metrics = packed_data["metrics"]
                B = packed_data["B"]
                collated_data = packed_data["collated_data"]
                num_total_tokens += packed_data["num_new_tokens"]
                training_rollouts_data = packed_data.get("training_rollouts_data")
                if B == 0:
                    print("[Main Thread] 🤡 After packing, there is not enough data to train")
                    continue

            # ------------------------------------------------------------------------------------------------
            # Train the model
            update_ref_policy_future = []
            with Timer("[Main Thread] 🗡️ Training"):
                metrics_list: List[dict[str, float]] = ray.get(
                    [
                        policy_group.models[i].train.remote(
                            **collated_data[i],
                            pad_token_id=tokenizer.pad_token_id,
                            num_mini_batches=args.num_mini_batches,
                        )
                        for i in range(args.world_size)
                    ]
                )
                if (
                    args.ref_policy_update_freq is not None
                    and training_step % args.ref_policy_update_freq == 0
                    and args.alpha > 0
                ):
                    update_ref_policy_future.extend(
                        [policy_group.models[i].update_ref_policy.remote() for i in range(args.world_size)]
                    )

                average_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
                metrics = {
                    "episode": episode,
                    "training_step": training_step,
                    "val/num_total_tokens": num_total_tokens,
                    "epoch": episode / args.num_samples_per_prompt_rollout / len(train_dataset),
                    "tokens_per_second": num_total_tokens / (time.time() - start_time),
                    **data_thread_metrics,
                    **average_metrics,
                }
                scalar_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, float) or isinstance(value, int):
                        writer.add_scalar(key, value, episode)
                        scalar_metrics[key] = value
                    if isinstance(value, np.ndarray) or isinstance(value, list):
                        if len(value) > 0:
                            writer.add_histogram(key, value, episode)
                print_rich_single_line_metrics(scalar_metrics)

                # Log training rollouts if available (thread-safe logging in main thread)
                if training_rollouts_data is not None and args.with_tracking:
                    print(f"[Main Thread] 📊 Logging {training_rollouts_data['num_samples']} training rollouts to wandb")
                    
                    # Create a simplified table for wandb with key information
                    rollout_table_data = []
                    for entry in training_rollouts_data["rollout_entries"]:
                        # Create a summary of spans for display
                        spans_summary = []
                        for span_info in entry["spans_info"]:
                            spans_text = []
                            for start, end in span_info["spans"]:
                                if start < len(entry["response"]) and end <= len(entry["response"]):
                                    span_text = entry["response"][start:end]
                                    spans_text.append(f'"{span_text}"')
                            spans_summary.append({
                                "spans_text": " | ".join(spans_text),
                                "score": span_info["score"],
                                "advantage": span_info["advantage"],
                                "reward_group_id": span_info["reward_group_id"],
                            })
                        
                        # Create token advantage summary (show only non-zero advantages)
                        nonzero_token_advs = [(td["token_text"], td["advantage"]) 
                                            for td in entry["token_details"] 
                                            if abs(td["advantage"]) > 1e-6]
                        token_adv_summary = " | ".join([f'"{text}": {adv:.3f}' 
                                                      for text, adv in nonzero_token_advs[:10]])  # Limit to first 10
                        if len(nonzero_token_advs) > 10:
                            token_adv_summary += f" ... (+{len(nonzero_token_advs) - 10} more)"
                        
                        rollout_table_data.append({
                            "prompt": entry["prompt"][:200] + "..." if len(entry["prompt"]) > 200 else entry["prompt"],
                            "response": entry["response"][:300] + "..." if len(entry["response"]) > 300 else entry["response"],
                            "total_score": entry["total_score"],
                            "avg_advantage": entry["avg_advantage"],
                            "advantage_std": entry["advantage_std"],
                            "num_tokens": entry["num_tokens"],
                            "finish_reason": entry["finish_reason"],
                            "dataset": entry["dataset"],
                            "spans_summary": str(spans_summary)[:500] + "..." if len(str(spans_summary)) > 500 else str(spans_summary),
                            "token_advantages": token_adv_summary[:500] + "..." if len(token_adv_summary) > 500 else token_adv_summary,
                        })
                    
                    # Log to wandb
                    train_df = pd.DataFrame(rollout_table_data)
                    wandb.log({"training_rollouts": wandb.Table(dataframe=train_df)}, step=episode)
                    
                    # Also log detailed rollout data as a JSON artifact for deeper analysis
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(training_rollouts_data, f, indent=2)
                        temp_path = f.name
                    
                    artifact = wandb.Artifact(f"training_rollouts_step_{training_step}", type="training_data")
                    artifact.add_file(temp_path, name=f"rollouts_step_{training_step}.json")
                    wandb.log_artifact(artifact)
                    
                    # Clean up temp file
                    os.unlink(temp_path)

                if args.save_freq > 0 and training_step % args.save_freq == 0:
                    with Timer("[Main Thread] 🗡️ Saving model"):
                        checkpoint_dir = f"{args.output_dir}_checkpoints"
                        step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
                        print(f"Saving model at step {training_step} to {step_dir}")
                        ray.get([policy_group.models[i].save_model.remote(step_dir) for i in range(args.world_size)])
                        if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                            leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
                            for i in range(args.world_size):
                                policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                                    step_dir, leaderboard_name, wandb_url, training_step
                                )
                if (
                    args.checkpoint_state_freq > 0
                    and training_step % args.checkpoint_state_freq == 0
                    and args.checkpoint_state_dir is not None
                ):
                    with Timer("[Main Thread] 🗡️ Saving checkpoint state"):
                        client_state = {"training_step": training_step}
                        ray.get(
                            [
                                policy_group.models[i].save_checkpoint_state.remote(
                                    args.checkpoint_state_dir, client_state
                                )
                                for i in range(args.world_size)
                            ]
                        )
                        print(f"Saved checkpoint state at step {training_step} to {args.checkpoint_state_dir}")

            if len(update_ref_policy_future) > 0:
                with Timer("[Main Thread] 🔃 Updating reference policy"):
                    ray.get(update_ref_policy_future)

            # ------------------------------------------------------------------------------------------------
            # Optionally evaluate the model
            try:
                # timeout 0.01 if this is the last training step or we're not evaluating
                # otherwise, wait to get the last evaluation generations (long timeout just in case)
                timeout = 0.01 if (training_step < args.num_training_steps or args.eval_freq < 0) else 100
                eval_responses, eval_finish_reasons, masks, eval_infos = evaluation_inference_results_Q.get(
                    timeout=timeout
                )
                print("[Main Thread] 📊 Evaluation responses received")

                eval_sequence_lengths = np.array([len(response) for response in eval_responses])
                eval_decoded_responses = tokenizer.batch_decode(eval_responses, skip_special_tokens=True)
                eval_stop_rate = sum(int(finish_reason == "stop") for finish_reason in eval_finish_reasons) / len(
                    eval_finish_reasons
                )

                # get and log evaluation metrics
                eval_scores, eval_reward_metrics = asyncio.run(
                    reward_fn(
                        eval_responses,
                        eval_decoded_responses,
                        eval_ground_truths,
                        eval_dataset_names,
                        eval_finish_reasons,
                        eval_infos,
                    )
                )
                eval_reward_metrics = {f"eval/{key}": val for key, val in eval_reward_metrics.items()}
                # Extract scalar scores from FinegrainedRewardOutput objects
                scalar_eval_scores = []
                for score_output in eval_scores:
                    if hasattr(score_output, 'finegrained_scores'):
                        # Sum all finegrained scores for this response
                        total_score = sum(score_obj.score for score_obj in score_output.finegrained_scores)
                        scalar_eval_scores.append(total_score)
                    else:
                        # Fallback for scalar scores
                        scalar_eval_scores.append(float(score_output))
                
                eval_metrics = {
                    "eval/scores": np.array(scalar_eval_scores).mean() if scalar_eval_scores else 0.0,
                    "eval/sequence_lengths": eval_sequence_lengths.mean(),
                    "eval/sequence_lengths_min": eval_sequence_lengths.min(),
                    "eval/sequence_lengths_max": eval_sequence_lengths.max(),
                    "eval/stop_rate": eval_stop_rate,
                    **eval_reward_metrics,
                }
                print_rich_single_line_metrics(eval_metrics)
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, episode)
                table = {}
                table["prompt"] = tokenizer.batch_decode(eval_prompt_token_ids)
                table["response"] = eval_decoded_responses
                table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
                table["scores"] = scalar_eval_scores
                table["ground_truth"] = eval_ground_truths
                df = pd.DataFrame(table)
                if args.with_tracking:
                    wandb.log({"sample_completions": wandb.Table(dataframe=df)})
                else:
                    print_rich_table(df.iloc[:1])
                del table
            except Empty:
                print("[Main Thread] 🙈 Evaluation responses not received")

        print(f"Saving final model at step {training_step} to {args.output_dir}")
        with Timer("[Main Thread] 🗡️ Saving model"):
            ray.get([policy_group.models[i].save_model.remote(args.output_dir) for i in range(args.world_size)])
            if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                leaderboard_name = args.hf_repo_revision
                for i in range(args.world_size):
                    policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                        args.output_dir, leaderboard_name, wandb_url, training_step
                    )

    except Exception as e:
        print(f"Training error occurred: {str(e)}")
        print(traceback.format_exc())
        try:
            asyncio.run(cleanup_all_llm_judge_clients())
            print("✅ LLM judge clients cleaned up")
        except Exception as cleanup_error:
            print(f"Warning: Error during LLM judge cleanup: {cleanup_error}")

        # Clean up MCP subprocess
        if mcp_process is not None:
            try:
                print("🧹 Cleaning up MCP server subprocess...")
                if mcp_process.poll() is None:
                    os.killpg(os.getpgid(mcp_process.pid), signal.SIGTERM)
                    time.sleep(2)
                    if mcp_process.poll() is None:
                        os.killpg(os.getpgid(mcp_process.pid), signal.SIGKILL)
                print("✅ MCP server subprocess cleaned up")
            except (OSError, ProcessLookupError) as cleanup_error:
                print(f"Warning: Error during MCP cleanup: {cleanup_error}")

        ray.shutdown()
        os._exit(1)
        raise  # Re-raise the exception after shutdown

    # Clean up threads
    thread.join()
    print("======== ✅ vllm generate thread ends =========")
    packing_thread.join()
    print("======== ✅ data preparation thread ends =========")

    try:
        asyncio.run(cleanup_all_llm_judge_clients())
        print("✅ LLM judge clients cleaned up")
    except Exception as cleanup_error:
        print(f"Warning: Error during LLM judge cleanup: {cleanup_error}")

    ray.shutdown()

    # Ai2 logic: we use /output to store the artifacts of the job, so we
    # make a copy of the model to `/output` in the end.
    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
    print("finished training")

    # Note: MCP subprocess cleanup is handled by atexit handler registered in launch_mcp_subprocess
    if mcp_process is not None:
        print("🧹 MCP server subprocess will be cleaned up by atexit handler")

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



if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, Args)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)

    reward_fn_mapping = build_all_verifiers(args)

    async def reward_fn(
        responses: List[torch.Tensor],
        decoded_responses: List[str],
        ground_truths: List[Union[str, List[str]]],
        datasets: List[str],
        finish_reasons: List[str],
        infos: List[List[int]],
        queries: Optional[List[str]] = None,
    ):
        num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = infos
        good_outputs = [
            len(tool_outputs[i]) > 0 and tool_calleds[i]
            for i in range(len(tool_outputs))
        ]
        
        # Check if we need finegrained rewards
        assert args.apply_finegrained_reward, "Finegrained rewards are not applied"
        # For finegrained rewards, we need to return (finegrained_scores, metrics)
        metrics = {}
        
        with Timer("[Data Preparation Thread] Calculating rewards -- 🏆 Applying finegrained reward"):
            finegrained_rewards, log_values = await apply_finegrained_reward(
                reward_fn_mapping,
                responses,
                decoded_responses,
                ground_truths,
                datasets,
                reward_mult=args.finegrained_reward,
                queries=queries,
                overwrite_reward_fn_tag=args.overwrite_reward_fn_tag,
                num_samples_per_prompt_rollout=args.num_samples_per_prompt_rollout,
            )
            
            # log finegrained reward log values
            for key, value in log_values.items():
                metrics[f"objective/reward_log_values/{key}"] = value
            
            # Directly compute advantages in the reward function
            scores_by_query_and_reward_group = defaultdict(lambda: defaultdict(list))
            for finegrained_reward in finegrained_rewards:
                for score_obj in finegrained_reward.finegrained_scores:
                    # score_obj is a FinegrainedScore object including attributes: score, effective_spans, reward_group_id, query_idx
                    scores_by_query_and_reward_group[score_obj.query_idx][score_obj.reward_group_id].append(score_obj.score)
            
            # Calculate average score per response for metrics
            reward_stats_by_query_and_reward_group = defaultdict(lambda: defaultdict(float))
            for query_idx, reward_group_id_to_scores in scores_by_query_and_reward_group.items():
                for reward_group_id, scores in reward_group_id_to_scores.items():
                    mean, std = np.mean(scores), np.std(scores) + 1e-8
                    reward_stats_by_query_and_reward_group[query_idx][reward_group_id] = (mean, std)
            
            # Calculate advantages
            for finegrained_reward in finegrained_rewards:
                for score_obj in finegrained_reward.finegrained_scores:
                    mean, std = reward_stats_by_query_and_reward_group[score_obj.query_idx][score_obj.reward_group_id]
                    score_obj.advantage = (score_obj.score - mean) / std
            
        return finegrained_rewards, metrics

    main(args, tokenizer_config, model_config, reward_fn)
