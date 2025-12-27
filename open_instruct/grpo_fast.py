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
import contextlib
import os
import pathlib
from concurrent import futures

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed

from open_instruct import utils

# isort: on
import asyncio
import dataclasses
import json
import logging
import math
import random
import shutil
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from queue import Empty, Full, Queue
from typing import Any, Literal

import backoff
import datasets
import numpy as np
import pandas as pd
import ray
import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
import wandb
from datasets import Dataset
from huggingface_hub import HfApi
from peft import PeftModel, get_peft_model_state_dict
from ray.util import queue as ray_queue
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, get_scheduler
from transformers.integrations import HfDeepSpeedConfig

from open_instruct import data_loader as data_loader_lib
from open_instruct import logger_utils, vllm_utils
from open_instruct.actor_manager import ActorManager
from open_instruct.data_types import (
    CollatedBatchData,
    GenerationResult,
    PromptRequest,
    RequestInfo,
    ShutdownSentinel,
    TokenStatistics,
)
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.ground_truth_utils import RewardConfig, build_all_verifiers, cleanup_all_llm_judge_clients
from open_instruct.model_utils import (
    Batch,
    ModelConfig,
    disable_dropout_in_model,
    entropy_from_logits,
    estimate_kl,
    get_olmo3_generation_config,
    load_ref_policy,
    log_softmax_and_gather,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
)
from open_instruct.rl_utils import PackedSequences, Timer, masked_mean, pack_sequences
from open_instruct.tool_utils import tools
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    RayProcess,
    _z3_params_to_fetch,
    calibrate_checkpoint_state_dir,
    clean_last_n_checkpoints_deepspeed,
    combine_reward_metrics,
    download_latest_checkpoint_from_gs,
    get_beaker_whoami,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_update_beaker_description,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    ray_get_with_progress,
    repeat_each,
    sync_gs_bucket,
)

logger = logger_utils.setup_logger(__name__)

INVALID_LOGPROB = 1.0
CHECKPOINT_COMPLETE_MARKER = ".checkpoint_complete"


@dataclass
class Args:
    # Dataset
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_eval_list: list[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    """A list of datasets (local or HF) to sample from for evaluation."""
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_mixer_eval_list_splits: list[str] = field(default_factory=lambda: ["test"])
    """The dataset splits to use for evaluation"""
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_max_length_filter_v1"])
    """The list of transform functions to apply to the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: str | None = None
    """The hash of the dataset configuration."""
    dataset_config_eval_hash: str | None = None
    """The hash of the dataset configuration for evaluation."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    shuffle_eval_dataset: bool = False
    """Whether to shuffle the evaluation dataset."""
    max_prompt_token_length: int = 256
    """The maximum prompt token length to use for the dataset"""
    system_prompt_override_file: str | None = None
    """Path to a text file containing a system prompt to override the dataset's system prompts"""

    # Experiment
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: str | None = None
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
    world_size: int | None = None
    """RUNTIME VALUE: The number of processes (GPUs) to use for training ONLY"""
    num_training_steps: int | None = None
    """RUNTIME VALUE: The number of training_steps to train"""
    local_eval_every: int = 100
    """Run evaluation after this many training steps. This controls in-loop evals, which reuse the generation/reward verifier setup. Set to -1 to disable."""
    save_freq: int = 200
    """How many train steps to save the model"""
    backend_timeout: int = 120
    """Timeout for inference/training backends in minutes. Default is 2 hours (120 min)."""

    # Generation
    response_length: int = 256
    """the length of the response"""
    temperature: float = 0.7
    """the sampling temperature"""
    num_unique_prompts_rollout: int = 16
    """The number of unique prompts during rollout"""
    num_samples_per_prompt_rollout: int = 4
    """the number of samples to generate per prompt during rollout, useful for easy-star"""
    stop_strings: list[str] | None = None
    """List of strings that stop the generation when they are generated.
    The returned output will not contain the stop strings."""
    # Algorithm
    async_steps: int = 1
    """Number of steps ahead to generate responses. Fully synchronous training is not supported, so async_steps must be greater than 0. The trainer learns from a policy up to async_steps old like Cleanba (https://arxiv.org/abs/2310.00036)"""
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
    truncated_importance_sampling_ratio_cap: float = 0.0
    """The maximum cap for truncated importance sampling ratio (0 means disabled)"""
    inflight_updates: bool = False
    """Enable immediate stopping of request processing when should_stop is set, allowing for quick pausing and resumption"""
    kl_estimator: Literal[0, 1, 2, 3] = 2
    """the KL estimator to use"""
    pack_length: int = 512
    """the length of the pack (you should prob set to the max length of the model)"""
    loss_denominator: str = "token"
    """Optional constant denominator for masked_mean; can be "token" or a float value.
    when "token", the loss is divided by the total number of tokens in the batch (standard LM training).
    when a float value, the loss is divided by this value (ideally, max tokens in batch, per Dr GRPO).
    """
    alpha: float = 0.6
    """The alpha value for doing polyak updates (ref_param = alpha * param + (1 - alpha) * ref_param)
    reference: [TR-DPO](https://huggingface.co/papers/2404.09656), but it's actually pretty commonly
    used. E.g., [TD3](https://arxiv.org/abs/1802.09477) uses https://github.com/vwxyzjn/cleanrl/blob/dcc289fc6f0bda492fa7360a155262cf826b12a5/cleanrl/td3_continuous_action.py#L269
    """
    ref_policy_update_freq: int | None = None
    """How many training steps to take before updating the reference policy."""
    load_ref_policy: bool = True
    """Whether to load and use a reference policy for KL penalty calculation."""
    advantage_normalization_type: Literal["standard", "centered"] = "standard"
    """The type of advantage normalization to use. Standard normalization is the default: it subtracts the mean and
    divides by the standard deviation. Centered normalization is the same but subtracts the mean only (e.g., used in
    DR.GRPO https://arxiv.org/pdf/2503.20783)."""
    mask_truncated_completions: bool = False
    """Whether to mask out truncated completions. Also called overlong filtering, from DAPO (https://arxiv.org/abs/2503.14476)."""
    loss_fn: Literal["dapo", "cispo"] = "dapo"
    """Whether to use DAPO or CISPO loss function."""

    active_sampling: bool = False
    """Whether to continue sampling responses until you get a full batch."""
    filter_zero_std_samples: bool = True
    """Whether to filter out prompts with zero reward std (all samples have the same score)."""
    no_resampling_pass_rate: float | None = None
    """If the response to a prompt is solved at a rate higher than this, do not resample this prompt again"""
    record_entropy: bool = False
    """whether to record the entropy of the policy during training. Uses extra memory."""
    use_vllm_logprobs: bool = False
    """whether to use vLLM's logprobs for training instead of calculating them via forward pass"""

    # Reward
    # -- r1 style format reward
    apply_r1_style_format_reward: bool = False
    """whether to add the R1 style format reward"""
    r1_style_format_reward: float = 1.0
    """the reward value for R1 style format reward"""
    additive_format_reward: bool = False
    """whether to add the format reward to the final reward"""

    # -- verifiable reward
    apply_verifiable_reward: bool = True
    """whether to apply verifiable reward"""
    verification_reward: float = 10.0
    """the reward value for verifiable responses"""
    remap_verifier: str = None
    """Remap verifier like string_f1=general-quality_ref. Currently can only remap once."""

    # -- llm verifiers
    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    """the model to use for the llm judge"""
    llm_judge_max_tokens: int = 2048
    """the max tokens to use for the llm judge"""
    llm_judge_max_context_length: int = 8192
    """the max context length to use for the llm judge"""
    llm_judge_temperature: float = 1.0
    """the temperature to use for the llm judge"""
    llm_judge_timeout: int = 60
    """the timeout to use for the llm judge"""

    # -- code verifier
    code_api_url: str = os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program"
    """the api url to use for the code verifier"""
    code_max_execution_time: float = 1.0
    """the max execution time to use for the code verifier"""
    code_pass_rate_reward_threshold: float = 0.0
    """the pass rate reward threshold for the code verifier. If pass rate is less than this threshold, reward is 0.0, otherwise reward is pass rate"""
    code_apply_perf_penalty: bool = False
    """whether to apply a performance penalty to the code verifier"""

    # -- max length verifier
    max_length_verifier_max_length: int = 32768
    """the max length to use for the max length verifier"""

    # -- non stop penalty
    non_stop_penalty: bool = False
    """whether to penalize responses which did not finish generation"""
    non_stop_penalty_value: float = 0.0
    """the reward value for responses which did not finish generation"""

    # Ray
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node (mostly for debugging purposes)"""
    num_learners_per_node: list[int] = field(default_factory=lambda: [1])
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
    deepspeed_zpg: int = 8
    """the deepspeed zpg value. Higher values are more memory efficient but slower. Set to 1 to disable zpg, which uses less memory but is significantly slower. Ideally is set to the number of GPUs per node (usually 8, default)."""
    deepspeed_offload_param: bool = False
    """whether to offload parameters to CPU (reduces GPU memory usage)"""
    deepspeed_offload_optimizer: bool = False
    """whether to offload optimizer states to CPU (reduces GPU memory usage)"""
    gather_whole_model: bool = True
    """whether to gather the whole model to boardcast (not doable for 70B but can be faster for 8B)"""
    enable_queue_dashboard: bool = True
    """whether to enable the ActorManager queue monitoring dashboard"""
    queue_dashboard_port: int | None = None
    """optional port for the dashboard server (if None, finds a free port automatically)"""

    # Experiment tracking
    verbose: bool = False
    """If toggled, debug output will be shown"""
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: str | None = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: str | None = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: str | None = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: str | None = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: str | None = None
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
    checkpoint_state_dir: str | None = None
    """Where to save the model checkpoint (if applicable)"""
    gs_checkpoint_state_dir: str | None = None
    """The actual `checkpoint_state_dir` to use (handling the case where gs_bucket_path is provided)"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: list[str] | None = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""
    oe_eval_beaker_image: str | None = None
    """the docker image for evaluation for oe-eval"""
    oe_eval_gpu_multiplier: int | None = None
    """multiply the gpus used for each oe-eval task"""
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    """the priority of auto-launched evaluation jobs"""
    eval_workspace: str = "ai2/tulu-3-results"
    """the workspace to launch evaluation jobs on"""
    send_slack_alerts: bool = False
    """Whether to send Slack alerts on training failures"""

    # Evaluation behavior
    eval_on_step_0: bool = False
    """Whether to run local evaluation at training step 0. Defaults to False."""

    # Tool settings
    tools: list[str] | None = None
    """If set, use the tool mapped to the string. Currently only supports `search` and `code`"""
    max_tool_calls: tuple[int, ...] = (5,)
    """Maximum number of tool calls allowed. If a tuple is provided, it must have length 1 (applies to all tools) or same length as tools (per-tool limit)."""
    mask_tool_use: bool = True
    """Whether to mask the tool output. By default on."""
    only_reward_good_outputs: bool = False
    """Whether to only reward good outputs. By default off. Useful to force the model to use the tool(s)."""

    # rl-rag specific settngs
    number_documents_to_search: int = 3
    """The maximum number of documents to retrieve for each query."""
    search_api_endpoint: str | None = None
    """The API endpoint for the search engine."""

    # code-tool specific settings
    code_tool_api_endpoint: str | None = None

    def __post_init__(self):
        if os.environ.get("VLLM_USE_V1") == "0":
            logger.warning("When using the v0 version of vLLM, caching is broken and will never be invalidated.")
            if self.vllm_enable_prefix_caching:
                raise ValueError("Prefix caching is currently not supported for v0.")
        if self.use_vllm_logprobs and self.truncated_importance_sampling_ratio_cap > 0.0:
            raise ValueError(
                "Cannot use both `use_vllm_logprobs` and `truncated_importance_sampling_ratio_cap`. "
                "use_vllm_logprobs sets old_logprobs to vLLM logprobs, making importance sampling pointless."
            )
        self.loss_denominator = utils.get_denominator(self.loss_denominator)
        assert self.num_samples_per_prompt_rollout > 0, "Number of samples per prompt must be greater than 0!"
        if self.num_samples_per_prompt_rollout == 1:
            logger.warning("num_samples_per_prompt_rollout is 1. This reduces GRPO to REINFORCE.")
        assert self.apply_verifiable_reward or self.apply_r1_style_format_reward or self.non_stop_penalty, (
            "At least one reward must be applied!"
        )
        # Ensure we have enough prompts for all VLLM engines
        if self.num_unique_prompts_rollout < self.vllm_num_engines:
            logger.warning(
                f"With num_unique_prompts_rollout={self.num_unique_prompts_rollout} < "
                f"vllm_num_engines={self.vllm_num_engines}, vllm will be generating data for multiple "
                "batches simultaneously. This is fine but might be unexpected behaviour."
            )
        # ensure enough samples for all ranks
        assert self.num_samples_per_prompt_rollout * self.num_unique_prompts_rollout >= sum(
            self.num_learners_per_node
        ), "You must have at least as many samples as training GPUs (DP ranks) for distributed training!"
        if self.stop_strings is None:
            self.stop_strings = []
        assert self.pack_length >= self.max_prompt_token_length + self.response_length, (
            "The `pack_length` needs to be greater than the sum of `max_prompt_token_length` and `response_length`!"
        )
        if self.checkpoint_state_freq > 0 and self.checkpoint_state_dir is None:
            raise ValueError("`checkpoint_state_dir` must be provided if `checkpoint_state_freq` is greater than 0!")
        if self.checkpoint_state_dir is not None and self.checkpoint_state_freq == -1:
            raise ValueError("`checkpoint_state_freq` must be greater than 0 if `checkpoint_state_dir` is provided!")

        if self.gs_checkpoint_state_dir is not None and not self.gs_checkpoint_state_dir.startswith("gs://"):
            raise ValueError(f"`gs_checkpoint_state_dir` must start with 'gs://', got: {self.gs_checkpoint_state_dir}")
        if self.gs_bucket_path is not None and not self.gs_bucket_path.startswith("gs://"):
            raise ValueError(f"`gs_bucket_path` must start with 'gs://', got: {self.gs_bucket_path}")

        if self.gs_bucket_path is not None and self.gs_checkpoint_state_dir is None:
            if self.checkpoint_state_dir is None:
                raise ValueError("`checkpoint_state_dir` must be provided when using `gs_bucket_path`!")
            checkpoint_dir_name = self.checkpoint_state_dir.rstrip("/")
            beaker_users = get_beaker_whoami()
            if beaker_users is not None:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{beaker_users}/{checkpoint_dir_name}"
            else:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{checkpoint_dir_name}"
            # On GCP, all checkpointing must happen on filestore.
            # TODO(finbarrtimbers): Change this so we can checkpoint to GCS.
            # TODO(finbarrtimbers): Move this logic to mason.py once we refactor config.
            if not checkpoint_dir_name.startswith("/filestore"):
                self.checkpoint_state_dir = f"/filestore{self.checkpoint_state_dir}"

        if self.checkpoint_state_dir is not None:
            if self.gs_checkpoint_state_dir is not None:
                download_latest_checkpoint_from_gs(self.gs_checkpoint_state_dir, self.checkpoint_state_dir)
            calibrate_checkpoint_state_dir(self.checkpoint_state_dir)
        if self.tools is not None and len(self.tools) > 0:
            for tool in self.tools:
                if tool not in ["search", "code"]:
                    raise ValueError(f"Tool {tool} is not supported. Supported tools are: search, code")
            assert len(self.tools) == len(set(self.tools)), "Duplicate tools are not allowed"
            if self.use_vllm_logprobs or self.truncated_importance_sampling_ratio_cap > 0.0:
                assert self.mask_tool_use, (
                    "Must mask tool use when using vLLM logprobs or truncated importance sampling."
                )
        if not self.load_ref_policy and self.beta != 0.0:
            raise ValueError(
                "When load_ref_policy=False, beta must be 0.0. "
                f"Got beta={self.beta}. Set --beta 0.0 or --load_ref_policy to use KL penalty."
            )

        # Figure out max possible RLVR score
        self.max_possible_score = 0
        if self.apply_verifiable_reward:
            self.max_possible_score += self.verification_reward
        if self.apply_r1_style_format_reward and self.additive_format_reward:
            self.max_possible_score += self.r1_style_format_reward

        if self.active_sampling:
            assert self.async_steps > 1, (
                "With active_sampling, you should set async_steps > 1 to account for filtering of the first batch. "
                "Otherwise, your generator only generates only one batch worth of prompts and a single filtered "
                "prompt will cause the trainer to stall waiting for more data  . "
            )
            assert self.filter_zero_std_samples, (
                "filter_zero_std_samples must be True when active_sampling is True. "
                "Active sampling requires filtering to work correctly."
            )
        if self.num_samples_per_prompt_rollout == 1 and self.filter_zero_std_samples:
            raise ValueError(
                "`filter_zero_std_samples` cannot be True when `num_samples_per_prompt_rollout` is 1, "
                "as the reward standard deviation will always be 0, causing all samples to be filtered."
            )
        if self.async_steps < 1:
            raise ValueError("`async_steps` must be greater than 0. Fully synchronous training is not supported.")


def collate_fn(tensors_list: list[torch.Tensor], pad_token_id: int, pin_memory: bool = True) -> torch.Tensor:
    padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors_list, batch_first=True, padding_value=pad_token_id)
    if pin_memory:
        padded_tensor = padded_tensor.pin_memory()
    return padded_tensor


@Timer("🔄 [Data Preparation Thread] Prepare collated data for each worker")
def prepare_collated_data_for_workers(
    packed_sequences: PackedSequences,
    world_size: int,
    per_device_train_batch_size: int,
    pad_token_id: int,
    pin_memory: bool = True,
) -> list[CollatedBatchData]:
    """Distributes and collates packed sequences for distributed training.

    Splits packed sequences across workers, randomly shuffles each worker's data,
    and collates into micro-batches for training.

    Args:
        packed_sequences: Packed training sequences containing query responses,
            attention masks, position IDs, advantages, response masks,
            and vllm logprobs.
        world_size: Number of distributed workers.
        per_device_train_batch_size: Batch size for each device's micro-batch.
        pad_token_id: Token ID used for padding sequences.
        pin_memory: Whether to pin memory for faster data transfer to GPU.

    Returns:
        List of dictionaries, one per worker, each containing collated tensors
        for query_responses, attention_masks, position_ids,
        advantages, response_masks, and vllm_logprobs.
    """
    B = len(packed_sequences.query_responses) // world_size  # essentially doing `drop_last=True`, which is fine.
    collated_data = []
    for i in range(world_size):
        per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
        per_device_packed_attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
        per_device_packed_position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
        per_device_packed_advantages = packed_sequences.advantages[B * i : B * (i + 1)]
        per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]
        per_device_packed_vllm_logprobs = packed_sequences.vllm_logprobs[B * i : B * (i + 1)]

        # Shuffle the batch and collate the data
        b_inds = np.random.permutation(len(per_device_packed_query_responses))
        collated_query_responses = []
        collated_attention_masks = []
        collated_position_ids = []
        collated_response_masks = []
        collated_advantages = []
        collated_vllm_logprobs = []
        for j in range(0, len(per_device_packed_query_responses), per_device_train_batch_size):
            micro_range = b_inds[j : j + per_device_train_batch_size]
            collated_query_responses.append(
                collate_fn([per_device_packed_query_responses[idx] for idx in micro_range], pad_token_id, pin_memory)
            )
            collated_attention_masks.append(
                collate_fn([per_device_packed_attention_masks[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_position_ids.append(
                collate_fn([per_device_packed_position_ids[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_response_masks.append(
                collate_fn([per_device_packed_response_masks[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_advantages.append(
                collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_vllm_logprobs.append(
                collate_fn([per_device_packed_vllm_logprobs[idx] for idx in micro_range], 0, pin_memory)
            )
        collated_data.append(
            CollatedBatchData(
                query_responses=collated_query_responses,
                attention_masks=collated_attention_masks,
                position_ids=collated_position_ids,
                advantages=collated_advantages,
                response_masks=collated_response_masks,
                vllm_logprobs=collated_vllm_logprobs,
            )
        )
    return collated_data


def to_device_inplace(tensors_list: list[torch.Tensor], device: torch.device):
    for i in range(len(tensors_list)):
        tensors_list[i] = tensors_list[i].to(device, non_blocking=True)


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def from_pretrained(
        self,
        args: Args,
        model_config: ModelConfig,
        beaker_config: BeakerRuntimeConfig,
        wandb_url: str,
        tokenizer: PreTrainedTokenizer,
    ) -> int:
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

        # Set seeds for this worker (different per rank to avoid correlation)
        worker_seed = args.seed + self.local_rank
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        # Pre-initialize torch.distributed WITHOUT device_id to avoid NCCL hangs.
        # DeepSpeed 0.17.3 and up sets device_id in init_process_group which can cause hangs
        # when multiple process groups exist (e.g., for weight sync to vLLM).
        # By initializing first, DeepSpeed will detect it and wrap it instead of re-initializing.
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=args.backend_timeout))
        deepspeed.init_distributed(timeout=timedelta(minutes=args.backend_timeout))

        ds_config = get_train_ds_config(
            offload=args.deepspeed_offload_param,
            adam_offload=args.deepspeed_offload_optimizer,
            stage=args.deepspeed_stage,
            bf16=True,
            zpg=args.deepspeed_zpg,
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
        logger.info(f"Deepspeed config: {dschf=}")

        self.policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
            **({"device_map": {"": self.local_rank}} if args.deepspeed_stage != 3 else {}),
        )
        disable_dropout_in_model(self.policy)
        self.policy.gradient_checkpointing_enable()
        if args.set_weight_decay_on_bias_and_norm:
            optim_params = get_optimizer_grouped_parameters(self.policy, args.weight_decay)
        else:
            optim_params = self.policy.parameters()
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
                logger.warning(
                    f"Skipping loading checkpoint state from {args.checkpoint_state_dir} because it does not exist!"
                )
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

                rng_states = states["rng_states"]
                torch.set_rng_state(rng_states["torch_cpu_rng_state"])
                np.random.set_state(rng_states["numpy_rng_state"])
                random.setstate(rng_states["python_rng_state"])

                if torch.cuda.is_available() and "torch_cuda_rng_states" in rng_states:
                    # device_str, e.g. "cuda:0"
                    for device_str, rng_state in rng_states["torch_cuda_rng_states"].items():
                        device_id = int(device_str.split(":")[1])
                        torch.cuda.set_rng_state(rng_state, device_id)
                    if "torch_cuda_rng_state_all" in rng_states:
                        torch.cuda.set_rng_state_all(rng_states["torch_cuda_rng_state_all"])

                logger.info(f"{self.rank=}: Restored RNG states from checkpoint")

                # Save reference policy path to load later (after ref_policy is initialized)
                self.ref_policy_checkpoint_path = None
                if args.load_ref_policy and states.get("ref_policy_saved", False):
                    ref_policy_dir = os.path.join(args.checkpoint_state_dir, "ref_policy")
                    model_path = os.path.join(ref_policy_dir, "pytorch_model.bin")
                    if os.path.exists(model_path):
                        self.ref_policy_checkpoint_path = model_path
                        logger.info(f"{self.rank=}: Will load reference policy from {model_path}")

                logger.info(
                    f"{self.rank=}: Loaded checkpoint from {args.checkpoint_state_dir} with {optimization_steps_done=}"
                )
        self.model.train()

        # reference model
        if args.load_ref_policy:
            ds_config, self.ref_policy_hf_ds_config = get_eval_ds_config(
                offload=False,
                # inference model only has stage 3 (sharding) or stage 0 (no sharding)
                # stage 2 is optimizer sharding which doesn't apply to inference
                stage=args.deepspeed_stage if args.deepspeed_stage == 3 else 0,
                bf16=True,
                per_device_train_batch_size=args.per_device_train_batch_size,
            )

            self.ref_policy: PreTrainedModel = load_ref_policy(
                model_config=model_config,
                ds_config=ds_config,
                deepspeed_stage=args.deepspeed_stage,
                local_rank=self.local_rank,
                device=self.device,
                rank=self.rank,
                checkpoint_path=self.ref_policy_checkpoint_path
                if hasattr(self, "ref_policy_checkpoint_path")
                else None,
                ref_policy_update_freq=args.ref_policy_update_freq,
                alpha=args.alpha,
            )
        self.local_metrics = utils.MetricsTracker(device=self.device)
        return optimization_steps_done

    def forward(
        self,
        model: PreTrainedModel,
        query_response: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_token_id: int,
        temperature: float,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        # For now, entropy is just for monitoring, and we don't pass gradients through it.
        entropy = None
        if return_entropy:
            with torch.no_grad():
                entropy = entropy_from_logits(logits)

        return logprob, entropy

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
                    timeout_minutes=self.args.backend_timeout,
                )
                for i, engine in enumerate(vllm_engines)
            ]
            torch.cuda.set_device(self.local_rank)
            self.model_update_group = vllm_utils.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
                timeout=timedelta(minutes=self.args.backend_timeout),
            )
            ray_get_with_progress(refs, desc="Initializing vLLM process groups", timeout=600)
        torch.distributed.barrier()

    def broadcast_to_vllm(self):
        # avoid OOM
        torch.cuda.empty_cache()
        # Ensure CUDA device is set before broadcast operations.
        # DeepSpeed 0.17.3+ sets device_id in init_process_group which affects NCCL device binding.
        torch.cuda.set_device(self.local_rank)
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
                                name, dtype=str(param.dtype), shape=shape, empty_cache=count == num_params
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
                            name, dtype=str(param.dtype), shape=shape, empty_cache=count == num_params
                        )
                        for engine in self.vllm_engines
                    ]
                    refss.extend(refs)
                with deepspeed.zero.GatheredParameters([param], enabled=self.args.deepspeed_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)

        # Return futures instead of blocking - let caller handle completion
        all_refs = []
        if torch.distributed.get_rank() == 0:
            all_refs.extend(refss)
        return all_refs

    def update_ref_policy(self):
        if not self.args.load_ref_policy:
            return
        for ref_param, param in zip(self.ref_policy.parameters(), self.model.parameters()):
            if self.args.deepspeed_stage == 3:
                with deepspeed.zero.GatheredParameters([param, ref_param], modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)
            else:
                ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)

    def compute_logprobs(
        self, model: PreTrainedModel, data_BT: CollatedBatchData, pad_token_id: int, use_grad: bool = False
    ) -> list[torch.Tensor]:
        logprobs_BT: list[torch.Tensor] = []

        context = contextlib.nullcontext() if use_grad else torch.no_grad()
        with context:
            for i in range(len(data_BT.query_responses)):
                logprob_BT, _ = self.forward(
                    model,
                    data_BT.query_responses[i],
                    data_BT.attention_masks[i],
                    data_BT.position_ids[i],
                    pad_token_id,
                    self.args.temperature,
                    return_entropy=False,
                )

                response_mask_BT = data_BT.response_masks[i]
                logprob_BT = torch.masked_fill(logprob_BT, ~response_mask_BT[:, 1:], INVALID_LOGPROB)
                logprobs_BT.append(logprob_BT)

                torch.cuda.empty_cache()

        return logprobs_BT

    def calculate_token_counts(self, accumulation_steps: int, data_BT: CollatedBatchData) -> dict[int, float]:
        accumulation_counts: dict[int, float] = {}
        local_counts = [mask[:, 1:].sum().float() for mask in data_BT.response_masks]
        if not local_counts:
            return accumulation_counts

        # do the all_reduce once to avoid calling each loop
        counts_tensor = torch.stack(local_counts)
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)

        for i, count in enumerate(counts_tensor):
            group_idx = i // accumulation_steps
            key = int(group_idx * accumulation_steps)
            accumulation_counts[key] = accumulation_counts.get(key, 0.0) + count.item()

        return accumulation_counts

    def train(self, data_BT: CollatedBatchData, pad_token_id: int) -> dict[str, float]:
        """Train the policy model on a batch of data.

        Args:
            data_BT: CollatedBatchData containing:
                - query_responses: Token IDs for query+response sequences (B, T)
                - attention_masks: Attention mask (1=valid, 0=padding) (B, T)
                - position_ids: Position indices for positional embeddings (B, T)
                - advantages: Advantage estimates for RL training (B, T)
                - response_masks: Binary mask for response tokens (B, T)
                - vllm_logprobs: Log probabilities from vLLM engine (B, T)
            pad_token_id: Token ID used for padding.

        Returns:
            Dictionary of training metrics (loss, KL, entropy, etc.).
        """
        for f in dataclasses.fields(data_BT):
            to_device_inplace(getattr(data_BT, f.name), self.device)
        data_BT.response_masks = [mask.bool() for mask in data_BT.response_masks]
        num_samples = len(data_BT)
        accumulation_steps = max(math.ceil(num_samples / self.args.num_mini_batches - 0.5), 1)
        leftover = num_samples % accumulation_steps
        if leftover > 0:
            data_BT = data_BT[:-leftover]
            logger.warning(f"{leftover} samples are dropped due to batch size {self.args.num_mini_batches}")

        num_mini_batches = len(data_BT.query_responses) // accumulation_steps

        ref_logprobs_BT: list[torch.Tensor] = []
        if self.args.load_ref_policy:
            with Timer("Inference Calculation", noop=self.rank != 0):
                ref_logprobs_BT = self.compute_logprobs(self.ref_policy, data_BT, pad_token_id, use_grad=False)

        # if we have multiple minibatches, we need to calculate the old logprobs for each minibatch
        # following gtrl scripts in just doing this on the current active policy, rather than use the logprobs
        # from the generator (note that async mode means these are a bit diff!)
        old_logprobs_BT: list[torch.Tensor | None] = [None for _ in range(len(data_BT.query_responses))]
        if num_mini_batches > 1:
            with Timer("Old logprobs Calculation", noop=self.rank != 0):
                local_old_logprobs_BT = None
                if not self.args.use_vllm_logprobs:
                    local_old_logprobs_BT = self.compute_logprobs(self.model, data_BT, pad_token_id, use_grad=False)

                with torch.no_grad():
                    for i in range(len(data_BT.query_responses)):
                        vllm_old_logprob_BT = data_BT.vllm_logprobs[i][:, 1:]
                        vllm_old_logprob_BT = torch.masked_fill(
                            vllm_old_logprob_BT, ~data_BT.response_masks[i][:, 1:], INVALID_LOGPROB
                        )
                        vllm_old_logprob_BT = torch.nan_to_num(vllm_old_logprob_BT, nan=INVALID_LOGPROB)

                        if self.args.use_vllm_logprobs:
                            old_logprobs_BT[i] = vllm_old_logprob_BT
                        else:
                            old_logprobs_BT[i] = local_old_logprobs_BT[i]

                        torch.cuda.empty_cache()

        local_step = 0
        num_samples = len(data_BT.query_responses)
        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        with Timer("[Training Processes] Loss calculation", noop=self.rank != 0):
            loss_stats_B: dict[str, torch.Tensor] = {
                "kl": torch.zeros(4, num_samples),
                "kl_loss": torch.zeros(num_samples),
                "pg_clipfrac": torch.zeros(num_samples),
                "pg_loss": torch.zeros(num_samples),
                "loss": torch.zeros(num_samples),
                "ratio": torch.zeros(num_samples),
                "entropy": torch.zeros(num_samples),
            }
            for epoch_idx in range(self.args.num_epochs):
                # Pre-compute total tokens for each accumulation group if using "token" normalization
                # This ensures all minibatches in an accumulation group are normalized by the same total
                if self.args.loss_denominator == "token":
                    accumulation_token_counts = self.calculate_token_counts(accumulation_steps, data_BT)
                else:
                    accumulation_token_counts = {
                        int(group_idx * accumulation_steps): self.args.loss_denominator
                        for group_idx in range((len(data_BT.query_responses) // accumulation_steps) + 1)
                    }

                for i in range(num_samples):
                    response_mask_BT = data_BT.response_masks[i][:, 1:]
                    # retrieve the loss denominator for the current batch
                    batch_start = (i // accumulation_steps) * accumulation_steps
                    loss_denominator = accumulation_token_counts[batch_start]
                    local_logprobs_BT, entropy_BT = self.forward(
                        self.model,
                        data_BT.query_responses[i],
                        data_BT.attention_masks[i],
                        data_BT.position_ids[i],
                        pad_token_id,
                        self.args.temperature,
                        return_entropy=self.args.record_entropy,
                    )
                    local_logprobs_BT = torch.masked_fill(local_logprobs_BT, ~response_mask_BT, INVALID_LOGPROB)
                    vllm_logprobs_BT = data_BT.vllm_logprobs[i][:, 1:]
                    vllm_logprobs_BT = torch.masked_fill(vllm_logprobs_BT, ~response_mask_BT, INVALID_LOGPROB)
                    vllm_logprobs_BT = torch.nan_to_num(vllm_logprobs_BT, nan=INVALID_LOGPROB)

                    # Compare vLLM logprobs with local logprobs
                    with torch.no_grad():
                        valid_mask_BT = response_mask_BT & ~torch.isnan(vllm_logprobs_BT)
                        logprob_diff_BT = (local_logprobs_BT - vllm_logprobs_BT).abs()
                        masked_diff_BT = torch.masked_fill(logprob_diff_BT, ~valid_mask_BT, 0.0)
                        mean_diff = masked_diff_BT.sum() / valid_mask_BT.sum() if valid_mask_BT.sum() > 0 else 0.0
                        max_diff = masked_diff_BT.max()
                        std_diff = masked_diff_BT[valid_mask_BT].std() if valid_mask_BT.sum() > 1 else 0.0

                        self.local_metrics["debug/vllm_vs_local_logprob_diff_mean"] = mean_diff.item()
                        self.local_metrics["debug/vllm_vs_local_logprob_diff_max"] = max_diff.item()
                        self.local_metrics["debug/vllm_vs_local_logprob_diff_std"] = std_diff.item()

                        reverse_kl_BT = torch.exp(vllm_logprobs_BT) * (vllm_logprobs_BT - local_logprobs_BT)
                        masked_reverse_kl_BT = torch.masked_fill(reverse_kl_BT, ~valid_mask_BT, 0.0)
                        mean_reverse_kl = (
                            masked_reverse_kl_BT.sum() / valid_mask_BT.sum() if valid_mask_BT.sum() > 0 else 0.0
                        )
                        self.local_metrics["debug/vllm_local_reverse_kl"] = mean_reverse_kl.item()

                    new_logprobs_BT = local_logprobs_BT

                    # Cache the old logprobs
                    if num_mini_batches > 1:
                        old_logprob_BT = old_logprobs_BT[i]
                    else:
                        with torch.no_grad():
                            if epoch_idx == 0:
                                if self.args.use_vllm_logprobs:
                                    old_logprobs_BT[i] = vllm_logprobs_BT
                                else:
                                    old_logprobs_BT[i] = local_logprobs_BT.detach()
                            old_logprob_BT = old_logprobs_BT[i]

                    old_logprobs_mask_BT = old_logprob_BT != INVALID_LOGPROB
                    assert torch.all(old_logprobs_mask_BT == response_mask_BT), (
                        f"Old logprobs mask should match response mask. "
                        f"old_mask sum={old_logprobs_mask_BT.sum()}, "
                        f"response_mask sum={response_mask_BT.sum()}"
                    )

                    # Calculate the policy's loss
                    logprobs_diff_BT = new_logprobs_BT - old_logprob_BT
                    ratio_BT = torch.exp(logprobs_diff_BT)
                    if self.args.loss_fn == "dapo":
                        pg_losses_BT = -data_BT.advantages[i][:, 1:] * ratio_BT
                        pg_losses2_BT = -data_BT.advantages[i][:, 1:] * torch.clamp(
                            ratio_BT, 1.0 - self.args.clip_lower, 1.0 + self.args.clip_higher
                        )
                    elif self.args.loss_fn == "cispo":
                        # cispo: directly clip ratio, no lower bound.
                        # reinforce loss, so multiply by new logprobs
                        pg_losses_BT = (
                            -data_BT.advantages[i][:, 1:]
                            * torch.clamp(ratio_BT.detach(), max=1.0 + self.args.clip_higher)
                            * new_logprobs_BT
                        )
                        pg_losses2_BT = pg_losses_BT
                    else:
                        raise ValueError(f"Invalid loss function: {self.args.loss_fn}")

                    # Apply truncated importance sampling if enabled
                    if self.args.truncated_importance_sampling_ratio_cap > 0 and vllm_logprobs_BT is not None:
                        old_logprobs_mask_BT = old_logprob_BT != INVALID_LOGPROB
                        vllm_logprobs_mask_BT = vllm_logprobs_BT != INVALID_LOGPROB

                        assert torch.all(old_logprobs_mask_BT == response_mask_BT), (
                            f"Old logprobs mask should match response mask. "
                            f"old_mask sum={old_logprobs_mask_BT.sum()}, "
                            f"response_mask sum={response_mask_BT.sum()}"
                        )
                        assert torch.all(vllm_logprobs_mask_BT == response_mask_BT), (
                            f"vLLM logprobs mask should match response mask. "
                            f"vllm_mask sum={vllm_logprobs_mask_BT.sum()}, "
                            f"response_mask sum={response_mask_BT.sum()}"
                        )

                        valid_mask_BT = response_mask_BT

                        # Initialize importance ratio to 1.0 (no effect) for all positions
                        tis_imp_ratio_BT = torch.ones_like(old_logprob_BT)

                        if valid_mask_BT.any():
                            # Calculate logprob difference only for valid positions
                            logprob_diff_is_BT = old_logprob_BT - vllm_logprobs_BT
                            # Clamp to prevent numerical overflow in exp
                            logprob_diff_is_BT = torch.where(
                                valid_mask_BT,
                                logprob_diff_is_BT.clamp(-10.0, 10.0),
                                torch.zeros_like(logprob_diff_is_BT),
                            )
                            # Compute importance ratio only for valid positions
                            tis_imp_ratio_BT = torch.where(
                                valid_mask_BT, torch.exp(logprob_diff_is_BT), tis_imp_ratio_BT
                            )
                            # Apply cap
                            tis_imp_ratio_BT = torch.clamp(
                                tis_imp_ratio_BT, max=self.args.truncated_importance_sampling_ratio_cap
                            )

                        # Apply importance sampling to losses
                        pg_losses_BT = pg_losses_BT * tis_imp_ratio_BT
                        pg_losses2_BT = pg_losses2_BT * tis_imp_ratio_BT

                    pg_loss_max_BT = torch.max(pg_losses_BT, pg_losses2_BT)

                    if self.args.load_ref_policy:
                        ref_logprob_BT = ref_logprobs_BT[i]
                        # Here we recalculate kl: we want the KL loss to backpropagate through the model
                        # We also clamp the KL loss to avoid numerical instability
                        # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
                        ref_logprobs_diff_BT = (new_logprobs_BT - ref_logprob_BT).clamp(-40.0, 40.0)
                        kl_4BT = estimate_kl(ref_logprobs_diff_BT, ratio_BT)
                        # grpo change: directly subtract KL in loss (add)
                        loss = masked_mean(
                            pg_loss_max_BT + self.args.beta * kl_4BT[self.args.kl_estimator],
                            response_mask_BT,
                            None,
                            loss_denominator,
                        )
                    else:
                        loss = masked_mean(pg_loss_max_BT, response_mask_BT, None, loss_denominator)

                    # we already took world size into account via the tokens
                    if dist.is_available() and dist.is_initialized():
                        loss *= dist.get_world_size()

                    # Clear CUDA cache before backward pass to free memory for reduce_scatter operations
                    torch.cuda.empty_cache()
                    self.model.backward(loss)
                    if (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                    local_step += 1
                    with torch.no_grad():
                        if args.load_ref_policy:
                            # NOTE: in packed implementation, kl calculation are averages over response tokens
                            loss_stats_B["kl"][:, i] = masked_mean(kl_4BT, response_mask_BT).float()
                            loss_stats_B["kl_loss"][i] = loss_stats_B["kl"][self.args.kl_estimator, i] * self.args.beta
                        loss_stats_B["pg_clipfrac"][i] = masked_mean(
                            (pg_losses2_BT > pg_losses_BT).float(), response_mask_BT
                        )
                        loss_stats_B["pg_loss"][i] = masked_mean(pg_loss_max_BT, response_mask_BT)
                        loss_stats_B["loss"][i] = loss
                        loss_stats_B["ratio"][i] = masked_mean(ratio_BT, response_mask_BT)
                        if self.args.record_entropy:
                            loss_stats_B["entropy"][i] = masked_mean(entropy_BT, response_mask_BT).float()

            with torch.no_grad():
                if args.load_ref_policy:
                    for j in range(4):
                        self.local_metrics[f"objective/kl{j}_avg"] = loss_stats_B["kl"][j].mean()
                    self.local_metrics["loss/kl_avg"] = loss_stats_B["kl_loss"].mean()
                self.local_metrics["loss/policy_avg"] = loss_stats_B["pg_loss"].mean()
                self.local_metrics["loss/total_avg"] = loss_stats_B["loss"].mean()
                self.local_metrics["policy/clipfrac_avg"] = loss_stats_B["pg_clipfrac"].mean()
                self.local_metrics["val/ratio"] = loss_stats_B["ratio"].mean()
                self.local_metrics["val/ratio_var"] = loss_stats_B["ratio"].var()
                if self.args.record_entropy:
                    self.local_metrics["policy/entropy_avg"] = loss_stats_B["entropy"].mean()
                self.local_metrics["lr"] = self.scheduler.get_last_lr()[0]
                return self.local_metrics.get_metrics_list()

    def save_checkpoint_state(self, checkpoint_state_dir: str, client_state: dict[str, Any]) -> None:
        args = self.args

        # Save comprehensive RNG states for each rank
        rng_states = {
            "torch_cpu_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        # Save CUDA RNG states for all devices
        if torch.cuda.is_available():
            rng_states["torch_cuda_rng_states"] = {
                f"cuda:{i}": torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            }
            rng_states["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        # Add RNG states to client_state
        client_state["rng_states"] = rng_states
        client_state["rank"] = self.rank

        # Save reference policy checkpoint (model only, no optimizer)
        if self.args.load_ref_policy:
            ref_policy_dir = os.path.join(checkpoint_state_dir, "ref_policy")
            os.makedirs(ref_policy_dir, exist_ok=True)

            # For reference policy, we save just the model weights
            # We can't use save_checkpoint because it would try to save DummyOptim
            # which doesn't have state_dict
            if self.rank == 0:
                # Only rank 0 saves the model state
                model_to_save = self.ref_policy.module if hasattr(self.ref_policy, "module") else self.ref_policy
                # Save the state dict
                torch.save(model_to_save.state_dict(), os.path.join(ref_policy_dir, "pytorch_model.bin"))
                logger.info(f"Saved reference policy model to {ref_policy_dir}")

            client_state["ref_policy_saved"] = True

        # Save the main model checkpoint with enhanced client state
        self.model.save_checkpoint(checkpoint_state_dir, client_state=client_state)

        # `save_checkpoint` needs to be called on all ranks, only rank 0 will have all the states
        if self.rank == 0:
            if args.keep_last_n_checkpoints >= 0:
                clean_last_n_checkpoints_deepspeed(checkpoint_state_dir, args.keep_last_n_checkpoints)

            # Sync to GCS if configured (check the actual target, not just gs_bucket_path)
            if args.gs_checkpoint_state_dir is not None:
                ray.remote(sync_gs_bucket).options(num_cpus=1).remote(
                    checkpoint_state_dir, args.gs_checkpoint_state_dir
                )

    def save_model(self, output_dir: str, chat_template_name: str, tokenizer: PreTrainedTokenizer) -> None:
        output_path = pathlib.Path(output_dir)
        marker_path = output_path / CHECKPOINT_COMPLETE_MARKER
        if marker_path.exists():
            logger.info(f"Checkpoint already complete at {output_dir}, skipping save")
            return

        model_to_save = self.model
        if chat_template_name is not None and "olmo" in chat_template_name:
            model_to_save.generation_config = get_olmo3_generation_config(tokenizer)

        if self.rank == 0:
            output_path.mkdir(parents=True, exist_ok=True)

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
                        get_peft_model_state_dict(model_to_save, output_state_dict), output_path / "adapter_model.bin"
                    )
            else:
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict)

            self.tokenizer.save_pretrained(output_dir)
            marker_path.touch()

    # we need this because we don't know which node is rank 0 is on
    def launch_ai2_evals_on_weka_wrapper(self, step_dir, leaderboard_name, wandb_url, training_step):
        args = self.args
        if self.rank == 0:
            ray.remote(launch_ai2_evals_on_weka).options(num_cpus=1).remote(
                path=step_dir,
                leaderboard_name=leaderboard_name,
                oe_eval_max_length=args.oe_eval_max_length,
                wandb_url=wandb_url,
                training_step=training_step,
                oe_eval_tasks=args.oe_eval_tasks,
                stop_strings=args.stop_strings,
                gs_bucket_path=args.gs_bucket_path,
                eval_priority=args.eval_priority,
                eval_workspace=args.eval_workspace,
                beaker_image=args.oe_eval_beaker_image,
                oe_eval_gpu_multiplier=args.oe_eval_gpu_multiplier,
            )


class ModelGroup:
    def __init__(
        self, pg: PlacementGroup, ray_process_cls: RayProcess, num_gpus_per_node: list[int], single_gpu_mode: bool
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
        results, _ = ray_get_with_progress(
            [master_policy.get_master_addr_port.remote()], desc="Getting master address"
        )
        (master_addr, master_port) = results[0]

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
            logger.debug(f"{rank=}, {world_size=}, {rank=}, {master_addr=}, {master_port=}")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node)
            )
            worker_policy = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(world_size, rank, 0, master_addr, master_port)
            self.models.append(worker_policy)


def calculate_utilization_metrics(
    model_dims: utils.ModelDims,
    prompt_lengths: list[int],
    response_lengths: list[int],
    total_generation_time: float,
    samples_per_prompt: int,
    num_engines: int,
    num_gpus_per_engine: int,
    training_time: float,
    num_training_gpus: int,
) -> dict:
    """Calculate MFU and MBU metrics for model inference and training.

    Args:
        model_dims: Model dimensions with device information
        prompt_lengths: List of prompt lengths
        response_lengths: List of response lengths
        total_generation_time: Total time taken for generation (for actor metrics)
        samples_per_prompt: Number of samples generated per prompt
        num_engines: Number of vLLM engines for inference
        num_gpus_per_engine: Number of GPUs assigned to each vLLM engine (tensor parallel size)
        training_time: Time taken for training step (for learner metrics)
        num_training_gpus: Number of GPUs used for training (for learner metrics)

    Returns:
        Dict with the following keys:
            - actor_mfu: Model FLOPs utilization for inference (percentage)
            - actor_mbu: Model bandwidth utilization for inference (percentage)
            - learner_mfu: Model FLOPs utilization for training (percentage)
    """
    assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
        f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
    )

    actor_metrics = model_dims.calculate_actor_utilization(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        total_generation_time=total_generation_time,
        samples_per_prompt=samples_per_prompt,
        num_engines=num_engines,
        num_gpus_per_engine=num_gpus_per_engine,
    )

    learner_metrics = model_dims.calculate_learner_utilization(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        training_time=training_time,
        samples_per_prompt=samples_per_prompt,
        num_training_gpus=num_training_gpus,
    )

    utilization_metrics = {f"actor_{k}": v for k, v in actor_metrics.items()}
    utilization_metrics["learner_mfu"] = learner_metrics["mfu"]

    return utilization_metrics


@dataclass
class BatchStatistics:
    prompt_lengths: list[int]
    response_lengths: list[int]
    filtered_prompts: int
    filtered_prompts_zero: int
    filtered_prompts_solved: int
    filtered_prompts_nonzero: int
    percent_solved_mean: float
    no_resampled_prompts: int
    total_prompts: int


def accumulate_inference_batches(
    inference_results_Q: ray_queue.Queue,
    args: Args,
    generation_config: vllm_utils.SamplingConfig,
    num_prompts: int,
    model_dims: utils.ModelDims,
    tokenizer: PreTrainedTokenizer,
    prompt_dataset: Dataset,
    data_loader: data_loader_lib.HFDataLoader | None = None,
    prompt_Q: ray_queue.Queue | None = None,
    actor_manager=None,
    timeout: float | None = None,
    active_sampling: bool = False,
    filter_zero_std_samples: bool = False,
    replenish_prompts: bool = False,
    no_resampling_pass_rate: float | None = None,
) -> tuple[GenerationResult, Batch, dict, BatchStatistics]:
    """Accumulate multiple inference results into a single training batch.

    Args:
        inference_results_Q: Queue containing individual GenerationResult objects (one per prompt)
        args: Arguments containing vllm_num_engines and batch size info
        generation_config: Generation config containing n (number of samples per prompt)
        num_prompts: Number of prompts to accumulate
        data_loader: Iterator over the dataloader for replenishing prompts. Required when
            replenish_prompts=True or no_resampling_pass_rate is set. Can be None for
            evaluation where all prompts are pre-queued.
        prompt_dataset: Dataset containing prompts
        prompt_Q: Queue containing prompts to send to generator. Required when
            replenish_prompts=True. Can be None for evaluation where no replenishment is needed.
        timeout: Optional timeout in seconds for queue get operations. If None, blocks indefinitely.
        active_sampling: Whether to continue sampling until we have sampled num_prompts prompts with non-zero std
        filter_zero_std_samples: Whether to filter samples with zero reward std
        replenish_prompts: Add a prompt back onto the prompt_Q after receiving a finished result
        no_resampling_pass_rate: Optional rate at which to note samples solved at greater than this rate
            and exclude them from further sampling

    Raises:
        queue.Empty: If timeout is specified and no data is available within timeout.

    Returns:
        Tuple of (combined_result, Batch with queries, ground_truths, datasets, prompt_lengths, response_lengths)
        or (ShutdownSentinel, None, None, None) if shutdown signal received
    """
    if no_resampling_pass_rate is not None:
        assert data_loader is not None, "no_resampling requires data_loader"

    if replenish_prompts:
        assert prompt_Q is not None and data_loader is not None and prompt_dataset is not None, (
            "replenish_prompts requires prompt_Q, data_loader, and prompt_dataset"
        )
    results = []
    all_queries = []
    all_ground_truths = []
    all_datasets = []
    all_raw_queries = []
    all_decoded_responses = []
    all_reward_metrics = []
    all_scores = []
    all_percent_solved = []
    total_filtered_prompts = 0
    filtered_prompt_zero = 0
    filtered_prompt_solved = 0
    filtered_prompt_nonzero = 0
    total_no_resampled = 0
    progress_bar = tqdm(
        total=num_prompts,
        desc=f"Accumulating Responses and Rewarding {num_prompts} prompts",
        bar_format="{l_bar}{bar}{r_bar}\n",
        disable=not args.verbose,
    )
    num_prompts_sampled = 0
    while num_prompts_sampled < num_prompts:
        result = inference_results_Q.get(timeout=timeout)

        if isinstance(result, ShutdownSentinel):
            return result, None, None, None

        # Validate that each individual result has the expected number of responses
        assert len(result.responses) == generation_config.n, (
            f"Mismatch: individual prompt result has {len(result.responses)} responses "
            f"but expected {generation_config.n} samples per prompt. "
            f"Prompt ID: {result.prompt_id}"
        )

        # Replenish generation queue with new prompt
        if replenish_prompts:
            add_prompt_to_generator(next(data_loader), prompt_Q, generation_config, is_eval=False)

        decoded_responses = tokenizer.batch_decode(result.responses, skip_special_tokens=True)

        percent_solved = np.mean(result.reward_scores).item() / args.max_possible_score
        # Don't resample prompt that was solved at more than no_resample_positive_rate
        if no_resampling_pass_rate is not None and percent_solved >= no_resampling_pass_rate:
            total_no_resampled += 1
            data_loader.exclude_index(result.dataset_index)
            logging.debug(
                f"[Data Preparation Thread] Prompt solved at {percent_solved}, total no resampled: {total_no_resampled}"
            )

        # Filter out zero std prompts
        if filter_zero_std_samples and np.std(result.reward_scores) == 0:
            # If we're not active sampling, still count this as a sample
            if not active_sampling:
                num_prompts_sampled += 1
                progress_bar.update(1)

            total_filtered_prompts += 1
            if result.reward_scores[0] == 0:
                filtered_prompt_zero += 1
            elif result.reward_scores[0] == args.max_possible_score:
                filtered_prompt_solved += 1
            else:
                filtered_prompt_nonzero += 1
            logging.debug(
                f"[Data Preparation Thread] Filtered prompt with reward std 0, total filtered {total_filtered_prompts}"
            )
            continue
        else:
            num_prompts_sampled += 1
            progress_bar.update(1)

        results.append(result)
        prompt_data = prompt_dataset[result.dataset_index]
        all_queries.extend(repeat_each([prompt_data[INPUT_IDS_PROMPT_KEY]], generation_config.n))
        all_ground_truths.extend(repeat_each([prompt_data[GROUND_TRUTHS_KEY]], generation_config.n))
        all_datasets.extend(repeat_each([prompt_data[VERIFIER_SOURCE_KEY]], generation_config.n))
        all_raw_queries.extend(repeat_each([prompt_data[RAW_PROMPT_KEY]], generation_config.n))
        all_decoded_responses.extend(decoded_responses)
        all_scores.extend(result.reward_scores)
        all_reward_metrics.append(result.reward_metrics)
        all_percent_solved.append(percent_solved)

    if len(results) == 0:
        logger.warning(
            "[Data Preparation Thread] All prompts were filtered during accumulation. "
            f"Filtered: {total_filtered_prompts} (zero std: {filtered_prompt_zero}, "
            f"solved: {filtered_prompt_solved}, nonzero: {filtered_prompt_nonzero})"
        )
        return None, None, None, None

    # Combine all results into a single GenerationResult
    combined_responses = []
    combined_finish_reasons = []
    combined_masks = []
    combined_num_calls = []
    combined_timeouts = []
    combined_tool_errors = []
    combined_tool_outputs = []
    combined_tool_runtimes = []
    combined_tool_calleds = []
    combined_logprobs = []

    earliest_start_time = float("inf")
    prompt_lengths = []
    response_lengths = []

    total_prompt_tokens = 0
    total_response_tokens = 0
    max_generation_time = 0

    for i, result in enumerate(results):
        combined_responses.extend(result.responses)
        combined_finish_reasons.extend(result.finish_reasons)
        combined_masks.extend(result.masks)
        combined_num_calls.extend(result.request_info.num_calls)
        combined_timeouts.extend(result.request_info.timeouts)
        combined_tool_errors.extend(result.request_info.tool_errors)
        combined_tool_outputs.extend(result.request_info.tool_outputs)
        combined_tool_runtimes.extend(result.request_info.tool_runtimes)
        combined_tool_calleds.extend(result.request_info.tool_calleds)

        combined_logprobs.extend(result.logprobs)

        earliest_start_time = min(earliest_start_time, result.start_time)

        prompt_lengths.append(len(all_queries[i * generation_config.n]))

        for response in result.responses:
            response_lengths.append(len(response))

        total_prompt_tokens += result.token_statistics.num_prompt_tokens
        total_response_tokens += result.token_statistics.num_response_tokens
        max_generation_time = max(max_generation_time, result.token_statistics.generation_time)

    # Use the maximum generation time across engines since they work in parallel
    # This avoids including queue overhead and accumulation time in MFU/MBU calculations
    total_generation_time = max_generation_time

    accumulated_stats = TokenStatistics(
        num_prompt_tokens=total_prompt_tokens,
        num_response_tokens=total_response_tokens,
        generation_time=total_generation_time,
        earliest_start_time=earliest_start_time,
    )

    # Create combined RequestInfo
    combined_request_info = RequestInfo(
        num_calls=combined_num_calls,
        timeouts=combined_timeouts,
        tool_errors=combined_tool_errors,
        tool_outputs=combined_tool_outputs,
        tool_runtimes=combined_tool_runtimes,
        tool_calleds=combined_tool_calleds,
    )

    # Create combined GenerationResult
    combined_result = GenerationResult(
        responses=combined_responses,
        finish_reasons=combined_finish_reasons,
        masks=combined_masks,
        request_info=combined_request_info,
        dataset_index=None,
        prompt_id=None,
        token_statistics=accumulated_stats,
        logprobs=combined_logprobs,
    )

    if actor_manager is not None:
        ray.get(actor_manager.report_token_statistics.remote(accumulated_stats))

    # Note: We don't have dataset_indices here, but they're not needed for the returned batch
    batch = Batch(
        queries=all_queries,
        ground_truths=all_ground_truths,
        datasets=all_datasets,
        raw_queries=all_raw_queries,
        decoded_responses=all_decoded_responses,
        indices=None,  # Not meaningful for combined results
        scores=all_scores,
    )

    combined_reward_metrics = combine_reward_metrics(all_reward_metrics)
    percent_solved_mean = np.mean(all_percent_solved) if all_percent_solved else 0.0

    batch_stats = BatchStatistics(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        filtered_prompts=total_filtered_prompts,
        filtered_prompts_zero=filtered_prompt_zero,
        filtered_prompts_solved=filtered_prompt_solved,
        filtered_prompts_nonzero=filtered_prompt_nonzero,
        percent_solved_mean=percent_solved_mean,
        no_resampled_prompts=total_no_resampled,
        total_prompts=len(results),
    )
    return combined_result, batch, combined_reward_metrics, batch_stats


def data_preparation_thread(
    inference_results_Q: ray_queue.Queue,
    prompt_Q: ray_queue.Queue,
    packed_sequences_Q: Queue,
    args: Args,
    tokenizer: PreTrainedTokenizer,
    num_training_steps: int,
    generation_config,
    resume_training_step: int,
    data_loader: data_loader_lib.HFDataLoader,
    train_dataset: Dataset,
    actor_manager=None,
    model_dims: utils.ModelDims = None,
):
    for training_step in range(resume_training_step, num_training_steps + 1):
        # Streaming accumulation: collect results as they arrive
        with Timer("🚀 [Data Preparation Thread] Getting response ids") as timer:
            result, batch, reward_metrics, batch_stats = accumulate_inference_batches(
                inference_results_Q,
                args,
                generation_config,
                num_prompts=args.num_unique_prompts_rollout,
                model_dims=model_dims,
                tokenizer=tokenizer,
                data_loader=data_loader,
                prompt_dataset=train_dataset,
                prompt_Q=prompt_Q,
                actor_manager=actor_manager,
                active_sampling=args.active_sampling,
                filter_zero_std_samples=args.filter_zero_std_samples,
                replenish_prompts=True,
                no_resampling_pass_rate=args.no_resampling_pass_rate,
            )
            if isinstance(result, ShutdownSentinel):
                logger.info("[Data Preparation Thread] Received shutdown sentinel, exiting")
                return
            if result is None:
                logger.info("[Data Preparation Thread] All prompts filtered, putting empty batch into queue")
                packed_sequences = PackedSequences(
                    query_responses=[],
                    attention_masks=[],
                    response_masks=[],
                    original_responses=[],
                    advantages=[],
                    position_ids=[],
                    vllm_logprobs=[],
                )
                collated_data = []
                packed_sequences_Q.put(
                    {
                        "packed_sequences": packed_sequences,
                        "collated_data": collated_data,
                        "metrics": {},
                        "responses_count": 0,
                        "num_new_tokens": 0,
                        "B": 0,
                        "prompt_lengths": [],
                        "response_lengths": [],
                        "num_filtered_prompts": 0,
                    }
                )
                continue

        getting_response_time = timer.duration
        scores = np.array(batch.scores)

        good_outputs = [
            len(result.request_info.tool_outputs[i]) > 0
            and result.request_info.tool_calleds[i]
            and not result.request_info.timeouts[i]
            and not result.request_info.tool_errors[i]
            for i in range(len(result.request_info.tool_outputs))
        ]
        scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
        mean_grouped_rewards = scores_per_prompt.mean(axis=-1)
        mean_grouped_rewards = np.repeat(mean_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0)
        std_grouped_rewards = scores_per_prompt.std(axis=-1)
        std_grouped_rewards = np.repeat(std_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0)
        if args.advantage_normalization_type == "standard":
            advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
        elif args.advantage_normalization_type == "centered":
            advantages = scores - mean_grouped_rewards
        else:
            raise ValueError(f"Invalid advantage normalization type: {args.advantage_normalization_type}")

        if args.mask_truncated_completions:
            stop_idxes = torch.tensor(
                [i for i in range(len(result.finish_reasons)) if result.finish_reasons[i] == "stop"]
            )
            num_truncated = len(result.finish_reasons) - len(stop_idxes)
            if num_truncated > 0:
                logger.info(
                    f"[Truncated completions filtering] Filtered {num_truncated} responses that didn't finish with 'stop'. "
                    f"Retention rate: {len(stop_idxes) / len(result.finish_reasons):.2%}"
                )
            scores = scores[stop_idxes]
            advantages = advantages[stop_idxes]
            batch = batch[stop_idxes.tolist()]
            result.responses = [result.responses[i] for i in stop_idxes]
            result.masks = [result.masks[i] for i in stop_idxes]
            result.finish_reasons = [result.finish_reasons[i] for i in stop_idxes]
            result.logprobs = [result.logprobs[i] for i in stop_idxes]

        with Timer("📦 [Data Preparation Thread] Packing sequences"):
            packed_sequences = pack_sequences(
                queries=batch.queries,
                responses=result.responses,
                masks=result.masks,
                pack_length=args.pack_length,
                pad_token_id=tokenizer.pad_token_id,
                vllm_logprobs=result.logprobs,
                mask_tool_use=args.mask_tool_use,
                min_num_batches=args.world_size,
            )
            num_new_tokens = sum(len(seq) for seq in packed_sequences.query_responses)
            # Vectorized advantage calculation: create a lookup array where each index corresponds to a response mask value
            # and each value is the corresponding advantage score: index 0 is set to 0 since response masks start from 1 (1-indexed)
            lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
            lookup_advantages[1:] = advantages
            packed_advantages = [
                torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
                for packed_mask in packed_sequences.response_masks
            ]
            packed_sequences.advantages = packed_advantages

        collated_data = prepare_collated_data_for_workers(
            packed_sequences, args.world_size, args.per_device_train_batch_size, tokenizer.pad_token_id
        )
        B = len(packed_sequences.query_responses) // args.world_size

        # Create a result package with metrics and data
        if len(result.responses) == 0:
            # Handle empty responses case
            # in this case, we won't log metrics, so it should be fine.
            metrics = {}
            logger.warning(f"No responses in batch {training_step}.")
        else:
            real_num_responses = len(result.responses)
            expected_num_responses = args.num_samples_per_prompt_rollout * args.num_unique_prompts_rollout

            unsolved_num_responses = (scores < args.max_possible_score).sum()
            sequence_lengths = np.array([len(response) for response in result.responses])
            sequence_length_solved = (
                np.array([]) if np.all(scores == 0) else np.array(sequence_lengths[scores == args.max_possible_score])
            )
            sequence_length_unsolved = (
                np.array([]) if np.all(scores == args.max_possible_score) else np.array(sequence_lengths[scores == 0])
            )
            stop_rate = sum(int(finish_reason == "stop") for finish_reason in result.finish_reasons) / len(
                result.finish_reasons
            )

            batch_metrics = asdict(batch_stats)
            batch_metrics_prefixed = {f"batch/{k}": v for k, v in batch_metrics.items()}

            metrics = {
                "scores": scores.mean(),
                "real_batch_size_ratio": real_num_responses / expected_num_responses,
                "unsolved_batch_size_ratio": unsolved_num_responses / real_num_responses,
                "packed_ratio": len(packed_sequences.query_responses) / real_num_responses,
                "val/solve_rate_hist": None,
                "val/total_reward_groups": real_num_responses / args.num_samples_per_prompt_rollout,
                "val/sequence_lengths": sequence_lengths.mean(),
                "val/sequence_lengths_min": sequence_lengths.min(),
                "val/sequence_lengths_max": sequence_lengths.max(),
                "val/sequence_lengths_unsolved": (
                    0 if len(sequence_length_unsolved) == 0 else sequence_length_unsolved.mean()
                ),
                "val/sequence_lengths_solved": (
                    0 if len(sequence_length_solved) == 0 else sequence_length_solved.mean()
                ),
                "val/sequence_lengths_unsolved_hist": sequence_length_unsolved,
                "val/sequence_lengths_solved_hist": sequence_length_solved,
                "val/stop_rate": stop_rate,
                "val/advantages_mean": advantages.mean(),
                "val/advantages_min": advantages.min(),
                "val/advantages_max": advantages.max(),
                "val/advantages_hist": advantages,
                "val/num_calls_rate": np.array(result.request_info.num_calls).mean(),
                "val/timeouts_rate": np.array(result.request_info.timeouts).mean(),
                "val/tool_errors_rate": np.array([len(item) > 0 for item in result.request_info.tool_errors]).mean(),
                "val/good_outputs_rate": np.array(good_outputs).mean(),
                "val/tool_runtimes_rate": np.array(result.request_info.tool_runtimes).mean(),
                "val/tool_calleds_rate": np.array(result.request_info.tool_calleds).mean(),
                "time/getting_response": getting_response_time,
                **reward_metrics,
                **batch_metrics_prefixed,
            }

            total_tokens = result.token_statistics.num_prompt_tokens + result.token_statistics.num_response_tokens
            metrics["val/actor_tokens_per_second"] = total_tokens / result.token_statistics.generation_time

        if args.save_traces:
            traces = {
                "scores": scores.tolist(),
                "finish_reasons": result.finish_reasons,
                "responses": result.responses,
                "training_step": training_step,
                **asdict(batch),  # Unpack all batch fields
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
                "responses_count": len(result.responses),
                "num_new_tokens": num_new_tokens,
                "B": B,
                "prompt_lengths": batch_stats.prompt_lengths,
                "response_lengths": batch_stats.response_lengths,
                "num_filtered_prompts": batch_stats.filtered_prompts,
            }
        )


def setup_runtime_variables(args: Args) -> Args:
    """Set up runtime variables for the experiment."""
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
    args.world_size = sum(args.num_learners_per_node)
    args.num_training_steps = args.total_episodes // (
        args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    )
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
    if args.with_tracking and args.wandb_entity is None:
        args.wandb_entity = maybe_use_ai2_wandb_entity()
    args.tool_use = args.tools is not None and len(args.tools) > 0
    return args


def setup_experiment_tracking(args: Args, tc: TokenizerConfig, model_config: ModelConfig):
    """Setup experiment tracking and seeds."""
    all_configs = {}
    beaker_config = None
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(tc), **asdict(model_config))

    wandb_url = None
    if args.with_tracking:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=all_configs,
            name=args.run_name,
            save_code=True,
            tags=[args.exp_name] + get_wandb_tags(),
        )
        wandb_url = wandb.run.get_url()
        maybe_update_beaker_description(wandb_url=wandb_url)

    return beaker_config, wandb_url


def setup_datasets(args: Args, tc: TokenizerConfig, tokenizer: PreTrainedTokenizer):
    """Set up training and evaluation datasets."""
    system_prompt_override = None
    if args.system_prompt_override_file is not None:
        logger.info(f"Loading system prompt override from {args.system_prompt_override_file}")
        with open(args.system_prompt_override_file) as f:
            system_prompt_override = f.read().strip()
        logger.info(f"System prompt overriden to:\n#####\n{system_prompt_override}\n#####\n")

    transform_fn_args = [
        {"system_prompt_override": system_prompt_override},
        {"max_prompt_token_length": args.max_prompt_token_length},
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
        system_prompt_override=system_prompt_override,
    )
    train_dataset = train_dataset.shuffle(seed=args.seed)

    if len(args.dataset_mixer_eval_list) > 0:
        eval_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_eval_list,
            dataset_mixer_list_splits=args.dataset_mixer_eval_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            hf_entity=args.hf_entity,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_eval_hash,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
            system_prompt_override=system_prompt_override,
        )
        if args.shuffle_eval_dataset:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)
    else:
        eval_dataset = None

    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)

    return train_dataset, eval_dataset


def load_tools(args: Args) -> dict[str, tools.Tool]:
    """Load tool instances based on args.tools configuration.

    Args:
        args: Parsed training arguments with tool configuration.

    Returns:
        A mapping from tool end strings to tool instances.

    Raises:
        ValueError: If an unknown tool is requested.
    """
    tool_objects: dict[str, tools.Tool] = {}
    if not args.tools:
        return tool_objects

    for tool in args.tools:
        if tool.lower() == "search":
            from open_instruct.search_utils.search_tool import SearchTool

            tool_instance = SearchTool(
                start_str="<query>",
                end_str="</query>",
                api_endpoint=args.search_api_endpoint,
                number_documents_to_search=args.number_documents_to_search,
            )
        elif tool.lower() == "code":
            tool_instance = tools.PythonCodeTool(
                start_str="<code>", end_str="</code>", api_endpoint=args.code_tool_api_endpoint
            )
        else:
            raise ValueError(f"Unknown tool: {tool}")

        tool_objects[tool_instance.end_str] = tool_instance

    return tool_objects


def create_model_and_optimizer(
    args: Args,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    beaker_config: BeakerRuntimeConfig,
    wandb_url: str,
    tokenizer: PreTrainedTokenizer,
    inference_results_Q: ray_queue.Queue,
    prompt_Q: ray_queue.Queue,
    evaluation_inference_results_Q: ray_queue.Queue,
    reward_config: RewardConfig,
    train_dataset,
    eval_dataset,
) -> tuple[ModelGroup, list[vllm_utils.LLMRayActor], dict[str, tools.Tool], int, int]:
    """Create the model, optimizer, and vLLM engines."""
    # Create placement group
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray_get_with_progress([pg.ready()], desc="Waiting for placement group")
    policy_group = ModelGroup(pg, PolicyTrainerRayProcess, args.num_learners_per_node, args.single_gpu_mode)
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits = [
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
        for model in policy_group.models
    ]

    tool_objects = load_tools(args)
    args.stop_strings.extend(tool_objects.keys())

    queues_to_monitor = {
        "Inference Results Queue": inference_results_Q,
        "Prompt Queue": prompt_Q,
        "Evaluation Queue": evaluation_inference_results_Q,
    }
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, args)

    # Create vLLM engines with queues
    vllm_engines = vllm_utils.create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.vllm_enforce_eager,
        tc.tokenizer_name_or_path,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        args.vllm_enable_prefix_caching,
        args.max_prompt_token_length + args.response_length,  # max_model_len, total length to generate
        args.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
        tools=tool_objects,
        max_tool_calls=args.max_tool_calls,
        mask_tool_use=args.mask_tool_use,
        prompt_queue=prompt_Q,
        results_queue=inference_results_Q,
        eval_results_queue=evaluation_inference_results_Q,
        actor_manager=actor_manager,
        inflight_updates=args.inflight_updates,
        reward_config=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    results, _ = ray_get_with_progress(inits, desc="Initializing models")
    resume_training_step = results[0] + 1
    episode = (resume_training_step - 1) * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    logger.info("======== ✅ all models and vLLM engines initialized =========")

    kv_cache_max_concurrency = ray.get(vllm_engines[0].get_kv_cache_info.remote())
    ray.get(actor_manager.set_kv_cache_max_concurrency.remote(kv_cache_max_concurrency))
    expected_batch_size = (
        args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout // args.vllm_num_engines
    )
    if kv_cache_max_concurrency < expected_batch_size:
        nodes_needed = (
            args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout // kv_cache_max_concurrency
        )
        logger.warning(
            f"kv_cache_max_concurrency ({kv_cache_max_concurrency}) is lower than "
            f"num_unique_prompts_rollout * num_samples_per_prompt_rollout // vllm_num_engines ({expected_batch_size}). "
            f"This means actors will have to run multiple sequential batches, hurting performance. "
            f"You might want to use more inference nodes ({nodes_needed} nodes to generate the entire batch simultaneously)."
        )

    ray_get_with_progress(
        [m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models],
        desc="Setting up model update group",
    )
    logger.info("======== ✅ model update group setup successfully =========")

    return policy_group, vllm_engines, tool_objects, resume_training_step, episode, actor_manager


def create_generation_configs(args: Args):
    """Create generation configs for training and evaluation."""
    generation_config = vllm_utils.SamplingConfig(
        temperature=args.temperature,
        top_p=args.vllm_top_p,
        max_tokens=args.response_length,
        n=args.num_samples_per_prompt_rollout,
        stop=args.stop_strings,
        seed=args.seed,
        logprobs=1,
    )
    eval_generation_config = dataclasses.replace(generation_config, n=1)
    return {"train": generation_config, "eval": eval_generation_config}


def add_prompt_to_generator(
    example: dict[str, Any], prompt_Q: ray_queue.Queue, generation_config, is_eval: bool
) -> None:
    """Add a prompt to the generation queue.

    Args:
        example: A dict containing:
            - INPUT_IDS_PROMPT_KEY: The tokenized prompt
            - dataset_index: Index into the original dataset
            - prompt_id: Unique identifier for this prompt (epoch_datasetIndex)
        param_prompt_Q: Queue to put the prompt request
        generation_config: Generation configuration
        is_eval: Whether this is an evaluation prompt
    """
    prompt_Q.put(
        PromptRequest(
            prompt=example[INPUT_IDS_PROMPT_KEY],
            generation_config=generation_config,
            dataset_index=example["dataset_index"],
            prompt_id=example["prompt_id"],
            is_eval=is_eval,
        )
    )


def load_data_from_packing_thread(
    packed_sequences_Q: Queue, num_total_tokens: int, stop_event: threading.Event, health_check_fn: Callable[[], None]
) -> tuple[list[dict[str, list[torch.Tensor]]] | None, dict[str, Any], int, int, list[int] | None, list[int] | None]:
    """Get the packed sequences with advantages from the packing thread."""
    with Timer("[Main Thread] 📦 Getting packed sequences from thread") as timer:
        while True:
            if stop_event.is_set():
                logger.warning("[Main Thread] Stop event detected while waiting for packed sequences")
                return None, {}, num_total_tokens, 0, None, None, 0
            try:
                # When running at 32k generation length, it typically takes 900s to generate data,
                # so you might see this fire a bunch of times. That's normal!
                packed_data = packed_sequences_Q.get(timeout=300)
                break
            except Empty:
                health_check_fn()
                logger.warning("[Main Thread] Timeout waiting for packed sequences. Retrying...")
        data_thread_metrics = packed_data["metrics"]
        B = packed_data["B"]
        collated_data = packed_data["collated_data"]
        num_step_tokens = packed_data["num_new_tokens"]
        num_total_tokens += num_step_tokens
        prompt_lengths = packed_data["prompt_lengths"]
        response_lengths = packed_data["response_lengths"]
        num_filtered_prompts = packed_data["num_filtered_prompts"]

    data_thread_metrics["time/trainer_idling"] = timer.duration
    if B == 0:
        logger.warning("[Main Thread] 🤡 After packing, there is not enough data to train")
        return None, data_thread_metrics, num_total_tokens, 0, None, None, 0
    return (
        collated_data,
        data_thread_metrics,
        num_total_tokens,
        num_step_tokens,
        prompt_lengths,
        response_lengths,
        num_filtered_prompts,
    )


def weight_sync_thread(
    args: Args,
    stop_event: threading.Event,
    weight_sync_trigger_event: threading.Event,
    policy_group: ModelGroup,
    actor_manager: ActorManager,
    weight_sync_metrics_Q: Queue,
    resume_training_step: int = 1,
):
    """Thread function that handles weight sync operations and actor manager coordination."""
    logger.info("[Weight Sync Thread] 🚀 Starting weight sync thread")
    if resume_training_step > 1:
        weight_sync_trigger_event.set()

    while not stop_event.is_set():
        # Wait for weight sync trigger from main thread
        if not weight_sync_trigger_event.wait(timeout=1.0):
            continue

        # Clear the event for next iteration
        weight_sync_trigger_event.clear()

        with Timer("[Weight Sync]") as timer:
            logger.debug("[Weight Sync Thread] Starting weight sync")

            # Set actors to stop
            ray.get(actor_manager.set_should_stop.remote(True))
            logger.debug("[Weight Sync Thread] Set should_stop to True for weight sync")

            # Broadcast weights to vLLM engines
            # First get the futures
            weight_broadcast_futures: list[ray.ObjectRef] = [m.broadcast_to_vllm.remote() for m in policy_group.models]

            # Wait for all weight updates to complete and collect individual timings
            _, actor_sync_times = ray_get_with_progress(
                weight_broadcast_futures,
                desc="[Weight Sync Thread] Waiting for weight updates to complete",
                enable=args.verbose,
            )

            # Allow actors to resume
            ray.get(actor_manager.set_should_stop.remote(False))
            logger.debug("[Weight Sync Thread] Set should_stop to False after weight sync")

        # Calculate distribution statistics
        sync_time_stats = {
            "time/weight_sync": timer.duration,
            "time/weight_sync_mean": np.mean(actor_sync_times),
            "time/weight_sync_min": np.min(actor_sync_times),
            "time/weight_sync_max": np.max(actor_sync_times),
            "time/weight_sync_median": np.median(actor_sync_times),
        }

        try:
            weight_sync_metrics_Q.put_nowait(sync_time_stats)
        except Full:
            logger.warning("[Weight Sync Thread] weight sync metrics queue full, skipping metric")

    logger.info("[Weight Sync Thread] 🛑 Stopping weight sync thread")


def one_training_step(
    args: Args,
    policy_group: ModelGroup,
    collated_data: list[dict[str, list[torch.Tensor]]],
    tokenizer: PreTrainedTokenizer,
    data_thread_metrics: dict[str, Any],
    episode: int,
    training_step: int,
    num_total_tokens: int,
    num_step_tokens: int,
    start_time: float,
    train_dataset: datasets.Dataset,
    training_start_time: float,
    wandb_url: str,
    chat_template_name: str,
    model_dims: utils.ModelDims,
    prompt_lengths: list[int],
    response_lengths: list[int],
    actor_manager: ActorManager | None = None,
) -> None:
    """Train the model for one step."""
    update_ref_policy_future = []
    with Timer("[Main Thread] 🗡️ Training") as train_timer:
        metrics_list, _ = ray_get_with_progress(
            [
                policy_group.models[i].train.remote(data_BT=collated_data[i], pad_token_id=tokenizer.pad_token_id)
                for i in range(args.world_size)
            ],
            desc=f"Running training step {training_step}",
        )
        if (
            args.load_ref_policy
            and args.ref_policy_update_freq is not None
            and training_step % args.ref_policy_update_freq == 0
            and args.alpha > 0
        ):
            update_ref_policy_future.extend(
                [policy_group.models[i].update_ref_policy.remote() for i in range(args.world_size)]
            )
            ray_get_with_progress(update_ref_policy_future, desc=f"Updating reference policy at step {training_step}")

    save_time = maybe_save_checkpoint(args, training_step, policy_group, chat_template_name, tokenizer, wandb_url)

    ray.get(actor_manager.report_training_step_time.remote(train_timer.duration))

    average_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
    step_time = time.perf_counter() - start_time
    total_training_time = time.perf_counter() - training_start_time

    total_generation_time = data_thread_metrics["time/getting_response"]

    utilization_metrics = calculate_utilization_metrics(
        model_dims=model_dims,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        total_generation_time=total_generation_time,
        samples_per_prompt=args.num_samples_per_prompt_rollout,
        num_engines=args.vllm_num_engines,
        num_gpus_per_engine=args.vllm_tensor_parallel_size,
        training_time=train_timer.duration,
        num_training_gpus=args.world_size,
    )

    metrics = {
        "episode": episode,
        "global_step": episode,
        "training_step": training_step,
        "val/num_total_tokens": num_total_tokens,
        "val/num_step_tokens": num_step_tokens,
        "epoch": episode / args.num_samples_per_prompt_rollout / len(train_dataset),
        "learner_tokens_per_second_overall": num_total_tokens / total_training_time,
        "learner_tokens_per_second_step": num_step_tokens / step_time,
        "time/total": step_time,
        "time/training": train_timer.duration,
        "time/saving": save_time,
        **data_thread_metrics,
        **average_metrics,
        **utilization_metrics,
    }
    # Print only scalar metrics
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, float | int)}
    print_rich_single_line_metrics(scalar_metrics)

    if args.with_tracking:
        # Convert array/list metrics to wandb histograms for logging
        for key, value in metrics.items():
            if (isinstance(value, np.ndarray | list)) and len(value) > 0:
                metrics[key] = wandb.Histogram(value)
        wandb.log(metrics, step=episode)


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def maybe_save_checkpoint(
    args: Args,
    training_step: int,
    policy_group: ModelGroup,
    chat_template_name: str,
    tokenizer: PreTrainedTokenizer,
    wandb_url: str,
) -> float:
    save_time = 0
    if args.save_freq > 0 and training_step % args.save_freq == 0 and (args.eval_on_step_0 or training_step > 1):
        with Timer("[Main Thread] 🗡️ Saving model") as timer:
            checkpoint_dir = f"{args.output_dir}_checkpoints"
            step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
            logger.info(f"Saving model at step {training_step} to {step_dir}")
            ray_get_with_progress(
                [
                    policy_group.models[i].save_model.remote(step_dir, chat_template_name, tokenizer)
                    for i in range(args.world_size)
                ],
                desc=f"Saving model at step {training_step}",
            )
            if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
                for i in range(args.world_size):
                    policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                        step_dir, leaderboard_name, wandb_url, training_step
                    )
        save_time = timer.duration

    return save_time


def maybe_evaluate(
    args: Args,
    training_step: int,
    evaluation_inference_results_Q: ray_queue.Queue,
    tokenizer,
    episode,
    eval_dataset: Dataset,
    eval_generation_config,
    model_dims: utils.ModelDims,
    actor_manager=None,
):
    """Optionally evaluate the model."""
    if eval_dataset is None:
        return

    try:
        # timeout 0.01 if this is not the last training step
        # otherwise, wait to get the last evaluation generations (long timeout just in case)
        timeout = 0.01 if training_step < args.num_training_steps else 100

        # Accumulate evaluation results from all vLLM engines
        eval_result, eval_batch, eval_reward_metrics, _ = accumulate_inference_batches(
            evaluation_inference_results_Q,
            args,
            eval_generation_config,
            num_prompts=len(eval_dataset),
            model_dims=model_dims,
            tokenizer=tokenizer,
            prompt_dataset=eval_dataset,
            actor_manager=actor_manager,
            timeout=timeout,
            active_sampling=False,
            filter_zero_std_samples=False,
            replenish_prompts=False,
        )

        logger.info("[Main Thread] 📊 Evaluation responses received")

        eval_sequence_lengths = np.array([len(response) for response in eval_result.responses])
        eval_stop_rate = sum(int(finish_reason == "stop") for finish_reason in eval_result.finish_reasons) / len(
            eval_result.finish_reasons
        )
        eval_reward_metrics = {f"eval/{key}": val for key, val in eval_reward_metrics.items()}
        eval_metrics = {
            "eval/scores": np.array(eval_batch.scores).mean(),
            "eval/sequence_lengths": eval_sequence_lengths.mean(),
            "eval/sequence_lengths_min": eval_sequence_lengths.min(),
            "eval/sequence_lengths_max": eval_sequence_lengths.max(),
            "eval/stop_rate": eval_stop_rate,
            **eval_reward_metrics,
        }

        total_tokens = (
            eval_result.token_statistics.num_prompt_tokens + eval_result.token_statistics.num_response_tokens
        )
        eval_metrics["eval/actor_tokens_per_second"] = total_tokens / eval_result.token_statistics.generation_time

        print_rich_single_line_metrics(eval_metrics)

        table = {}
        table["prompt"] = tokenizer.batch_decode(eval_batch.queries if eval_batch else [])
        table["response"] = eval_batch.decoded_responses
        table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
        table["scores"] = eval_batch.scores
        table["ground_truth"] = eval_batch.ground_truths if eval_batch else []
        df = pd.DataFrame(table)

        if args.with_tracking:
            eval_metrics["sample_completions"] = wandb.Table(dataframe=df)
            wandb.log(eval_metrics, step=episode)
        else:
            print_rich_table(df.iloc[:1])
        del table
    except Empty:
        logger.warning("[Main Thread] 🙈 Evaluation responses not received")


def save_final_model(
    args: Args,
    policy_group: ModelGroup,
    tokenizer: PreTrainedTokenizer,
    training_step: int,
    wandb_url: str,
    chat_template_name: str,
):
    """Save the final model and launch evaluation jobs if configured."""
    logger.info(f"Saving final model at step {training_step} to {args.output_dir}")
    with Timer("[Main Thread] 🗡️ Saving model"):
        ray_get_with_progress(
            [
                policy_group.models[i].save_model.remote(args.output_dir, chat_template_name, tokenizer)
                for i in range(args.world_size)
            ],
            desc="Saving final model",
        )
        if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
            leaderboard_name = args.hf_repo_revision
            for i in range(args.world_size):
                policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                    args.output_dir, leaderboard_name, wandb_url, training_step
                )


def make_tokenizer(tc: TokenizerConfig, model_config: ModelConfig):
    """Setup tokenizer with appropriate configuration."""
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
        logger.warning(warning)
    return tc.tokenizer


def cleanup_judge_clients():
    """Cleans up all LLM judge clients."""
    asyncio.run(cleanup_all_llm_judge_clients())
    logger.info("✅ LLM judge clients cleaned up")


def cleanup_training_resources(
    stop_event: threading.Event,
    executor: futures.ThreadPoolExecutor,
    queues: list[ray_queue.Queue],
    actor_manager: ActorManager,
) -> None:
    """Clean up all training resources including threads and Ray queues."""
    stop_event.set()

    logger.info("Signaling all actors to stop...")
    ray.get(actor_manager.set_should_stop.remote(True))
    logger.info("✅ Signaled all actors to stop")

    # Clean up ActorManager resources
    logger.info("Cleaning up ActorManager resources...")
    ray.get(actor_manager.cleanup.remote())
    logger.info("✅ ActorManager resources cleaned up")

    logger.info("Pushing shutdown sentinel to queues...")
    # Push sentinel to the first queue (inference_results_Q)
    if queues and len(queues) > 0:
        queues[0].put(ShutdownSentinel(), timeout=1)

    logger.info("Shutting down Ray queues...")
    if queues and len(queues) > 0:
        [queue.shutdown() for queue in queues]
    logger.info("Shutting down thread pool executor...")
    executor.shutdown(wait=True)

    # Clean up judge clients
    cleanup_judge_clients()

    # Shutdown Ray only from the main process (rank 0) or when DDP isn't initialized
    try:
        is_ddp = dist.is_available() and dist.is_initialized()
        is_rank0 = (not is_ddp) or (dist.get_rank() == 0)
        if is_rank0 and ray.is_initialized():
            logger.info("Shutting down Ray...")
            ray.shutdown()
            logger.info("✅ Ray shut down")
    except Exception as e:
        logger.warning(f"Ray shutdown failed: {e}")

    # Clean up distributed process group if it was initialized
    if dist.is_initialized():
        logger.info("Destroying process group...")
        dist.destroy_process_group()
        logger.info("✅ Process group destroyed")


def run_training(
    args,
    tokenizer,
    train_dataset,
    eval_dataset,
    policy_group,
    vllm_engines,
    generation_configs,
    data_loader,
    resume_training_step,
    episode,
    wandb_url,
    tc,
    stop_event,
    executor,
    inference_results_Q,
    prompt_Q,
    evaluation_inference_results_Q,
    packed_sequences_Q,
    weight_sync_metrics_Q,
    actor_manager: ActorManager,
    model_dims: utils.ModelDims,
    checkpoint_state=None,
):
    if resume_training_step > 1:
        logger.info(f"[Main Thread] Resuming training from step {resume_training_step}")

    logger.info("======== ✅ weight sync thread starts =========")
    weight_sync_trigger_event = threading.Event()
    weight_sync_thread_future = executor.submit(
        weight_sync_thread,
        args,
        stop_event,
        weight_sync_trigger_event,
        policy_group,
        actor_manager,
        weight_sync_metrics_Q,
        resume_training_step,
    )

    """Run the main training loop with worker threads."""
    ray_get_with_progress(
        [engine.ready.remote() for engine in vllm_engines], "Checking engines are ready to work", timeout=300
    )

    logger.info("======== ✅ data preparation thread starts =========")
    packing_future = executor.submit(
        data_preparation_thread,
        inference_results_Q,
        prompt_Q,
        packed_sequences_Q,
        args,
        tokenizer,
        args.num_training_steps,
        generation_configs["train"],
        resume_training_step,
        data_loader,
        train_dataset,
        actor_manager,
        model_dims,
    )

    def health_check_fn():
        [f.result() for f in [packing_future, weight_sync_thread_future] if f.done()]
        ray_get_with_progress(
            [engine.check_background_threads.remote() for engine in vllm_engines],
            desc="Checking vLLM engine health",
            enable=False,
        )

    # Send initial data to ensure we have a N-step offset.
    for _ in range(args.async_steps * args.num_unique_prompts_rollout):
        example = next(data_loader)
        add_prompt_to_generator(example, prompt_Q, generation_configs["train"], is_eval=False)
    if checkpoint_state and "num_total_tokens" in checkpoint_state:
        num_total_tokens = checkpoint_state["num_total_tokens"]
        logger.info(f"Restored num_total_tokens: {num_total_tokens}")
    else:
        num_total_tokens = 0

    if eval_dataset is not None:
        eval_data_loader = data_loader_lib.HFDataLoader(
            dataset=eval_dataset,
            batch_size=1,
            seed=args.seed,
            rank=0,
            world_size=1,
            work_dir=args.output_dir,
            automatic_reshuffle=False,
        )
    else:
        eval_data_loader = None
    training_start_time = time.perf_counter()  # Track overall training start time
    maybe_update_beaker_description(
        current_step=resume_training_step - 1,
        total_steps=args.num_training_steps,
        start_time=training_start_time,
        wandb_url=wandb_url,
    )
    for training_step in range(resume_training_step, args.num_training_steps + 1):
        start_time = time.perf_counter()

        # Check if any of the threads have raised an exception.
        health_check_start = time.perf_counter()
        health_check_fn()
        health_check_time = time.perf_counter() - health_check_start

        (
            collated_data,
            data_thread_metrics,
            num_total_tokens,
            num_step_tokens,
            prompt_lengths,
            response_lengths,
            num_filtered_prompts,
        ) = load_data_from_packing_thread(packed_sequences_Q, num_total_tokens, stop_event, health_check_fn)

        if (
            training_step % args.local_eval_every == 0
            and eval_data_loader is not None
            and (args.eval_on_step_0 or training_step > 1)
        ):
            for eval_example in iter(eval_data_loader):
                add_prompt_to_generator(eval_example, prompt_Q, generation_configs["eval"], is_eval=True)
        if collated_data is None:
            continue

        episode += args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout

        try:
            data_thread_metrics |= weight_sync_metrics_Q.get_nowait()
        except Empty:
            logger.info("[Main Thread] didn't get train generation metrics")

        data_thread_metrics["time/health_check"] = health_check_time

        one_training_step(
            args,
            policy_group,
            collated_data,
            tokenizer,
            data_thread_metrics,
            episode,
            training_step,
            num_total_tokens,
            num_step_tokens,
            start_time,
            train_dataset,
            training_start_time,
            wandb_url,
            tc.chat_template_name,
            model_dims,
            prompt_lengths,
            response_lengths,
            actor_manager,
        )

        # Checkpoint after one_training_step (or even if it was skipped)
        # This ensures we checkpoint progress even if the exact checkpoint step has no data
        if (
            args.checkpoint_state_freq > 0
            and training_step % args.checkpoint_state_freq == 0
            and args.checkpoint_state_dir is not None
        ):
            utils.warn_if_low_disk_space(args.checkpoint_state_dir, send_slack_alerts=args.send_slack_alerts)
            with Timer("[Main Thread] 🗡️ Saving checkpoint state"):
                # Save comprehensive client state including dataloader state
                client_state = {
                    "training_step": training_step,
                    "episode": episode,
                    "num_total_tokens": num_total_tokens,
                }

                # Save dataloader state
                if data_loader is not None:
                    client_state["dataloader_state"] = data_loader.state_dict()

                ray_get_with_progress(
                    [
                        policy_group.models[i].save_checkpoint_state.remote(args.checkpoint_state_dir, client_state)
                        for i in range(args.world_size)
                    ],
                    desc=f"Saving checkpoint state at step {training_step}",
                )
                logger.info(f"Saved checkpoint state at step {training_step} to {args.checkpoint_state_dir}")

        logger.debug(f"[Main Thread] Triggered weight sync for step {training_step}")
        weight_sync_trigger_event.set()

        maybe_evaluate(
            args,
            training_step,
            evaluation_inference_results_Q,
            tokenizer,
            episode,
            eval_dataset,
            generation_configs["eval"],
            model_dims,
            actor_manager,
        )

        maybe_update_beaker_description(
            current_step=training_step,
            total_steps=args.num_training_steps,
            start_time=training_start_time,
            wandb_url=wandb_url,
        )

    if resume_training_step > args.num_training_steps:
        raise ValueError(f"Training didn't run since {resume_training_step=} > {args.num_training_steps=}")

    save_final_model(args, policy_group, tokenizer, training_step, wandb_url, tc.chat_template_name)


def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig):
    tokenizer = make_tokenizer(tc, model_config)
    args = setup_runtime_variables(args)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

    beaker_config, wandb_url = setup_experiment_tracking(args, tc, model_config)

    train_dataset, eval_dataset = setup_datasets(args, tc, tokenizer)

    if len(train_dataset) < (needed := max(args.async_steps, 1) * args.num_unique_prompts_rollout):
        raise ValueError(
            f"Train dataset is too small! Is {len(train_dataset)} prompts, but {needed} are needed to have enough prompts for bsz and prefill. Try reducing async_steps or num_unique_prompts_rollout, or increasing the dataset size."
        )

    if args.cache_dataset_only:
        return

    pprint([args, model_config])

    # Initialize Ray before creating Ray objects
    ray.init(dashboard_host="0.0.0.0", runtime_env={"excludes": [".git/"], "env_vars": dict(os.environ)})

    # Create Ray queues.
    # Since we now send/receive individual prompts, queue size should accommodate
    # - all prompts from async_steps + 1 training steps
    # - all eval prompts
    num_eval_prompts = len(eval_dataset) if eval_dataset is not None else 0
    queue_size = (args.async_steps + 1) * args.num_unique_prompts_rollout + num_eval_prompts
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    prompt_Q = ray_queue.Queue(maxsize=queue_size)
    # We don't care if we ever hit the max, so we let the queue be unbounded.
    evaluation_inference_results_Q = ray_queue.Queue()

    reward_config = RewardConfig(
        apply_r1_style_format_reward=args.apply_r1_style_format_reward,
        r1_style_format_reward=args.r1_style_format_reward,
        apply_verifiable_reward=args.apply_verifiable_reward,
        verification_reward=args.verification_reward,
        non_stop_penalty=args.non_stop_penalty,
        non_stop_penalty_value=args.non_stop_penalty_value,
        only_reward_good_outputs=args.only_reward_good_outputs,
        additive_format_reward=args.additive_format_reward,
        verifier_functions=build_all_verifiers(args),
    )

    policy_group, vllm_engines, tool_objects, resume_training_step, episode, actor_manager = (
        create_model_and_optimizer(
            args,
            tc,
            model_config,
            beaker_config,
            wandb_url,
            tokenizer,
            inference_results_Q,
            prompt_Q,
            evaluation_inference_results_Q,
            reward_config,
            train_dataset,
            eval_dataset,
        )
    )

    # Get the model dimensions from one of the engines without loading weights
    model_dims = ray.get(vllm_engines[0].get_model_dims.remote())

    generation_configs = create_generation_configs(args)

    checkpoint_state = None
    if args.checkpoint_state_dir and os.path.exists(args.checkpoint_state_dir):
        # Try to load the checkpoint state from the first rank
        checkpoint_path = os.path.join(args.checkpoint_state_dir, "global_0", "state.pt")
        if os.path.exists(checkpoint_path):
            checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded checkpoint state from {checkpoint_path}")

            episode = checkpoint_state["episode"]
            logger.info(f"Restored episode count: {episode}")

    data_loader = data_loader_lib.HFDataLoader(
        dataset=train_dataset,
        batch_size=1,
        seed=args.seed,
        rank=0,
        world_size=1,
        work_dir=args.output_dir,
        automatic_reshuffle=True,
    )

    if checkpoint_state and "dataloader_state" in checkpoint_state:
        data_loader.load_state_dict(checkpoint_state["dataloader_state"])
        logger.info("Restored dataloader state from checkpoint")

    # Create additional queues (main queues already created above)
    packed_sequences_Q = Queue(maxsize=args.async_steps)
    weight_sync_metrics_Q = Queue(maxsize=args.async_steps)

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="grpo")

    try:
        episode = run_training(
            args,
            tokenizer,
            train_dataset,
            eval_dataset,
            policy_group,
            vllm_engines,
            generation_configs,
            data_loader,
            resume_training_step,
            episode,
            wandb_url,
            tc,
            stop_event,
            executor,
            inference_results_Q,
            prompt_Q,
            evaluation_inference_results_Q,
            packed_sequences_Q,
            weight_sync_metrics_Q,
            actor_manager,
            model_dims,
            checkpoint_state,
        )

        if args.push_to_hub and (not dist.is_initialized() or dist.get_rank() == 0):
            push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    except Exception as e:
        if args.send_slack_alerts:
            utils.send_slack_message(f"<!here> A RL job has died. Error message: {e}.")
        raise
    finally:
        cleanup_training_resources(
            stop_event, executor, [inference_results_Q, prompt_Q, evaluation_inference_results_Q], actor_manager
        )

    # Ai2 logic: we use /output to store the artifacts of the job, so we
    # make a copy of the model to `/output` in the end.
    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
        and os.path.isdir(args.output_dir)
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
    logger.info("finished training")

    # Check for runtime leaks before exiting
    logger.info("Checking for runtime leaks...")

    utils.check_runtime_leaks()


if __name__ == "__main__":
    utils.check_oe_eval_internal()

    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, Args)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)

    main(args, tokenizer_config, model_config)
