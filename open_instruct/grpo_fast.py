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
from concurrent import futures

# We need to set NCCL_CUMEM_ENABLE=0 for performance reasons; see:
# https://github.com/vllm-project/vllm/issues/5723#issuecomment-2554389656
os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
try:
    import deepspeed

    # @vwxyzjn: when importing on CPU-only machines, we get the following error:
    # RuntimeError: 0 active drivers ([]). There should only be one.
    # so we need to catch the exception and do nothing
    # https://github.com/deepspeedai/DeepSpeed/issues/7028
except Exception:
    pass

from open_instruct import utils

# isort: on
import asyncio
import json
import math
import random
import shutil
import socket
import threading
import time
from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional

import datasets
import numpy as np
import pandas as pd
import ray
import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
import vllm
import wandb
from huggingface_hub import HfApi
from peft import PeftModel, get_peft_model_state_dict
from ray.util import queue as ray_queue
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, get_scheduler
from transformers.integrations import HfDeepSpeedConfig

from open_instruct import logger_utils, vllm_utils3
from open_instruct.actor_manager import ActorManager
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
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
    Batch,
    ModelConfig,
    apply_verifiable_reward,
    disable_dropout_in_model,
    entropy_from_logits,
    get_olmo3_generation_config,
    log_softmax_and_gather,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
)
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.rl_utils2 import Timer, pack_sequences
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    RayProcess,
    _z3_params_to_fetch,
    calibrate_checkpoint_state_dir,
    clean_last_n_checkpoints_deepspeed,
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

api = HfApi()
INVALID_LOGPROB = 1.0


class ShutdownSentinel:
    """Sentinel value to signal thread shutdown via queue."""


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
    local_eval_every: int = 100
    """Run evaluation after this many training steps. This controls in-loop evals, which reuse the generation/reward verifier setup. Set to -1 to disable."""
    save_freq: int = 200
    """How many train steps to save the model"""
    allow_world_padding: bool = False
    """Whether to allow world padding. This is useful for model sweeps, but wastes compute."""
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
    stop_strings: Optional[List[str]] = None
    """List of strings that stop the generation when they are generated.
    The returned output will not contain the stop strings."""
    use_fp8_kv_cache: bool = False
    """Whether to use fp8 kv cache. This is useful for larger models or olmo."""

    # Algorithm
    async_steps: int = 1
    """Number of steps ahead to generate responses. Set to 0 to make the code synchronous. Values greater than 0 learn from a policy up to async_steps old like Cleanba (https://arxiv.org/abs/2310.00036)"""
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
    advantage_normalization_type: Literal["standard", "centered"] = "standard"
    """The type of advantage normalization to use. Standard normalization is the default: it subtracts the mean and
    divides by the standard deviation. Centered normalization is the same but subtracts the mean only (e.g., used in
    DR.GRPO https://arxiv.org/pdf/2503.20783)."""
    mask_truncated_completions: bool = False
    """Whether to mask out truncated completions. Also called overlong filtering, from DAPO (https://arxiv.org/abs/2503.14476)."""

    fill_completions: bool = False
    """Whether to refill the batchsize with after filtering."""

    record_entropy: bool = False
    """whether to record the entropy of the policy during training. Uses extra memory."""

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
    num_learners_per_node: List[int] = field(default_factory=lambda: [1])
    """number of GPU deepspeed learners per node (e.g., --num_learners_per_node 2 4 means 2 learner processes
    on the first node and 4 learner processes on the second node; each process will have 1 GPU)"""
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    inference_batch_size: Optional[int] = None
    """inference batch size per vLLM engine. If None, calculated as ceil(num_unique_prompts_rollout / vllm_num_engines) * num_samples_per_prompt_rollout"""
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
    enable_queue_dashboard: bool = True
    """whether to enable the ActorManager queue monitoring dashboard"""
    queue_dashboard_port: Optional[int] = None
    """optional port for the dashboard server (if None, finds a free port automatically)"""

    # Experiment tracking
    verbose: bool = False
    """If toggled, debug output will be shown"""
    update_progress_every: int = 10
    """How often to update the progress bar (in steps)."""
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
    oe_eval_beaker_image: Optional[str] = None
    """the docker image for evaluation for oe-eval"""
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    """the priority of auto-launched evaluation jobs"""

    # Evaluation behavior
    eval_on_step_0: bool = False
    """Whether to run local evaluation at training step 0. Defaults to False."""

    # Tool settings
    tools: Optional[List[str]] = None
    """If set, use the tool mapped to the string. Currently only supports `search` and `code`"""
    max_tool_calls: List[int] = field(default_factory=lambda: [5])
    """Maximum number of tool calls allowed. If a list is provided, it must have length 1 (applies to all tools) or same length as tools (per-tool limit)."""
    mask_tool_use: bool = True
    """Whether to mask the tool output. By default on."""
    only_reward_good_outputs: bool = False
    """Whether to only reward good outputs. By default off. Useful to force the model to use the tool(s)."""

    # rl-rag specific settngs
    number_documents_to_search: int = 3
    """The maximum number of documents to retrieve for each query."""
    search_api_endpoint: Optional[str] = None
    """The API endpoint for the search engine."""

    # code-tool specific settings
    code_tool_api_endpoint: Optional[str] = None

    def __post_init__(self):
        if os.environ.get("VLLM_USE_V1") == "0":
            logger.warning("When using the v0 version of vLLM, caching is broken and will never be invalidated.")
            if self.vllm_enable_prefix_caching:
                raise ValueError("Prefix caching is currently not supported for v0.")
        assert self.num_samples_per_prompt_rollout > 0, "Number of samples per prompt must be greater than 0!"
        if self.num_samples_per_prompt_rollout == 1:
            logger.warning("num_samples_per_prompt_rollout is 1. This reduces GRPO to REINFORCE.")
        assert self.apply_verifiable_reward or self.apply_r1_style_format_reward or self.non_stop_penalty, (
            "At least one reward must be applied!"
        )
        # Ensure we have enough prompts for all VLLM engines
        if self.num_unique_prompts_rollout < self.vllm_num_engines:
            raise ValueError(
                f"{self.num_unique_prompts_rollout=} must be >= {self.vllm_num_engines=} to avoid empty batches."
            )
        # Initialize stop_strings if None
        if self.stop_strings is None:
            self.stop_strings = []
        if self.inference_batch_size is None:
            total_prompts = self.num_samples_per_prompt_rollout * self.num_unique_prompts_rollout
            self.inference_batch_size = max(1, math.ceil(total_prompts / self.vllm_num_engines))
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

        if self.checkpoint_state_dir is not None:
            if self.gs_checkpoint_state_dir is not None:
                download_latest_checkpoint_from_gs(self.gs_checkpoint_state_dir, self.checkpoint_state_dir)
            calibrate_checkpoint_state_dir(self.checkpoint_state_dir)
        if self.tools is not None and len(self.tools) > 0:
            for tool in self.tools:
                if tool not in ["search", "code"]:
                    raise ValueError(f"Tool {tool} is not supported. Supported tools are: search, code")
            assert len(self.tools) == len(set(self.tools)), "Duplicate tools are not allowed"


def next_batch(dataset_indices: List[int], dataset: datasets.Dataset) -> Batch:
    """Extract next batch of data based on indices."""
    data_next = dataset[dataset_indices]
    return Batch(
        queries=data_next[INPUT_IDS_PROMPT_KEY],
        ground_truths=data_next[GROUND_TRUTHS_KEY],
        datasets=data_next[VERIFIER_SOURCE_KEY],
        raw_queries=data_next[RAW_PROMPT_KEY],
        indices=dataset_indices,
    )


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

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the iterator for checkpointing."""
        return {
            "index": self.index,
            "data": self.data.copy(),  # Current shuffled order
            "rng_state": self.rng.bit_generator.state,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the iterator state from a checkpoint."""
        self.index = state["index"]
        self.data = state["data"].copy()
        self.rng.bit_generator.state = state["rng_state"]


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

        # Set seeds for this worker (different per rank to avoid correlation)
        worker_seed = args.seed + self.local_rank
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        deepspeed.init_distributed(timeout=timedelta(minutes=args.backend_timeout))

        ds_config = get_train_ds_config(offload=False, adam_offload=False, stage=args.deepspeed_stage, bf16=True)
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
                if states.get("ref_policy_saved", False):
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
        logger.info(f"DeepSpeed config: {dschf=}")

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

        # Load reference policy checkpoint if available
        if hasattr(self, "ref_policy_checkpoint_path") and self.ref_policy_checkpoint_path:
            state_dict = torch.load(self.ref_policy_checkpoint_path, map_location=self.device)
            if hasattr(self.ref_policy, "module"):
                # If wrapped by DeepSpeed
                self.ref_policy.module.load_state_dict(state_dict)
            else:
                self.ref_policy.load_state_dict(state_dict)
            logger.info(f"{self.rank=}: Loaded reference policy checkpoint from {self.ref_policy_checkpoint_path}")
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
            self.model_update_group = vllm_utils3.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
                timeout=timedelta(minutes=self.args.backend_timeout),
            )
            ray_get_with_progress(refs, desc="Initializing vLLM process groups", timeout=60)
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

        # Return futures instead of blocking - let caller handle completion
        all_refs = []
        if torch.distributed.get_rank() == 0:
            all_refs.extend(refss)
        return all_refs

    def update_ref_policy(self):
        for ref_param, param in zip(self.ref_policy.parameters(), self.model.parameters()):
            if self.args.deepspeed_stage == 3:
                with deepspeed.zero.GatheredParameters([param, ref_param], modifier_rank=0):
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
        # accumulation steps should always be at least 1
        accumulation_steps = max(math.ceil(len(collated_query_responses) / num_mini_batches - 0.5), 1)
        leftover = len(collated_query_responses) % accumulation_steps
        if leftover > 0:
            collated_query_responses = collated_query_responses[0:-leftover]
            collated_tool_masks = collated_tool_masks[0:-leftover]
            collated_attention_masks = collated_attention_masks[0:-leftover]
            collated_position_ids = collated_position_ids[0:-leftover]
            collated_advantages = collated_advantages[0:-leftover]
            collated_response_masks = collated_response_masks[0:-leftover]
            logger.warning(f"{leftover} samples are dropped due to batch size {num_mini_batches}")

        # recalculate the "real" number of mini-batches
        num_mini_batches = len(collated_query_responses) // accumulation_steps

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
                    ref_logprob, _ = self.forward(
                        self.ref_policy,
                        query_response,
                        attention_mask,
                        position_id,
                        pad_token_id,
                        args.temperature,
                        return_entropy=False,
                    )
                    if args.mask_tool_use and args.tool_use:
                        # mask logprobs for tool tokens
                        response_mask = response_mask.bool() & tool_mask.bool()
                    else:
                        response_mask = response_mask.bool()
                    ref_logprob = torch.masked_fill(ref_logprob, ~response_mask[:, 1:], INVALID_LOGPROB)
                    collated_ref_logprobs.append(ref_logprob)
                    torch.cuda.empty_cache()
        # if we have multiple minibatches, we need to calculate the old logprobs for each minibatch
        # following gtrl scripts in just doing this on the current active policy, rather than use the logprobs
        # from the generator (note that async mode means these are a bit diff!)
        old_logprobs = [None for _ in range(len(collated_query_responses))]
        if num_mini_batches > 1:
            with Timer("Old logprobs Calculation", noop=self.rank != 0):
                with torch.no_grad():
                    for i in range(len(collated_query_responses)):
                        query_response = collated_query_responses[i]
                        tool_mask = collated_tool_masks[i]
                        attention_mask = collated_attention_masks[i]
                        position_id = collated_position_ids[i]
                        response_mask = collated_response_masks[i]
                        old_logprob, _ = self.forward(
                            self.model,
                            query_response,
                            attention_mask,
                            position_id,
                            pad_token_id,
                            args.temperature,
                            return_entropy=False,
                        )
                        if args.mask_tool_use and args.tool_use:
                            response_mask = response_mask.bool() & tool_mask.bool()
                        else:
                            response_mask = response_mask.bool()
                        old_logprob = torch.masked_fill(old_logprob, ~response_mask[:, 1:], INVALID_LOGPROB)
                        old_logprobs[i] = old_logprob
                        torch.cuda.empty_cache()

        local_step = 0
        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        with Timer("[Training Processes] Loss calculation", noop=self.rank != 0):
            kl1_stats = torch.zeros(len(collated_query_responses))
            kl2_stats = torch.zeros(len(collated_query_responses))
            kl3_stats = torch.zeros(len(collated_query_responses))
            kl4_stats = torch.zeros(len(collated_query_responses))
            kl_loss_stats = torch.zeros(len(collated_query_responses))
            pg_clipfrac_stats = torch.zeros(len(collated_query_responses))
            pg_loss_stats = torch.zeros(len(collated_query_responses))
            loss_stats = torch.zeros(len(collated_query_responses))
            ratio_stats = torch.zeros(len(collated_query_responses))
            entropy_stats = torch.zeros(len(collated_query_responses))
            for epoch_idx in range(args.num_epochs):
                for i in range(len(collated_query_responses)):
                    mb_ref_logprob = collated_ref_logprobs[i]
                    mb_query_responses = collated_query_responses[i]
                    mb_tool_mask = collated_tool_masks[i]
                    mb_advantages = collated_advantages[i]
                    mb_response_masks = collated_response_masks[i]
                    mb_response_masks_bool = mb_response_masks[:, 1:].bool()
                    # if masking snippets, do it here.
                    if args.mask_tool_use and args.tool_use:
                        mb_response_masks_bool = mb_response_masks[:, 1:].bool() & mb_tool_mask[:, 1:].bool()
                    mb_attention_mask = collated_attention_masks[i]
                    mb_position_id = collated_position_ids[i]
                    mb_new_logprobs, mb_entropy = self.forward(
                        self.model,
                        mb_query_responses,
                        mb_attention_mask,
                        mb_position_id,
                        pad_token_id,
                        args.temperature,
                        return_entropy=args.record_entropy,
                    )
                    mb_new_logprobs = torch.masked_fill(mb_new_logprobs, ~mb_response_masks_bool, INVALID_LOGPROB)

                    # Cache the old logprobs
                    if num_mini_batches > 1:
                        mb_old_logprobs = old_logprobs[i]
                    else:
                        with torch.no_grad():
                            if epoch_idx == 0:
                                old_logprobs[i] = mb_new_logprobs
                            mb_old_logprobs = old_logprobs[i].detach()

                    # Calculate the policy's loss
                    logprobs_diff = mb_new_logprobs - mb_old_logprobs
                    ratio = torch.exp(logprobs_diff)
                    pg_losses = -mb_advantages[:, 1:] * ratio
                    pg_losses2 = -mb_advantages[:, 1:] * torch.clamp(
                        ratio, 1.0 - args.clip_lower, 1.0 + args.clip_higher
                    )
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    # Here we recalculate kl: we want the KL loss to backpropagate through the model
                    # We also clamp the KL loss to avoid numerical instability
                    # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
                    ref_logprobs_diff = (mb_new_logprobs - mb_ref_logprob).clamp(-40.0, 40.0)
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

                    # grpo change: directly subtract KL in loss (add)
                    loss = masked_mean(pg_loss_max + (args.beta * kl), mb_response_masks_bool, args.masked_mean_axis)
                    loss = loss / accumulation_steps
                    self.model.backward(loss)
                    if (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                    local_step += 1
                    with torch.no_grad():
                        # NOTE: in packed implementation, kl calculation are averages over response tokens
                        kl1_stats[i] = masked_mean(kl1, mb_response_masks_bool, args.masked_mean_axis).float()
                        kl2_stats[i] = masked_mean(kl2, mb_response_masks_bool, args.masked_mean_axis).float()
                        kl3_stats[i] = masked_mean(kl3, mb_response_masks_bool, args.masked_mean_axis).float()
                        kl4_stats[i] = masked_mean(kl4, mb_response_masks_bool, args.masked_mean_axis).float()
                        if args.kl_estimator == "kl1":
                            kl_loss_stats[i] = kl1_stats[i] * args.beta
                        elif args.kl_estimator == "kl2":
                            kl_loss_stats[i] = kl2_stats[i] * args.beta
                        elif args.kl_estimator == "kl3":
                            kl_loss_stats[i] = kl3_stats[i] * args.beta
                        elif args.kl_estimator == "kl4":
                            kl_loss_stats[i] = kl4_stats[i] * args.beta
                        pg_clipfrac_stats[i] = masked_mean(
                            (pg_losses2 > pg_losses).float(), mb_response_masks_bool, args.masked_mean_axis
                        )
                        pg_loss_stats[i] = masked_mean(pg_loss_max, mb_response_masks_bool, args.masked_mean_axis)
                        loss_stats[i] = loss
                        ratio_stats[i] = masked_mean(ratio, mb_response_masks_bool, args.masked_mean_axis)
                        if args.record_entropy:
                            # Calculate entropy statistics
                            entropy_stats[i] = masked_mean(
                                mb_entropy, mb_response_masks_bool, args.masked_mean_axis
                            ).float()

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
                if args.record_entropy:
                    self.local_metrics.add("policy/entropy_avg", entropy_stats.mean())
                self.local_metrics.add("lr", self.scheduler.get_last_lr()[0])
                return self.local_metrics.get_metrics_list()

    def save_checkpoint_state(self, checkpoint_state_dir: str, client_state: Dict[str, Any]) -> None:
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
        if hasattr(self, "ref_policy") and self.ref_policy is not None:
            ref_policy_dir = os.path.join(checkpoint_state_dir, "ref_policy")
            os.makedirs(ref_policy_dir, exist_ok=True)

            # For reference policy, we save just the model weights
            # We can't use save_checkpoint because it would try to save DummyOptim
            # which doesn't have state_dict
            if self.rank == 0:
                # Only rank 0 saves the model state
                if hasattr(self.ref_policy, "module"):
                    # If wrapped by DeepSpeed, get the underlying module
                    model_to_save = self.ref_policy.module
                else:
                    model_to_save = self.ref_policy

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
        model_to_save = self.model
        if "olmo" in chat_template_name:
            # New chat template has no bos token, and two eos tokens: <|im_end|> and <|endoftext|>
            model_to_save.generation_config = get_olmo3_generation_config(tokenizer)

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
                args.oe_eval_beaker_image,
            )


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
        master_addr, master_port = ray_get_with_progress(
            [master_policy.get_master_addr_port.remote()], desc="Getting master address"
        )[0]

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


class PendingQueriesMap:
    """Thread-safe map for tracking pending queries with reference counting."""

    def __init__(self):
        self._map = {}  # dataset_idx -> (query, ground_truth, dataset, count)
        self._lock = threading.Lock()

    def insert(self, dataset_idx, query, ground_truth, dataset, raw_query):
        """Insert or increment count for a dataset index."""
        with self._lock:
            if dataset_idx in self._map:
                # Already exists - just increment count
                existing_query, existing_ground_truth, existing_dataset, existing_raw_query, count = self._map[
                    dataset_idx
                ]
                self._map[dataset_idx] = (
                    existing_query,
                    existing_ground_truth,
                    existing_dataset,
                    existing_raw_query,
                    count + 1,
                )
            else:
                # New entry - count starts at 1
                self._map[dataset_idx] = (query, ground_truth, dataset, raw_query, 1)

    def insert_many(self, dataset_indices, queries, ground_truths, datasets, raw_queries):
        """Insert or increment count for multiple dataset indices at once."""
        with self._lock:
            for i, dataset_idx in enumerate(dataset_indices):
                current_raw_query = raw_queries[i]

                if dataset_idx in self._map:
                    # Already exists - just increment count
                    existing_query, existing_ground_truth, existing_dataset, existing_raw_query, count = self._map[
                        dataset_idx
                    ]
                    self._map[dataset_idx] = (
                        existing_query,
                        existing_ground_truth,
                        existing_dataset,
                        existing_raw_query,
                        count + 1,
                    )
                else:
                    # New entry - count starts at 1
                    self._map[dataset_idx] = (queries[i], ground_truths[i], datasets[i], current_raw_query, 1)

    def pop(self, dataset_idx):
        """Retrieve data and decrement count. Removes entry when count reaches 0."""
        with self._lock:
            if dataset_idx not in self._map:
                raise RuntimeError(f"Dataset index {dataset_idx} not found in pending_queries_map")

            query, ground_truth, dataset, raw_query, count = self._map[dataset_idx]

            if count > 1:
                # More results expected - just decrement
                self._map[dataset_idx] = (query, ground_truth, dataset, raw_query, count - 1)
            else:
                # Last result - remove entry
                del self._map[dataset_idx]

            return query, ground_truth, dataset, raw_query

    def __len__(self):
        """Return the number of entries in the map."""
        with self._lock:
            return len(self._map)

    def __contains__(self, dataset_idx):
        """Check if a dataset index is in the map."""
        with self._lock:
            return dataset_idx in self._map

    def __getitem__(self, dataset_idx):
        """Get the value for a dataset index."""
        with self._lock:
            return self._map[dataset_idx]

    def keys(self):
        """Return a view of the keys in the map."""
        with self._lock:
            return list(self._map.keys())


def accumulate_inference_batches(
    inference_results_Q: ray_queue.Queue,
    pending_queries_map: PendingQueriesMap,
    args: Args,
    training_step: int,
    generation_config: vllm.SamplingParams,
    num_prompts: int,
    actor_manager=None,
    timeout: Optional[float] = None,
) -> tuple[GenerationResult, Batch]:
    """Accumulate multiple inference results into a single training batch.

    Args:
        inference_results_Q: Queue containing individual GenerationResult objects (one per prompt)
        pending_queries_map: PendingQueriesMap instance for thread-safe query tracking
        args: Arguments containing vllm_num_engines and batch size info
        training_step: Current training step for error reporting
        generation_config: Generation config containing n (number of samples per prompt)
        num_prompts: Number of prompts to accumulate
        timeout: Optional timeout in seconds for queue get operations. If None, blocks indefinitely.

    Raises:
        queue.Empty: If timeout is specified and no data is available within timeout.

    Returns:
        Tuple of (combined_result, Batch with queries, ground_truths, datasets) or (ShutdownSentinel, None) if shutdown signal received
    """
    results = []
    all_queries = []
    all_ground_truths = []
    all_datasets = []
    all_raw_queries = []
    for i in tqdm(
        range(num_prompts),
        total=num_prompts,
        desc=f"Accumulating results from {num_prompts} prompts",
        bar_format="{l_bar}{bar}{r_bar}\n",
        disable=not args.verbose,
    ):
        result = inference_results_Q.get(timeout=timeout)

        if isinstance(result, ShutdownSentinel):
            return result, None

        # Validate that each individual result has the expected number of responses
        assert len(result.responses) == generation_config.n, (
            f"Mismatch: individual prompt result has {len(result.responses)} responses "
            f"but expected {generation_config.n} samples per prompt. "
            f"Dataset index: {result.dataset_index}, Training step: {training_step}"
        )

        query, ground_truth, dataset, raw_query = pending_queries_map.pop(result.dataset_index)

        results.append(result)
        all_queries.append(query)
        all_ground_truths.append(ground_truth)
        all_datasets.append(dataset)
        all_raw_queries.append(raw_query)

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

    # Initialize accumulated token statistics
    accumulated_stats = TokenStatistics(num_prompt_tokens=0, num_response_tokens=0, generation_time=0)

    for result in results:
        combined_responses.extend(result.responses)
        combined_finish_reasons.extend(result.finish_reasons)
        combined_masks.extend(result.masks)
        combined_num_calls.extend(result.request_info.num_calls)
        combined_timeouts.extend(result.request_info.timeouts)
        combined_tool_errors.extend(result.request_info.tool_errors)
        combined_tool_outputs.extend(result.request_info.tool_outputs)
        combined_tool_runtimes.extend(result.request_info.tool_runtimes)
        combined_tool_calleds.extend(result.request_info.tool_calleds)

        if result.token_statistics:
            accumulated_stats.num_prompt_tokens += result.token_statistics.num_prompt_tokens
            accumulated_stats.num_response_tokens += result.token_statistics.num_response_tokens
            accumulated_stats.generation_time = max(
                accumulated_stats.generation_time, result.token_statistics.generation_time
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
        dataset_index=None,  # Not meaningful for combined result
        token_statistics=accumulated_stats,
    )

    if actor_manager is not None:
        ray.get(actor_manager.report_token_statistics.remote(accumulated_stats))

    # Note: We don't have dataset_indices here, but they're not needed for the returned batch
    batch = Batch(
        queries=all_queries,
        ground_truths=all_ground_truths,
        datasets=all_datasets,
        raw_queries=all_raw_queries,
        indices=None,  # Not meaningful for combined results
    )
    return combined_result, batch


def data_preparation_thread(
    reward_fn: Callable,
    inference_results_Q: ray_queue.Queue,  # Ray queue
    packed_sequences_Q: Queue,
    pending_queries_map: dict,
    args: Args,
    tokenizer: PreTrainedTokenizer,
    num_training_steps: int,
    generation_config,
    resume_training_step: int,
    actor_manager=None,
):
    for training_step in range(resume_training_step, num_training_steps + 1):
        # Streaming accumulation: collect results as they arrive
        with Timer("🚀 [Data Preparation Thread] Getting response ids") as timer:
            result, batch = accumulate_inference_batches(
                inference_results_Q,
                pending_queries_map,
                args,
                training_step,
                generation_config,
                num_prompts=args.num_unique_prompts_rollout,
                actor_manager=actor_manager,
            )
            if isinstance(result, ShutdownSentinel):
                logger.info("[Data Preparation Thread] Received shutdown sentinel, exiting")
                return

        getting_response_time = timer.duration

        # ------------------------------------------------------------------------------------------------
        # Pack sequences
        if args.num_samples_per_prompt_rollout > 1:
            batch = Batch(
                queries=repeat_each(batch.queries, args.num_samples_per_prompt_rollout),
                ground_truths=repeat_each(batch.ground_truths, args.num_samples_per_prompt_rollout),
                datasets=repeat_each(batch.datasets, args.num_samples_per_prompt_rollout),
                raw_queries=repeat_each(batch.raw_queries, args.num_samples_per_prompt_rollout),
                indices=repeat_each(batch.indices, args.num_samples_per_prompt_rollout) if batch.indices else None,
            )
            good_outputs = [
                len(result.request_info.tool_outputs[i]) > 0
                and result.request_info.tool_calleds[i]
                and not result.request_info.timeouts[i]
                and not result.request_info.tool_errors[i]
                for i in range(len(result.request_info.tool_outputs))
            ]
        for i in range(len(result.finish_reasons)):
            # edge case: sometimes it outputs eos immediately, and we get an empty response
            # in that case, we need to add the eos token to the response
            # note that this also adds eos to the end of reponses that stopped for other reasons.
            if result.finish_reasons[i] == "stop" and (
                len(result.responses[i]) == 0 or result.responses[i][-1] != tokenizer.eos_token_id
            ):
                result.responses[i].append(tokenizer.eos_token_id)
                result.masks[i].append(1)  # never mask the eos token for
        with Timer("🔥 [Data Preparation Thread] Decoding responses", noop=True):
            decoded_responses = tokenizer.batch_decode(result.responses, skip_special_tokens=True)
            decoded_queries = batch.raw_queries
            stop_rate = sum(int(finish_reason == "stop") for finish_reason in result.finish_reasons) / len(
                result.finish_reasons
            )

        with Timer("💰 [Data Preparation Thread] Calculating rewards and advantages"):
            scores, reward_metrics = asyncio.run(
                reward_fn(
                    result.responses,
                    decoded_responses,
                    batch,
                    result.finish_reasons,
                    result.request_info,
                    decoded_queries,
                )
            )
            scores = np.array(scores)
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

        with Timer("📦 [Data Preparation Thread] Filtering sequences"):
            # Here we get the max possible score for each prompt, and see how many prompts are unsolved
            max_possible_score = 0
            if args.apply_verifiable_reward:
                max_possible_score += args.verification_reward
            if args.apply_r1_style_format_reward and args.additive_format_reward:
                max_possible_score += args.r1_style_format_reward
            unsolved_batch_size_ratio = ((scores != max_possible_score) > 0).sum() / len(scores)
            # In GRPO, if the std of grouped rewards is 0, then there is zero gradient for the batch
            # of args.num_samples_per_prompt_rollout responses, so we need to filter out those batches
            non_zero_std_mask = scores_per_prompt.std(axis=-1) != 0
            real_batch_size_ratio = non_zero_std_mask.sum() * args.num_samples_per_prompt_rollout / len(scores)
            expanded_mask = np.repeat(non_zero_std_mask, args.num_samples_per_prompt_rollout)
            non_zero_gradient_index = np.where(expanded_mask)[0]

            # Log zero-gradient filtering statistics
            num_zero_std_prompts = (~non_zero_std_mask).sum()
            num_filtered_responses = len(scores) - len(non_zero_gradient_index)
            if num_filtered_responses > 0:
                logger.info(
                    f"[Zero-gradient filtering] Filtered {num_zero_std_prompts} prompts with zero std "
                    f"({num_filtered_responses} responses). Retention rate: {len(non_zero_gradient_index) / len(scores):.2%}"
                )

            advantages = advantages[non_zero_gradient_index]
            original_batch_size = len(scores)
            scores = scores[non_zero_gradient_index]
            responses = [result.responses[i] for i in non_zero_gradient_index]
            masks = [result.masks[i] for i in non_zero_gradient_index]
            batch = batch[non_zero_gradient_index.tolist()]
            finish_reasons = [result.finish_reasons[i] for i in non_zero_gradient_index]
            if args.mask_truncated_completions:
                stop_idxes = torch.tensor([i for i in range(len(finish_reasons)) if finish_reasons[i] == "stop"])
                num_truncated = len(finish_reasons) - len(stop_idxes)
                if num_truncated > 0:
                    logger.info(
                        f"[Truncated completions filtering] Filtered {num_truncated} responses that didn't finish with 'stop'. "
                        f"Retention rate: {len(stop_idxes) / len(finish_reasons):.2%}"
                    )
                scores = scores[stop_idxes]
                advantages = advantages[stop_idxes]
                responses = [responses[i] for i in stop_idxes]
                masks = [masks[i] for i in stop_idxes]
                batch = batch[stop_idxes.tolist()]
                finish_reasons = [finish_reasons[i] for i in stop_idxes]

            if args.fill_completions:
                with Timer("⏱ [Data Preparation Thread] Refill completions"):
                    current_batch_size = len(scores)
                    original_prompt_cnt = original_batch_size // args.num_samples_per_prompt_rollout
                    current_prompt_cnt = current_batch_size // args.num_samples_per_prompt_rollout
                    need_to_fill_prompt = original_prompt_cnt - current_prompt_cnt
                    k = args.num_samples_per_prompt_rollout

                    if need_to_fill_prompt > 0 and current_prompt_cnt > 0:
                        scores_matrix = scores.reshape(current_prompt_cnt, k)
                        stds = scores_matrix.std(axis=1) + 1e-8
                        probs = stds / stds.sum()

                        logger.info(
                            f"[Refill completions] Need to fill {need_to_fill_prompt} prompts to maintain batch size. "
                            f"Original: {original_prompt_cnt}, Current: {current_prompt_cnt}"
                        )

                        sampled_prompt_ids = np.random.choice(
                            current_prompt_cnt, size=need_to_fill_prompt, replace=True, p=probs
                        )

                        sampled_indices = []
                        for pid in sampled_prompt_ids:
                            start = pid * k
                            sampled_indices.extend(range(start, start + k))

                        advantages = np.concatenate([advantages, advantages[sampled_indices]])
                        scores = np.concatenate([scores, scores[sampled_indices]])
                        responses += [responses[i] for i in sampled_indices]
                        masks += [masks[i] for i in sampled_indices]

                        sampled_batch = batch[sampled_indices]

                        batch = Batch(
                            queries=batch.queries + sampled_batch.queries,
                            ground_truths=batch.ground_truths + sampled_batch.ground_truths,
                            datasets=batch.datasets + sampled_batch.datasets,
                            indices=batch.indices + sampled_batch.indices if batch.indices is not None else None,
                        )

                        finish_reasons += [finish_reasons[i] for i in sampled_indices]

                        logger.info(
                            f"📊 Duplicated {need_to_fill_prompt} prompts from {len(sampled_indices)} total responses"
                        )

            # Count groups with all zero rewards
            all_zero_groups = (scores_per_prompt == 0).all(axis=-1).sum()
            total_groups = len(scores_per_prompt)
            logger.info(
                f"[Reward Summary] Groups with all zero rewards: {all_zero_groups}/{total_groups} "
                f"({all_zero_groups / total_groups:.1%})"
            )

        with Timer("📦 [Data Preparation Thread] Packing sequences"):
            packed_sequences = pack_sequences(
                queries=batch.queries,
                responses=responses,
                masks=masks,
                pack_length=args.pack_length,
                pad_token_id=tokenizer.pad_token_id,
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

        # if we have less batches than world size, we need to pad out so each world is fine
        # ideally, you should avoid this since its wasting computation.
        if args.allow_world_padding:
            with Timer("🤺 [Data Preparation Thread] Padding sequences for world size"):
                shortfall = args.world_size - len(packed_sequences.query_responses)
                if shortfall > 0:
                    logger.warning(
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
                    collated_advantages.append(
                        collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0)
                    )
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
        if len(responses) == 0:
            # Handle empty responses case
            # in this case, we won't log metrics, so it should be fine.
            metrics = {}
        else:
            sequence_lengths = np.array([len(response) for response in responses])
            sequence_length_solved = (
                np.array([]) if np.all(scores == 0) else np.array(sequence_lengths[scores == max_possible_score])
            )
            sequence_length_unsolved = (
                np.array([]) if np.all(scores == max_possible_score) else np.array(sequence_lengths[scores == 0])
            )

            # Use the already calculated reward summary metrics for wandb
            all_zero_groups_ratio = all_zero_groups / total_groups if total_groups > 0 else 0

            metrics = {
                "scores": np.array(scores).mean(),
                "real_batch_size_ratio": real_batch_size_ratio,
                "unsolved_batch_size_ratio": unsolved_batch_size_ratio,
                "packed_ratio": len(packed_sequences.query_responses) / len(responses) if len(responses) > 0 else 0,
                "val/all_zero_reward_groups": all_zero_groups,
                "val/all_zero_reward_groups_ratio": all_zero_groups_ratio,
                "val/total_reward_groups": total_groups,
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
            }

        if args.save_traces:
            traces = {
                "scores": scores.tolist(),
                "finish_reasons": finish_reasons,
                "responses": responses,
                "training_step": training_step,
                **asdict(batch),  # Unpack all batch fields
                **reward_metrics,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            with open(f"{args.output_dir}/traces_{args.run_name}.jsonl", "a") as f:
                json.dump(traces, f)
                f.write("\n")

        if len(responses) == 0:
            logger.warning(f"No responses in batch {training_step}.")

        # Put the packed sequences and metrics into the output queue
        packed_sequences_Q.put(
            {
                "packed_sequences": packed_sequences,  # for debugging purposes
                "collated_data": collated_data,
                "metrics": metrics,
                "responses_count": len(responses),
                "num_new_tokens": num_new_tokens,
                "B": B,
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
    if args.with_tracking:
        if args.wandb_entity is None:
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
        if args.shuffle_eval_dataset:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)

    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)

    return train_dataset, eval_dataset


def create_model_and_optimizer(
    args: Args,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    beaker_config: BeakerRuntimeConfig,
    wandb_url: str,
    tokenizer: PreTrainedTokenizer,
    inference_results_Q: ray_queue.Queue,
    param_prompt_Q: ray_queue.Queue,
    evaluation_inference_results_Q: ray_queue.Queue,
) -> tuple[ModelGroup, list[vllm_utils3.LLMRayActor], dict, int, int]:
    """Create the model, optimizer, and vLLM engines."""
    # Create placement group
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray_get_with_progress([pg.ready()], desc="Waiting for placement group")
    inits = []
    policy_group = ModelGroup(pg, PolicyTrainerRayProcess, args.num_learners_per_node, args.single_gpu_mode)
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits.extend(
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
        for model in policy_group.models
    )

    # Set up tools
    max_len = args.max_prompt_token_length + args.response_length
    tool_objects = {}
    if args.tools:
        for tool in args.tools:
            if tool.lower() == "search":
                from open_instruct.search_utils.search_tool import SearchTool

                tool = SearchTool(
                    start_str="<query>",
                    end_str="</query>",
                    api_endpoint=args.search_api_endpoint,
                    number_documents_to_search=args.number_documents_to_search,
                )
                tool_objects[tool.end_str] = tool
                # Add tool end string to stop_strings
                args.stop_strings.append(tool.end_str)
            elif tool.lower() == "code":
                from open_instruct.tool_utils.tool_vllm import PythonCodeTool

                tool = PythonCodeTool(start_str="<code>", end_str="</code>", api_endpoint=args.code_tool_api_endpoint)
                tool_objects[tool.end_str] = tool
                # Add tool end string to stop_strings
                args.stop_strings.append(tool.end_str)
            else:
                raise ValueError(f"Unknown tool: {tool}")

    queues_to_monitor = {
        "Inference Results Queue": inference_results_Q,
        "Param Prompt Queue": param_prompt_Q,
        "Evaluation Queue": evaluation_inference_results_Q,
    }
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, args)

    # Create vLLM engines with queues
    vllm_engines = vllm_utils3.create_vllm_engines(
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
        prompt_queue=param_prompt_Q,
        results_queue=inference_results_Q,
        eval_results_queue=evaluation_inference_results_Q,
        actor_manager=actor_manager,
        inference_batch_size=args.inference_batch_size,
        use_fp8_kv_cache=args.use_fp8_kv_cache,
        verbose=args.verbose,
    )

    resume_training_step = ray_get_with_progress(inits, desc="Initializing models")[0] + 1
    episode = (resume_training_step - 1) * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    logger.info("======== ✅ all models and vLLM engines initialized =========")

    # Get and set KV cache max concurrency from the first engine (all engines have the same config)
    # fp8 kv cache for now forces v0 engine and breaks this.
    if vllm_engines and not args.use_fp8_kv_cache:
        kv_cache_max_concurrency = ray.get(vllm_engines[0].get_kv_cache_info.remote())
        ray.get(actor_manager.set_kv_cache_max_concurrency.remote(kv_cache_max_concurrency))
    else:
        # dummy value
        ray.get(actor_manager.set_kv_cache_max_concurrency.remote(-1))

    ray_get_with_progress(
        [m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models],
        desc="Setting up model update group",
    )
    logger.info("======== ✅ model update group setup successfully =========")

    return policy_group, vllm_engines, tool_objects, resume_training_step, episode, actor_manager


def create_generation_configs(args: Args):
    """Create generation configs for training and evaluation."""
    generation_config = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.vllm_top_p,  # prevent rare out-of-vocab tokens with qwen
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        n=args.num_samples_per_prompt_rollout,
        stop=args.stop_strings,
        seed=args.seed,
        # IMPORTANT: Set output_kind to FINAL_ONLY to ensure vLLM V1 properly handles n>1
        # With the default CUMULATIVE mode, vLLM V1 returns separate outputs for each
        # completion, making it difficult to aggregate them correctly. FINAL_ONLY mode
        # ensures all n completions are returned together in a single output.
        output_kind=vllm.sampling_params.RequestOutputKind.FINAL_ONLY,
    )
    eval_generation_config = generation_config.clone()
    eval_generation_config.temperature = 0.0
    eval_generation_config.n = 1
    return {"train": generation_config, "eval": eval_generation_config}


def split_and_insert_batch(
    batch: Batch,
    training_step: int,
    pending_queries_map: PendingQueriesMap,
    param_prompt_Q: ray_queue.Queue,
    generation_config,
    is_eval: bool,
) -> None:
    """Split a batch into multiple inference batches and insert individual prompts into queues and mapping."""
    for idx, query, ground_truth, dataset, raw_query in zip(
        batch.indices, batch.queries, batch.ground_truths, batch.datasets, batch.raw_queries
    ):
        pending_queries_map.insert(idx, query, ground_truth, dataset, raw_query)
        param_prompt_Q.put(
            PromptRequest(
                prompt=query,
                generation_config=generation_config,
                training_step=training_step,
                dataset_index=idx,
                is_eval=is_eval,
            )
        )


def load_data_from_packing_thread(
    packed_sequences_Q: Queue, num_total_tokens: int, stop_event: threading.Event, health_check_fn: Callable[[], None]
):
    """Get the packed sequences with advantages from the packing thread."""
    with Timer("[Main Thread] 📦 Getting packed sequences from thread") as timer:
        while True:
            if stop_event.is_set():
                logger.warning("[Main Thread] Stop event detected while waiting for packed sequences")
                return None, {}, num_total_tokens
            try:
                packed_data = packed_sequences_Q.get(timeout=30.0)
                break
            except Empty:
                # check that everything is still alive
                health_check_fn()
                logger.warning("[Main Thread] Timeout waiting for packed sequences. Retrying...")
        data_thread_metrics = packed_data["metrics"]
        B = packed_data["B"]
        collated_data = packed_data["collated_data"]
        num_total_tokens += packed_data["num_new_tokens"]

    data_thread_metrics["time/trainer_idling"] = timer.duration
    if B == 0:
        logger.warning("[Main Thread] 🤡 After packing, there is not enough data to train")
        return None, data_thread_metrics, num_total_tokens
    return collated_data, data_thread_metrics, num_total_tokens


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
            weight_broadcast_futures: List[ray.ObjectRef] = [m.broadcast_to_vllm.remote() for m in policy_group.models]

            # Wait for all weight updates to complete
            ray_get_with_progress(
                weight_broadcast_futures,
                desc="[Weight Sync Thread] Waiting for weight updates to complete",
                enable=args.verbose,
            )

            # Allow actors to resume
            ray.get(actor_manager.set_should_stop.remote(False))
            logger.debug("[Weight Sync Thread] Set should_stop to False after weight sync")

        try:
            weight_sync_metrics_Q.put_nowait({"time/weight_sync": timer.duration})
        except Full:
            logger.warning("[Weight Sync Thread] weight sync metrics queue full, skipping metric")

    logger.info("[Weight Sync Thread] 🛑 Stopping weight sync thread")


def generate_thread(args, vllm_engines, resume_training_step, stop_event, generate_metrics_Q):
    """Thread function that repeatedly calls process_from_queue on vllm engines."""
    logger.info("[Generate Thread] 🚀 Starting generation thread")
    while not stop_event.is_set():
        with Timer("🔥 Generation time") as timer:
            processed_results = ray_get_with_progress(
                [engine.process_from_queue.remote(timeout=20) for engine in vllm_engines],
                desc="[Generate Thread] Waiting for vLLM engines to process",
                enable=args.verbose,
            )
            num_processed = sum(int(result) for result in processed_results)
            # Suppress timing output if nothing was processed
            if num_processed == 0:
                timer.noop = True
        if num_processed > 0:
            try:
                generate_metrics_Q.put_nowait({"time/generation": timer.duration})
            except Full:
                logger.warning("[Generate Thread] generate metrics queue full, skipping metric")
    logger.info("[Generate Thread] 🛑 Stopping generation thread")


def one_training_step(
    args: Args,
    policy_group: ModelGroup,
    collated_data,
    tokenizer,
    data_thread_metrics,
    episode,
    training_step,
    num_total_tokens,
    start_time,
    train_dataset,
    wandb_url,
    chat_template_name,
    actor_manager=None,
    iter_dataloader=None,
):
    """Train the model for one step."""
    update_ref_policy_future = []
    with Timer("[Main Thread] 🗡️ Training") as train_timer:
        metrics_list: List[dict[str, float]] = ray_get_with_progress(
            [
                policy_group.models[i].train.remote(
                    **collated_data[i], pad_token_id=tokenizer.pad_token_id, num_mini_batches=args.num_mini_batches
                )
                for i in range(args.world_size)
            ],
            desc=f"Running training step {training_step}",
        )
        if (
            args.ref_policy_update_freq is not None
            and training_step % args.ref_policy_update_freq == 0
            and args.alpha > 0
        ):
            update_ref_policy_future.extend(
                [policy_group.models[i].update_ref_policy.remote() for i in range(args.world_size)]
            )

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
        save_time += timer.duration

    if len(update_ref_policy_future) > 0:
        with Timer("[Main Thread] 🔃 Updating reference policy"):
            ray_get_with_progress(update_ref_policy_future, desc="Updating reference policy")

    ray.get(actor_manager.report_training_step_time.remote(train_timer.duration))

    average_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
    total_time = time.perf_counter() - start_time
    metrics = {
        "episode": episode,
        "global_step": episode,
        "training_step": training_step,
        "val/num_total_tokens": num_total_tokens,
        "epoch": episode / args.num_samples_per_prompt_rollout / len(train_dataset),
        "tokens_per_second": num_total_tokens / total_time,
        "time/total": total_time,
        "time/training": train_timer.duration,
        "time/saving": save_time,
        **data_thread_metrics,
        **average_metrics,
    }
    # Print only scalar metrics
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (float, int))}
    print_rich_single_line_metrics(scalar_metrics)

    if args.with_tracking:
        # Convert array/list metrics to wandb histograms for logging
        for key, value in metrics.items():
            if isinstance(value, np.ndarray) or isinstance(value, list):
                if len(value) > 0:
                    metrics[key] = wandb.Histogram(value)
        wandb.log(metrics, step=episode)


def maybe_evaluate(
    args: Args,
    training_step: int,
    evaluation_inference_results_Q: ray_queue.Queue,  # Ray queue
    tokenizer,
    reward_fn,
    episode,
    eval_pending_queries_map: PendingQueriesMap,
    eval_generation_config,
    generate_metrics_Q: Queue,
    num_eval_prompts: int,
    actor_manager=None,
):
    """Optionally evaluate the model."""
    try:
        # timeout 0.01 if this is the last training step or we're not evaluating
        # otherwise, wait to get the last evaluation generations (long timeout just in case)
        timeout = 0.01 if (training_step < args.num_training_steps or args.local_eval_every < 0) else 100

        # Accumulate evaluation results from all vLLM engines
        eval_result, eval_batch = accumulate_inference_batches(
            evaluation_inference_results_Q,
            eval_pending_queries_map,
            args,
            training_step,
            eval_generation_config,
            num_prompts=num_eval_prompts,
            actor_manager=actor_manager,
            timeout=timeout,
        )

        logger.info("[Main Thread] 📊 Evaluation responses received")

        eval_generate_metrics = {}
        try:
            eval_generate_metrics = generate_metrics_Q.get_nowait()
        except Empty:
            logger.info("[Main Thread] didn't get eval generation metrics")

        eval_sequence_lengths = np.array([len(response) for response in eval_result.responses])
        eval_decoded_responses = tokenizer.batch_decode(eval_result.responses, skip_special_tokens=True)
        eval_stop_rate = sum(int(finish_reason == "stop") for finish_reason in eval_result.finish_reasons) / len(
            eval_result.finish_reasons
        )

        # get and log evaluation metrics
        eval_scores, eval_reward_metrics = asyncio.run(
            reward_fn(
                eval_result.responses,
                eval_decoded_responses,
                eval_batch if eval_batch else Batch(queries=[], ground_truths=[], datasets=[], indices=None),
                eval_result.finish_reasons,
                eval_result.request_info,
            )
        )
        eval_reward_metrics = {f"eval/{key}": val for key, val in eval_reward_metrics.items()}
        eval_metrics = {
            "eval/scores": np.array(eval_scores).mean(),
            "eval/sequence_lengths": eval_sequence_lengths.mean(),
            "eval/sequence_lengths_min": eval_sequence_lengths.min(),
            "eval/sequence_lengths_max": eval_sequence_lengths.max(),
            "eval/stop_rate": eval_stop_rate,
            **eval_reward_metrics,
        }
        if "time/generation" in eval_generate_metrics:
            eval_metrics["eval/generation_time"] = eval_generate_metrics["time/generation"]
        print_rich_single_line_metrics(eval_metrics)

        table = {}
        table["prompt"] = tokenizer.batch_decode(eval_batch.queries if eval_batch else [])
        table["response"] = eval_decoded_responses
        table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
        table["scores"] = eval_scores
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


def make_reward_fn(args: Args) -> Callable:
    """Create a reward function based on the provided arguments."""
    reward_fn_mapping = build_all_verifiers(args)

    async def reward_fn(
        responses: List[torch.Tensor],
        decoded_responses: List[str],
        batch: Batch,
        finish_reasons: List[str],
        infos: List[List[int]],
        queries: Optional[List[str]] = None,
    ) -> List[float]:
        timeouts = infos.timeouts
        tool_errors = infos.tool_errors
        tool_outputs = infos.tool_outputs
        tool_calleds = infos.tool_calleds
        good_outputs = [
            len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
            for i in range(len(tool_outputs))
        ]
        scores = [0] * len(decoded_responses)
        metrics = {}

        if args.apply_r1_style_format_reward:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🧮 Calculating format reward"):
                format_scores = soft_format_reward_func(decoded_responses, args.r1_style_format_reward)
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]
                metrics["val/format_scores"] = np.array(format_scores).mean()

        if args.apply_verifiable_reward:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🏆 Applying verifiable reward"):
                verifiable_rewards, per_func_rewards = await apply_verifiable_reward(
                    reward_fn_mapping,
                    responses,
                    decoded_responses,
                    batch,
                    reward_mult=args.verification_reward,
                    queries=queries,
                )
                if len(verifiable_rewards) != len(scores):
                    raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")
                # slightly complex combo of good outputs and additive format reward
                for i in range(len(verifiable_rewards)):
                    if not args.only_reward_good_outputs or (good_outputs[i] and args.only_reward_good_outputs):
                        if args.apply_r1_style_format_reward and args.additive_format_reward:
                            scores[i] = verifiable_rewards[i] + scores[i]
                        elif args.apply_r1_style_format_reward and not args.additive_format_reward:
                            scores[i] = verifiable_rewards[i] if format_scores[i] == 1 else 0
                        else:
                            scores[i] = verifiable_rewards[i]
                np_verifiable_rewards = np.array(verifiable_rewards)
                metrics["objective/verifiable_reward"] = np_verifiable_rewards.mean()
                metrics["objective/verifiable_correct_rate"] = (np_verifiable_rewards > 0.0).mean()
                # reshuffle around per_func rewards
                per_func_lists = defaultdict(list)
                for reward_dict in per_func_rewards:
                    for key, value in reward_dict.items():
                        per_func_lists[key].append(value)
                # log per function rewards
                for key, value in per_func_lists.items():
                    np_value = np.array(value)
                    metrics[f"objective/{key}_reward"] = np_value.mean()
                    metrics[f"objective/{key}_correct_rate"] = (np_value > 0.0).mean()

        # this gets applied at the very end since it replaces (rather than adds to) the existing reward.
        if args.non_stop_penalty:
            with Timer("[Data Preparation Thread] Calculating rewards -- 🦖 Applying non stop penalty"):
                assert len(finish_reasons) == len(scores)
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = args.non_stop_penalty_value

        return scores, metrics

    return reward_fn


def cleanup_judge_clients():
    """Cleans up all LLM judge clients and shutdown Ray."""
    asyncio.run(cleanup_all_llm_judge_clients())
    logger.info("✅ LLM judge clients cleaned up")
    ray.shutdown()
    logger.info("✅ Ray shut down")


def cleanup_training_resources(
    stop_event: threading.Event,
    executor: futures.ThreadPoolExecutor,
    queues: list[ray_queue.Queue],
    actor_manager: ActorManager,
) -> None:
    """Clean up all training resources including threads and Ray queues."""
    # Signal generate_thread to stop
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

    # Clean up distributed process group if it was initialized
    if dist.is_initialized():
        logger.info("Destroying process group...")
        dist.destroy_process_group()
        logger.info("✅ Process group destroyed")


def run_training(
    args,
    tokenizer,
    train_dataset,
    eval_batch,
    policy_group,
    vllm_engines,
    generation_configs,
    iter_dataloader,
    reward_fn,
    resume_training_step,
    episode,
    wandb_url,
    tc,
    stop_event,
    executor,
    inference_results_Q,
    param_prompt_Q,
    evaluation_inference_results_Q,
    packed_sequences_Q,
    pending_queries_map,
    eval_pending_queries_map,
    generate_metrics_Q,
    weight_sync_metrics_Q,
    actor_manager: ActorManager,
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
        reward_fn,
        inference_results_Q,
        packed_sequences_Q,
        pending_queries_map,
        args,
        tokenizer,
        args.num_training_steps,
        generation_configs["train"],
        resume_training_step,
        actor_manager,
    )

    logger.info("======== ✅ generation thread starts =========")
    generation_future = executor.submit(
        generate_thread, args, vllm_engines, resume_training_step, stop_event, generate_metrics_Q
    )

    # setup health check function to check that everything is still alive
    def health_check_fn():
        [f.result() for f in [packing_future, generation_future, weight_sync_thread_future] if f.done()]

    # Send initial data to ensure we have a N-step offset.
    for _ in range(args.async_steps):
        dataset_indices = next(iter_dataloader)
        batch = next_batch(dataset_indices, train_dataset)
        split_and_insert_batch(
            batch,
            resume_training_step,
            pending_queries_map,
            param_prompt_Q,
            generation_configs["train"],
            is_eval=False,
        )
    if checkpoint_state and "num_total_tokens" in checkpoint_state:
        num_total_tokens = checkpoint_state["num_total_tokens"]
        logger.info(f"Restored num_total_tokens: {num_total_tokens}")
    else:
        num_total_tokens = 0

    num_total_tokens = 0
    training_start_time = time.time()  # Track overall training start time
    for training_step in range(resume_training_step, args.num_training_steps + 1):
        start_time = time.perf_counter()

        if (
            training_step == resume_training_step
            or training_step % args.update_progress_every == 0
            or training_step == args.num_training_steps
        ):
            maybe_update_beaker_description(
                current_step=training_step,
                total_steps=args.num_training_steps,
                start_time=training_start_time,
                wandb_url=wandb_url,
            )

        # Check if any of the threads have raised an exception.
        health_check_fn()

        logger.debug(f"[Main Thread] Triggered weight sync for step {training_step}")
        weight_sync_trigger_event.set()

        episode += args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
        batch = next_batch(next(iter_dataloader), train_dataset)
        split_and_insert_batch(
            batch, training_step, pending_queries_map, param_prompt_Q, generation_configs["train"], is_eval=False
        )
        if (
            training_step % args.local_eval_every == 0
            and eval_batch is not None
            and (args.eval_on_step_0 or training_step > 1)
        ):
            split_and_insert_batch(
                eval_batch,
                training_step,
                eval_pending_queries_map,
                param_prompt_Q,
                generation_configs["eval"],
                is_eval=True,
            )

        # The generate_thread is now handling vLLM processing asynchronously
        collated_data, data_thread_metrics, num_total_tokens = load_data_from_packing_thread(
            packed_sequences_Q, num_total_tokens, stop_event, health_check_fn
        )
        if collated_data is None:
            continue

        for metrics_Q in [generate_metrics_Q, weight_sync_metrics_Q]:
            try:
                data_thread_metrics |= metrics_Q.get_nowait()
            except Empty:
                logger.info("[Main Thread] didn't get train generation metrics")

        one_training_step(
            args,
            policy_group,
            collated_data,
            tokenizer,
            data_thread_metrics,
            episode,
            training_step,
            num_total_tokens,
            start_time,
            train_dataset,
            wandb_url,
            tc.chat_template_name,
            actor_manager,
            iter_dataloader,
        )

        # Checkpoint after one_training_step (or even if it was skipped)
        # This ensures we checkpoint progress even if the exact checkpoint step has no data
        if (
            args.checkpoint_state_freq > 0
            and training_step % args.checkpoint_state_freq == 0
            and args.checkpoint_state_dir is not None
        ):
            with Timer("[Main Thread] 🗡️ Saving checkpoint state"):
                # Save comprehensive client state including ShufflingIterator state
                client_state = {
                    "training_step": training_step,
                    "episode": episode,
                    "num_total_tokens": num_total_tokens,
                }

                # Save ShufflingIterator state
                if iter_dataloader is not None:
                    client_state["shuffling_iterator_state"] = iter_dataloader.get_state()

                ray_get_with_progress(
                    [
                        policy_group.models[i].save_checkpoint_state.remote(args.checkpoint_state_dir, client_state)
                        for i in range(args.world_size)
                    ],
                    desc=f"Saving checkpoint state at step {training_step}",
                )
                logger.info(f"Saved checkpoint state at step {training_step} to {args.checkpoint_state_dir}")

        maybe_evaluate(
            args,
            training_step,
            evaluation_inference_results_Q,
            tokenizer,
            reward_fn,
            episode,
            eval_pending_queries_map,
            generation_configs["eval"],
            generate_metrics_Q,
            len(eval_batch.queries) if eval_batch else 0,
            actor_manager,
        )

    if resume_training_step > args.num_training_steps:
        raise ValueError(f"Training didn't run since {resume_training_step=} > {args.num_training_steps=}")

    save_final_model(args, policy_group, tokenizer, training_step, wandb_url, tc.chat_template_name)


def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig):
    tokenizer = make_tokenizer(tc, model_config)
    args = setup_runtime_variables(args)
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
    ray.init(dashboard_host="0.0.0.0")

    # Create Ray queues.
    # Since we now send/receive individual prompts, queue size should accommodate
    # all prompts from async_steps + 1 training steps
    queue_size = (args.async_steps + 1) * args.num_unique_prompts_rollout
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    param_prompt_Q = ray_queue.Queue(maxsize=queue_size)
    # We don't care if we ever hit the max, so we let the queue be unbounded.
    evaluation_inference_results_Q = ray_queue.Queue()

    policy_group, vllm_engines, tool_objects, resume_training_step, episode, actor_manager = (
        create_model_and_optimizer(
            args,
            tc,
            model_config,
            beaker_config,
            wandb_url,
            tokenizer,
            inference_results_Q,
            param_prompt_Q,
            evaluation_inference_results_Q,
        )
    )

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

    train_dataset_idxs = np.arange(len(train_dataset))
    iter_dataloader = ShufflingIterator(train_dataset_idxs, args.num_unique_prompts_rollout, seed=args.seed)

    if checkpoint_state and "shuffling_iterator_state" in checkpoint_state:
        iter_dataloader.set_state(checkpoint_state["shuffling_iterator_state"])
        logger.info("Restored ShufflingIterator state from checkpoint")

    # Create additional queues (main queues already created above)
    packed_sequences_Q = Queue(maxsize=args.async_steps)
    pending_queries_map = PendingQueriesMap()
    eval_pending_queries_map = PendingQueriesMap()
    generate_metrics_Q = Queue(maxsize=args.async_steps)
    weight_sync_metrics_Q = Queue(maxsize=args.async_steps)

    if eval_dataset is None:
        eval_batch = None
    else:
        eval_dataset_indices = list(range(len(eval_dataset)))
        eval_batch = next_batch(eval_dataset_indices, eval_dataset)
    reward_fn = make_reward_fn(args)

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="grpo")

    try:
        episode = run_training(
            args,
            tokenizer,
            train_dataset,
            eval_batch,
            policy_group,
            vllm_engines,
            generation_configs,
            iter_dataloader,
            reward_fn,
            resume_training_step,
            episode,
            wandb_url,
            tc,
            stop_event,
            executor,
            inference_results_Q,
            param_prompt_Q,
            evaluation_inference_results_Q,
            packed_sequences_Q,
            pending_queries_map,
            eval_pending_queries_map,
            generate_metrics_Q,
            weight_sync_metrics_Q,
            actor_manager,
            checkpoint_state,
        )
    finally:
        cleanup_training_resources(
            stop_event, executor, [inference_results_Q, param_prompt_Q, evaluation_inference_results_Q], actor_manager
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

    accelerator = Namespace()
    accelerator.is_main_process = True  # hack
    if args.push_to_hub:
        logger.info("Pushing model to hub")
        push_folder_to_hub(accelerator, args.output_dir, args.hf_repo_id, args.hf_repo_revision)

    # Check for runtime leaks before exiting
    logger.info("Checking for runtime leaks...")

    utils.check_runtime_leaks()


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, Args)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)

    main(args, tokenizer_config, model_config)
