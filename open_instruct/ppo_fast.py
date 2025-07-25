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
# ---------------------------------------------------------------------
# The file referenced a lot of code from the ORZ project:
# https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
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
import json
import os
import shutil
import socket
import threading
import time
import traceback
from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import Callable, Iterator, List, Literal, Optional, Union

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
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)
from transformers.integrations import HfDeepSpeedConfig
from vllm import SamplingParams

from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
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
from open_instruct.rl_utils2 import Timer, calculate_advantages_packed, pack_sequences
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    RayProcess,
    _z3_params_to_fetch,
    extract_user_query,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)
from open_instruct.vllm_utils3 import create_vllm_engines, init_process_group

api = HfApi()
INVALID_LOGPROB = 1.0
INVALID_VALUE = 0.0  # to play nicely with debugging output
# torch.set_printoptions(precision=2, sci_mode=False)


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
    value_learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer for the value function"""
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
    """this sets how many in-loop evals we do during training. in-loop evals reuse the generation/reward verifier setup."""
    local_eval_freq: Optional[int] = None
    """this controls the number of in-loop evals, which reuses the generation/reward verifier setup. don't set this directly, but set via num_evals."""
    save_freq: int = 200
    """How many train steps to save the model"""

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
    value_num_mini_batches: int = 1
    """Number of minibatches to split a batch into for the value function"""
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    cliprange: float = 0.2
    """the clip range"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1.0
    """the discount factor"""
    lam: float = 1.0
    """the lambda value for GAE"""
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
    value_model_name_or_path: str = "EleutherAI/pythia-160m"
    """the name or path to the value model"""
    value_model_revision: Optional[str] = None
    """the revision of the value model"""
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

    # -- llm verifiers reward
    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    """the model to use for the llm judge"""
    llm_judge_max_tokens: int = 2048
    """the max tokens to use for the llm judge"""
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
    """Maximum number of tool calls allowed. Can be either a single integer (applies to all tools) or a list of integers
    with length 1 (applies to all tools) or matching the length of the tools list (per-tool limit)."""
    mask_tool_use: bool = True
    """Whether to mask the tool output. By default on."""
    only_reward_good_outputs: bool = False
    """Whether to only reward good outputs from the tools or not."""

    # rl-rag specific settngs
    number_documents_to_search: int = 3
    """The maximum number of documents to retrieve for each query."""
    search_api_endpoint: Optional[str] = None
    """The API endpoint for the search engine."""

    # code-tool specific settings
    code_tool_api_endpoint: Optional[str] = None

    def __post_init__(self):
        assert self.apply_verifiable_reward or self.apply_r1_style_format_reward or self.non_stop_penalty, (
            "At least one reward must be applied!"
        )
        assert self.pack_length >= self.max_prompt_token_length + self.response_length, (
            "The `pack_length` needs to be greater than the sum of `max_prompt_token_length` and `response_length`!"
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
        self.args = args
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.beaker_config = beaker_config
        self.wandb_url = wandb_url
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(self.local_rank)
        deepspeed.init_distributed()

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
        # Have to use FusedAdam for offloading6 and backloading
        from deepspeed.ops.adam import FusedAdam  # DeepSpeedCPUAdam

        if args.set_weight_decay_on_bias_and_norm:
            optim_params = get_optimizer_grouped_parameters(self.policy, args.weight_decay)
        else:
            optim_params = self.policy.parameters()
        self.optimizer = FusedAdam(
            optim_params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay
        )
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
            args.value_model_name_or_path,
            revision=args.value_model_revision,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )

        value_head = self.value_model.score
        config = AutoConfig.from_pretrained(
            args.value_model_name_or_path, revision=args.value_model_revision, trust_remote_code=True
        )
        print("initialize value_head for ZeRO-3 reward model training.")
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

        disable_dropout_in_model(self.value_model)
        self.value_model.gradient_checkpointing_enable()
        if args.set_weight_decay_on_bias_and_norm:
            optim_params = get_optimizer_grouped_parameters(self.value_model, args.weight_decay)
        else:
            optim_params = self.value_model.parameters()
        self.value_optimizer = FusedAdam(
            optim_params, lr=args.value_learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay
        )
        value_scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=self.value_optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_scheduler_steps,
        )
        self.value_model, self.value_optimizer, _, self.value_scheduler = deepspeed.initialize(
            model=self.value_model,
            optimizer=self.value_optimizer,
            config=ds_config,
            lr_scheduler=value_scheduler,
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

        self.offload_to_cpu(self.model)
        self.offload_to_cpu(self.value_model)
        optimization_steps_done = 0
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
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits
        logits /= temperature + 1e-7
        logprob = log_softmax_and_gather(logits, input_ids[:, 1:])

        # fow now, entropy is just for monitoring, and we don't pass gradients through it.
        entropy = None
        if return_entropy:
            with torch.no_grad():
                entropy = entropy_from_logits(logits)

        return logprob, entropy

    def forward_value(
        self,
        model: PreTrainedModel,
        query_response: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_token_id: int,
    ) -> torch.Tensor:
        # Replace pad tokens with 0s so that we don't run into index out of bounds errors
        padding_mask = query_response != pad_token_id
        input_ids = torch.masked_fill(query_response, ~padding_mask, 0)
        lm_backbone = getattr(model, model.base_model_prefix)
        output = lm_backbone(
            input_ids=input_ids[:, :-1],
            # @vwxyzjn: without clamp, we get index out of bounds errors; TODO: investigate
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
            output_hidden_states=True,
        )
        reward_logits = model.score(output.hidden_states[-1])  # (batch_size, sequence_length)
        return reward_logits

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
        # @vwxyzjn: magic taken from https://github.com/huggingface/trl/issues/2840#issuecomment-2662747485
        # to avoid assert not param.ds_active_sub_modules, param.ds_summary()
        for param in self.model.parameters():
            param.ds_active_sub_modules.clear()
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
        collated_query_responses: torch.Tensor,
        collated_attention_masks: torch.Tensor,
        collated_position_ids: torch.Tensor,
        collated_response_masks: torch.Tensor,
        collated_dones: torch.Tensor,
        collated_rewards: torch.Tensor,
        pad_token_id: int,
        num_mini_batches: int,
        value_num_mini_batches: int,
        gamma: float,
        lam: float,
    ):
        args = self.args
        # Flatten tensors to cpu list for adv calculation
        cpu_collated_dones = sum([item.clamp(0, 1).tolist() for item in collated_dones], [])
        cpu_collated_rewards = sum([item.tolist() for item in collated_rewards], [])
        cpu_collated_response_masks = sum([item.clamp(0, 1).tolist() for item in collated_response_masks], [])
        max_len = max([len(item) for item in cpu_collated_dones])

        # Put to device
        to_device_inplace(collated_query_responses, self.device)
        to_device_inplace(collated_attention_masks, self.device)
        to_device_inplace(collated_position_ids, self.device)
        to_device_inplace(collated_response_masks, self.device)
        accumulation_steps = max(1, len(collated_query_responses) // (num_mini_batches))
        value_accumulation_steps = max(1, len(collated_query_responses) // (value_num_mini_batches))

        # Calculate the logprob of the reference policy
        with Timer("Ref Logprob Calculation", noop=self.rank != 0):
            collated_ref_logprobs = []
            with torch.no_grad():
                for i in range(len(collated_query_responses)):
                    query_response = collated_query_responses[i]
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
                    ref_logprob = torch.masked_fill(ref_logprob, ~response_mask[:, 1:].bool(), INVALID_LOGPROB)
                    collated_ref_logprobs.append(ref_logprob)
                    torch.cuda.empty_cache()

        old_logprobs = [None for _ in range(len(collated_query_responses))]
        with Timer("Old Logprob Calculation", noop=self.rank != 0):
            with torch.no_grad():
                for i in range(len(collated_query_responses)):
                    query_response = collated_query_responses[i]
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
                    old_logprob = torch.masked_fill(old_logprob, ~response_mask[:, 1:].bool(), INVALID_LOGPROB)
                    old_logprobs[i] = old_logprob
                    torch.cuda.empty_cache()
        with Timer("Backload Value Model", noop=self.rank != 0):
            self.backload_to_gpu(self.value_model)
        with Timer("Value Calculation", noop=self.rank != 0):
            collated_values = []
            with torch.no_grad():
                for i in range(len(collated_query_responses)):
                    query_response = collated_query_responses[i]
                    attention_mask = collated_attention_masks[i]
                    position_id = collated_position_ids[i]
                    response_mask = collated_response_masks[i]
                    value = self.forward_value(
                        self.value_model, query_response, attention_mask, position_id, pad_token_id
                    ).squeeze(-1)
                    value = torch.masked_fill(value, ~response_mask[:, 1:].bool(), INVALID_VALUE)
                    collated_values.append(value)

        with Timer("Advantage Calculation", noop=self.rank != 0):
            with torch.no_grad():
                # Pad cpu lists to max_len; then do advantage calculation
                cpu_collated_values = sum([item.cpu().tolist() for item in collated_values], [])
                cpu_collated_dones = [item + [0] * (max_len - len(item)) for item in cpu_collated_dones]
                cpu_collated_rewards = [item + [0] * (max_len - len(item)) for item in cpu_collated_rewards]
                cpu_collated_response_masks = [
                    item + [0] * (max_len - len(item)) for item in cpu_collated_response_masks
                ]
                # minus 1 in `cpu_collated_values` because we already did the calculation based on [:-1] and masking on [1:]
                cpu_collated_values = [
                    item + [INVALID_VALUE] * (max_len - 1 - len(item)) for item in cpu_collated_values
                ]
                adv, ret = calculate_advantages_packed(
                    np.stack(cpu_collated_values),
                    np.stack(cpu_collated_rewards)[:, 1:],
                    gamma,
                    lam,
                    np.stack(cpu_collated_dones)[:, 1:],
                    np.stack(cpu_collated_response_masks)[:, 1:],
                )

                # Calculate the mean and std of the advantages
                adv_gpu = torch.from_numpy(adv).to(self.device)
                response_masks_gpu = torch.from_numpy(np.stack(cpu_collated_response_masks)[:, 1:]).to(self.device)
                adv_sum = (adv_gpu * response_masks_gpu).sum()
                adv_abs_sum = (torch.abs(adv_gpu) * response_masks_gpu).sum()
                mask_sum = response_masks_gpu.sum()
                dist.all_reduce(adv_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(adv_abs_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(mask_sum, op=dist.ReduceOp.SUM)
                adv_mean = adv_sum / mask_sum
                adv_abs_mean = adv_abs_sum / mask_sum
                adv_variance = (torch.square(adv_gpu - adv_mean) * response_masks_gpu).sum()
                dist.all_reduce(adv_variance, op=dist.ReduceOp.SUM)
                adv_variance /= mask_sum
                adv_std = torch.sqrt(adv_variance)

                # Normalize the advantages
                adv = (adv_gpu - adv_mean) / (adv_std + 1e-8)
                collated_advantages = []
                collated_returns = []
                offset = 0
                for i in range(len(collated_query_responses)):
                    batch_size, seq_len = collated_query_responses[i].shape
                    collated_advantages.append(adv[offset : offset + batch_size, : seq_len - 1])
                    collated_returns.append(torch.from_numpy(ret[offset : offset + batch_size, : seq_len - 1]))
                    offset += batch_size
                to_device_inplace(collated_advantages, self.device)
                to_device_inplace(collated_returns, self.device)
                to_device_inplace(collated_values, self.device)

        with Timer("[Training Processes] Value Loss calculation", noop=self.rank != 0):
            value_losses = torch.zeros(len(collated_query_responses))
            local_step = 0
            value_optimizer_step = 0
            for epoch_idx in range(args.num_epochs):
                for i in range(len(collated_query_responses)):
                    mb_query_responses = collated_query_responses[i]
                    mb_response_masks = collated_response_masks[i]
                    mb_response_masks_bool = mb_response_masks[:, 1:].bool()
                    mb_attention_mask = collated_attention_masks[i]
                    mb_position_id = collated_position_ids[i]
                    mb_values = collated_values[i]
                    mb_return = collated_returns[i]
                    vpred = self.forward_value(
                        self.value_model, mb_query_responses, mb_attention_mask, mb_position_id, pad_token_id
                    ).squeeze(-1)
                    vpred = torch.masked_fill(vpred, ~mb_response_masks_bool, INVALID_VALUE)
                    vpredclipped = torch.clamp(
                        vpred, mb_values - args.cliprange_value, mb_values + args.cliprange_value
                    )
                    vf_losses1 = torch.square(vpred - mb_return)
                    vf_losses2 = torch.square(vpredclipped - mb_return)
                    vf_loss_max = torch.max(vf_losses1, vf_losses2)
                    loss = 0.5 * masked_mean(vf_loss_max, mb_response_masks_bool)
                    loss = loss / value_accumulation_steps
                    self.value_model.backward(loss)
                    with torch.no_grad():
                        value_losses[i] = loss
                    if (local_step + 1) % value_accumulation_steps == 0:
                        self.value_model.step()
                        value_optimizer_step += 1
                    if (local_step + 1) // value_accumulation_steps == value_num_mini_batches:
                        break
                    local_step += 1

            with torch.no_grad():
                self.local_metrics.add("loss/value", value_losses.mean())
        with Timer("Offload Value Model", noop=self.rank != 0):
            self.offload_to_cpu(self.value_model)

        with Timer("Backload Model", noop=self.rank != 0):
            self.backload_to_gpu(self.model)

        with Timer("[Training Processes] Policy Loss calculation", noop=self.rank != 0):
            local_step = 0
            policy_optimizer_step = 0
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
                    mb_advantages = collated_advantages[i]
                    mb_response_masks = collated_response_masks[i]
                    mb_response_masks_bool = mb_response_masks[:, 1:].bool()
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

                    # If we have one minibatch, we can just re-use the initial logprobs.
                    # otherwise, we use the logprobs we already calculated.
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
                    pg_losses = -mb_advantages * ratio
                    pg_losses2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
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
                        policy_optimizer_step += 1
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
                self.local_metrics.add("lr", self.scheduler.get_last_lr()[0])
                self.local_metrics.add("lr_value", self.value_scheduler.get_last_lr()[0])
                self.local_metrics.add("policy_optimizer_step", policy_optimizer_step)
                self.local_metrics.add("value_optimizer_step", value_optimizer_step)
                self.local_metrics.add("val/adv_mean", adv_mean)
                self.local_metrics.add("val/adv_abs_mean", adv_abs_mean)
                self.local_metrics.add("val/adv_std", adv_std)
                if args.record_entropy:
                    self.local_metrics.add("policy/entropy_avg", entropy_stats.mean())
                self.local_metrics.add("policy/entropy_var", entropy_stats.var())
                metrics_list = self.local_metrics.get_metrics_list()
                # metrics_list["val/advantages_mean"] = adv.mean()
                # metrics_list["val/advantages_min"] = adv.min()
                # metrics_list["val/advantages_max"] = adv.max()
                # metrics_list["val/advantages_std"] = adv.std()

        with Timer("Offload Model", noop=self.rank != 0):
            self.offload_to_cpu(self.model)
        return metrics_list

    def save_model(self, output_dir: str, chat_template_name: str, tokenizer: PreTrainedTokenizer) -> None:
        model_to_save = self.model
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # save model weights for ZeRO2/3
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        if "olmo" in chat_template_name:
            # New chat template has no bos token, and two eos tokens: <|im_end|> and <|endoftext|>
            model_to_save.generation_config = get_olmo3_generation_config(tokenizer)

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


def vllm_generate_thread(
    vllm_engines: List[ray.actor.ActorHandle],
    generation_config: SamplingParams,
    eval_generation_config: SamplingParams,
    inference_results_Q: Queue,
    param_prompt_Q: Queue,
    num_training_steps: int,
    eval_prompt_token_ids: Optional[List[int]],
    evaluation_inference_results_Q: Queue,
    local_eval_freq: int,
    resume_training_step: int = 1,
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
            if args.tool_use:
                masks.extend([out.mask for output in outputs for out in output.outputs])
                num_calls.extend([out.num_calls for output in outputs for out in output.outputs])
                timeouts.extend([out.timeout for output in outputs for out in output.outputs])
                tool_errors.extend([out.tool_error for output in outputs for out in output.outputs])
                tool_outputs.extend([out.tool_output for output in outputs for out in output.outputs])
                tool_runtimes.extend([out.tool_runtime for output in outputs for out in output.outputs])
                tool_calleds.extend([out.tool_called for output in outputs for out in output.outputs])
        # if not using the tool, mask is all 1s
        if not args.tool_use:
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
        if eval_prompt_token_ids is not None and (training_step - 1) % local_eval_freq == 0:
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

        with Timer("💰 [Data Preparation Thread] Calculating rewards"):
            scores, reward_metrics = asyncio.run(
                reward_fn(
                    responses, decoded_responses, ground_truths, datasets, finish_reasons, infos, decoded_queries
                )
            )
            scores = np.array(scores)
            scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
            non_zero_std_mask = scores_per_prompt.std(axis=-1) != 0
            real_batch_size_ratio = non_zero_std_mask.sum() * args.num_samples_per_prompt_rollout / len(scores)

        with Timer("📦 [Data Preparation Thread] Packing sequences"):
            packed_sequences = pack_sequences(
                queries=queries,
                responses=responses,
                masks=masks,
                pack_length=args.pack_length,
                pad_token_id=tokenizer.pad_token_id,
            )
            num_new_tokens = sum(len(seq) for seq in packed_sequences.query_responses)
            lookup_rewards = np.zeros(len(scores) + 1, dtype=np.float32)
            lookup_rewards[1:] = scores
            packed_rewards = [
                torch.tensor(lookup_rewards[packed_done], dtype=torch.float32)
                for packed_done in packed_sequences.dones
            ]
            packed_sequences.rewards = packed_rewards

        with Timer("🔄 [Data Preparation Thread] Prepare collated data for each worker"):
            B = (
                len(packed_sequences.query_responses) // args.world_size
            )  # essentially doing `drop_last=True`, which is fine.
            collated_data = []
            for i in range(args.world_size):
                per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
                per_device_packed_attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
                per_device_packed_position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
                per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]
                per_device_packed_dones = packed_sequences.dones[B * i : B * (i + 1)]
                per_device_packed_rewards = packed_sequences.rewards[B * i : B * (i + 1)]

                # Shuffle the batch and collate the data
                # b_inds = np.random.permutation(len(per_device_packed_query_responses))
                b_inds = np.arange(len(per_device_packed_query_responses))
                collated_query_responses = []
                collated_attention_masks = []
                collated_position_ids = []
                collated_response_masks = []
                collated_dones = []
                collated_rewards = []
                for j in range(0, len(per_device_packed_query_responses), args.per_device_train_batch_size):
                    micro_range = b_inds[j : j + args.per_device_train_batch_size]
                    collated_query_responses.append(
                        collate_fn(
                            [per_device_packed_query_responses[idx] for idx in micro_range], tokenizer.pad_token_id
                        )
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
                    collated_dones.append(collate_fn([per_device_packed_dones[idx] for idx in micro_range], 0))
                    collated_rewards.append(collate_fn([per_device_packed_rewards[idx] for idx in micro_range], 0))
                collated_data.append(
                    {
                        "collated_query_responses": collated_query_responses,
                        "collated_attention_masks": collated_attention_masks,
                        "collated_position_ids": collated_position_ids,
                        "collated_response_masks": collated_response_masks,
                        "collated_dones": collated_dones,
                        "collated_rewards": collated_rewards,
                    }
                )

        # Create a result package with metrics and data
        max_possible_score = 0
        if args.apply_verifiable_reward:
            max_possible_score += args.verification_reward
        if args.apply_r1_style_format_reward and args.additive_format_reward:
            max_possible_score += args.r1_style_format_reward
        sequence_lengths = np.array([len(response) for response in responses])
        sequence_length_solved = (
            np.array([]) if np.all(scores == 0) else np.array(sequence_lengths[scores == max_possible_score])
        )
        sequence_length_unsolved = (
            np.array([]) if np.all(scores == max_possible_score) else np.array(sequence_lengths[scores == 0])
        )
        metrics = {
            "scores": np.array(scores).mean(),
            "real_batch_size_ratio": real_batch_size_ratio,
            "packed_ratio": len(packed_sequences.query_responses) / len(responses),
            "val/sequence_lengths": sequence_lengths.mean(),
            "val/sequence_lengths_min": sequence_lengths.min(),
            "val/sequence_lengths_max": sequence_lengths.max(),
            "val/sequence_lengths_unsolved": (
                0 if len(sequence_length_unsolved) == 0 else sequence_length_unsolved.mean()
            ),
            "val/sequence_lengths_solved": 0 if len(sequence_length_solved) == 0 else sequence_length_solved.mean(),
            "val/sequence_lengths_unsolved_hist": sequence_length_unsolved,
            "val/sequence_lengths_solved_hist": sequence_length_solved,
            "val/stop_rate": stop_rate,
            "val/num_calls_rate": np.array(num_calls).mean(),
            "val/timeouts_rate": np.array(timeouts).mean(),
            "val/tool_errors_rate": np.array([len(item) > 0 for item in tool_errors]).mean(),
            "val/good_outputs_rate": np.array(good_outputs).mean(),
            "val/tool_runtimes_rate": np.array(tool_runtimes).mean(),
            "val/tool_calleds_rate": np.array(tool_calleds).mean(),
            **reward_metrics,
        }

        if args.save_traces:
            traces = {
                "scores": scores.tolist(),
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
            }
        )


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
    if args.local_eval_freq is not None:
        raise ValueError("local_eval_freq should not be set manually; it will be computed automatically")
    args.local_eval_freq = max(1, args.num_training_steps // args.num_evals)
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
    if args.cache_dataset_only:
        return

    # ------------------------------------------------------------
    # Runtime setups and quick logging
    pprint([args, model_config])

    # ------------------------------------------------------------
    # Create the model and optimizer
    ray.init(dashboard_host="0.0.0.0")  # enable debugging from a different machine (e.g., phobos)
    pg = None
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    inits = []
    policy_group = ModelGroup(pg, PolicyTrainerRayProcess, args.num_learners_per_node, args.single_gpu_mode)
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
                from open_instruct.search_utils.search_tool import SearchTool

                tool = SearchTool(
                    start_str="<query>",
                    end_str="</query>",
                    api_endpoint=args.search_api_endpoint,
                    number_documents_to_search=args.number_documents_to_search,
                )
                tool_objects[tool.end_str] = tool
            elif tool.lower() == "code":
                from open_instruct.tool_utils.tool_vllm import PythonCodeTool

                tool = PythonCodeTool(start_str="<code>", end_str="</code>", api_endpoint=args.code_tool_api_endpoint)
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
        top_p=0.98,  # prevent rare out-of-vocab tokens with qwen
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        n=args.num_samples_per_prompt_rollout,
        stop=stop_strings,
    )
    eval_generation_config = SamplingParams(
        temperature=0.0,
        top_p=0.98,  # prevent rare out-of-vocab tokens with qwen
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
        eval_dataset_names = eval_dataset[:num_eval_samples][VERIFIER_SOURCE_KEY]
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
            args.local_eval_freq,
            resume_training_step,
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
    datasets_next = data_next[VERIFIER_SOURCE_KEY]
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
                    datasets_next = data_next[VERIFIER_SOURCE_KEY]
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
                    datasets_next = data_next[VERIFIER_SOURCE_KEY]
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
                            value_num_mini_batches=args.value_num_mini_batches,
                            gamma=args.gamma,
                            lam=args.lam,
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

                if args.save_freq > 0 and training_step % args.save_freq == 0:
                    with Timer("[Main Thread] 🗡️ Saving model"):
                        checkpoint_dir = f"{args.output_dir}_checkpoints"
                        step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
                        print(f"Saving model at step {training_step} to {step_dir}")
                        ray.get(
                            [
                                policy_group.models[i].save_model.remote(step_dir, tc.chat_template_name, tokenizer)
                                for i in range(args.world_size)
                            ]
                        )
                        if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                            leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
                            for i in range(args.world_size):
                                policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                                    step_dir, leaderboard_name, wandb_url, training_step
                                )

            if len(update_ref_policy_future) > 0:
                with Timer("[Main Thread] 🔃 Updating reference policy"):
                    ray.get(update_ref_policy_future)

            # ------------------------------------------------------------------------------------------------
            # Optionally evaluate the model
            try:
                # timeout 0.01 if this is the last training step or we're not evaluating
                # otherwise, wait to get the last evaluation generations (long timeout just in case)
                timeout = 0.01 if (training_step < args.num_training_steps or args.local_eval_freq < 0) else 100
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
                        None,  # queries not available for evaluation
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
                print_rich_single_line_metrics(eval_metrics)
                for key, value in eval_metrics.items():
                    writer.add_scalar(key, value, episode)
                table = {}
                table["prompt"] = tokenizer.batch_decode(eval_prompt_token_ids)
                table["response"] = eval_decoded_responses
                table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
                table["scores"] = eval_scores
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
            ray.get(
                [
                    policy_group.models[i].save_model.remote(args.output_dir, tc.chat_template_name, tokenizer)
                    for i in range(args.world_size)
                ]
            )
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

    accelerator = Namespace()
    accelerator.is_main_process = True  # hack
    if args.push_to_hub:
        print("Pushing model to hub")
        push_folder_to_hub(accelerator, args.output_dir, args.hf_repo_id, args.hf_repo_revision)


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
    ) -> List[float]:
        num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = infos
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
                    ground_truths,
                    datasets,
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

    main(args, tokenizer_config, model_config, reward_fn)
