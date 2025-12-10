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
"""
On-Policy Distillation Trainer

This module implements on-policy distillation as described in:
https://thinkingmachines.ai/blog/on-policy-distillation/

The key idea is to:
1. Sample trajectories from the student model
2. Get per-token logprobs from an external teacher model (via OpenAI-compatible API)
3. Compute reverse KL divergence: KL(π_student || π_teacher) = log π_student - log π_teacher
4. Use negative reverse KL as per-token advantage for RL training

This differs from standard distillation which trains on teacher-generated sequences.
On-policy distillation learns from the student's own outputs, combining on-policy
relevance with dense per-token supervision.

TODO: Support different tokenizers between student and teacher. Currently assumes
the same tokenizer is used for both models. To support different tokenizers:
1. Decode student tokens to text
2. Re-encode with teacher tokenizer
3. Get teacher logprobs
4. Align logprobs back to student tokens (handle different tokenization boundaries)
"""

# isort: off
import contextlib
import os
import threading
import time
from argparse import Namespace
from concurrent import futures
from dataclasses import asdict
from queue import Empty, Queue

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed

from open_instruct import utils

# isort: on
import asyncio
import dataclasses
import json
import math
import socket
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

import aiohttp
import backoff
import numpy as np
import ray
import torch
import torch.distributed as dist
import wandb
from huggingface_hub import HfApi
from openai import AsyncOpenAI
from ray.util import queue as ray_queue
from ray.util.placement_group import placement_group
from rich.pretty import pprint
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, get_scheduler
from transformers.integrations import HfDeepSpeedConfig

from open_instruct import data_loader as data_loader_lib
from open_instruct import logger_utils, vllm_utils
from open_instruct.data_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.dataset_transformation import (
    INPUT_IDS_PROMPT_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.grpo_fast import ModelGroup
from open_instruct.model_utils import (
    Batch,
    ModelConfig,
    disable_dropout_in_model,
    entropy_from_logits,
    log_softmax_and_gather,
    print_rich_single_line_metrics,
    push_folder_to_hub,
)
from open_instruct.rl_utils import PackedSequences, Timer, masked_mean, pack_sequences
from open_instruct.utils import (
    ArgumentParserPlus,
    RayProcess,
    _z3_params_to_fetch,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_update_beaker_description,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    ray_get_with_progress,
)

logger = logger_utils.setup_logger(__name__)

INVALID_LOGPROB = 1.0


@dataclass
class DistillationArgs:
    """Arguments for on-policy distillation training."""

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
    exp_name: str = "distillation"
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
    """RUNTIME VALUE: The number of processes (GPUs) to use"""
    num_training_steps: int | None = None
    """RUNTIME VALUE: The number of training_steps to train"""
    local_eval_every: int = 100
    """Run evaluation after this many training steps."""
    save_freq: int = 200
    """How many train steps to save the model"""
    allow_world_padding: bool = False
    """Whether to allow world padding."""
    backend_timeout: int = 120
    """Timeout for inference/training backends in minutes."""

    # Generation
    response_length: int = 256
    """the length of the response"""
    temperature: float = 0.7
    """the sampling temperature"""
    num_unique_prompts_rollout: int = 16
    """The number of unique prompts during rollout"""
    num_samples_per_prompt_rollout: int = 1
    """the number of samples to generate per prompt during rollout"""
    stop_strings: list[str] | None = None
    """List of strings that stop the generation when they are generated."""

    # Algorithm - On-Policy Distillation specific
    async_steps: int = 1
    """Number of steps ahead to generate responses."""
    num_epochs: int = 1
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    beta: float = 1.0
    """Weight for the KL divergence loss (higher = more distillation influence)"""
    clip_lower: float = 0.2
    """the lower clip range"""
    clip_higher: float = 0.2
    """the higher clip range"""
    pack_length: int = 512
    """the length of the pack"""
    loss_denominator: str = "token"
    """Optional constant denominator for masked_mean."""
    use_vllm_logprobs: bool = True
    """whether to use vLLM's logprobs for training instead of calculating them via forward pass"""
    record_entropy: bool = False
    """whether to record the entropy of the policy during training."""

    # Teacher model configuration
    teacher_api_base: str = "http://localhost:8000/v1"
    """Base URL for the teacher model's OpenAI-compatible API"""
    teacher_api_key: str = "EMPTY"
    """API key for the teacher model (use 'EMPTY' for local vLLM servers)"""
    teacher_model_name: str | None = None
    """Model name to use in API calls (if None, will be auto-detected)"""
    teacher_max_concurrent_requests: int = 64
    """Maximum number of concurrent requests to the teacher API"""
    teacher_timeout: float = 120.0
    """Timeout in seconds for teacher API requests"""
    teacher_temperature: float = 1.0
    """Temperature for teacher model (usually 1.0 for logprob computation)"""

    # Ray
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node"""
    num_learners_per_node: list[int] = field(default_factory=lambda: [1])
    """number of GPU deepspeed learners per node"""
    vllm_num_engines: int = 1
    """number of vLLM Engines"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine"""
    vllm_enforce_eager: bool = False
    """whether to enforce eager mode for vLLM"""
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
    """the deepspeed zpg value"""
    deepspeed_offload_param: bool = False
    """whether to offload parameters to CPU"""
    deepspeed_offload_optimizer: bool = False
    """whether to offload optimizer states to CPU"""
    gather_whole_model: bool = True
    """whether to gather the whole model to boardcast"""

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
    """The id of the saved model in the Hugging Face Hub"""
    hf_repo_revision: str | None = None
    """The revision of the saved model in the Hugging Face Hub"""
    hf_repo_url: str | None = None
    """The url of the saved model in the Hugging Face Hub"""
    output_dir: str = "output"
    """Where to save the model"""
    save_traces: bool = False
    """Whether to save learning data traces"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""

    def __post_init__(self):
        self.loss_denominator = utils.get_denominator(self.loss_denominator)
        assert self.num_samples_per_prompt_rollout > 0, "Number of samples per prompt must be greater than 0!"


class TeacherClient:
    """Async client for getting logprobs from an external teacher model via OpenAI-compatible API.

    TODO: Support different tokenizers between student and teacher. Currently assumes
    the same tokenizer is used for both models.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str = "EMPTY",
        model_name: str | None = None,
        max_concurrent_requests: int = 64,
        timeout: float = 120.0,
        temperature: float = 1.0,
    ):
        self.client = AsyncOpenAI(base_url=api_base, api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.temperature = temperature
        self._model_name_resolved = False

    async def _resolve_model_name(self) -> str:
        """Auto-detect model name from the API if not specified."""
        if self.model_name is not None:
            return self.model_name

        if not self._model_name_resolved:
            models = await self.client.models.list()
            if models.data:
                self.model_name = models.data[0].id
                self._model_name_resolved = True
                logger.info(f"Auto-detected teacher model: {self.model_name}")
            else:
                raise ValueError("Could not auto-detect teacher model name and none was provided")

        return self.model_name

    @backoff.on_exception(
        backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError, Exception), max_tries=3, max_time=300
    )
    async def get_logprobs_for_sequence(self, prompt_tokens: list[int], response_tokens: list[int]) -> list[float]:
        """Get per-token logprobs from the teacher for a given prompt+response sequence.

        Args:
            prompt_tokens: Token IDs for the prompt
            response_tokens: Token IDs for the response (we want logprobs for these)

        Returns:
            List of logprobs, one per response token
        """
        async with self.semaphore:
            model_name = await self._resolve_model_name()

            # Combine prompt and response - we'll ask for logprobs on the full sequence
            # but only extract the response portion
            full_sequence = prompt_tokens + response_tokens

            try:
                # Use completions API with echo to get logprobs for existing tokens
                response = await self.client.completions.create(
                    model=model_name,
                    prompt=full_sequence,
                    max_tokens=0,  # We don't want to generate new tokens
                    temperature=self.temperature,
                    logprobs=1,  # Return top-1 logprob for each token
                    echo=True,  # Echo back the prompt with logprobs
                )

                # Extract logprobs for the response portion
                if response.choices and response.choices[0].logprobs:
                    all_logprobs = response.choices[0].logprobs.token_logprobs
                    # The first token has no logprob (it's the start)
                    # Response tokens start at len(prompt_tokens)
                    if all_logprobs is not None:
                        # Skip prompt tokens and get response token logprobs
                        response_logprobs = all_logprobs[len(prompt_tokens) :]
                        # Handle None values (first token)
                        response_logprobs = [lp if lp is not None else 0.0 for lp in response_logprobs]
                        return response_logprobs

                # Fallback: return zeros if we couldn't get logprobs
                logger.warning("Could not get logprobs from teacher, returning zeros")
                return [0.0] * len(response_tokens)

            except Exception as e:
                logger.error(f"Error getting teacher logprobs: {e}")
                raise

    async def get_batch_logprobs(
        self, prompt_tokens_batch: list[list[int]], response_tokens_batch: list[list[int]]
    ) -> list[list[float]]:
        """Get logprobs for a batch of sequences.

        Args:
            prompt_tokens_batch: List of prompt token sequences
            response_tokens_batch: List of response token sequences

        Returns:
            List of logprob lists, one per sequence
        """
        tasks = [
            self.get_logprobs_for_sequence(prompt, response)
            for prompt, response in zip(prompt_tokens_batch, response_tokens_batch)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to get logprobs for sequence {i}: {result}")
                final_results.append([0.0] * len(response_tokens_batch[i]))
            else:
                final_results.append(result)

        return final_results


def compute_reverse_kl_advantages(
    student_logprobs: np.ndarray, teacher_logprobs: np.ndarray, response_masks: np.ndarray
) -> np.ndarray:
    """Compute per-token reverse KL divergence as advantages.

    The reverse KL is: KL(π_student || π_teacher) = log π_student - log π_teacher

    We use the NEGATIVE of this as the advantage, because we want to minimize KL
    (i.e., maximize -KL).

    Args:
        student_logprobs: Student model logprobs [batch, seq_len]
        teacher_logprobs: Teacher model logprobs [batch, seq_len]
        response_masks: Binary mask for response tokens [batch, seq_len]

    Returns:
        Per-token advantages [batch, seq_len]
    """
    # Reverse KL = log(student) - log(teacher)
    reverse_kl = student_logprobs - teacher_logprobs

    # Advantage = -reverse_kl (we want to minimize KL, so maximize -KL)
    advantages = -reverse_kl

    # Mask out non-response tokens
    advantages = advantages * response_masks

    return advantages


def to_device_inplace(tensors: list[torch.Tensor], device: torch.device):
    """Move a list of tensors to a device in place."""
    for i, t in enumerate(tensors):
        if isinstance(t, torch.Tensor):
            tensors[i] = t.to(device)


@dataclass
class DistillationCollatedBatchData:
    """Container for collated batch data with teacher logprobs for distillation."""

    query_responses: list[torch.Tensor]
    attention_masks: list[torch.Tensor]
    position_ids: list[torch.Tensor]
    advantages: list[torch.Tensor]  # Per-token -reverse_KL advantages
    response_masks: list[torch.Tensor]
    vllm_logprobs: list[torch.Tensor]  # Student logprobs from generation
    teacher_logprobs: list[torch.Tensor]  # Teacher logprobs

    def __getitem__(self, idx: int | slice) -> "DistillationCollatedBatchData":
        return DistillationCollatedBatchData(**{f.name: getattr(self, f.name)[idx] for f in dataclasses.fields(self)})

    def __len__(self) -> int:
        return len(self.query_responses)


class PolicyTrainerRayProcess(RayProcess):
    """Ray process for policy training with on-policy distillation."""

    def __init__(
        self,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str,
        master_port: int,
        model_config: ModelConfig,
        tokenizer: PreTrainedTokenizer,
        args: DistillationArgs,
    ):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.args = args
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(self.device)
        self.local_metrics = utils.MetricsTracker(device=self.device)

    @classmethod
    def from_pretrained(
        cls,
        world_size: int,
        rank: int,
        local_rank: int,
        master_addr: str,
        master_port: int,
        model_config: ModelConfig,
        tokenizer: PreTrainedTokenizer,
        args: DistillationArgs,
    ) -> tuple["PolicyTrainerRayProcess", int]:
        """Create a PolicyTrainerRayProcess from a pretrained model."""
        trainer = cls(world_size, rank, local_rank, master_addr, master_port, model_config, tokenizer, args)
        optimization_steps_done = trainer._initialize_model(model_config, args)
        return trainer, optimization_steps_done

    def _initialize_model(self, model_config: ModelConfig, args: DistillationArgs) -> int:
        """Initialize model, optimizer, and scheduler."""
        # Set up DeepSpeed config
        ds_config = get_train_ds_config(
            offload_optimizer=args.deepspeed_offload_optimizer,
            offload_param=args.deepspeed_offload_param,
            stage=args.deepspeed_stage,
            zpg=args.deepspeed_zpg,
            bf16=True,
            per_device_train_batch_size=args.per_device_train_batch_size,
            grad_accum=max(1, args.num_mini_batches),
        )

        # Required for DeepSpeed integration with HuggingFace
        _hf_ds_config = HfDeepSpeedConfig(ds_config)  # noqa: F841

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
        disable_dropout_in_model(self.model)

        if model_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Setup optimizer
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model, args.weight_decay, set_on_bias_and_norm=args.set_weight_decay_on_bias_and_norm
        )

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)

        # Calculate training steps
        num_training_steps = args.num_training_steps or args.total_episodes

        # Setup scheduler
        warm_up_steps = args.warm_up_steps
        if args.warmup_ratio > 0:
            warm_up_steps = int(num_training_steps * args.warmup_ratio)

        self.scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_training_steps,
        )

        # Initialize with DeepSpeed
        self.model, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model, optimizer=optimizer, config=ds_config, lr_scheduler=self.scheduler
        )

        self.model.train()
        return 0

    def forward(
        self,
        model: PreTrainedModel,
        query_response: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_token_id: int,
        temperature: float,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to compute logprobs."""
        # Replace pad tokens with 0s to avoid index out of bounds
        padding_mask = query_response != pad_token_id
        input_ids = torch.masked_fill(query_response, ~padding_mask, 0)

        output = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )

        logits = output.logits
        logits /= temperature + 1e-7
        logprob = log_softmax_and_gather(logits, input_ids[:, 1:])

        entropy = None
        if return_entropy:
            with torch.no_grad():
                entropy = entropy_from_logits(logits)

        return logprob, entropy

    def train(self, data_BT: DistillationCollatedBatchData, pad_token_id: int) -> dict[str, float]:
        """Train on a batch using on-policy distillation.

        The loss is the policy gradient loss using per-token negative reverse KL
        as the advantage (discount=0, immediate token optimization).
        """
        args = self.args

        # Move data to device
        for f in dataclasses.fields(data_BT):
            to_device_inplace(getattr(data_BT, f.name), self.device)
        data_BT.response_masks = [mask.bool() for mask in data_BT.response_masks]

        num_samples = len(data_BT)
        accumulation_steps = max(math.ceil(num_samples / args.num_mini_batches - 0.5), 1)
        leftover = num_samples % accumulation_steps
        if leftover > 0:
            data_BT = data_BT[:-leftover]
            logger.warning(f"{leftover} samples dropped due to batch size {args.num_mini_batches}")

        # Compute old logprobs if needed for multiple mini-batches
        old_logprobs_BT: list[torch.Tensor | None] = [None for _ in range(len(data_BT.query_responses))]
        if args.num_mini_batches > 1 and not args.use_vllm_logprobs:
            with Timer("Old logprobs Calculation", noop=self.rank != 0), torch.no_grad():
                for i in range(len(data_BT.query_responses)):
                    logprob, _ = self.forward(
                        self.model,
                        data_BT.query_responses[i],
                        data_BT.attention_masks[i],
                        data_BT.position_ids[i],
                        pad_token_id,
                        args.temperature,
                    )
                    old_logprobs_BT[i] = logprob

        local_step = 0
        num_samples = len(data_BT.query_responses)

        # Track loss stats
        loss_stats_B = {
            "reverse_kl": torch.zeros(num_samples),
            "pg_clipfrac": torch.zeros(num_samples),
            "pg_loss": torch.zeros(num_samples),
            "loss": torch.zeros(num_samples),
            "ratio": torch.zeros(num_samples),
            "entropy": torch.zeros(num_samples),
        }

        # Calculate token counts for loss normalization
        if args.loss_denominator == "token":
            local_counts = [mask[:, 1:].sum().float() for mask in data_BT.response_masks]
            if local_counts:
                counts_tensor = torch.stack(local_counts)
                dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
                accumulation_token_counts = {}
                for i, count in enumerate(counts_tensor):
                    group_idx = i // accumulation_steps
                    key = int(group_idx * accumulation_steps)
                    accumulation_token_counts[key] = accumulation_token_counts.get(key, 0.0) + count.item()
        else:
            accumulation_token_counts = {
                int(group_idx * accumulation_steps): args.loss_denominator
                for group_idx in range((num_samples // accumulation_steps) + 1)
            }

        with Timer("[Training] Loss calculation", noop=self.rank != 0):
            for epoch_idx in range(args.num_epochs):
                for i in range(num_samples):
                    response_mask_BT = data_BT.response_masks[i][:, 1:]

                    # Get loss denominator
                    batch_start = (i // accumulation_steps) * accumulation_steps
                    loss_denominator = accumulation_token_counts.get(batch_start, 1.0)

                    # Forward pass
                    local_logprobs_BT, entropy_BT = self.forward(
                        self.model,
                        data_BT.query_responses[i],
                        data_BT.attention_masks[i],
                        data_BT.position_ids[i],
                        pad_token_id,
                        args.temperature,
                        return_entropy=args.record_entropy,
                    )
                    local_logprobs_BT = torch.masked_fill(local_logprobs_BT, ~response_mask_BT, INVALID_LOGPROB)

                    # Get vLLM logprobs
                    vllm_logprobs_BT = data_BT.vllm_logprobs[i][:, 1:]
                    vllm_logprobs_BT = torch.masked_fill(vllm_logprobs_BT, ~response_mask_BT, INVALID_LOGPROB)
                    vllm_logprobs_BT = torch.nan_to_num(vllm_logprobs_BT, nan=INVALID_LOGPROB)

                    # Determine old logprobs
                    if args.num_mini_batches > 1:
                        old_logprob_BT = vllm_logprobs_BT if args.use_vllm_logprobs else old_logprobs_BT[i]
                    else:
                        with torch.no_grad():
                            if epoch_idx == 0:
                                if args.use_vllm_logprobs:
                                    old_logprobs_BT[i] = vllm_logprobs_BT
                                else:
                                    old_logprobs_BT[i] = local_logprobs_BT.detach()
                            old_logprob_BT = old_logprobs_BT[i]

                    # Calculate policy loss with per-token KL advantages
                    logprobs_diff_BT = local_logprobs_BT - old_logprob_BT
                    ratio_BT = torch.exp(logprobs_diff_BT)

                    # Advantages are pre-computed negative reverse KL
                    advantages_BT = data_BT.advantages[i][:, 1:]

                    pg_losses_BT = -advantages_BT * ratio_BT
                    pg_losses2_BT = -advantages_BT * torch.clamp(
                        ratio_BT, 1.0 - args.clip_lower, 1.0 + args.clip_higher
                    )

                    pg_loss_max_BT = torch.max(pg_losses_BT, pg_losses2_BT)

                    # Final loss
                    loss = masked_mean(pg_loss_max_BT, response_mask_BT, None, loss_denominator)

                    # Scale by world size (already accounted for via tokens)
                    if dist.is_available() and dist.is_initialized():
                        loss *= dist.get_world_size()

                    # Backward pass
                    torch.cuda.empty_cache()
                    self.model.backward(loss)

                    if (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                    local_step += 1

                    # Track stats
                    with torch.no_grad():
                        # Compute reverse KL for logging
                        teacher_logprobs_BT = data_BT.teacher_logprobs[i][:, 1:]
                        reverse_kl_BT = local_logprobs_BT - teacher_logprobs_BT
                        loss_stats_B["reverse_kl"][i] = masked_mean(reverse_kl_BT, response_mask_BT)

                        loss_stats_B["pg_clipfrac"][i] = masked_mean(
                            (pg_losses2_BT > pg_losses_BT).float(), response_mask_BT
                        )
                        loss_stats_B["pg_loss"][i] = masked_mean(pg_loss_max_BT, response_mask_BT)
                        loss_stats_B["loss"][i] = loss
                        loss_stats_B["ratio"][i] = masked_mean(ratio_BT, response_mask_BT)
                        if args.record_entropy and entropy_BT is not None:
                            loss_stats_B["entropy"][i] = masked_mean(entropy_BT, response_mask_BT).float()

        # Aggregate metrics
        with torch.no_grad():
            self.local_metrics["objective/reverse_kl_avg"] = loss_stats_B["reverse_kl"].mean()
            self.local_metrics["loss/policy_avg"] = loss_stats_B["pg_loss"].mean()
            self.local_metrics["loss/total_avg"] = loss_stats_B["loss"].mean()
            self.local_metrics["policy/clipfrac_avg"] = loss_stats_B["pg_clipfrac"].mean()
            self.local_metrics["val/ratio"] = loss_stats_B["ratio"].mean()
            self.local_metrics["val/ratio_var"] = loss_stats_B["ratio"].var()
            if args.record_entropy:
                self.local_metrics["policy/entropy_avg"] = loss_stats_B["entropy"].mean()
            self.local_metrics["lr"] = self.scheduler.get_last_lr()[0]

            return self.local_metrics.get_metrics_list()

    def setup_model_update_group(self, vllm_engines):
        """Setup process group for model weight synchronization with vLLM engines."""
        self.vllm_engines = vllm_engines
        if self.rank == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]

            vllm_num_engines = self.args.vllm_num_engines
            vllm_tensor_parallel_size = self.args.vllm_tensor_parallel_size
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
        """Broadcast model weights to vLLM engines."""
        torch.cuda.empty_cache()
        model = self.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        refss = []

        if self.args.gather_whole_model:
            with deepspeed.zero.GatheredParameters(model.parameters(), enabled=self.args.deepspeed_stage == 3):
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
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
        else:
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

        all_refs = []
        if torch.distributed.get_rank() == 0:
            all_refs.extend(refss)
        return all_refs

    def save_model(self, output_dir: str, tokenizer: PreTrainedTokenizer) -> None:
        """Save the model to disk."""
        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # Gather and save model weights
        if self.args.deepspeed_stage == 3:
            state_dict = _z3_params_to_fetch(self.model.module.named_parameters())
            with deepspeed.zero.GatheredParameters(state_dict, enabled=True):
                if self.rank == 0:
                    self.model.module.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
        else:
            if self.rank == 0:
                self.model.module.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)


def prepare_collated_data_for_workers(
    packed_sequences: PackedSequences,
    world_size: int,
    per_device_train_batch_size: int,
    pad_token_id: int,
    teacher_logprobs_packed: list[torch.Tensor],
) -> list[DistillationCollatedBatchData]:
    """Prepare collated batch data for distributed workers."""
    num_sequences = len(packed_sequences.query_responses)
    sequences_per_worker = num_sequences // world_size

    collated_data = []
    for worker_idx in range(world_size):
        start_idx = worker_idx * sequences_per_worker
        end_idx = start_idx + sequences_per_worker

        worker_query_responses = []
        worker_attention_masks = []
        worker_position_ids = []
        worker_advantages = []
        worker_response_masks = []
        worker_vllm_logprobs = []
        worker_teacher_logprobs = []

        for i in range(start_idx, end_idx, per_device_train_batch_size):
            batch_end = min(i + per_device_train_batch_size, end_idx)

            # Stack tensors for this mini-batch
            qr_batch = [torch.tensor(packed_sequences.query_responses[j]) for j in range(i, batch_end)]
            am_batch = [torch.tensor(packed_sequences.attention_masks[j]) for j in range(i, batch_end)]
            pi_batch = [torch.tensor(packed_sequences.position_ids[j]) for j in range(i, batch_end)]
            adv_batch = [packed_sequences.advantages[j] for j in range(i, batch_end)]
            rm_batch = [torch.tensor(packed_sequences.response_masks[j]) for j in range(i, batch_end)]
            vl_batch = [torch.tensor(packed_sequences.vllm_logprobs[j]) for j in range(i, batch_end)]
            tl_batch = [teacher_logprobs_packed[j] for j in range(i, batch_end)]

            # Pad to same length within batch
            max_len = max(qr.shape[0] for qr in qr_batch)

            def pad_tensor(t, max_len, pad_value=0):
                if t.shape[0] < max_len:
                    pad = torch.full((max_len - t.shape[0],), pad_value, dtype=t.dtype)
                    return torch.cat([t, pad])
                return t

            qr_padded = torch.stack([pad_tensor(qr, max_len, pad_token_id) for qr in qr_batch])
            am_padded = torch.stack([pad_tensor(am, max_len, 0) for am in am_batch])
            pi_padded = torch.stack([pad_tensor(pi, max_len, 0) for pi in pi_batch])
            adv_padded = torch.stack([pad_tensor(adv, max_len, 0.0) for adv in adv_batch])
            rm_padded = torch.stack([pad_tensor(rm, max_len, 0) for rm in rm_batch])
            vl_padded = torch.stack([pad_tensor(vl, max_len, float("nan")) for vl in vl_batch])
            tl_padded = torch.stack([pad_tensor(tl, max_len, 0.0) for tl in tl_batch])

            worker_query_responses.append(qr_padded)
            worker_attention_masks.append(am_padded)
            worker_position_ids.append(pi_padded)
            worker_advantages.append(adv_padded)
            worker_response_masks.append(rm_padded)
            worker_vllm_logprobs.append(vl_padded)
            worker_teacher_logprobs.append(tl_padded)

        collated_data.append(
            DistillationCollatedBatchData(
                query_responses=worker_query_responses,
                attention_masks=worker_attention_masks,
                position_ids=worker_position_ids,
                advantages=worker_advantages,
                response_masks=worker_response_masks,
                vllm_logprobs=worker_vllm_logprobs,
                teacher_logprobs=worker_teacher_logprobs,
            )
        )

    return collated_data


def accumulate_inference_batches(
    inference_results_Q: ray_queue.Queue, num_prompts: int, timeout: float = 300.0
) -> tuple[GenerationResult, Batch] | None:
    """Accumulate generation results from the vLLM queue."""
    all_responses = []
    all_finish_reasons = []
    all_masks = []
    all_logprobs = []
    all_queries = []
    all_decoded_responses = []

    results_received = 0
    start_time = time.time()

    while results_received < num_prompts:
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time <= 0:
            logger.warning(f"Timeout waiting for inference results. Got {results_received}/{num_prompts}")
            break

        try:
            result: GenerationResult = inference_results_Q.get(timeout=min(remaining_time, 30.0))
            all_responses.extend(result.responses)
            all_finish_reasons.extend(result.finish_reasons)
            all_masks.extend(result.masks)
            if result.logprobs:
                all_logprobs.extend(result.logprobs)
            results_received += 1
        except Empty:
            continue

    if not all_responses:
        return None

    # Combine into single GenerationResult
    combined_result = GenerationResult(
        responses=all_responses,
        finish_reasons=all_finish_reasons,
        masks=all_masks,
        request_info=RequestInfo(
            num_calls=[], timeouts=[], tool_errors=[], tool_outputs=[], tool_runtimes=[], tool_calleds=[]
        ),
        dataset_index=None,
        prompt_id=None,
        token_statistics=TokenStatistics(num_prompt_tokens=0, num_response_tokens=0, generation_time=0.0),
        logprobs=all_logprobs,
    )

    batch = Batch(
        queries=all_queries,
        ground_truths=[],
        datasets=[],
        raw_queries=[],
        decoded_responses=all_decoded_responses,
        indices=None,
        scores=[],
    )

    return combined_result, batch


def data_preparation_thread(
    inference_results_Q: ray_queue.Queue,
    param_prompt_Q: ray_queue.Queue,
    packed_sequences_Q: Queue,
    args: DistillationArgs,
    tokenizer: PreTrainedTokenizer,
    num_training_steps: int,
    generation_config,
    data_loader: data_loader_lib.HFDataLoader,
    teacher_client: TeacherClient,
):
    """Thread that processes generation results and prepares training data."""
    for training_step in range(1, num_training_steps + 1):
        with Timer(f"[Data Preparation Thread] Step {training_step}") as timer:
            # Collect generation results
            all_responses = []
            all_masks = []
            all_logprobs = []
            all_finish_reasons = []

            for _ in range(args.num_unique_prompts_rollout):
                try:
                    result: GenerationResult = inference_results_Q.get(timeout=300)
                    all_responses.extend(result.responses)
                    all_finish_reasons.extend(result.finish_reasons)
                    all_masks.extend(result.masks)
                    if result.logprobs:
                        all_logprobs.extend(result.logprobs)
                except Empty:
                    logger.warning("[Data Preparation Thread] Timeout waiting for inference results")
                    continue

            if not all_responses:
                logger.warning("[Data Preparation Thread] No responses received, skipping step")
                packed_sequences_Q.put({"collated_data": None, "metrics": {}, "num_new_tokens": 0})
                continue

            # Get teacher logprobs for all sequences
            # Need to extract prompts and responses for teacher API
            prompt_tokens_batch = []
            response_tokens_batch = []

            # We need to reconstruct prompts - for now we'll use the response tokens only
            # TODO: Get actual prompt tokens from the generation result
            for response in all_responses:
                # For now, use empty prompt (teacher will compute logprobs for response only)
                # In practice, you'd want to store and retrieve the actual prompts
                prompt_tokens_batch.append([])
                response_tokens_batch.append(response)

            # Get teacher logprobs asynchronously
            with Timer("[Data Preparation Thread] Getting teacher logprobs"):
                teacher_logprobs_batch = asyncio.run(
                    teacher_client.get_batch_logprobs(prompt_tokens_batch, response_tokens_batch)
                )

            # Compute per-token reverse KL advantages
            # advantage = -(student_logprob - teacher_logprob) = teacher_logprob - student_logprob
            advantages = []
            for student_lp, teacher_lp in zip(all_logprobs, teacher_logprobs_batch):
                # Ensure same length
                min_len = min(len(student_lp), len(teacher_lp))
                student_lp = student_lp[:min_len]
                teacher_lp = teacher_lp[:min_len]

                # Compute advantage: -KL = -(log_student - log_teacher) = log_teacher - log_student
                adv = [t - s for s, t in zip(student_lp, teacher_lp)]
                advantages.append(sum(adv) / len(adv) if adv else 0.0)  # Use mean as scalar advantage

            # Pack sequences
            with Timer("[Data Preparation Thread] Packing sequences"):
                # Create dummy queries for packing (will be updated with actual queries)
                dummy_queries = [[tokenizer.bos_token_id]] * len(all_responses)

                packed_sequences = pack_sequences(
                    queries=dummy_queries,
                    responses=all_responses,
                    masks=all_masks,
                    pack_length=args.pack_length,
                    pad_token_id=tokenizer.pad_token_id,
                    vllm_logprobs=all_logprobs,
                )

                # Create lookup array for vectorized advantage assignment
                lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
                lookup_advantages[1:] = advantages
                packed_advantages = [
                    torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
                    for packed_mask in packed_sequences.response_masks
                ]
                packed_sequences.advantages = packed_advantages

                # Create teacher logprobs in packed format
                # For each packed sequence, map teacher logprobs using response mask
                teacher_logprobs_packed = []
                for packed_mask in packed_sequences.response_masks:
                    # packed_mask contains indices (1-indexed) into the original responses
                    packed_teacher_lp = []
                    for idx in packed_mask:
                        if idx == 0:  # Padding
                            packed_teacher_lp.append(0.0)
                        else:
                            # idx is 1-indexed, so idx-1 gives the response index
                            resp_idx = idx - 1
                            if resp_idx < len(teacher_logprobs_batch):
                                # Use mean teacher logprob for this response
                                tlp = teacher_logprobs_batch[resp_idx]
                                packed_teacher_lp.append(np.mean(tlp) if tlp else 0.0)
                            else:
                                packed_teacher_lp.append(0.0)
                    teacher_logprobs_packed.append(torch.tensor(packed_teacher_lp, dtype=torch.float32))

            # Prepare collated data for workers
            collated_data = prepare_collated_data_for_workers(
                packed_sequences,
                args.world_size,
                args.per_device_train_batch_size,
                tokenizer.pad_token_id,
                teacher_logprobs_packed,
            )

            num_new_tokens = sum(len(seq) for seq in packed_sequences.query_responses)

            # Compute metrics
            sequence_lengths = np.array([len(response) for response in all_responses])
            metrics = {
                "val/sequence_lengths": sequence_lengths.mean(),
                "val/sequence_lengths_min": sequence_lengths.min() if len(sequence_lengths) > 0 else 0,
                "val/sequence_lengths_max": sequence_lengths.max() if len(sequence_lengths) > 0 else 0,
                "val/advantages_mean": np.mean(advantages),
                "val/num_responses": len(all_responses),
                "time/data_preparation": timer.duration,
            }

            # Send prompts for next batch
            for _ in range(args.num_unique_prompts_rollout):
                example = next(data_loader)
                param_prompt_Q.put(
                    PromptRequest(
                        prompt=example[INPUT_IDS_PROMPT_KEY],
                        generation_config=generation_config,
                        dataset_index=example.get("dataset_index", 0),
                        prompt_id=example.get("prompt_id", ""),
                        is_eval=False,
                    )
                )

            # Put packed sequences and metrics into the output queue
            packed_sequences_Q.put(
                {"collated_data": collated_data, "metrics": metrics, "num_new_tokens": num_new_tokens}
            )

            if args.save_traces:
                traces = {
                    "advantages": [float(a) for a in advantages],
                    "finish_reasons": all_finish_reasons,
                    "responses": all_responses,
                    "training_step": training_step,
                }
                os.makedirs(args.output_dir, exist_ok=True)
                with open(f"{args.output_dir}/traces_{args.run_name}.jsonl", "a") as f:
                    json.dump(traces, f)
                    f.write("\n")


def setup_runtime_variables(args: DistillationArgs) -> DistillationArgs:
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
    if args.push_to_hub:
        if args.hf_repo_id is None:
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    if args.with_tracking and args.wandb_entity is None:
        args.wandb_entity = maybe_use_ai2_wandb_entity()
    return args


def setup_experiment_tracking(args: DistillationArgs, tc: TokenizerConfig, model_config: ModelConfig):
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


def setup_datasets(args: DistillationArgs, tc: TokenizerConfig, tokenizer: PreTrainedTokenizer):
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

    eval_dataset = None
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

    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)

    return train_dataset, eval_dataset


def create_model_and_optimizer(
    args: DistillationArgs,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    tokenizer: PreTrainedTokenizer,
    inference_results_Q: ray_queue.Queue,
    param_prompt_Q: ray_queue.Queue,
) -> tuple[ModelGroup, list[vllm_utils.LLMRayActor], int]:
    """Create the model, optimizer, and vLLM engines."""
    # Create placement group
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray_get_with_progress([pg.ready()], desc="Waiting for placement group")

    inits = []
    policy_group = ModelGroup(pg, PolicyTrainerRayProcess, args.num_learners_per_node, args.single_gpu_mode)
    inits.extend(model.from_pretrained.remote(args, model_config, tokenizer) for model in policy_group.models)

    # Create vLLM engines
    max_len = args.max_prompt_token_length + args.response_length
    vllm_engines = vllm_utils.create_vllm_engines(
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
        prompt_queue=param_prompt_Q,
        results_queue=inference_results_Q,
    )

    results, _ = ray_get_with_progress(inits, desc="Initializing models")
    resume_training_step = results[0] + 1
    logger.info("======== ✅ all models and vLLM engines initialized =========")

    ray_get_with_progress(
        [m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models],
        desc="Setting up model update group",
    )
    logger.info("======== ✅ model update group setup successfully =========")

    return policy_group, vllm_engines, resume_training_step


def create_generation_config(args: DistillationArgs):
    """Create generation config for training."""
    return vllm_utils.SamplingConfig(
        temperature=args.temperature,
        top_p=args.vllm_top_p,
        max_tokens=args.response_length,
        n=args.num_samples_per_prompt_rollout,
        stop=args.stop_strings,
        seed=args.seed,
        logprobs=1,
    )


def run_training(
    args: DistillationArgs,
    tokenizer: PreTrainedTokenizer,
    train_dataset,
    eval_dataset,
    policy_group: ModelGroup,
    generation_config,
    data_loader: data_loader_lib.HFDataLoader,
    resume_training_step: int,
    teacher_client: TeacherClient,
):
    """Run the main training loop."""
    # Create queues
    inference_results_Q = ray_queue.Queue(maxsize=(args.async_steps + 1) * args.num_unique_prompts_rollout)
    param_prompt_Q = ray_queue.Queue(maxsize=(args.async_steps + 1) * args.num_unique_prompts_rollout)
    packed_sequences_Q = Queue(maxsize=args.async_steps)

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="distillation")

    # Start data preparation thread
    executor.submit(
        data_preparation_thread,
        inference_results_Q,
        param_prompt_Q,
        packed_sequences_Q,
        args,
        tokenizer,
        args.num_training_steps,
        generation_config,
        data_loader,
        teacher_client,
    )

    # Send initial prompts
    for _ in range(args.async_steps * args.num_unique_prompts_rollout):
        example = next(data_loader)
        param_prompt_Q.put(
            PromptRequest(
                prompt=example[INPUT_IDS_PROMPT_KEY],
                generation_config=generation_config,
                dataset_index=example.get("dataset_index", 0),
                prompt_id=example.get("prompt_id", ""),
                is_eval=False,
            )
        )

    num_total_tokens = 0
    training_start_time = time.perf_counter()

    for training_step in range(resume_training_step, args.num_training_steps + 1):
        start_time = time.perf_counter()

        # Get packed data from preparation thread
        try:
            packed_data = packed_sequences_Q.get(timeout=600)
        except Empty:
            logger.error("[Main Thread] Timeout waiting for packed sequences")
            break

        collated_data = packed_data["collated_data"]
        data_thread_metrics = packed_data["metrics"]
        num_step_tokens = packed_data["num_new_tokens"]
        num_total_tokens += num_step_tokens

        if collated_data is None:
            logger.warning(f"[Main Thread] No data for step {training_step}, skipping")
            continue

        # Train
        with Timer("[Main Thread] Training") as train_timer:
            metrics_list, _ = ray_get_with_progress(
                [
                    policy_group.models[i].train.remote(data_BT=collated_data[i], pad_token_id=tokenizer.pad_token_id)
                    for i in range(args.world_size)
                ],
                desc=f"Running training step {training_step}",
            )

        # Broadcast updated weights to vLLM
        with Timer("[Main Thread] Weight sync"):
            weight_broadcast_futures = [m.broadcast_to_vllm.remote() for m in policy_group.models]
            ray_get_with_progress(weight_broadcast_futures, desc="Broadcasting weights to vLLM")

        # Aggregate metrics
        average_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in metrics_list[0]}
        step_time = time.perf_counter() - start_time
        total_training_time = time.perf_counter() - training_start_time

        metrics = {
            "training_step": training_step,
            "val/num_total_tokens": num_total_tokens,
            "val/num_step_tokens": num_step_tokens,
            "epoch": training_step * args.num_unique_prompts_rollout / len(train_dataset),
            "learner_tokens_per_second_overall": num_total_tokens / total_training_time,
            "learner_tokens_per_second_step": num_step_tokens / step_time,
            "time/total": step_time,
            "time/training": train_timer.duration,
            **data_thread_metrics,
            **average_metrics,
        }

        print_rich_single_line_metrics(metrics)

        if args.with_tracking:
            wandb.log(metrics, step=training_step)

        # Save checkpoint
        if args.save_freq > 0 and training_step % args.save_freq == 0:
            checkpoint_dir = f"{args.output_dir}_checkpoints/step_{training_step}"
            logger.info(f"Saving checkpoint at step {training_step} to {checkpoint_dir}")
            ray_get_with_progress(
                [policy_group.models[i].save_model.remote(checkpoint_dir, tokenizer) for i in range(args.world_size)],
                desc=f"Saving checkpoint at step {training_step}",
            )

        # Evaluation
        if args.local_eval_every > 0 and training_step % args.local_eval_every == 0 and eval_dataset is not None:
            logger.info(f"[Main Thread] Running evaluation at step {training_step}")
            # TODO: Implement evaluation loop

    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    ray_get_with_progress(
        [policy_group.models[i].save_model.remote(args.output_dir, tokenizer) for i in range(args.world_size)],
        desc="Saving final model",
    )

    # Cleanup
    stop_event.set()
    executor.shutdown(wait=True)


def make_tokenizer(tc: TokenizerConfig, model_config: ModelConfig) -> PreTrainedTokenizer:
    """Setup tokenizer with appropriate configuration."""
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    return tc.tokenizer


def main(args: DistillationArgs, tc: TokenizerConfig, model_config: ModelConfig):
    """Main entry point for on-policy distillation training."""
    tokenizer = make_tokenizer(tc, model_config)
    args = setup_runtime_variables(args)

    _beaker_config, _wandb_url = setup_experiment_tracking(args, tc, model_config)

    train_dataset, eval_dataset = setup_datasets(args, tc, tokenizer)

    if args.cache_dataset_only:
        return

    pprint([args, model_config])

    # Initialize Ray
    ray.init(dashboard_host="0.0.0.0", runtime_env={"excludes": [".git/"], "env_vars": dict(os.environ)})

    # Create queues
    queue_size = (args.async_steps + 1) * args.num_unique_prompts_rollout
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    param_prompt_Q = ray_queue.Queue(maxsize=queue_size)

    # Create model and vLLM engines
    policy_group, _vllm_engines, resume_training_step = create_model_and_optimizer(
        args, tc, model_config, tokenizer, inference_results_Q, param_prompt_Q
    )

    # Create generation config
    generation_config = create_generation_config(args)

    # Create data loader
    data_loader = data_loader_lib.HFDataLoader(
        dataset=train_dataset,
        batch_size=1,
        seed=args.seed,
        rank=0,
        world_size=1,
        work_dir=args.output_dir,
        automatic_reshuffle=True,
    )

    # Initialize teacher client
    teacher_client = TeacherClient(
        api_base=args.teacher_api_base,
        api_key=args.teacher_api_key,
        model_name=args.teacher_model_name,
        max_concurrent_requests=args.teacher_max_concurrent_requests,
        timeout=args.teacher_timeout,
        temperature=args.teacher_temperature,
    )
    logger.info(f"Teacher client initialized with base URL: {args.teacher_api_base}")

    try:
        run_training(
            args,
            tokenizer,
            train_dataset,
            eval_dataset,
            policy_group,
            generation_config,
            data_loader,
            resume_training_step,
            teacher_client,
        )
    finally:
        # Cleanup
        if ray.is_initialized():
            ray.shutdown()

    # Push to hub
    if args.push_to_hub:
        accelerator = Namespace()
        accelerator.is_main_process = True
        push_folder_to_hub(accelerator, args.output_dir, args.hf_repo_id, args.hf_repo_revision)

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = ArgumentParserPlus((DistillationArgs, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    main(args, tokenizer_config, model_config)
