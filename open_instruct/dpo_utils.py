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
DPO utils
Adapted from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py
"""

import contextlib
import enum
import functools
import hashlib
import json
import os
import pathlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import DataCollatorForSeq2Seq
from transformers.training_args import _convert_str_dict

from open_instruct import logger_utils, model_utils, utils
from open_instruct.dataset_transformation import (
    TOKENIZED_PREFERENCE_DATASET_KEYS,
    TokenizerConfig,
    compute_config_hash,
    load_dataset_configs,
)
from open_instruct.padding_free_collator import PAD_VALUES, calculate_per_token_logps, pad_to_length
from open_instruct.padding_free_collator import concatenated_inputs as pf_concatenated_inputs
from open_instruct.padding_free_collator import get_batch_logps as pf_get_batch_logps

logger = logger_utils.setup_logger(__name__)


def config_to_json_serializable(obj: object) -> object:
    """Convert config object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: config_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [config_to_json_serializable(v) for v in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


class DPOLossType(enum.StrEnum):
    dpo = "dpo"
    dpo_norm = "dpo_norm"
    simpo = "simpo"
    wpo = "wpo"

    @property
    def is_average_loss(self) -> bool:
        return self in (DPOLossType.simpo, DPOLossType.dpo_norm)

    @property
    def needs_reference_model(self) -> bool:
        return self in (DPOLossType.dpo, DPOLossType.dpo_norm, DPOLossType.wpo)

    @property
    def computes_reward_metrics(self) -> bool:
        return self in (DPOLossType.dpo, DPOLossType.dpo_norm)


@dataclass
class TrackingConfig:
    """Base configuration for experiment tracking."""

    exp_name: str = "dpo_experiment"
    """The name of this experiment"""
    run_name: str | None = None
    """A unique name of this run"""
    seed: int = 42
    """Random seed for initialization and dataset shuffling."""
    add_seed_and_date_to_exp_name: bool = True
    """Append the seed and date to exp_name"""


@dataclass
class DPOConfig:
    """Configuration for DPO-specific hyperparameters."""

    beta: float = 0.1
    """Beta parameter for DPO loss."""
    loss_type: DPOLossType = DPOLossType.dpo
    """Type of DPO loss to use. Options are 'dpo', 'dpo_norm', 'simpo', 'wpo'."""
    gamma_beta_ratio: float = 0.3
    """Gamma to beta ratio for SimPO loss. Not used for DPO loss."""
    label_smoothing: float = 0.0
    """Label smoothing for DPO/SimPO loss. Default is 0 (no smoothing)."""
    load_balancing_loss: bool = False
    """Whether to include a load balancing loss (for OLMoE) or not."""
    load_balancing_weight: float = 0.001
    """Weight for load balancing loss if applicable."""
    concatenated_forward: bool = True
    """Whether to concatenate chosen and rejected for DPO training."""
    packing: bool = False
    """Whether to use packing/padding-free collation."""


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    num_epochs: int = 2
    """Total number of training epochs to perform."""
    per_device_train_batch_size: int = 8
    """Batch size per GPU/TPU core/CPU for training."""
    gradient_accumulation_steps: int = 1
    """Number of updates steps to accumulate before performing a backward/update pass."""
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    warmup_ratio: float = 0.03
    """Linear warmup over warmup_ratio fraction of total steps."""
    weight_decay: float = 0.0
    """Weight decay for AdamW if we apply some."""
    max_grad_norm: float = -1
    """Maximum gradient norm for clipping. -1 means no clipping."""
    max_seq_length: int = 2048
    """The maximum total input sequence length after tokenization."""
    lr_scheduler_type: str = "linear"
    """The scheduler type to use for learning rate adjustment."""
    max_train_steps: int | None = None
    """If set, overrides the number of training steps. Otherwise, num_epochs is used."""
    activation_memory_budget: float = 1.0
    """Memory budget for activation checkpointing (0.0-1.0).

    A practical "just turn it on" default is `0.5` (somewhat arbitrary, but works well in practice):
    any value < 1.0 enables budget-mode checkpointing. Higher values use more memory and are
    typically faster; lower values use less memory and are typically slower, so use the highest
    value your hardware can support. See: https://pytorch.org/blog/activation-checkpointing-techniques/.
    """
    use_8bit_optimizer: bool = False
    """Use 8bit optimizer from bitsandbytes."""
    dpo_use_paged_optimizer: bool = False
    """Use paged optimizer from bitsandbytes."""
    fused_optimizer: bool = True
    """Whether to use fused AdamW or not."""
    tensor_parallel_degree: int = 1
    """Tensor parallelism degree. Default 1 (disabled)."""
    context_parallel_degree: int = 1
    """Context parallelism degree. Default 1 (disabled)."""
    cache_logprobs_only: bool = False
    """Exit after building the reference logprobs cache (for benchmarking)."""
    compile_model: bool = True
    """Whether to apply torch.compile to model blocks."""
    shard_degree: int | None = None
    """FSDP shard degree. None means auto-detect."""
    num_replicas: int | None = None
    """Number of FSDP replicas. None means auto-detect."""


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-wildchat-reused-on-policy-8b", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    transform_fn: list[str] = field(
        default_factory=lambda: ["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""
    target_columns: list[str] = field(default_factory=lambda: TOKENIZED_PREFERENCE_DATASET_KEYS)
    """The columns to use for the dataset."""
    cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    skip_cache: bool = False
    """Whether to skip the cache."""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    config_hash: str | None = None
    """The hash of the dataset configuration."""


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training."""

    use_lora: bool = False
    """If True, will use LORA to train the model."""
    lora_rank: int = 64
    """The rank of lora."""
    lora_alpha: float = 16
    """The alpha parameter of lora."""
    lora_dropout: float = 0.1
    """The dropout rate of lora modules."""


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""

    logging_steps: int | None = None
    """Log the training loss and learning rate every logging_steps steps."""
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "open_instruct_internal"
    """The wandb project name"""
    wandb_entity: str | None = None
    """The entity (team) of wandb's project"""
    report_to: str | list[str] = "all"
    """The integration(s) to report results and logs to."""


@dataclass
class HubConfig:
    """Configuration for Hugging Face Hub integration."""

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


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""

    output_dir: str = "output/"
    """The output directory where the model predictions and checkpoints will be written."""
    checkpointing_steps: int | str = 500
    """Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."""
    keep_last_n_checkpoints: int = 3
    """How many checkpoints to keep in the output directory. -1 for all."""
    resume_from_checkpoint: str | None = None
    """If the training should continue from a checkpoint folder."""


@dataclass
class EvalConfig:
    """Configuration for evaluation and deployment."""

    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: list[str] | None = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """The max generation length for evaluation for oe-eval"""
    oe_eval_gpu_multiplier: int | None = None
    """The multiplier for the number of GPUs for evaluation"""
    eval_workspace: str | None = "ai2/tulu-3-results"
    """The workspace to launch evaluation jobs on"""
    eval_priority: Literal["low", "normal", "high"] | None = "high"
    """The priority of auto-launched evaluation jobs"""


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    model_name_or_path: str | None = None
    """The model checkpoint for weights initialization."""
    use_flash_attn: bool = True
    """Whether to use flash attention in the model training"""
    attn_backend: str = "auto"
    """Attention backend for OLMo-core models. Options: flash_2, flash_3, auto."""
    model_revision: str | None = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""
    low_cpu_mem_usage: bool = False
    """Create the model as an empty shell, then materialize parameters when pretrained weights are loaded."""


REFERENCE_LOGPROBS_CACHE_PATH = os.environ.get(
    "REFERENCE_LOGPROBS_CACHE_PATH", "/weka/oe-adapt-default/allennlp/deletable_reference_logprobs_cache"
)

torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class ExperimentConfig(
    TrackingConfig,
    ModelConfig,
    DPOConfig,
    TrainingConfig,
    DatasetConfig,
    LoRAConfig,
    LoggingConfig,
    HubConfig,
    CheckpointConfig,
    EvalConfig,
):
    """
    Full arguments class for all fine-tuning jobs.
    """

    _VALID_DICT_FIELDS = ["additional_model_arguments"]

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    do_not_randomize_output_dir: bool = False
    """By default the output directory will be randomized"""
    config_name: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    additional_model_arguments: dict | str | None = field(
        default_factory=dict, metadata={"help": "A dictionary of additional model args used to construct the model."}
    )
    sync_each_batch: bool = False
    """Optionaly sync grads every batch when using grad accumulation. Can significantly reduce memory costs."""
    dataset_name: str | None = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_mixer: dict | None = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mix_dir: str | None = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_seq_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    resume_from_checkpoint: str | None = field(
        default=None, metadata={"help": "If the training should continue from a checkpoint folder."}
    )
    save_to_hub: str | None = field(
        default=None, metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"}
    )
    use_liger_kernel: bool = field(default=False, metadata={"help": "Whether to use LigerKernel for training."})
    profiling: bool = field(default=False, metadata={"help": "Enable torch profiler to trace training steps."})
    hf_metadata_dataset: str | None = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""

    zero_stage: int | None = field(
        default=None,
        metadata={
            "help": "DeepSpeed ZeRO optimization stage (0, 1, 2, or 3). If None, DeepSpeed config must be provided via accelerate launch."
        },
    )
    offload_optimizer: bool = field(
        default=False,
        metadata={"help": "Offload optimizer states to CPU to save GPU memory. Only used if zero_stage is set."},
    )
    offload_param: bool = field(
        default=False, metadata={"help": "Offload parameters to CPU to save GPU memory. Only used with zero_stage 3."}
    )
    zero_hpz_partition_size: int = field(
        default=8, metadata={"help": "Hierarchical partition size for ZeRO stage 3. Only used with zero_stage 3."}
    )

    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: list[str] | None = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""
    oe_eval_gpu_multiplier: int | None = None
    """the multiplier for the number of GPUs for evaluation"""
    eval_workspace: str | None = "ai2/tulu-3-results"
    """The workspace to launch evaluation jobs on"""
    eval_priority: str | None = "high"
    """The priority of auto-launched evaluation jobs"""

    @property
    def forward_fn(self) -> Callable:
        fn = concatenated_forward if self.concatenated_forward else separate_forward
        if self.packing:
            if not self.concatenated_forward:
                raise NotImplementedError("separate forward not implemented for packing/padding-free")
            fn = functools.partial(fn, packing=True)
        return fn

    def __post_init__(self):
        if isinstance(self.loss_type, str):
            self.loss_type = DPOLossType(self.loss_type)

        if self.dataset_name is None and self.dataset_mixer is None and self.mixer_list is None:
            raise ValueError("Need either a dataset name, dataset mixer, or a training file.")
        if (
            (self.dataset_name is not None and (self.dataset_mixer is not None or self.mixer_list is not None))
            or (self.dataset_name is not None)
            or (self.dataset_mixer is not None and self.mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")

        for dict_feld in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, dict_feld)
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, dict_feld, loaded_dict)

        if self.zero_stage is not None:
            if self.zero_stage not in [0, 1, 2, 3]:
                raise ValueError(f"zero_stage must be 0, 1, 2, or 3, got {self.zero_stage}")
            if self.offload_param and self.zero_stage != 3:
                raise ValueError("offload_param can only be used with zero_stage 3")


FlatArguments = ExperimentConfig


def compute_reference_cache_hash(args: ExperimentConfig, tc: TokenizerConfig) -> str:
    """Compute deterministic hash for reference logprobs cache from ExperimentConfig."""
    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
    dcs = load_dataset_configs(
        args.mixer_list, args.mixer_list_splits, args.transform_fn, transform_fn_args, args.target_columns
    )
    dataset_config_hash = args.config_hash or compute_config_hash(dcs, tc)
    config_str = json.dumps(
        {
            "concatenated_forward": args.concatenated_forward,
            "dataset_config_hash": dataset_config_hash,
            "loss_type": args.loss_type,
            "max_train_samples": args.max_train_samples,
            "model_name_or_path": args.model_name_or_path,
            "model_revision": args.model_revision,
            "packing": args.packing,
            "use_lora": args.use_lora,
            "use_qlora": args.use_qlora,
        },
        sort_keys=True,
    )
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def build_reference_logprobs_cache(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    average_log_prob: bool,
    forward_fn: Callable,
    full_dataset_size: int,
    device: torch.device,
    cache_path: pathlib.Path,
    is_main_process: bool,
    model_dims: utils.ModelDims,
    use_lora: bool = False,
    disable_adapter_context: Callable[[], contextlib.AbstractContextManager] | None = None,
) -> model_utils.TensorCache:
    """Build a TensorCache with reference logprobs by computing logprobs once for all samples.

    Args:
        model: The model to compute logprobs with.
        dataloader: DataLoader providing batches with 'index' key.
        average_log_prob: Whether to average log probs over sequence length.
        forward_fn: Forward function to compute logprobs.
        full_dataset_size: Total number of samples in the dataset.
        device: Device to place tensors on.
        cache_path: Path to save/load cache from.
        is_main_process: Whether this is the main process.
        use_lora: Whether LoRA is enabled (requires disable_adapter_context).
        disable_adapter_context: Callable returning context manager to disable LoRA adapter.

    Returns:
        TensorCache containing 'chosen_logps' and 'rejected_logps' tensors.
    """
    if cache_path.exists():
        logger.info(f"Loading reference logprobs cache from {cache_path}")
        return model_utils.TensorCache.from_disk(cache_path, device=device)

    if is_main_process:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        test_file = cache_path.parent / f".write_test_{cache_path.stem}"
        try:
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise RuntimeError(
                f"Cannot write to cache directory {cache_path.parent}: {e}. "
                f"Set REFERENCE_LOGPROBS_CACHE_PATH to a writable location."
            ) from e
    if dist.is_initialized():
        dist.barrier()

    model.eval()
    chosen_tensor = torch.full((full_dataset_size,), float("-inf"), dtype=torch.float32, device=device)
    rejected_tensor = torch.full((full_dataset_size,), float("-inf"), dtype=torch.float32, device=device)

    total_tokens = 0
    total_examples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, disable=not is_main_process, desc="Caching reference logprobs")
        for batch in pbar:
            batch_start = time.perf_counter()
            if use_lora and disable_adapter_context is not None:
                with disable_adapter_context():
                    chosen_logps, rejected_logps, _ = forward_fn(model, batch, average_log_prob=average_log_prob)
            else:
                chosen_logps, rejected_logps, _ = forward_fn(model, batch, average_log_prob=average_log_prob)

            chosen_tensor[batch["index"]] = chosen_logps
            rejected_tensor[batch["index"]] = rejected_logps

            bs = len(batch["index"])
            if "chosen_cu_seq_lens_k" in batch:
                chosen_actual = batch["chosen_cu_seq_lens_k"][-1].item()
                rejected_actual = batch["rejected_cu_seq_lens_k"][-1].item()
                batch_tokens = chosen_actual + rejected_actual
            else:
                batch_tokens = batch["chosen_input_ids"].numel() + batch["rejected_input_ids"].numel()
            total_tokens += batch_tokens
            total_examples += bs

            if "chosen_cu_seq_lens_k" in batch:
                chosen_cu = batch["chosen_cu_seq_lens_k"]
                rejected_cu = batch["rejected_cu_seq_lens_k"]
                chosen_lengths = (chosen_cu[1:] - chosen_cu[:-1]).tolist()
                rejected_lengths = (rejected_cu[1:] - rejected_cu[:-1]).tolist()
            else:
                chosen_lengths = [batch["chosen_input_ids"].shape[1]] * bs
                rejected_lengths = [batch["rejected_input_ids"].shape[1]] * bs
            pbar.set_postfix(
                {
                    "avg_tok/ex": f"{total_tokens / total_examples:.0f}",
                    "MFU%": f"{model_dims.calculate_mfu(chosen_lengths + rejected_lengths, time.perf_counter() - batch_start):.1f}",
                    "mem_GB": f"{torch.cuda.max_memory_allocated() / 1e9:.1f}",
                    "mem%": f"{torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.0f}",
                }
            )

    if dist.is_initialized():
        dist.all_reduce(chosen_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(rejected_tensor, op=dist.ReduceOp.MAX)

    missing_chosen = torch.where(chosen_tensor == float("-inf"))[0]
    missing_rejected = torch.where(rejected_tensor == float("-inf"))[0]
    if len(missing_chosen) > 0 or len(missing_rejected) > 0:
        missing_indices = torch.unique(torch.cat([missing_chosen, missing_rejected]))
        raise RuntimeError(
            f"Missing {len(missing_indices)} indices during reference logprobs caching. "
            f"First 10: {missing_indices[:10].tolist()}"
        )

    model.train()
    cache = model_utils.TensorCache(tensors={"chosen_logps": chosen_tensor, "rejected_logps": rejected_tensor})

    cache_mem_bytes = sum(t.numel() * t.element_size() for t in cache.tensors.values())
    cache_mem_gib = cache_mem_bytes / (1024**3)
    if device.type == "cuda":
        cache_mem_pct = 100 * cache_mem_bytes / torch.cuda.get_device_properties(device).total_memory
        logger.info(f"Reference logprobs cached, using {cache_mem_gib:.2f} GiB of GPU RAM ({cache_mem_pct:.1f}%).")
    else:
        logger.info(f"Reference logprobs cached, using {cache_mem_gib:.2f} GiB of RAM.")

    if is_main_process:
        logger.info(f"Saving reference logprobs cache to {cache_path}")
        cache.to_disk(cache_path)

    if dist.is_initialized():
        dist.barrier()

    return cache


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    reference_free: bool = False,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model
            for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model
            for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model
            for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model
            for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something
            in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model
            and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, mean_chosen_rewards, mean_rejected_rewards).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    chosen_rewards = (beta * (policy_chosen_logps - reference_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - reference_rejected_logps)).detach()

    return losses, chosen_rewards, rejected_rewards


def wpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    chosen_loss_mask: torch.Tensor,
    rejected_loss_mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Weighted Preference Optimization (WPO) loss.
    Paper: https://arxiv.org/abs/2406.11827

    WPO extends DPO by weighting the loss based on the policy model's confidence,
    computed from the average log probabilities of chosen and rejected responses.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for chosen responses.
        policy_rejected_logps: Log probabilities of the policy model for rejected responses.
        reference_chosen_logps: Log probabilities of the reference model for chosen responses.
        reference_rejected_logps: Log probabilities of the reference model for rejected responses.
        beta: Temperature parameter for the loss.
        label_smoothing: Label smoothing parameter.
        chosen_loss_mask: Boolean mask for chosen response tokens.
        rejected_loss_mask: Boolean mask for rejected response tokens.

    Returns:
        A tuple of (losses, mean_chosen_rewards, mean_rejected_rewards).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # compute average logps and use them to compute the weights
    policy_chosen_logps_average = (policy_chosen_logps * chosen_loss_mask).sum(-1) / chosen_loss_mask.sum(-1)
    policy_rejected_logps_average = (policy_rejected_logps * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1)
    policy_weights = torch.clamp(torch.exp(policy_chosen_logps_average + policy_rejected_logps_average), max=1)

    logits = pi_logratios - ref_logratios

    losses = (
        -F.logsigmoid(beta * logits) * (1 - label_smoothing) * policy_weights
        - F.logsigmoid(-beta * logits) * label_smoothing * policy_weights
    )

    chosen_rewards = (beta * (policy_chosen_logps - reference_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - reference_rejected_logps)).detach()

    return losses, chosen_rewards, rejected_rewards


# From https://github.com/princeton-nlp/SimPO/blob/main/scripts/simpo_trainer.py#L560C1-L595C56
def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    beta: float,
    gamma_beta_ratio: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the SimPO loss for a batch of policy model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of (losses, mean_chosen_rewards, mean_rejected_rewards).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    logits = pi_logratios - gamma_beta_ratio

    # sigmoid loss type from SimPO.
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = (beta * policy_chosen_logps).detach()
    rejected_rewards = (beta * policy_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def compute_loss(
    args: DPOConfig,
    batch: dict[str, torch.Tensor],
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_cache: model_utils.TensorCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_type = args.loss_type

    if loss_type in (DPOLossType.dpo, DPOLossType.dpo_norm):
        assert reference_cache is not None
        ref_logps = reference_cache[batch["index"]]
        return dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_logps["chosen_logps"],
            ref_logps["rejected_logps"],
            beta=args.beta,
            label_smoothing=args.label_smoothing,
        )
    elif loss_type == DPOLossType.simpo:
        return simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            beta=args.beta,
            gamma_beta_ratio=args.gamma_beta_ratio,
            label_smoothing=args.label_smoothing,
        )
    elif loss_type == DPOLossType.wpo:
        assert reference_cache is not None
        ref_logps = reference_cache[batch["index"]]
        return wpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_logps["chosen_logps"],
            ref_logps["rejected_logps"],
            beta=args.beta,
            label_smoothing=args.label_smoothing,
            chosen_loss_mask=batch["chosen_labels"] != -100,
            rejected_loss_mask=batch["rejected_labels"] != -100,
        )
    raise ValueError(f"Unknown loss type: {loss_type}")


def _get_batch_logps(
    per_token_logps: torch.Tensor, labels: torch.Tensor, average_log_prob: bool = False
) -> torch.Tensor:
    """Aggregate per-token log probabilities into per-sequence log probabilities.

    Args:
        per_token_logps: Per-token log probabilities where position i contains
            log p(labels[i+1] | x_i). Shape: (batch_size, sequence_length)
        labels: Labels used to build the loss mask.
            Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token.
            Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum
            log probabilities of the given labels under the given logits.
    """
    per_token_logps = per_token_logps[:, :-1]
    loss_mask = labels[:, 1:] != -100

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def process_batch(batch: dict[str, list | torch.Tensor], prefix: str, pad_value: int = 0) -> dict[str, torch.Tensor]:
    """Process either chosen or rejected inputs separately.

    Args:
        batch: Input batch dictionary
        prefix: Either 'chosen' or 'rejected'
        pad_value: Value to use for padding (0 for input_ids, -100 for labels)

    Returns:
        Processed batch dictionary for the specified prefix
    """
    processed = {}
    for k in batch:
        if k.startswith(prefix) and isinstance(batch[k], torch.Tensor):
            new_key = k.replace(prefix + "_", "")
            processed[new_key] = batch[k]
    return processed


def concatenated_inputs(batch: dict[str, list | torch.Tensor]) -> dict[str, torch.Tensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids'
            and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    chosen_input_ids: torch.Tensor = batch["chosen_input_ids"]  # type: ignore[assignment]
    rejected_input_ids: torch.Tensor = batch["rejected_input_ids"]  # type: ignore[assignment]
    max_length = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])
    concatenated_batch: dict[str, torch.Tensor] = {}
    for k in batch:
        v = batch[k]
        if k.startswith("chosen") and isinstance(v, torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(v, max_length, pad_value=pad_value)
    for k in batch:
        v = batch[k]
        if k.startswith("rejected") and isinstance(v, torch.Tensor):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (concatenated_batch[concatenated_key], pad_to_length(v, max_length, pad_value=pad_value)), dim=0
            )
    return concatenated_batch


def concatenated_forward(
    model: nn.Module,
    batch: dict[str, list | torch.Tensor],
    average_log_prob: bool = False,
    output_router_logits: bool = False,
    packing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    Uses HuggingFace model interface: model(**inputs) returns outputs with .logits attribute.

    Args:
        model: The model to run (HuggingFace-style model).
        batch: Dictionary containing chosen and rejected inputs.
        average_log_prob: Whether to average the log probabilities.
        output_router_logits: Whether to output router logits for MoE models.
        packing: Whether to use padding-free packing.

    Returns:
        Tuple of (chosen_logps, rejected_logps, aux_loss).
    """
    if not packing:
        concatenated_batch = concatenated_inputs(batch)
    else:
        concatenated_batch, bs = pf_concatenated_inputs(batch)

    inputs = {
        k.replace("concatenated_", ""): v
        for k, v in concatenated_batch.items()
        if k.startswith("concatenated_") and not k.endswith("labels")
    }
    if output_router_logits:
        outputs = model(**inputs, output_router_logits=True)
        logits = outputs.logits
        aux_loss = outputs.aux_loss
    else:
        logits = model(**inputs).logits
        aux_loss = None

    concatenated_labels = concatenated_batch["concatenated_labels"]
    per_token_logps = calculate_per_token_logps(logits, concatenated_labels)

    if not packing:
        all_logps = _get_batch_logps(per_token_logps, concatenated_labels, average_log_prob=average_log_prob)
        bs = batch["chosen_input_ids"].shape[0]
    else:
        all_logps = pf_get_batch_logps(
            per_token_logps, concatenated_labels, inputs["cu_seq_lens_k"], average_log_prob=average_log_prob
        )
    chosen_logps = all_logps[:bs]
    rejected_logps = all_logps[bs:]
    return chosen_logps, rejected_logps, aux_loss


def separate_forward(
    model: nn.Module,
    batch: dict[str, list | torch.Tensor],
    average_log_prob: bool = False,
    output_router_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the model on chosen and rejected inputs separately.

    Uses HuggingFace model interface: model(**inputs) returns outputs with .logits attribute.

    Args:
        model: The model to run (HuggingFace-style model).
        batch: Dictionary containing chosen and rejected inputs.
        average_log_prob: Whether to average the log probabilities.
        output_router_logits: Whether to output router logits for MoE models.

    Returns:
        Tuple of (chosen_logps, rejected_logps, aux_loss).
    """
    chosen_batch = process_batch(batch, "chosen")

    if output_router_logits:
        chosen_outputs = model(
            input_ids=chosen_batch["input_ids"],
            attention_mask=chosen_batch["attention_mask"],
            output_router_logits=True,
        )
        chosen_logits = chosen_outputs.logits.to(torch.float32)
        chosen_aux_loss = chosen_outputs.aux_loss
    else:
        chosen_logits = model(
            input_ids=chosen_batch["input_ids"], attention_mask=chosen_batch["attention_mask"]
        ).logits.to(torch.float32)
        chosen_aux_loss = None

    chosen_per_token_logps = calculate_per_token_logps(chosen_logits, chosen_batch["labels"])
    chosen_logps = _get_batch_logps(chosen_per_token_logps, chosen_batch["labels"], average_log_prob=average_log_prob)
    del chosen_batch, chosen_logits, chosen_per_token_logps
    if output_router_logits:
        del chosen_outputs
    torch.cuda.empty_cache()

    rejected_batch = process_batch(batch, "rejected")

    if output_router_logits:
        rejected_outputs = model(
            input_ids=rejected_batch["input_ids"],
            attention_mask=rejected_batch["attention_mask"],
            output_router_logits=True,
        )
        rejected_logits = rejected_outputs.logits.to(torch.float32)
        rejected_aux_loss = rejected_outputs.aux_loss
    else:
        rejected_logits = model(
            input_ids=rejected_batch["input_ids"], attention_mask=rejected_batch["attention_mask"]
        ).logits.to(torch.float32)
        rejected_aux_loss = None

    rejected_per_token_logps = calculate_per_token_logps(rejected_logits, rejected_batch["labels"])
    rejected_logps = _get_batch_logps(
        rejected_per_token_logps, rejected_batch["labels"], average_log_prob=average_log_prob
    )
    del rejected_batch, rejected_logits, rejected_per_token_logps
    if output_router_logits:
        del rejected_outputs
    torch.cuda.empty_cache()

    if output_router_logits and chosen_aux_loss is not None and rejected_aux_loss is not None:
        aux_loss = torch.cat([chosen_aux_loss, rejected_aux_loss], dim=0)
    else:
        aux_loss = None
    return chosen_logps, rejected_logps, aux_loss


def concatenated_forward_olmo(
    model: nn.Module,
    batch: dict[str, list | torch.Tensor],
    average_log_prob: bool = False,
    packing: bool = False,
    output_router_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

    We do this to avoid doing two forward passes, because it's faster for FSDP.
    Uses OLMo-core Transformer interface. Passes labels so the LM head handles DTensor
    correctly under TP+compile, then computes per-token log-probs on the local logits shard
    to avoid materializing the full (S, vocab) tensor.

    Args:
        model: The model to run (OLMo-core style model).
        batch: Dictionary containing chosen and rejected inputs.
        average_log_prob: Whether to average the log probabilities.
        packing: Whether to use padding-free packing.
        output_router_logits: Unused for OLMo-core models (MoE aux loss handled separately).

    Returns:
        Tuple of (chosen_logps, rejected_logps, aux_loss). aux_loss is always None for OLMo-core.
    """
    del output_router_logits
    if not packing:
        concatenated_batch = concatenated_inputs(batch)
    else:
        concatenated_batch, bs = pf_concatenated_inputs(batch)

    concatenated_labels = concatenated_batch["concatenated_labels"]
    output = model(concatenated_batch["concatenated_input_ids"], labels=concatenated_labels)
    per_token_logps = output.loss

    if not packing:
        all_logps = _get_batch_logps(per_token_logps, concatenated_labels, average_log_prob=average_log_prob)
        bs = batch["chosen_input_ids"].shape[0]
    else:
        all_logps = pf_get_batch_logps(
            per_token_logps,
            concatenated_labels,
            concatenated_batch["concatenated_cu_seq_lens_k"],
            average_log_prob=average_log_prob,
        )

    chosen_logps = all_logps[:bs]
    rejected_logps = all_logps[bs:]
    return chosen_logps, rejected_logps, None


def separate_forward_olmo(
    model: nn.Module,
    batch: dict[str, list | torch.Tensor],
    average_log_prob: bool = False,
    output_router_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the model on chosen and rejected inputs separately.

    Uses OLMo-core Transformer interface: model(input_ids) returns logits tensor directly.
    Note: OLMo-core handles MoE aux loss via compute_auxiliary_metrics() in the train module.

    Args:
        model: The model to run (OLMo-core style model).
        batch: Dictionary containing chosen and rejected inputs.
        average_log_prob: Whether to average the log probabilities.
        output_router_logits: Unused for OLMo-core models (MoE aux loss handled separately).

    Returns:
        Tuple of (chosen_logps, rejected_logps, aux_loss). aux_loss is always None for OLMo-core.
    """
    del output_router_logits
    chosen_batch = process_batch(batch, "chosen")
    chosen_output = model(chosen_batch["input_ids"], labels=chosen_batch["labels"])

    chosen_logps = _get_batch_logps(chosen_output.loss, chosen_batch["labels"], average_log_prob=average_log_prob)
    del chosen_batch
    torch.cuda.empty_cache()

    rejected_batch = process_batch(batch, "rejected")
    rejected_output = model(rejected_batch["input_ids"], labels=rejected_batch["labels"])

    rejected_logps = _get_batch_logps(
        rejected_output.loss, rejected_batch["labels"], average_log_prob=average_log_prob
    )
    del rejected_batch
    torch.cuda.empty_cache()

    return chosen_logps, rejected_logps, None


@dataclass
class DataCollatorForSeq2SeqDPO(DataCollatorForSeq2Seq):
    """
    Alternate version of the hf DataCollatorForSeq2Seq for use with DPO.
    adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L517C1
    """

    def __call__(self, features, return_tensors=None):
        def filter_batch(match_string, features):
            filtered = []
            for f in features:
                item = {}
                for k, v in f.items():
                    if match_string in k:
                        key = k.replace(match_string, "")
                        if isinstance(v, np.ndarray):
                            item[key] = torch.as_tensor(v)
                        elif isinstance(v, list):
                            item[key] = torch.tensor(v)
                        else:
                            item[key] = v
                filtered.append(item)
            return filtered

        chosen_features = super().__call__(filter_batch("chosen_", features), return_tensors=return_tensors)
        rejected_features = super().__call__(filter_batch("rejected_", features), return_tensors=return_tensors)
        result = {}
        for k in chosen_features:
            result["chosen_" + k] = chosen_features[k]
        for k in rejected_features:
            result["rejected_" + k] = rejected_features[k]
        if "index" in features[0]:
            result["index"] = torch.tensor([f["index"] for f in features])
        max_len = max(result["chosen_input_ids"].shape[1], result["rejected_input_ids"].shape[1])
        for prefix in ["chosen_", "rejected_"]:
            for key in ["input_ids", "attention_mask", "labels"]:
                full_key = f"{prefix}{key}"
                pad_value = PAD_VALUES.get(key, self.tokenizer.pad_token_id)
                result[full_key] = pad_to_length(result[full_key], max_len, pad_value)
        result["input_ids"] = torch.cat([result["chosen_input_ids"], result["rejected_input_ids"]], dim=0)
        result["attention_mask"] = torch.cat(
            [result["chosen_attention_mask"], result["rejected_attention_mask"]], dim=0
        )
        return result
