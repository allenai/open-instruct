"""Shared configuration classes for DPO training scripts.

This module contains base dataclasses that are shared between dpo.py (OLMo-core)
and dpo_tune_cache.py (Accelerate/DeepSpeed) implementations.
"""

from dataclasses import dataclass, field
from typing import Literal

from open_instruct.dataset_transformation import TOKENIZED_PREFERENCE_DATASET_KEYS


@dataclass
class ExperimentConfig:
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
class ModelConfig:
    """Configuration for model loading."""

    model_name_or_path: str | None = None
    """The model checkpoint for weights initialization."""
    use_flash_attn: bool = True
    """Whether to use flash attention in the model training"""
    model_revision: str | None = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""
    low_cpu_mem_usage: bool = False
    """Create the model as an empty shell, then materialize parameters when pretrained weights are loaded."""


@dataclass
class DPOHyperparamsConfig:
    """Configuration for DPO-specific hyperparameters."""

    dpo_beta: float = 0.1
    """Beta parameter for DPO loss."""
    dpo_loss_type: str = "dpo"
    """Type of DPO loss to use. Options are 'dpo', 'dpo_norm', 'simpo', 'wpo'."""
    dpo_gamma_beta_ratio: float = 0.3
    """Gamma to beta ratio for SimPO loss. Not used for DPO loss."""
    dpo_label_smoothing: float = 0.0
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
    gradient_checkpointing: bool = False
    """Turn on gradient checkpointing. Saves memory but slows training."""
    use_8bit_optimizer: bool = False
    """Use 8bit optimizer from bitsandbytes."""
    dpo_use_paged_optimizer: bool = False
    """Use paged optimizer from bitsandbytes."""
    fused_optimizer: bool = True
    """Whether to use fused AdamW or not."""


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    dataset_mixer_list: list[str] = field(
        default_factory=lambda: ["allenai/tulu-3-wildchat-reused-on-policy-8b", "1.0"]
    )
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_PREFERENCE_DATASET_KEYS)
    """The columns to use for the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    dataset_config_hash: str | None = None
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
    checkpointing_steps: int | str | None = None
    """Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."""
    keep_last_n_checkpoints: int = 3
    """How many checkpoints to keep in the output directory. -1 for all."""
    resume_from_checkpoint: str | None = None
    """If the training should continue from a checkpoint folder."""


@dataclass
class EvalConfig:
    """Configuration for evaluation jobs."""

    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
