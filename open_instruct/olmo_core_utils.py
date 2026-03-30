"""
OLMo-core utility functions, shared training configurations, and model configuration mappings.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.distributed as dist
import transformers
from olmo_core.distributed.utils import is_distributed
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.backend import has_flash_attn_3
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.train.callbacks import CheckpointerCallback

from open_instruct import logger_utils, utils
from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu

logger = logger_utils.setup_logger(__name__)


@dataclass
class TrackingConfig:
    """Base configuration for experiment tracking."""

    exp_name: str = "experiment"
    """The name of this experiment"""
    run_name: str | None = None
    """A unique name of this run"""
    seed: int = 42
    """Random seed for initialization and dataset shuffling."""


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    model_name_or_path: str | None = None
    """The model checkpoint for weights initialization."""
    config_name: str | None = None
    """Pretrained config name or path if not the same as model_name."""
    use_flash_attn: bool = True
    """Whether to use flash attention in the model training"""
    attn_backend: str = "auto"
    """Attention backend for OLMo-core models. Options: flash_2, flash_3, auto."""
    model_revision: str | None = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""
    low_cpu_mem_usage: bool = False
    """Create the model as an empty shell, then materialize parameters when pretrained weights are loaded."""


@dataclass
class BaseTrainingConfig:
    """Shared training hyperparameters used by SFT, DPO, and other OLMo-core trainers."""

    num_epochs: int = 1
    """Total number of training epochs to perform."""
    per_device_train_batch_size: int = 8
    """Batch size per GPU/TPU core/CPU for training."""
    gradient_accumulation_steps: int = 1
    """Number of updates steps to accumulate before performing a backward/update pass."""
    learning_rate: float = 2e-5
    """The initial learning rate for the optimizer."""
    warmup_ratio: float = 0.1
    """Linear warmup over warmup_ratio fraction of total steps."""
    weight_decay: float = 0.0
    """Weight decay for AdamW if we apply some."""
    max_grad_norm: float = -1
    """Maximum gradient norm for clipping. -1 means no clipping."""
    max_seq_length: int = 4096
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
    compile_model: bool = True
    """Whether to apply torch.compile to model blocks."""


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    mixer_list: list[str] = field(default_factory=list)
    """A list of datasets (local or HF) to sample from."""
    mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    transform_fn: list[str] = field(default_factory=list)
    """The list of transform functions to apply to the dataset."""
    target_columns: list[str] = field(default_factory=list)
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
    hf_entity: str | None = None
    """The user or org name for dataset caching on the Hugging Face Hub."""


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment tracking."""

    logging_steps: int = 1
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
class CheckpointConfig:
    """Configuration for checkpointing."""

    output_dir: str = "output/"
    """The output directory where the model predictions and checkpoints will be written."""
    checkpointing_steps: int | str = 500
    """Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."""
    ephemeral_save_interval: int | None = None
    """Temporary checkpoint cadence for OLMo-core trainers. Must be lower than checkpointing_steps when set."""
    keep_last_n_checkpoints: int = 3
    """How many checkpoints to keep in the output directory. -1 for all."""
    resume_from_checkpoint: str | None = None
    """If the training should continue from a checkpoint folder."""


def build_checkpointer_callback(
    checkpointing_steps: int | str, ephemeral_save_interval: int | None, save_async: bool = True
) -> CheckpointerCallback:
    """Construct a CheckpointerCallback with shared Open Instruct defaults."""
    return CheckpointerCallback(
        save_interval=int(checkpointing_steps), ephemeral_save_interval=ephemeral_save_interval, save_async=save_async
    )


OLMO_MODEL_CONFIG_MAP: dict[str, str] = {
    "allenai/OLMo-2-0425-1B": "olmo2_1B_v2",
    "allenai/OLMo-2-1124-7B": "olmo2_7B",
    "allenai/OLMo-2-1124-13B": "olmo2_13B",
    "allenai/OLMo-2-0325-32B": "olmo2_32B",
    "allenai/Olmo-3-1025-7B": "olmo3_7B",
    "allenai/OLMoE-1B-7B-0924": "olmoe_1B_7B",
    "Qwen/Qwen3-0.6B": "qwen3_0_6B",
    "Qwen/Qwen3-0.6B-Base": "qwen3_0_6B",
    "Qwen/Qwen3-1.7B": "qwen3_1_7B",
    "Qwen/Qwen3-4B": "qwen3_4B",
    "Qwen/Qwen3-8B": "qwen3_8B",
    "Qwen/Qwen3-14B": "qwen3_14B",
    "Qwen/Qwen3-32B": "qwen3_32B",
}


def get_transformer_config(model_name_or_config: str, vocab_size: int, attn_backend: str) -> TransformerConfig:
    """Get the appropriate TransformerConfig for a given model name or config name.

    Args:
        model_name_or_config: HuggingFace model name, path, or direct config name (e.g., 'olmo3_7B').
        vocab_size: Vocabulary size for the model.
        attn_backend: Attention backend name (e.g., 'flash_2', 'flash_3', 'torch').

    Returns:
        TransformerConfig for the specified model.

    Raises:
        ValueError: If model/config not found.
    """
    config_name = OLMO_MODEL_CONFIG_MAP.get(model_name_or_config)
    if config_name is None:
        config_name = model_name_or_config

    if not hasattr(TransformerConfig, config_name):
        available_models = ", ".join(OLMO_MODEL_CONFIG_MAP.keys())
        available_configs = [
            name for name in dir(TransformerConfig) if name.startswith(("olmo", "qwen")) and not name.startswith("_")
        ]
        raise ValueError(
            f"Model/config '{model_name_or_config}' not found. "
            f"Available models: {available_models}. "
            f"Available config names: {', '.join(available_configs)}"
        )
    return getattr(TransformerConfig, config_name)(
        vocab_size=vocab_size, attn_backend=AttentionBackendName(attn_backend)
    )


def resolve_attn_backend(attn_backend: str) -> str:
    if attn_backend == "auto":
        device_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
        is_h100 = "h100" in device_name or "h800" in device_name
        attn_backend = "flash_3" if (is_h100 and has_flash_attn_3()) else "flash_2"
        logger.info(f"Auto-detected attn_backend={attn_backend} for device: {device_name}")
    return attn_backend


def setup_model(
    model_name_or_path: str, config_name: str | None, attn_backend: str
) -> tuple[Transformer, TransformerConfig]:
    hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    vocab_size = hf_config.vocab_size
    logger.info(f"Building OLMo-core model with vocab_size={vocab_size}")
    resolved_backend = resolve_attn_backend(attn_backend)
    model_config = get_transformer_config(config_name or model_name_or_path, vocab_size, attn_backend=resolved_backend)
    model = model_config.build(init_device="cpu")
    return model, model_config


def load_dataset_distributed(
    args: DatasetConfig, tc: TokenizerConfig, transform_fn_args: list[dict], is_main_process: bool
):
    def _load():
        return get_cached_dataset_tulu(
            dataset_mixer_list=args.mixer_list,
            dataset_mixer_list_splits=args.mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.target_columns,
            dataset_cache_mode=args.cache_mode,
            dataset_config_hash=args.config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.local_cache_dir,
            dataset_skip_cache=args.skip_cache,
        )

    if is_main_process:
        dataset = _load()
    if is_distributed():
        dist.barrier()
    if not is_main_process:
        dataset = _load()
    return dataset  # noqa: F821 -- always bound: either is_main_process or not


def setup_tokenizer_and_cache(model_config: ModelConfig, dataset_config: DatasetConfig, tc: TokenizerConfig):
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer
    dataset_config.local_cache_dir = os.path.abspath(dataset_config.local_cache_dir)
    if utils.is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            dataset_config.local_cache_dir = beaker_cache_dir
    return tokenizer


def save_state_dict_as_hf(model_config, state_dict, save_dir, original_model_name_or_path, tokenizer):
    try:
        unwrapped_model = model_config.build(init_device="cpu")
        unwrapped_model.load_state_dict(state_dict)
        save_hf_model(save_dir=save_dir, model_state_dict=state_dict, model=unwrapped_model, save_overwrite=True)
    except NotImplementedError as exc:
        logger.warning(
            "Falling back to raw state_dict save because HF export is unsupported for this OLMo-core model: %s", exc
        )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(save_dir, "model_state_dict.pt"))
    tokenizer.save_pretrained(save_dir)
    original_config = transformers.AutoConfig.from_pretrained(original_model_name_or_path)
    original_config.save_pretrained(save_dir)
