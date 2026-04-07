"""
OLMo-core utility functions, shared training configurations, and model configuration mappings.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.distributed as dist
import transformers
from olmo_core.data import TokenizerConfig as OLMoCoreTokenizerConfig
from olmo_core.distributed.utils import is_distributed
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.train.train_module.transformer import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
)
from olmo_core.train.train_module.transformer.config import TransformerContextParallelConfig

from open_instruct import logger_utils, model_utils, utils
from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu

logger = logger_utils.setup_logger(__name__)


@dataclass
class ExperimentConfig:
    """Base configuration for experiment identity and metadata."""

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
    attn_implementation: AttentionBackendName | None = None
    """Which attention implementation to use. If None, auto-detects the best available."""
    model_revision: str | None = None
    """The specific model version to use (can be a branch name, tag name or commit id)."""
    low_cpu_mem_usage: bool = False
    """Create the model as an empty shell, then materialize parameters when pretrained weights are loaded."""
    rope_scaling_factor: float | None = None
    """YaRN RoPE scaling factor. When set, applies YaRN RoPE scaling to the model."""
    rope_scaling_beta_fast: float = 32
    """YaRN RoPE beta_fast parameter."""
    rope_scaling_beta_slow: float = 1
    """YaRN RoPE beta_slow parameter."""
    rope_scaling_old_context_len: int = 8192
    """YaRN RoPE old_context_len parameter."""

    def __post_init__(self):
        if self.attn_implementation is None:
            self.attn_implementation = model_utils.detect_attn_implementation()


@dataclass
class TrainingConfig:
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
    fused_optimizer: bool = True
    """Whether to use fused AdamW."""
    global_batch_size_tokens: int | None = None
    """Global batch size in tokens. When set, used directly for NumpyDataLoaderConfig instead of
    computing from per_device_batch_size * grad_accum * world_size * seq_len."""
    data_seed: int | None = None
    """Separate seed for the data loader. If None, uses the experiment tracking seed."""
    cp_degree: int | None = None
    """Context parallelism degree. When set, enables context parallelism."""
    cp_strategy: Literal["llama3", "zig_zag"] = "llama3"
    """Context parallelism strategy."""
    ac_mode: TransformerActivationCheckpointingMode = TransformerActivationCheckpointingMode.budget
    """Activation checkpointing mode."""
    ac_modules: list[str] | None = None
    """Modules to checkpoint when ac_mode='selected_modules'. E.g., ['blocks.*.feed_forward']."""


def build_ac_config(training: TrainingConfig) -> TransformerActivationCheckpointingConfig | None:
    if training.ac_mode == TransformerActivationCheckpointingMode.selected_modules:
        if not training.ac_modules:
            raise ValueError("ac_modules must be set when ac_mode='selected_modules'")
        return TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules, modules=training.ac_modules
        )
    if training.ac_mode == TransformerActivationCheckpointingMode.budget:
        if training.activation_memory_budget < 1.0 and training.compile_model:
            return TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.budget,
                activation_memory_budget=training.activation_memory_budget,
            )
        return None
    raise ValueError(f"Unknown ac_mode: {training.ac_mode!r}")


def build_cp_config(training: TrainingConfig) -> TransformerContextParallelConfig | None:
    if training.cp_degree is None:
        return None
    if training.cp_strategy == "llama3":
        return TransformerContextParallelConfig.llama3(degree=training.cp_degree)
    elif training.cp_strategy == "zig_zag":
        return TransformerContextParallelConfig.zig_zag(degree=training.cp_degree)
    else:
        raise ValueError(f"Unknown cp_strategy: {training.cp_strategy}")


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
    numpy_dataset_path: str | None = None
    """Path to pre-tokenized numpy dataset directory."""


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
    wandb_group_name: str | None = None
    """Optional W&B group name used to group related runs together."""


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""

    output_dir: str = "output/"
    """The output directory where the model predictions and checkpoints will be written."""
    checkpointing_steps: int = 500
    """Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."""
    ephemeral_save_interval: int | None = None
    """Temporary checkpoint cadence for OLMo-core trainers. Must be lower than checkpointing_steps when set."""
    keep_last_n_checkpoints: int = 3
    """How many checkpoints to keep in the output directory. -1 for all."""
    resume_from_checkpoint: str | None = None
    """If the training should continue from a checkpoint folder."""


def build_checkpointer_callback(
    checkpointing_steps: int, ephemeral_save_interval: int | None, save_async: bool = True
) -> CheckpointerCallback:
    """Construct a CheckpointerCallback with shared Open Instruct defaults."""
    return CheckpointerCallback(
        save_interval=checkpointing_steps, ephemeral_save_interval=ephemeral_save_interval, save_async=save_async
    )


def is_hf_checkpoint(path: str) -> bool:
    """Detect whether a model path is a HuggingFace checkpoint (vs olmo-core format).

    Returns True for HF hub IDs (e.g. 'allenai/Olmo-3-1025-7B'), local/weka paths
    containing config.json, and paths with a '-hf' component. Returns False for
    olmo-core distributed checkpoints.
    """
    if not os.path.isabs(path):
        return True
    if os.path.isfile(os.path.join(path, "config.json")):
        return True
    parts = path.replace("\\", "/").split("/")
    return any("-hf" in part for part in parts)


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


def setup_model(model_config_args: ModelConfig, init_device: str = "cpu") -> tuple[Transformer, TransformerConfig]:
    model_name_or_path = model_config_args.model_name_or_path
    if is_hf_checkpoint(model_name_or_path):
        hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vocab_size = hf_config.vocab_size
    else:
        assert model_config_args.config_name is not None, (
            "--config_name is required when model_name_or_path is an olmo-core checkpoint"
        )
        vocab_size = OLMoCoreTokenizerConfig.dolma2().padded_vocab_size()
    logger.info(f"Building OLMo-core model with vocab_size={vocab_size}")
    model_config = get_transformer_config(
        model_config_args.config_name or model_name_or_path,
        vocab_size,
        attn_backend=model_config_args.attn_implementation,
    )
    if model_config_args.rope_scaling_factor is not None:
        model_config = model_config.with_rope_scaling(
            YaRNRoPEScalingConfig(
                factor=model_config_args.rope_scaling_factor,
                beta_fast=model_config_args.rope_scaling_beta_fast,
                beta_slow=model_config_args.rope_scaling_beta_slow,
                old_context_len=model_config_args.rope_scaling_old_context_len,
            )
        )
    model = model_config.build(init_device=init_device)
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

    # Rank 0 loads and caches the dataset to disk first. The barrier ensures
    # the cache is fully written before other ranks call _load(), which then
    # reads from the disk cache instead of all racing to download/process in
    # parallel.
    if is_main_process:
        dataset = _load()
    if is_distributed():
        dist.barrier()
    if not is_main_process:
        dataset = _load()
    return dataset


def setup_tokenizer_and_cache(model_config: ModelConfig, dataset_config: DatasetConfig, tc: TokenizerConfig):
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer
    dataset_config.local_cache_dir = os.path.abspath(dataset_config.local_cache_dir)
    if utils.is_beaker_job():
        dataset_config.local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
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
