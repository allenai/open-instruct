"""
OLMo-core utility functions, shared training configurations, and model configuration mappings.
"""

import datetime
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import accelerate
import torch
import torch.distributed as dist
import transformers
from olmo_core import optim as olmo_optim
from olmo_core.data import TokenizerConfig as OLMoCoreTokenizerConfig
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf import convert as olmo_hf_convert
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.train import callbacks as train_callbacks
from olmo_core.train import prepare_training_environment
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.train.train_module.transformer import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
)
from olmo_core.train.train_module.transformer.config import TransformerContextParallelConfig
from torch.distributed.tensor import DTensor, distribute_tensor

from open_instruct import logger_utils, model_utils, olmo_core_callbacks, utils
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
    data_loader_seed: int | None = None
    """Separate seed for data loader instance ordering. If None, uses seed."""


@dataclass(kw_only=True)
class ModelConfig:
    """Configuration for model loading."""

    model_name_or_path: str
    """The model checkpoint for weights initialization."""
    config_name: str | None = None
    """Pretrained config name or path if not the same as model_name."""
    attn_implementation: AttentionBackendName | None = None
    """Which attention implementation to use. If None, auto-detects the best available."""
    loss_implementation: LMLossImplementation = LMLossImplementation.default
    """LM loss implementation (e.g. 'fused_linear' for Liger FLCE). Defaults to olmo-core's default.
    Only consulted when labels are passed into the model (SFT); DPO computes loss from logits outside the lm_head."""
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
    max_grad_norm: float | None = None
    """Maximum gradient norm for clipping. None means no clipping."""
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
    activation_checkpointing_mode: Literal["budget", "selected_modules"] = "budget"
    """Activation checkpointing mode.

    "budget" uses torch.compile's partitioner with `activation_memory_budget` (requires compilation,
    and cannot checkpoint through opaque custom ops such as the GDN `fla` kernels). "selected_modules"
    wraps the individual block submodules in `activation_checkpointing_modules` (by default all of
    them, including the GDN mixer), which keeps compile *outside* the checkpoint boundary (the
    supported order) while still recovering full-block memory savings, letting `torch.compile`
    coexist with checkpointing at long sequences. olmo-core's "full" mode (wrapping whole blocks) is
    intentionally unsupported: its recompute re-enters the compiled block's backward and fails an
    inductor stride guard for GDN models.
    """
    activation_checkpointing_modules: list[str] = field(
        default_factory=lambda: [
            "blocks.*.attention_norm",
            "blocks.*.attention",
            "blocks.*.attention_residual_stream",
            "blocks.*.feed_forward_norm",
            "blocks.*.feed_forward",
            "blocks.*.feed_forward_residual_stream",
        ]
    )
    """Module-name globs to wrap when `activation_checkpointing_mode` is "selected_modules".

    Defaults to every transformer-block submodule, including the GDN mixer (`blocks.*.attention`).
    Wrapping submodules individually (rather than the whole block) keeps `torch.compile` *outside*
    the checkpoint boundary, which is the order compile supports, while still recovering full-block
    activation memory. The opaque `fla` kernel's recompute metadata check is suppressed by passing
    `determinism_check="none"` through the activation checkpointing config.
    """
    compile_model: bool = True
    """Whether to apply torch.compile to model blocks."""
    fused_optimizer: bool = True
    """Whether to use fused AdamW."""
    cp_degree: int | None = None
    """Context parallelism degree. When set, enables context parallelism."""
    cp_strategy: Literal["llama3", "zig_zag", "ulysses"] = "llama3"
    """Context parallelism strategy."""


def build_ac_config(
    activation_memory_budget: float, compile_model: bool, mode: str = "budget", modules: list[str] | None = None
) -> TransformerActivationCheckpointingConfig | None:
    if mode == "selected_modules":
        # The OLMo-hybrid small suite's maintainer OLMo-core fork predates the configurable
        # determinism check (needed to suppress the opaque fla kernel's recompute metadata
        # check); only pass it when the installed olmo-core supports it.
        kwargs = {}
        if any(f.name == "determinism_check" for f in fields(TransformerActivationCheckpointingConfig)):
            kwargs["determinism_check"] = "none"
        return TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules, modules=modules, **kwargs
        )
    if activation_memory_budget < 1.0 and compile_model:
        return TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget, activation_memory_budget=activation_memory_budget
        )
    return None


def build_cp_config(training: TrainingConfig) -> TransformerContextParallelConfig | None:
    if training.cp_degree is None:
        return None
    if training.cp_strategy == "llama3":
        return TransformerContextParallelConfig.llama3(degree=training.cp_degree)
    elif training.cp_strategy == "zig_zag":
        return TransformerContextParallelConfig.zig_zag(degree=training.cp_degree)
    elif training.cp_strategy == "ulysses":
        return TransformerContextParallelConfig.ulysses(degree=training.cp_degree)
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


def setup_distributed_env(seed: int, timeout: datetime.timedelta | None = None) -> tuple[int, int, bool]:
    """Initialize the olmo-core training environment and return (global_rank, world_size, is_main_process)."""
    kwargs: dict[str, Any] = {"seed": seed}
    if timeout is not None:
        kwargs["timeout"] = timeout
    prepare_training_environment(**kwargs)
    global_rank = get_rank() if is_distributed() else 0
    world_size = get_world_size() if is_distributed() else 1
    is_main_process = global_rank == 0
    logger_utils.setup_logger(rank=global_rank)
    return global_rank, world_size, is_main_process


def build_scheduler(lr_scheduler_type: str, warmup_ratio: float, num_training_steps: int):
    """Build an olmo-core LR scheduler from a scheduler-type string."""
    warmup_steps = int(num_training_steps * warmup_ratio)
    if lr_scheduler_type == "cosine":
        return olmo_optim.CosWithWarmup(warmup_steps=warmup_steps)
    if lr_scheduler_type == "linear":
        return olmo_optim.LinearWithWarmup(warmup_steps=warmup_steps, alpha_f=0.0)
    if lr_scheduler_type == "constant":
        return olmo_optim.ConstantWithWarmup(warmup_steps=warmup_steps)
    raise ValueError(f"Unknown lr_scheduler_type: {lr_scheduler_type!r}")


def reload_hf_checkpoint_after_parallelization(train_module, model_name_or_path: str, work_dir: str) -> None:
    """Reload HF weights into a parallelized train_module.

    TransformerTrainModule.__init__ calls parallelize_model which calls init_weights,
    reinitializing all model weights. This reloads the HF checkpoint on top.
    """
    logger.info("Reloading HuggingFace weights after parallelization...")
    sd = train_module.model.state_dict()
    load_hf_model_with_hybrid_support(model_name_or_path, sd, work_dir=work_dir)
    train_module.model.load_state_dict(sd)


def build_base_callbacks(
    config_dict: dict,
    run_name: str | None,
    checkpointing_steps: int,
    ephemeral_save_interval: int | None,
    with_tracking: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    save_async: bool = True,
) -> dict[str, Any]:
    """Build the callbacks shared across SFT and DPO: beaker, gpu monitor, checkpointer, and optional wandb."""
    result: dict[str, Any] = {
        "beaker": olmo_core_callbacks.BeakerCallbackV2(config=config_dict),
        "gpu_monitor": train_callbacks.GPUMemoryMonitorCallback(),
        "checkpointer": build_checkpointer_callback(
            checkpointing_steps, ephemeral_save_interval, save_async=save_async
        ),
    }
    if with_tracking and wandb_project:
        result["wandb"] = train_callbacks.WandBCallback(
            name=run_name,
            entity=wandb_entity,
            project=wandb_project,
            config=config_dict,
            enabled=True,
            cancel_check_interval=10,
        )
    return result


def is_hf_checkpoint(path: str) -> bool:
    """Detect whether a model path is a HuggingFace checkpoint (vs olmo-core format).

    Returns True for HF hub IDs (e.g. 'allenai/Olmo-3-1025-7B'), local/weka paths
    containing config.json, and paths with a '-hf' component. Returns False for
    olmo-core distributed checkpoints.
    """
    if os.path.isdir(path):
        return os.path.isfile(os.path.join(path, "config.json"))
    parts = path.replace("\\", "/").split("/")
    if any("-hf" in part for part in parts):
        return True
    return not os.path.isabs(path)


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
    "Qwen/Qwen3-4B-Base": "qwen3_4B",
    "Qwen/Qwen3-8B": "qwen3_8B",
    "Qwen/Qwen3-14B": "qwen3_14B",
    "Qwen/Qwen3-32B": "qwen3_32B",
}


def _override_attn_backend(config_dict: Any, attn_backend: str) -> None:
    """Recursively override any `AttentionConfig.backend` field in-place.

    Maintainer-pipeline checkpoints (e.g. OLMo-hybrid small suite) bake in the backend
    (e.g. `flash_3`, H100-only) used at training time. Force it to match the requested
    `attn_backend` so the config also builds on other hardware (e.g. `flash_2`).
    """
    if isinstance(config_dict, dict):
        if config_dict.get("_CLASS_", "").endswith("AttentionConfig") and "backend" in config_dict:
            config_dict["backend"] = attn_backend
        for value in config_dict.values():
            _override_attn_backend(value, attn_backend)
    elif isinstance(config_dict, list):
        for item in config_dict:
            _override_attn_backend(item, attn_backend)


def _load_native_transformer_config(model_name_or_path: str, attn_backend: str | None) -> TransformerConfig | None:
    """Load a `TransformerConfig` from a maintainer-pipeline checkpoint's own config.

    Maintainer pipelines (e.g. OLMo-hybrid small suite) write a full HF-format checkpoint
    (suffixed `-hf`) next to the olmo-core-native checkpoint it was converted from. The
    native checkpoint's `config.json` embeds the exact `TransformerConfig` (as a `_CLASS_`-
    tagged dict) used to train it, so we can deserialize it directly instead of requiring a
    hardcoded named preset (there is no `olmo_hybrid_small` preset in `TransformerConfig`).

    Returns None if `model_name_or_path` doesn't look like such a checkpoint.
    """
    path = Path(model_name_or_path)
    if not path.is_dir() or not path.name.endswith("-hf"):
        return None
    native_dir = path.with_name(path.name[: -len("-hf")])
    config_path = native_dir / "config.json"
    if not config_path.is_file():
        return None
    with open(config_path) as f:
        data = json.load(f)
    model_dict = data.get("model")
    if model_dict is None:
        return None
    if attn_backend is not None:
        _override_attn_backend(model_dict, str(AttentionBackendName(attn_backend)))
    logger.info(f"Loading native TransformerConfig from '{config_path}'")
    return TransformerConfig.from_dict(model_dict)


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
        native_config = _load_native_transformer_config(model_name_or_config, attn_backend)
        if native_config is not None:
            native_config.vocab_size = vocab_size
            return native_config

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


def setup_model(
    model_config_args: ModelConfig, tc: TokenizerConfig | None = None, init_device: str = "cpu"
) -> tuple[Transformer, TransformerConfig]:
    model_name_or_path = model_config_args.model_name_or_path
    if is_hf_checkpoint(model_name_or_path):
        logger.info(f"Detected HuggingFace checkpoint at {model_name_or_path}")
        hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vocab_size = hf_config.vocab_size
    else:
        logger.info(f"Detected olmo-core checkpoint at {model_name_or_path}")
        assert model_config_args.config_name is not None, (
            "--config_name is required when model_name_or_path is an olmo-core checkpoint"
        )
        assert tc is not None, "tc (TokenizerConfig) is required for olmo-core checkpoints to derive vocab_size"
        vocab_size = to_oc_tokenizer_config(tc).padded_vocab_size()
    logger.info(f"Building OLMo-core model with vocab_size={vocab_size}")
    model_config = get_transformer_config(
        model_config_args.config_name or model_name_or_path,
        vocab_size,
        attn_backend=model_config_args.attn_implementation,
    )
    model_config.lm_head.loss_implementation = LMLossImplementation(model_config_args.loss_implementation)
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


def to_oc_tokenizer_config(tc: TokenizerConfig) -> OLMoCoreTokenizerConfig:
    """Map open-instruct TokenizerConfig to olmo-core's TokenizerConfig for NumpyFSL loading.

    Prefers the curated `dolma2()` preset for the dolma2/OLMo-2 family (its
    `padded_vocab_size()` rounding matches the trainer expectations); otherwise
    builds one directly from the HF tokenizer's special tokens.
    """
    identifier = tc.tokenizer_name_or_path or ""
    dolma2_markers = ("dolma2", "OLMo-2-1124", "OLMo-2-0325", "OLMo-2-0425")
    if any(marker in identifier for marker in dolma2_markers):
        return OLMoCoreTokenizerConfig.dolma2()
    eos_id = tc.tokenizer.eos_token_id
    bos_id = tc.tokenizer.bos_token_id
    # olmo-core's doc-boundary scanner requires an EOS *immediately followed by* BOS
    # when both are set; when the tokenizer reuses the same id for both (e.g.
    # olmo-3-tokenizer-instruct-dev has eos==bos==100257), that pattern never
    # appears in practice, so fall back to the EOS-only scanner.
    if bos_id == eos_id:
        bos_id = None
    return OLMoCoreTokenizerConfig(
        vocab_size=tc.tokenizer.vocab_size,
        eos_token_id=eos_id,
        pad_token_id=tc.tokenizer.pad_token_id,
        bos_token_id=bos_id,
        identifier=identifier or None,
    )


# olmo-core <-> HF weight-name mappings for the OLMo-hybrid small suite (`olmo_hybrid_small`
# architecture, a GatedDeltaNet linear-attention + full-attention hybrid). These match the
# maintainer's own `convert_olmo_hybrid_small_weights_to_hf.py` in the transformers fork.
# The olmo-core fork's generic `load_hf_model`/`convert_state_to_hf` only know the standard
# (non-hybrid) key mappings, so we implement the conversion ourselves.
HYBRID_SMALL_MODEL_TYPE = "olmo_hybrid_small"

_HYBRID_SMALL_TOP_LEVEL_TO_HF: dict[str, str] = {
    "embeddings.weight": "model.embed_tokens.weight",
    "embedding_norm.weight": "model.embed_norm.weight",
    "lm_head.norm.weight": "model.norm.weight",
    "lm_head.w_out.weight": "lm_head.weight",
}

# Per-block mappings shared by both layer types (peri-norm blocks + SwiGLU MLP).
_HYBRID_SMALL_COMMON_LAYER_TO_HF: dict[str, str] = {
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "feed_forward_norm.weight": "ffn_layernorm.weight",
    "post_feed_forward_norm.weight": "post_feedforward_layernorm.weight",
}

# GatedDeltaNet linear-attention blocks.
_HYBRID_SMALL_GDN_LAYER_TO_HF: dict[str, str] = {
    "attention.w_q.weight": "linear_attn.q_proj.weight",
    "attention.w_k.weight": "linear_attn.k_proj.weight",
    "attention.w_v.weight": "linear_attn.v_proj.weight",
    "attention.w_out.weight": "linear_attn.o_proj.weight",
    "attention.w_g.weight": "linear_attn.g_proj.weight",
    "attention.w_a.weight": "linear_attn.a_proj.weight",
    "attention.w_b.weight": "linear_attn.b_proj.weight",
    "attention.o_norm.weight": "linear_attn.o_norm.weight",
    "attention.q_conv1d.weight": "linear_attn.q_conv1d.weight",
    "attention.k_conv1d.weight": "linear_attn.k_conv1d.weight",
    "attention.v_conv1d.weight": "linear_attn.v_conv1d.weight",
    "attention.A_log": "linear_attn.A_log",
    "attention.dt_bias": "linear_attn.dt_bias",
}

# Gated full-attention blocks.
_HYBRID_SMALL_FULL_ATTN_LAYER_TO_HF: dict[str, str] = {
    "attention.w_q.weight": "self_attn.q_proj.weight",
    "attention.w_k.weight": "self_attn.k_proj.weight",
    "attention.w_v.weight": "self_attn.v_proj.weight",
    "attention.w_out.weight": "self_attn.o_proj.weight",
    "attention.w_g.weight": "self_attn.attn_gate.weight",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
}

_HYBRID_SMALL_BLOCK_RE = re.compile(r"blocks\.(\d+)\.(.*)")


def is_hybrid_small_checkpoint(model_name_or_path: str) -> bool:
    """Whether an HF checkpoint uses the OLMo-hybrid small suite architecture."""
    hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    return getattr(hf_config, "model_type", None) == HYBRID_SMALL_MODEL_TYPE


def _hybrid_small_layer_mapping(layer_type: str) -> dict[str, str]:
    mapping = dict(_HYBRID_SMALL_COMMON_LAYER_TO_HF)
    mapping.update(
        _HYBRID_SMALL_GDN_LAYER_TO_HF if layer_type == "linear_attention" else _HYBRID_SMALL_FULL_ATTN_LAYER_TO_HF
    )
    return mapping


def _convert_hybrid_small_state_from_hf(
    hf_state_dict: dict[str, torch.Tensor], n_layers: int
) -> dict[str, torch.Tensor]:
    """Convert an HF `olmo_hybrid_small` state dict to olmo-core naming.

    Per-layer types are detected by the presence of the GDN-specific `A_log` parameter.
    """
    layer_types = [
        "linear_attention" if f"model.layers.{i}.linear_attn.A_log" in hf_state_dict else "full_attention"
        for i in range(n_layers)
    ]
    converted = {oc_key: hf_state_dict[hf_key] for oc_key, hf_key in _HYBRID_SMALL_TOP_LEVEL_TO_HF.items()}
    for layer_i, layer_type in enumerate(layer_types):
        for oc_key, hf_key in _hybrid_small_layer_mapping(layer_type).items():
            converted[f"blocks.{layer_i}.{oc_key}"] = hf_state_dict[f"model.layers.{layer_i}.{hf_key}"]
    return converted


def convert_hybrid_small_state_to_hf(state_dict: dict[str, torch.Tensor], n_layers: int) -> dict[str, torch.Tensor]:
    """Convert an olmo-core `olmo_hybrid_small` state dict to HF naming (inverse of the above)."""
    layer_types = [
        "linear_attention" if f"blocks.{i}.attention.A_log" in state_dict else "full_attention"
        for i in range(n_layers)
    ]
    converted = {hf_key: state_dict[oc_key] for oc_key, hf_key in _HYBRID_SMALL_TOP_LEVEL_TO_HF.items()}
    for layer_i, layer_type in enumerate(layer_types):
        for oc_key, hf_key in _hybrid_small_layer_mapping(layer_type).items():
            converted[f"model.layers.{layer_i}.{hf_key}"] = state_dict[f"blocks.{layer_i}.{oc_key}"]
    return converted


def make_hybrid_small_name_mapper(state_dict_keys: Iterable[str]) -> Callable[[str], str]:
    """Build an olmo-core -> HF weight-name mapper for `olmo_hybrid_small` models.

    Used during GRPO weight sync to translate learner parameter names to the HF names the
    vLLM model expects. The mapping is layer-type dependent (`attention.w_q` becomes
    `linear_attn.q_proj` in GDN blocks but `self_attn.q_proj` in full-attention blocks), so
    GDN layers are detected up front from the GDN-specific `A_log` key.
    """
    gdn_layers = {
        int(match.group(1))
        for key in state_dict_keys
        if (match := _HYBRID_SMALL_BLOCK_RE.match(key)) and match.group(2) == "attention.A_log"
    }

    def mapper(name: str) -> str:
        if name in _HYBRID_SMALL_TOP_LEVEL_TO_HF:
            return _HYBRID_SMALL_TOP_LEVEL_TO_HF[name]
        match = _HYBRID_SMALL_BLOCK_RE.match(name)
        if match:
            layer_idx, rest = match.group(1), match.group(2)
            layer_type = "linear_attention" if int(layer_idx) in gdn_layers else "full_attention"
            mapping = _hybrid_small_layer_mapping(layer_type)
            if rest in mapping:
                return f"model.layers.{layer_idx}.{mapping[rest]}"
        return name

    return mapper


def load_hf_model_with_hybrid_support(
    model_name_or_path: str, model_state_dict: dict[str, Any], work_dir: str
) -> None:
    """Load HF checkpoint weights into an olmo-core state dict, with OLMo-hybrid small support.

    The olmo-core fork's `load_hf_model` only implements the generic (non-hybrid) HF ->
    olmo-core weight conversion, which doesn't know about `olmo_hybrid_small`'s GatedDeltaNet
    linear-attention layers. Dispatch to our own conversion for that architecture; delegate to
    olmo-core otherwise.

    Note: entries of ``model_state_dict`` may be *replaced* (not copied into in-place), so
    callers must apply the result with ``model.load_state_dict(model_state_dict)``.
    """
    hf_config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if getattr(hf_config, "model_type", None) != HYBRID_SMALL_MODEL_TYPE:
        load_hf_model(model_name_or_path, model_state_dict, work_dir=work_dir)
        return

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
    converted = _convert_hybrid_small_state_from_hf(hf_model.state_dict(), hf_config.num_hidden_layers)
    for key in sorted(converted.keys()):
        state = converted[key]
        current = model_state_dict[key]
        if isinstance(current, DTensor):
            state = distribute_tensor(state, current.device_mesh, current.placements)
        model_state_dict[key] = state


def verify_can_save_as_hf(model_config: TransformerConfig, original_model_name_or_path: str) -> None:
    """Fail fast if the run cannot later be exported to HF format.

    Builds the olmo-core model and the target HF model on meta, runs the
    state-dict converter, and verifies the converted keys exactly cover the HF
    model's expected parameters. Raises before any training starts.
    """
    hf_config = transformers.AutoConfig.from_pretrained(original_model_name_or_path, trust_remote_code=True)
    olmo_core_model = model_config.build(init_device="meta")
    olmo_core_state = olmo_core_model.state_dict()

    if getattr(hf_config, "model_type", None) == HYBRID_SMALL_MODEL_TYPE:
        converted = convert_hybrid_small_state_to_hf(olmo_core_state, hf_config.num_hidden_layers)
    else:
        converted = olmo_hf_convert.convert_state_to_hf(hf_config, olmo_core_state)

    with accelerate.init_empty_weights():
        hf_model = transformers.AutoModelForCausalLM.from_config(hf_config)
    expected = set(hf_model.state_dict().keys())
    produced = set(converted.keys())

    missing = expected - produced
    extra = produced - expected
    if missing or extra:
        raise RuntimeError(
            f"HF export is not implemented for {original_model_name_or_path} "
            f"(model_type={getattr(hf_config, 'model_type', None)}). "
            f"Missing keys: {sorted(missing)}. Extra keys: {sorted(extra)}."
        )
    logger.info(
        f"Verified HF export works for {original_model_name_or_path} "
        f"(model_type={getattr(hf_config, 'model_type', None)}, {len(expected)} params)."
    )


def save_state_dict_as_hf(
    state_dict: dict[str, torch.Tensor],
    save_dir: str,
    original_model_name_or_path: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> None:
    """Convert an olmo-core state dict to HuggingFace format and save it to disk.

    Loads the target HF config from ``original_model_name_or_path``, converts the
    olmo-core ``state_dict`` keys/shapes to the matching HF layout, materializes
    an HF model with those weights, and writes the model + tokenizer to
    ``save_dir``.
    """
    hf_config = transformers.AutoConfig.from_pretrained(original_model_name_or_path, trust_remote_code=True)
    if getattr(hf_config, "model_type", None) == HYBRID_SMALL_MODEL_TYPE:
        converted = convert_hybrid_small_state_to_hf(state_dict, hf_config.num_hidden_layers)
    else:
        converted = olmo_hf_convert.convert_state_to_hf(hf_config, state_dict)
    converted = {k: v.contiguous() for k, v in converted.items()}

    with accelerate.init_empty_weights():
        hf_model = transformers.AutoModelForCausalLM.from_config(hf_config)
    hf_model.load_state_dict(converted, assign=True)

    os.makedirs(save_dir, exist_ok=True)
    hf_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def doc_lens_from_attention_mask(attention_mask_BS: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """Convert an integer-coded packed attention_mask to OLMo-core ``doc_lens`` / ``max_doc_lens``.

    ``attention_mask_BS`` is a ``(batch_size, sequence_length)`` integer tensor produced by the
    packed-sequence collator: each token's value is the 1-indexed document ID it belongs to
    (1, 2, 3, ...) within its row, and 0 marks padding. Tokens sharing a value belong to the same
    document and are expected to be contiguous along the sequence dimension.

    Padding spans (zeros in ``attention_mask_BS``) are emitted as their own segments: OLMo-core
    flattens ``doc_lens`` into a single cumulative offset across the batch, so dropping padding
    would shift subsequent rows' document boundaries into the previous row's padding region.
    """
    batch_size = attention_mask_BS.size(0)
    rows = [torch.unique_consecutive(attention_mask_BS[i], return_counts=True)[1] for i in range(batch_size)]
    max_docs = max(t.numel() for t in rows)
    doc_lens_BD = torch.zeros((batch_size, max_docs), dtype=torch.int32, device=attention_mask_BS.device)
    for i, t in enumerate(rows):
        doc_lens_BD[i, : t.numel()] = t.to(torch.int32)
    max_doc_lens_B = doc_lens_BD.max(dim=1).values.tolist()
    return doc_lens_BD, max_doc_lens_B


def doc_lens_from_cu_seq_lens(cu_seq_lens_k_D1: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, list[int]]:
    """Convert flash-attn ``cu_seq_lens_k`` to OLMo-core ``doc_lens`` / ``max_doc_lens``.

    ``cu_seq_lens_k_D1`` is a ``(num_docs + 1,)`` cumulative-offset tensor with a leading 0.
    ``seq_len`` is the (possibly padded) length of the packed row; any trailing pad span is
    emitted as its own segment so ``doc_lens`` sums to ``seq_len``, matching OLMo-core's
    requirement that document boundaries cover the full sequence.
    """
    seq_lens_D = cu_seq_lens_k_D1.diff().to(torch.int32)
    total = int(cu_seq_lens_k_D1[-1].item())
    if total < seq_len:
        pad_D = torch.tensor([seq_len - total], dtype=torch.int32, device=seq_lens_D.device)
        seq_lens_D = torch.cat([seq_lens_D, pad_D])
    doc_lens_BD = seq_lens_D.unsqueeze(0)
    max_doc_lens_B = [int(doc_lens_BD.max().item())]
    return doc_lens_BD, max_doc_lens_B
