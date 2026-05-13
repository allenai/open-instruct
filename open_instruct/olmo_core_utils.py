"""
OLMo-core utility functions, shared training configurations, and model configuration mappings.
"""

import datetime
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.distributed as dist
import transformers
from olmo_core import optim as olmo_optim
from olmo_core.data import TokenizerConfig as OLMoCoreTokenizerConfig
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.checkpoint import load_hf_model, save_hf_model
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
    compile_model: bool = True
    """Whether to apply torch.compile to model blocks."""
    fused_optimizer: bool = True
    """Whether to use fused AdamW."""
    cp_degree: int | None = None
    """Context parallelism degree. When set, enables context parallelism."""
    cp_strategy: Literal["llama3", "zig_zag", "ulysses"] = "llama3"
    """Context parallelism strategy."""


def build_ac_config(
    activation_memory_budget: float, compile_model: bool
) -> TransformerActivationCheckpointingConfig | None:
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
    load_hf_model(model_name_or_path, sd, work_dir=work_dir)
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
