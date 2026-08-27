"""Config dataclasses and builders for multimodal (Molmo stage 2) SFT on OLMo-core.

A thin transcription of OLMo-core's ``src/scripts/train/Molmo2-Stage2.py`` into
open-instruct's ArgumentParserPlus dataclass style (docs/design/multimodal_sft.md §3).
The model, train module, collator, and data loader are all OLMo-core's own; nothing
here subclasses them.
"""

import dataclasses
import typing

from olmo_core.config import DType
from olmo_core.data.multimodal import MultimodalCollatorConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode, TransformerBlockConfig
from olmo_core.nn.vision import MultimodalLM, MultimodalLMConfig, molmo2_loader
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride, PerGroupScheduler
from olmo_core.train.train_module import (
    MultimodalTransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device
from transformers import AutoConfig, AutoModelForImageTextToText

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


@dataclasses.dataclass(kw_only=True)
class MultimodalModelConfig:
    """Which multimodal model to train and where its initial weights come from."""

    base_hf_model_id: str = "allenai/Molmo2-4B"
    """HF repo that provides the model config (via ``molmo2_config_from_hf_config``
    unless ``model_preset`` is set), the tokenizer, and — when ``model_name_or_path``
    is unset — the initial weights."""

    model_name_or_path: str | None = None
    """An OLMo-core checkpoint dir (e.g. a stage-1 caption-pretraining run) to
    initialize from via the trainer's ``load_path``. None initializes from
    ``base_hf_model_id`` through the molmo2 loader instead."""

    model_preset: str | None = None
    """A ``MultimodalLMConfig`` classmethod name (``molmo2_4B``, ``molmo2_8B``, and —
    once the Olmo 3 backbone workstream lands upstream — ``molmo3_7B``). None derives
    the config from ``base_hf_model_id``'s HF config."""

    residual_dropout: float = 0.1
    """Stage2 fine-tuning setting: ``config.lm.block.dropout`` (mm_olmo pairs
    llm.residual_dropout=0.1 with response_residual_dropout=0.0)."""

    tokenizer_name_or_path: str | None = None
    """Defaults to ``base_hf_model_id``."""

    trust_remote_code: bool = True


@dataclasses.dataclass
class MultimodalTrainingConfig:
    """Stage2-parity training hyperparameters (Molmo2-Stage2.py values as defaults)."""

    max_seq_length: int = 16384
    global_batch_instances: int = 128
    rank_microbatch_instances: int = 2
    learning_rate: float = 1e-5
    """LM learning rate; the connector and vision tower get their own groups."""
    connector_lr: float = 5e-6
    vision_lr: float = 5e-6
    warmup_steps: int = 200
    alpha_f: float = 0.1
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-6
    max_grad_norm: float = 1.0
    z_loss_multiplier: float = 1e-4
    max_train_steps: int = 300_000
    freeze_params: list[str] = dataclasses.field(default_factory=list)
    """fnmatch globs frozen before optimizer construction, e.g. ["vision.*"]."""
    response_logits_only: bool = True
    compile_model: bool = True
    compile_vision: bool = True
    compile_connector: bool = True
    vision_activation_checkpointing: bool = True
    connector_activation_checkpointing: bool = True
    ac_block_interval: int = 2
    """LM activation checkpointing: every Nth block (selected_blocks mode)."""
    dp_shard_degree: int | None = None
    """FSDP shard degree. None: shard within a node and replicate across nodes (HSDP)
    when the world is larger than one node, plain FSDP otherwise — mirroring
    ``olmo_core_finetune.py``. Stage2 itself always uses plain FSDP on one node."""


def build_multimodal_model_config(model: MultimodalModelConfig) -> MultimodalLMConfig:
    """Stage2 ``_build_model_config``: preset or HF-derived config, plus dropout."""
    molmo2_loader.ensure_default_rope_registered()
    if model.model_preset is not None:
        preset = getattr(MultimodalLMConfig, model.model_preset, None)
        if preset is None or not callable(preset):
            raise ValueError(f"MultimodalLMConfig has no preset {model.model_preset!r}")
        config = preset()
    else:
        hf_config = AutoConfig.from_pretrained(model.base_hf_model_id, trust_remote_code=model.trust_remote_code)
        config = molmo2_loader.molmo2_config_from_hf_config(hf_config)
    if isinstance(config.lm.block, dict):
        # Per-block-index config dicts (olmo_core declares dict[str, TransformerBlockConfig]).
        blocks = typing.cast(list[TransformerBlockConfig], list(config.lm.block.values()))
    else:
        blocks = [config.lm.block]
    for block in blocks:
        block.dropout = model.residual_dropout
    return config


def setup_multimodal_model(model: MultimodalModelConfig) -> tuple[MultimodalLM, MultimodalLMConfig, bool]:
    """Build the model and load initial weights, BEFORE train-module wrapping.

    Returns (model, model_config, defer_load): ``defer_load`` is True when weights
    come from an OLMo-core checkpoint via the trainer's ``load_path``
    (``model_name_or_path``); the model is then materialized empty with tied word
    embeddings restored, exactly like Stage2's ``train()``. Otherwise HF weights from
    ``base_hf_model_id`` are loaded here through the molmo2 loader.
    """
    model_config = build_multimodal_model_config(model)
    lm = model_config.build(init_device="meta")
    if model.model_name_or_path is not None:
        logger.info("Deferring weight init to trainer load_path=%s", model.model_name_or_path)
        lm.to_empty(device=get_default_device())
        # `to_empty` breaks weight tying (Molmo2-4B). Restore the share *before* FSDP
        # wrapping and the checkpoint load so both state-dict keys fill one parameter.
        molmo2_loader.retie_word_embeddings(lm)
        return lm, model_config, True

    logger.info("Loading HF weights from %s ...", model.base_hf_model_id)
    hf = AutoModelForImageTextToText.from_pretrained(model.base_hf_model_id, trust_remote_code=model.trust_remote_code)
    molmo2_loader.reinit_rope_buffers(hf)
    converted = molmo2_loader.molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), model_config)
    del hf
    lm.to_empty(device=get_default_device())
    lm.load_state_dict(converted, strict=False)
    del converted
    molmo2_loader.retie_word_embeddings(lm)
    return lm, model_config, False


def build_multimodal_train_module_config(
    training: MultimodalTrainingConfig, *, world_size: int, gpus_per_node: int
) -> MultimodalTransformerTrainModuleConfig:
    """Stage2's optimizer/scheduler/parallelism block, with multi-node HSDP added."""
    dp_shard_degree = training.dp_shard_degree or min(gpus_per_node, world_size)
    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp if world_size > dp_shard_degree else DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
    )
    if world_size > dp_shard_degree:
        dp_config.shard_degree = dp_shard_degree

    component_scheduler = CosWithWarmup(warmup=training.warmup_steps, alpha_f=training.alpha_f)
    return MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=training.rank_microbatch_instances * training.max_seq_length,
        max_sequence_length=training.max_seq_length,
        optim=AdamWConfig(
            lr=training.learning_rate,
            betas=(training.adam_beta1, training.adam_beta2),
            eps=training.adam_eps,
            weight_decay=training.weight_decay,
            group_overrides=[
                OptimGroupOverride(
                    params=["connector.*"],
                    opts=dict(lr=training.connector_lr, weight_decay=0.0, scheduler_name="connector"),
                ),
                OptimGroupOverride(
                    params=["vision.*"], opts=dict(lr=training.vision_lr, weight_decay=0.0, scheduler_name="vision")
                ),
            ],
        ),
        z_loss_multiplier=training.z_loss_multiplier,
        max_grad_norm=training.max_grad_norm,
        compile_model=training.compile_model,
        compile_vision=training.compile_vision,
        compile_connector=training.compile_connector,
        vision_activation_checkpointing=training.vision_activation_checkpointing,
        connector_activation_checkpointing=training.connector_activation_checkpointing,
        autocast_precision=DType.bfloat16,
        scheduler=PerGroupScheduler(
            schedulers={"connector": component_scheduler, "vision": component_scheduler}, default=component_scheduler
        ),
        dp_config=dp_config,
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_blocks, block_interval=training.ac_block_interval
        ),
        freeze_params=training.freeze_params or None,
        response_logits_only=training.response_logits_only,
    )


def build_multimodal_collator_config(tokenizer, max_seq_length: int) -> MultimodalCollatorConfig:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path!r} has no pad token id")
    return MultimodalCollatorConfig(
        pad_token_id=pad_token_id, label_ignore_index=-100, pad_sequence_length=max_seq_length
    )


def global_batch_size_tokens(training: MultimodalTrainingConfig) -> int:
    return training.global_batch_instances * training.max_seq_length
