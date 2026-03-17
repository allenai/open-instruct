"""
OLMo-core utility functions and configuration mappings.

This module provides common utilities for working with OLMo-core models,
including model configuration mappings and helper functions.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import transformers
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim.config import INITIAL_LR_FIELD, LR_FIELD, OptimConfig
from olmo_core.utils import move_to_device

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

OLMO_MODEL_CONFIG_MAP: dict[str, str] = {
    "allenai/OLMo-2-0425-1B": "olmo2_1B_v2",
    "allenai/OLMo-2-1124-7B": "olmo2_7B",
    "allenai/OLMo-2-1124-13B": "olmo2_13B",
    "allenai/OLMo-2-0325-32B": "olmo2_32B",
    "allenai/Olmo-3-1025-7B": "olmo3_7B",
    "allenai/OLMoE-1B-7B-0924": "olmoe_1B_7B",
    "Qwen/Qwen3-0.6B": "qwen3_0_6B",
    "Qwen/Qwen3-1.7B": "qwen3_1_7B",
    "Qwen/Qwen3-4B": "qwen3_4B",
    "Qwen/Qwen3-8B": "qwen3_8B",
    "Qwen/Qwen3-14B": "qwen3_14B",
    "Qwen/Qwen3-32B": "qwen3_32B",
}


def get_transformer_config(
    model_name_or_config: str, vocab_size: int, attn_backend: str | None = None
) -> TransformerConfig:
    """Get the appropriate TransformerConfig for a given model name or config name.

    Args:
        model_name_or_config: HuggingFace model name, path, or direct config name (e.g., 'olmo3_7B').
        vocab_size: Vocabulary size for the model.
        attn_backend: Attention backend name (e.g., 'flash_2', 'flash_3'). If None, uses default.

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
    kwargs: dict = {"vocab_size": vocab_size}
    if attn_backend is not None:
        kwargs["attn_backend"] = AttentionBackendName(attn_backend)
    return getattr(TransformerConfig, config_name)(**kwargs)


@dataclass
class MuonConfig(OptimConfig):
    lr: float = 0.02
    mu: float = 0.95
    weight_decay: float = 0.0
    epsilon: float = 1e-8
    adjust_lr: str = "spectral_norm"
    nesterov: bool = False
    adamw_lr: float | None = None
    lm_head_lr_scale: bool = True
    d_model: int = 0

    @classmethod
    def optimizer(cls):
        from dion import Muon  # noqa: PLC0415

        return Muon

    def build(self, model: nn.Module, strict: bool = True) -> torch.optim.Optimizer:
        adamw_lr = self.adamw_lr if self.adamw_lr is not None else self.lr

        muon_params = []
        adamw_params = []
        embedding_params = []
        lm_head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("embeddings"):
                embedding_params.append(param)
            elif name.startswith("lm_head"):
                lm_head_params.append(param)
            elif "blocks" in name and param.ndim == 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        lm_head_lr = adamw_lr
        if self.lm_head_lr_scale and self.d_model > 0:
            lm_head_lr = adamw_lr / math.sqrt(self.d_model)

        param_groups: list[dict[str, Any]] = []
        if muon_params:
            param_groups.append({"params": muon_params})
        if adamw_params:
            param_groups.append({"params": adamw_params, "algorithm": "adamw", "lr": adamw_lr})
        if embedding_params:
            param_groups.append({"params": embedding_params, "algorithm": "adamw", "lr": adamw_lr})
        if lm_head_params:
            param_groups.append({"params": lm_head_params, "algorithm": "adamw", "lr": lm_head_lr})

        distributed_mesh = None
        for param in model.parameters():
            if hasattr(param, "device_mesh"):
                distributed_mesh = param.device_mesh
                break

        from dion import Muon  # noqa: PLC0415

        optim = Muon(
            param_groups,
            distributed_mesh=distributed_mesh,
            lr=self.lr,
            mu=self.mu,
            weight_decay=self.weight_decay,
            epsilon=self.epsilon,
            adjust_lr=self.adjust_lr,
            nesterov=self.nesterov,
        )

        fixed_fields_per_group: list[dict[str, Any]] = [{} for _ in optim.param_groups]
        for fixed_fields, group in zip(fixed_fields_per_group, optim.param_groups):
            lr = group.get(LR_FIELD, self.lr)
            if self.compile:
                group[LR_FIELD] = move_to_device(torch.tensor(lr), self.device)
            else:
                group[LR_FIELD] = lr
            group.setdefault(INITIAL_LR_FIELD, lr)
            for k in self.fixed_fields:
                if k in group:
                    fixed_fields[k] = group[k]

        logger.info(f"Building Muon optimizer with {len(optim.param_groups)} param group(s)...")
        for g_idx, group in enumerate(optim.param_groups):
            group_fields_list = "\n - ".join([f"{k}: {v}" for k, v in group.items() if k != "params"])
            if group_fields_list:
                logger.info(f"Group {g_idx}, {len(group['params'])} parameter(s):\n - {group_fields_list}")
            else:
                logger.info(f"Group {g_idx}, {len(group['params'])} parameter(s)")

        if self.compile:
            logger.info("Compiling optimizer step...")
            optim.step = torch.compile(optim.step)

        def reset_fixed_fields(opt: torch.optim.Optimizer):
            for ff, group in zip(fixed_fields_per_group, opt.param_groups):
                group.update(ff)

        optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return optim


def save_state_dict_as_hf(model_config, state_dict, save_dir, original_model_name_or_path, tokenizer):
    unwrapped_model = model_config.build(init_device="cpu")
    unwrapped_model.load_state_dict(state_dict)
    save_hf_model(save_dir=save_dir, model_state_dict=state_dict, model=unwrapped_model, save_overwrite=True)
    tokenizer.save_pretrained(save_dir)
    original_config = transformers.AutoConfig.from_pretrained(original_model_name_or_path)
    original_config.save_pretrained(save_dir)
