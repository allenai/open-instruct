# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/transformers_plugin/src/olmoe3_transformers_plugin/__init__.py
"""Transformers registration for the OLMoE3 KDA + LatentMoE ladder."""

from __future__ import annotations

from typing import TYPE_CHECKING

MODEL_TYPE = "olmo3moe"
VISION_MODEL_TYPE = "olmo3moe_vision"
_REGISTERED = False


def _safe_register(register_fn, *args) -> None:
    try:
        register_fn(*args, exist_ok=True)
        return
    except TypeError:
        pass
    except ValueError as exc:
        if "already" in str(exc).lower() and "register" in str(exc).lower():
            return
        raise
    try:
        register_fn(*args)
    except ValueError as exc:
        if "already" not in str(exc).lower() or "register" not in str(exc).lower():
            raise


def register() -> None:
    """Register config, base model, and causal LM with Transformers Auto classes.

    Also registers the multimodal export's ``olmo3moe_vision`` config and its whole-model
    wrapper (``Olmo3MoeVisionForConditionalGeneration``, whose forward runs the text model).
    """

    global _REGISTERED
    if _REGISTERED:
        return
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM, Olmo3MoeModel
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    from .hf_vision.configuration_olmo3moe_vision import Olmo3MoeVisionConfig
    from .hf_vision.modeling_olmo3moe_vision import Olmo3MoeVisionForConditionalGeneration

    _safe_register(AutoConfig.register, MODEL_TYPE, Olmo3MoeConfig)
    _safe_register(AutoModel.register, Olmo3MoeConfig, Olmo3MoeModel)
    _safe_register(AutoModelForCausalLM.register, Olmo3MoeConfig, Olmo3MoeForCausalLM)
    _safe_register(AutoConfig.register, VISION_MODEL_TYPE, Olmo3MoeVisionConfig)
    _safe_register(
        AutoModel.register, Olmo3MoeVisionConfig, Olmo3MoeVisionForConditionalGeneration
    )
    _safe_register(
        AutoModelForCausalLM.register,
        Olmo3MoeVisionConfig,
        Olmo3MoeVisionForConditionalGeneration,
    )
    _REGISTERED = True


if TYPE_CHECKING:
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import (
        Olmo3MoeForCausalLM,
        Olmo3MoeModel,
        Olmo3MoePreTrainedModel,
    )


def __getattr__(name: str):
    if name == "Olmo3MoeConfig":
        from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig

        return Olmo3MoeConfig
    if name in ("Olmo3MoeForCausalLM", "Olmo3MoeModel", "Olmo3MoePreTrainedModel"):
        from olmo_core.nn.moe.v2.hf import modeling_olmo3moe

        return getattr(modeling_olmo3moe, name)
    if name == "Olmo3MoeVisionConfig":
        from .hf_vision.configuration_olmo3moe_vision import Olmo3MoeVisionConfig

        return Olmo3MoeVisionConfig
    if name in ("Olmo3MoeVisionForConditionalGeneration", "Olmo3MoeVisionPreTrainedModel"):
        from .hf_vision import modeling_olmo3moe_vision

        return getattr(modeling_olmo3moe_vision, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MODEL_TYPE",
    "VISION_MODEL_TYPE",
    "Olmo3MoeConfig",
    "Olmo3MoeForCausalLM",
    "Olmo3MoeModel",
    "Olmo3MoePreTrainedModel",
    "Olmo3MoeVisionConfig",
    "Olmo3MoeVisionForConditionalGeneration",
    "Olmo3MoeVisionPreTrainedModel",
    "register",
]
