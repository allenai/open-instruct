# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/__init__.py
"""vLLM registration hook for OLMoE3."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ARCHITECTURE = "Olmo3MoeForCausalLM"
VISION_ARCHITECTURE = "Olmo3MoeVisionForConditionalGeneration"
VISION_MODEL_TYPE = "olmo3moe_vision"

# Lazy "<module>:<class>" targets: vLLM imports them in its inspection subprocess, so neither
# module is imported here and a vision-side import problem cannot break the text executor.
MODEL_TARGETS = {
    ARCHITECTURE: "olmoe3_vllm_plugin.model:Olmo3MoeForCausalLM",
    VISION_ARCHITECTURE: "olmoe3_vllm_plugin.vision:Olmo3MoeVisionForConditionalGeneration",
}


def _vision_config_registered() -> bool:
    """Whether the transformers plugin registered the whole-model export's HF config."""
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError:
        return False
    return VISION_MODEL_TYPE in CONFIG_MAPPING


def register() -> None:
    """Register the HF configs and the lazy vLLM executors (text, and whole-model LM-only)."""
    # Not a declared dependency: the package isn't on PyPI, and eval jobs install each
    # plugin separately from this repo at the evaluated ref.
    try:
        import olmoe3_transformers_plugin
    except ModuleNotFoundError as exc:
        if exc.name != "olmoe3_transformers_plugin":
            raise
        raise ImportError(
            "olmoe3-vllm-plugin requires olmoe3-transformers-plugin. Install it from the same "
            "scaling-ladders ref: #subdirectory=ladders/olmoe3/transformers_plugin"
        ) from exc

    olmoe3_transformers_plugin.register()

    from vllm import ModelRegistry

    for architecture, target in MODEL_TARGETS.items():
        ModelRegistry.register_model(architecture, target)
        logger.warning("Registered %s with the OLMoE3 vLLM plugin", architecture)

    # Eval jobs read config.json with trust_remote_code=False, so a multimodal export only
    # resolves when the transformers plugin registers `olmo3moe_vision` (it does from the ref
    # that added the multimodal converter); the text model never depends on it.
    if not _vision_config_registered():
        logger.warning(
            "olmoe3-transformers-plugin did not register model_type %r; %s can only serve "
            "exports whose config.json is readable without it (trust_remote_code=True).",
            VISION_MODEL_TYPE,
            VISION_ARCHITECTURE,
        )


__all__ = ["ARCHITECTURE", "MODEL_TARGETS", "VISION_ARCHITECTURE", "VISION_MODEL_TYPE", "register"]
