"""Register the release architecture and its earlier small-suite model names."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers_plugin import configuration_mainline_ladder, modeling_mainline_ladder, register

HYBRID_MODEL_TYPES = frozenset({"mainline_ladder", "olmo_hybrid_small"})


class OlmoHybridSmallConfig(configuration_mainline_ladder.MainlineLadderConfig):
    """Compatibility name for checkpoints written before the release rename."""

    model_type = "olmo_hybrid_small"


class OlmoHybridSmallModel(modeling_mainline_ladder.MainlineLadderModel):
    config_class = OlmoHybridSmallConfig


class OlmoHybridSmallForCausalLM(modeling_mainline_ladder.MainlineLadderForCausalLM):
    config_class = OlmoHybridSmallConfig


def register_models() -> None:
    """Register both names in every driver/trainer process before HF loading."""
    register()
    AutoConfig.register("olmo_hybrid_small", OlmoHybridSmallConfig, exist_ok=True)
    AutoModel.register(OlmoHybridSmallConfig, OlmoHybridSmallModel, exist_ok=True)
    AutoModelForCausalLM.register(OlmoHybridSmallConfig, OlmoHybridSmallForCausalLM, exist_ok=True)
