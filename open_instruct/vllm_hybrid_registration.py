"""vLLM worker registration for the earlier small-suite checkpoint name."""

from vllm import ModelRegistry

from open_instruct import hybrid_models


def register() -> None:
    hybrid_models.register_models()
    ModelRegistry.register_model("OlmoHybridSmallForCausalLM", "vllm_plugin.mainline_ladder:MainlineLadderForCausalLM")
