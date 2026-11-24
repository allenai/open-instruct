# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/transformers_plugin/src/olmoe3_transformers_plugin/hf_vision/configuration_olmo3moe_vision.py
"""HF config of a multimodal OLMoE3 export: an ``olmo3moe`` LM plus an olmo-core vision stack."""

from typing import Any, Dict, List, Optional

from transformers.configuration_utils import PretrainedConfig

try:  # Bundled in an exported checkpoint next to the text team's files.
    from .configuration_olmo3moe import Olmo3MoeConfig
except ImportError:  # Installed plugin package: the text config lives in olmo-core.
    from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig

MODEL_TYPE = "olmo3moe_vision"
ARCHITECTURE = "Olmo3MoeVisionForConditionalGeneration"


class Olmo3MoeVisionConfig(PretrainedConfig):
    """
    Configuration of a whole multimodal model exported from an olmo-core
    ``MultimodalLMConfig`` checkpoint.

    ``text_config`` is exactly the :class:`Olmo3MoeConfig` a text-only export of the same
    language model produces. ``vision_config`` and ``connector_config`` are the olmo-core
    ``VisionEncoderConfig`` / ``VisionConnectorConfig`` dictionaries, kept verbatim, and
    ``olmo_core_config`` is the source checkpoint's full ``config.json``.

    :param text_config: The language model's ``olmo3moe`` config (dict or instance).
    :param vision_config: The olmo-core vision encoder config as a dict.
    :param connector_config: The olmo-core vision connector config as a dict.
    :param image_patch_token_id: LM vocabulary id of the ``<im_patch>`` placeholder token.
    :param vit_layers: ViT hidden-state layers concatenated before the connector.
    :param olmo_core_config: The source checkpoint's experiment config.
    """

    model_type = MODEL_TYPE
    sub_configs = {"text_config": Olmo3MoeConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: Optional[Any] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        connector_config: Optional[Dict[str, Any]] = None,
        image_patch_token_id: Optional[int] = None,
        vit_layers: Optional[List[int]] = None,
        olmo_core_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_model_type = text_config.pop("model_type", Olmo3MoeConfig.model_type)
            if text_model_type != Olmo3MoeConfig.model_type:
                raise ValueError(
                    f"text_config.model_type must be {Olmo3MoeConfig.model_type!r}, "
                    f"got {text_model_type!r}"
                )
            text_config = Olmo3MoeConfig(**text_config)
        elif text_config is None:
            text_config = Olmo3MoeConfig()
        # Set before the base initializer, whose validation already consults the text config.
        self.text_config = text_config
        self.vision_config = dict(vision_config) if vision_config is not None else {}
        self.connector_config = dict(connector_config) if connector_config is not None else {}
        self.image_patch_token_id = image_patch_token_id
        self.vit_layers = list(vit_layers) if vit_layers is not None else None
        self.olmo_core_config = olmo_core_config
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs) -> PretrainedConfig:
        """The language model's config, for tools (e.g. vLLM) that size the model from it."""
        text_config = getattr(self, "text_config", None)
        return text_config if text_config is not None else self


__all__ = ["ARCHITECTURE", "MODEL_TYPE", "Olmo3MoeVisionConfig"]
