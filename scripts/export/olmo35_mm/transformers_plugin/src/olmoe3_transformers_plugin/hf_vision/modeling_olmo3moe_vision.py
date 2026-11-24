# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/transformers_plugin/src/olmoe3_transformers_plugin/hf_vision/modeling_olmo3moe_vision.py
"""HF wrapper of a multimodal OLMoE3 export: the text model runs; vision is held, not run.

``Olmo3MoeVisionForConditionalGeneration`` owns the whole exported model:

* ``language_model``: the text team's :class:`Olmo3MoeForCausalLM`, built from
  ``config.text_config`` and loaded from the ``language_model.*`` tensors;
* ``vision_tower`` and ``connector``: parameter trees whose state-dict names and shapes are
  those of olmo-core's ``VisionTransformer`` and ``VisionConnector`` (``vision_tower.*`` and
  ``connector.*`` tensors), so the export carries every weight of the checkpoint.

Only the text forward is implemented: it runs the language model. Passing images raises
:class:`NotImplementedError`; vision inference is out of scope for this export.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:  # Bundled in an exported checkpoint next to the text team's files.
    # transformers copies only the files a remote module imports directly, so the text
    # model's own dependencies are imported here too, to travel with it.
    from .configuration_olmo3moe import Olmo3MoeConfig  # noqa: F401
    from .configuration_olmo3moe_vision import Olmo3MoeVisionConfig
    from .modeling_olmo3moe import Olmo3MoeForCausalLM
    from .moe_numerics import core_combine, core_swiglu  # noqa: F401
except ImportError:  # Installed plugin package: the text model lives in olmo-core.
    from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM

    from .configuration_olmo3moe_vision import Olmo3MoeVisionConfig


def _vision_dtype(config: Dict[str, Any]) -> torch.dtype:
    name = config.get("dtype", "float32")
    if isinstance(name, torch.dtype):
        return name
    return getattr(torch, str(name))


class Olmo3MoeVisionAttention(nn.Module):
    """Parameters of olmo-core's ``ViTAttention`` (``wq``/``wk``/``wv``/``wo`` with biases)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dtype = _vision_dtype(config)
        emb = int(config["image_emb_dim"])
        heads = int(config["image_num_heads"])
        kv_heads = int(config["image_num_key_value_heads"])
        head_dim = int(config["image_head_dim"])
        self.wq = nn.Linear(emb, heads * head_dim, bias=True, dtype=dtype)
        self.wk = nn.Linear(emb, kv_heads * head_dim, bias=True, dtype=dtype)
        self.wv = nn.Linear(emb, kv_heads * head_dim, bias=True, dtype=dtype)
        self.wo = nn.Linear(heads * head_dim, emb, bias=True, dtype=dtype)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Vision inference is not implemented for this export")


class Olmo3MoeVisionMLP(nn.Module):
    """Parameters of olmo-core's ``ViTMLP`` (``w1``/``w2`` with biases)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dtype = _vision_dtype(config)
        emb = int(config["image_emb_dim"])
        mlp = int(config["image_mlp_dim"])
        self.w1 = nn.Linear(emb, mlp, bias=True, dtype=dtype)
        self.w2 = nn.Linear(mlp, emb, bias=True, dtype=dtype)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Vision inference is not implemented for this export")


class Olmo3MoeVisionBlock(nn.Module):
    """Parameters of olmo-core's standard pre-LN ``ViTBlock``."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dtype = _vision_dtype(config)
        emb = int(config["image_emb_dim"])
        eps = float(config.get("image_norm_eps", 1e-5))
        self.attn_norm = nn.LayerNorm(emb, eps=eps, dtype=dtype)
        self.ffn_norm = nn.LayerNorm(emb, eps=eps, dtype=dtype)
        self.attn = Olmo3MoeVisionAttention(config)
        self.ffn = Olmo3MoeVisionMLP(config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Vision inference is not implemented for this export")


class Olmo3MoeVisionTower(nn.Module):
    """
    Parameters of olmo-core's ``VisionTransformer`` under their olmo-core names.

    :param config: The olmo-core ``VisionEncoderConfig`` as a dict.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        block_name = (config.get("block") or {}).get("name", "standard")
        if block_name != "standard":
            raise NotImplementedError(f"Unsupported vision block {block_name!r}")
        dtype = _vision_dtype(config)
        emb = int(config["image_emb_dim"])
        patch = int(config["image_patch_size"])
        self.patch_embedding = nn.Linear(
            patch * patch * 3,
            emb,
            bias=bool(config.get("patch_embedding_bias", False)),
            dtype=dtype,
        )
        self.class_embedding: Optional[nn.Parameter] = None
        if config.get("use_cls_token", True):
            self.class_embedding = nn.Parameter(torch.zeros(emb, dtype=dtype))
        self.positional_embedding = nn.Parameter(
            torch.zeros(int(config["image_num_pos"]), emb, dtype=dtype)
        )
        self.pre_ln: Optional[nn.LayerNorm] = None
        if config.get("use_pre_ln", True):
            self.pre_ln = nn.LayerNorm(
                emb, eps=float(config.get("image_norm_eps", 1e-5)), dtype=dtype
            )
        self.blocks = nn.ModuleList(
            [Olmo3MoeVisionBlock(config) for _ in range(int(config["image_num_layers"]))]
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Vision inference is not implemented for this export")


class Olmo3MoeVisionPooling(nn.Module):
    """Parameters of olmo-core's connector ``_PoolingCrossAttention``."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dtype = _vision_dtype(config)
        emb = int(config["image_emb_dim"])
        input_dim = int(config.get("num_input_layers", 1)) * emb
        heads = int(config["image_num_heads"])
        kv_heads = int(config["image_num_key_value_heads"])
        head_dim = int(config["image_head_dim"])
        self.wq = nn.Linear(input_dim, heads * head_dim, bias=True, dtype=dtype)
        self.wk = nn.Linear(input_dim, kv_heads * head_dim, bias=True, dtype=dtype)
        self.wv = nn.Linear(input_dim, kv_heads * head_dim, bias=True, dtype=dtype)
        self.wo = nn.Linear(heads * head_dim, emb, bias=True, dtype=dtype)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Vision inference is not implemented for this export")


class Olmo3MoeVisionProjectorMLP(nn.Module):
    """Parameters of olmo-core's connector ``_ConnectorMLP`` (SwiGLU ``w1``/``w2``/``w3``)."""

    def __init__(self, input_dim: int, hidden: int, output_dim: int, dtype: torch.dtype):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden, bias=False, dtype=dtype)
        self.w3 = nn.Linear(input_dim, hidden, bias=False, dtype=dtype)
        self.w2 = nn.Linear(hidden, output_dim, bias=False, dtype=dtype)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Vision inference is not implemented for this export")


class Olmo3MoeVisionConnector(nn.Module):
    """
    Parameters of olmo-core's ``VisionConnector`` under their olmo-core names.

    :param config: The olmo-core ``VisionConnectorConfig`` as a dict.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        dtype = _vision_dtype(config)
        emb = int(config["image_emb_dim"])
        pooling_input_dim = int(config.get("num_input_layers", 1)) * emb
        pooling_type = config.get("pooling_type", "attention_meanq")
        self.pooling: Optional[Olmo3MoeVisionPooling] = None
        if pooling_type == "attention_meanq":
            self.pooling = Olmo3MoeVisionPooling(config)
            projector_input_dim = emb
        elif pooling_type == "none":
            projector_input_dim = pooling_input_dim
        else:
            raise NotImplementedError(f"Unsupported pooling type {pooling_type!r}")
        projector_type = config.get("projector_type", "mlp")
        output_dim = int(config["output_dim"])
        if projector_type == "mlp":
            hidden = int(config.get("mlp_hidden_size") or 4 * emb)
            self.projector: nn.Module = Olmo3MoeVisionProjectorMLP(
                projector_input_dim, hidden, output_dim, dtype
            )
        elif projector_type == "linear":
            self.projector = nn.Linear(projector_input_dim, output_dim, bias=False, dtype=dtype)
        else:
            raise NotImplementedError(f"Unsupported projector type {projector_type!r}")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Vision inference is not implemented for this export")


class Olmo3MoeVisionPreTrainedModel(PreTrainedModel):
    config: Olmo3MoeVisionConfig
    config_class = Olmo3MoeVisionConfig
    base_model_prefix = "language_model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Olmo3MoeDecoderLayer", "Olmo3MoeVisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        # The text model relies on the base initializer (it also fills the rotary embeddings'
        # non-persistent ``inv_freq`` buffers); only the vision tower adds bare parameters.
        if isinstance(module, Olmo3MoeVisionTower):
            std = float(self.config.vision_config.get("initializer_range", 0.02))
            module.positional_embedding.data.normal_(mean=0.0, std=std)
            if module.class_embedding is not None:
                module.class_embedding.data.zero_()
            return
        super()._init_weights(module)


class Olmo3MoeVisionForConditionalGeneration(Olmo3MoeVisionPreTrainedModel, GenerationMixin):
    """
    The whole multimodal model. The text forward runs ``language_model``; image inputs are
    rejected with :class:`NotImplementedError`.
    """

    def __init__(self, config: Olmo3MoeVisionConfig):
        super().__init__(config)
        self.vision_tower = Olmo3MoeVisionTower(config.vision_config)
        self.connector = Olmo3MoeVisionConnector(config.connector_config)
        self.language_model = Olmo3MoeForCausalLM(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        images: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pooled_patches_idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Run the language model on text.

        :raises NotImplementedError: If any image input is given.
        """
        if images is not None or pixel_values is not None or pooled_patches_idx is not None:
            raise NotImplementedError(
                "Vision inference is not implemented for this export; it runs text only"
            )
        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


__all__ = [
    "Olmo3MoeVisionConnector",
    "Olmo3MoeVisionForConditionalGeneration",
    "Olmo3MoeVisionPreTrainedModel",
    "Olmo3MoeVisionTower",
]
