"""Olmo Hybrid (gated DeltaNet + attention) support for the olmo-core training path.

olmo-core can *export* a hybrid model to HF format (``convert_hybrid_state_to_hf``)
but it has neither a :class:`TransformerConfig` preset for Olmo Hybrid 7B nor the
inverse HF -> olmo-core conversion needed to fine-tune from ``allenai/Olmo-Hybrid-7B``.
Both live here until they land upstream.

Two things to know about this architecture, because neither matches Olmo 3:

* It is **NoPE**. ``config.json`` carries ``rope_parameters = {"rope_theta": null}``
  and HF's ``OlmoHybrid`` builds no rotary embedding, so the full-attention layers
  have no positional encoding at all -- position comes from the GDN layers. Do not
  pass ``--rope_scaling_factor``.
* The two block types use **different norm conventions**: GDN blocks are pre-norm
  and attention blocks are reordered-norm (Olmo 2 style). The HF key names differ
  accordingly, which is why the per-layer key maps below are split in two.
"""

from typing import Any

import torch
import transformers
from olmo_core.config import DType
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig, AttentionType
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerBlockType, TransformerConfig

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

OLMO_HYBRID_MODEL_TYPE = "olmo_hybrid"

GDN_BLOCK_KEY = "gdn"
ATTN_BLOCK_KEY = "attn"

#: Non-block keys, HF -> olmo-core.
HF_TO_OLMO_CORE_SHARED_KEYS: dict[str, str] = {
    "model.embed_tokens.weight": "embeddings.weight",
    "model.norm.weight": "lm_head.norm.weight",
    "lm_head.weight": "lm_head.w_out.weight",
}

#: GDN ("linear_attention") layers, HF suffix -> olmo-core suffix.
#:
#: These names are taken from ``allenai/Olmo-Hybrid-7B`` itself. Note that
#: olmo-core's own ``HYBRID_GDN_LAYER_KEY_MAP`` disagrees with the released
#: checkpoint on two of them -- it expects ``linear_attn.out_proj`` and
#: ``linear_attn.norm`` where the checkpoint has ``o_proj`` and ``o_norm``.
HF_TO_OLMO_CORE_GDN_KEYS: dict[str, str] = {
    "linear_attn.q_proj.weight": "attention.w_q.weight",
    "linear_attn.k_proj.weight": "attention.w_k.weight",
    "linear_attn.v_proj.weight": "attention.w_v.weight",
    "linear_attn.a_proj.weight": "attention.w_a.weight",
    "linear_attn.b_proj.weight": "attention.w_b.weight",
    "linear_attn.g_proj.weight": "attention.w_g.weight",
    "linear_attn.o_proj.weight": "attention.w_out.weight",
    "linear_attn.q_conv1d.weight": "attention.q_conv1d.weight",
    "linear_attn.k_conv1d.weight": "attention.k_conv1d.weight",
    "linear_attn.v_conv1d.weight": "attention.v_conv1d.weight",
    "linear_attn.o_norm.weight": "attention.o_norm.weight",
    "linear_attn.A_log": "attention.A_log",
    "linear_attn.dt_bias": "attention.dt_bias",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "feed_forward_norm.weight",
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
}

#: Full-attention layers, HF suffix -> olmo-core suffix. These blocks are
#: reordered-norm, so the norms land *after* the sequence mixer and the MLP.
HF_TO_OLMO_CORE_ATTN_KEYS: dict[str, str] = {
    "self_attn.q_proj.weight": "attention.w_q.weight",
    "self_attn.k_proj.weight": "attention.w_k.weight",
    "self_attn.v_proj.weight": "attention.w_v.weight",
    "self_attn.o_proj.weight": "attention.w_out.weight",
    "self_attn.q_norm.weight": "attention.q_norm.weight",
    "self_attn.k_norm.weight": "attention.k_norm.weight",
    "post_attention_layernorm.weight": "attention_norm.weight",
    "post_feedforward_layernorm.weight": "feed_forward_norm.weight",
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
}


def olmo_hybrid_like(
    *,
    d_model: int,
    vocab_size: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    linear_num_key_heads: int,
    linear_num_value_heads: int,
    linear_key_head_dim: int,
    linear_value_head_dim: int,
    linear_conv_kernel_dim: int = 4,
    linear_allow_neg_eigval: bool = True,
    layer_norm_eps: float = 1e-6,
    attn_backend: AttentionBackendName = AttentionBackendName.flash_2,
    dtype: DType = DType.float32,
    **kwargs: Any,
) -> TransformerConfig:
    """Build an Olmo-Hybrid-style config: three GDN blocks per full-attention block.

    The attention blocks carry no RoPE (see the module docstring) and use full-width
    (not per-head) QK norm, matching Olmo 2/3.
    """
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=layer_norm_eps, bias=False, dtype=dtype)
    feed_forward = FeedForwardConfig(hidden_size=intermediate_size, bias=False, dtype=dtype)

    gdn_block = TransformerBlockConfig(
        name=TransformerBlockType.default,
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=linear_num_key_heads,
            n_v_heads=linear_num_value_heads,
            head_dim=linear_key_head_dim,
            expand_v=linear_value_head_dim / linear_key_head_dim,
            allow_neg_eigval=linear_allow_neg_eigval,
            conv_size=linear_conv_kernel_dim,
            norm_eps=layer_norm_eps,
            dtype=dtype,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )
    attn_block = TransformerBlockConfig(
        name=TransformerBlockType.reordered_norm,
        sequence_mixer=AttentionConfig(
            name=AttentionType.default,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            bias=False,
            rope=None,
            qk_norm=layer_norm,
            use_head_qk_norm=False,
            backend=attn_backend,
            dtype=dtype,
        ),
        feed_forward=feed_forward,
        layer_norm=layer_norm,
    )
    return TransformerConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        n_layers=n_layers,
        block={GDN_BLOCK_KEY: gdn_block, ATTN_BLOCK_KEY: attn_block},
        block_pattern=[GDN_BLOCK_KEY, GDN_BLOCK_KEY, GDN_BLOCK_KEY, ATTN_BLOCK_KEY],
        lm_head=LMHeadConfig(layer_norm=layer_norm, bias=False, dtype=dtype),
        dtype=dtype,
        tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
        **kwargs,
    )


def olmo3_hybrid_7B(vocab_size: int, **kwargs: Any) -> TransformerConfig:
    """The config for ``allenai/Olmo-Hybrid-7B``.

    32 layers, 24 GDN + 8 full-attention (at indices 3, 7, ..., 31).
    """
    return olmo_hybrid_like(
        d_model=3840,
        vocab_size=vocab_size,
        n_layers=32,
        n_heads=30,
        n_kv_heads=30,
        head_dim=128,
        intermediate_size=11008,
        linear_num_key_heads=30,
        linear_num_value_heads=30,
        linear_key_head_dim=96,
        linear_value_head_dim=192,
        **kwargs,
    )


#: Config names resolvable by ``--config_name`` that olmo-core does not provide.
LOCAL_TRANSFORMER_CONFIGS = {"olmo3_hybrid_7B": olmo3_hybrid_7B}


def layer_types_from_hf_config(hf_config: transformers.PretrainedConfig) -> list[str]:
    """Return the per-layer ``linear_attention``/``full_attention`` list."""
    layer_types = getattr(hf_config, "layer_types", None)
    if not layer_types:
        raise ValueError(f"HF config for {OLMO_HYBRID_MODEL_TYPE} is missing 'layer_types'")
    return list(layer_types)


def convert_hybrid_state_from_hf(hf_state: dict[str, Any], layer_types: list[str]) -> dict[str, torch.Tensor]:
    """Convert an HF ``olmo_hybrid`` state dict to olmo-core format.

    This is a pure renaming: every parameter has identical shape and layout on both
    sides, so no fusing, splitting or transposing is needed.
    """
    olmo_state: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for hf_key, value in hf_state.items():
        shared = HF_TO_OLMO_CORE_SHARED_KEYS.get(hf_key)
        if shared is not None:
            olmo_state[shared] = value
            continue

        prefix, _, suffix = hf_key.partition(".")
        if prefix != "model" or not suffix.startswith("layers."):
            unmapped.append(hf_key)
            continue
        _, layer_str, block_suffix = suffix.split(".", 2)
        layer_idx = int(layer_str)
        key_map = (
            HF_TO_OLMO_CORE_GDN_KEYS if layer_types[layer_idx] == "linear_attention" else HF_TO_OLMO_CORE_ATTN_KEYS
        )
        olmo_suffix = key_map.get(block_suffix)
        if olmo_suffix is None:
            unmapped.append(hf_key)
            continue
        olmo_state[f"blocks.{layer_idx}.{olmo_suffix}"] = value

    if unmapped:
        raise KeyError(f"{len(unmapped)} HF keys could not be mapped to olmo-core, e.g. {sorted(unmapped)[:5]}")
    return olmo_state


def convert_hybrid_state_to_hf(state_dict: dict[str, Any], layer_types: list[str]) -> dict[str, torch.Tensor]:
    """Convert an olmo-core hybrid state dict back to HF ``olmo_hybrid`` format.

    olmo-core ships its own ``convert_hybrid_state_to_hf``, but its GDN key map
    disagrees with the released checkpoint (see ``HF_TO_OLMO_CORE_GDN_KEYS``), so
    this inverts the maps here instead.
    """
    shared = {v: k for k, v in HF_TO_OLMO_CORE_SHARED_KEYS.items()}
    gdn = {v: k for k, v in HF_TO_OLMO_CORE_GDN_KEYS.items()}
    attn = {v: k for k, v in HF_TO_OLMO_CORE_ATTN_KEYS.items()}

    hf_state: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for olmo_key, value in state_dict.items():
        if olmo_key in shared:
            hf_state[shared[olmo_key]] = value
            continue
        prefix, _, rest = olmo_key.partition(".")
        if prefix != "blocks":
            unmapped.append(olmo_key)
            continue
        layer_str, _, block_suffix = rest.partition(".")
        key_map = gdn if layer_types[int(layer_str)] == "linear_attention" else attn
        hf_suffix = key_map.get(block_suffix)
        if hf_suffix is None:
            unmapped.append(olmo_key)
            continue
        hf_state[f"model.layers.{layer_str}.{hf_suffix}"] = value

    if unmapped:
        raise KeyError(f"{len(unmapped)} olmo-core keys could not be mapped to HF, e.g. {sorted(unmapped)[:5]}")
    return hf_state


def load_hf_hybrid_model(model_name_or_path: str, model_state_dict: dict[str, Any]) -> None:
    """Load HF ``olmo_hybrid`` weights into an olmo-core state dict, in place.

    Mirrors ``olmo_core.nn.hf.checkpoint.load_hf_model``, which cannot be used here
    because ``convert_state_from_hf`` has no ``olmo_hybrid`` branch and would fall
    through to the llama-style key templates.
    """
    # Load in the checkpoint's own dtype, as olmo-core's load_hf_model does. Forcing
    # float32 here would double host RAM on every rank; load_state_dict casts to the
    # parameter dtype anyway.
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    layer_types = layer_types_from_hf_config(hf_model.config)
    converted = convert_hybrid_state_from_hf(hf_model.state_dict(), layer_types)
    del hf_model

    missing = sorted(set(model_state_dict) - set(converted))
    unexpected = sorted(set(converted) - set(model_state_dict))
    if missing or unexpected:
        raise KeyError(
            f"Converted state does not match the olmo-core model: "
            f"{len(missing)} missing (e.g. {missing[:5]}), {len(unexpected)} unexpected (e.g. {unexpected[:5]})"
        )

    for key, value in converted.items():
        target = model_state_dict[key]
        if isinstance(target, torch.distributed.tensor.DTensor):
            model_state_dict[key] = torch.distributed.tensor.distribute_tensor(
                value, target.device_mesh, target.placements
            )
        else:
            model_state_dict[key] = value
    logger.info(f"Loaded {len(converted)} Olmo Hybrid tensors from {model_name_or_path}")
