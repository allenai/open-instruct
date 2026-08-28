"""Olmo Hybrid (gated DeltaNet + attention) support for the olmo-core training path.

Holds the ``TransformerConfig`` preset for Olmo Hybrid 7B and the HF <-> olmo-core
state conversion for ``model_type: olmo_hybrid``, neither of which olmo-core has.
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
    attn_backend: AttentionBackendName = AttentionBackendName.flash_2,
    dtype: DType = DType.float32,
) -> TransformerConfig:
    """Build an Olmo-Hybrid-style config: three GDN blocks per full-attention block.

    Only the geometry varies between hybrid checkpoints, so everything else is
    fixed below. The attention blocks use full-width (not per-head) QK norm,
    matching Olmo 2/3.
    """
    layer_norm = LayerNormConfig(name=LayerNormType.rms, eps=1e-6, bias=False, dtype=dtype)
    feed_forward = FeedForwardConfig(hidden_size=intermediate_size, bias=False, dtype=dtype)

    gdn_block = TransformerBlockConfig(
        name=TransformerBlockType.default,
        sequence_mixer=GatedDeltaNetConfig(
            n_heads=linear_num_key_heads,
            n_v_heads=linear_num_value_heads,
            head_dim=linear_key_head_dim,
            expand_v=linear_value_head_dim / linear_key_head_dim,
            allow_neg_eigval=True,
            conv_size=4,
            # NOT rms_norm_eps: HF hardcodes 1e-5 for the GDN output norm (see
            # OlmoHybridGatedDeltaNet.__init__), which is also olmo-core's default.
            norm_eps=1e-5,
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
            # NoPE: config.json carries rope_parameters = {"rope_theta": null} and HF
            # builds no rotary embedding, so these layers have no positional encoding.
            # A RoPE here is one the trained weights never saw; do not pass
            # --rope_scaling_factor either.
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
        tie_word_embeddings=False,
    )


def olmo3_hybrid_7B(
    vocab_size: int, attn_backend: AttentionBackendName = AttentionBackendName.flash_2
) -> TransformerConfig:
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
        attn_backend=attn_backend,
    )


#: Config names resolvable by ``--config_name`` that olmo-core does not provide.
LOCAL_TRANSFORMER_CONFIGS = {"olmo3_hybrid_7B": olmo3_hybrid_7B}


def layer_types_from_hf_config(hf_config: transformers.PretrainedConfig) -> list[str]:
    """Return the per-layer ``linear_attention``/``full_attention`` list."""
    layer_types = getattr(hf_config, "layer_types", None)
    if not layer_types:
        raise ValueError(f"HF config for {OLMO_HYBRID_MODEL_TYPE} is missing 'layer_types'")
    return list(layer_types)


LAYER_PREFIXES = {"hf": "model.layers.", "olmo-core": "blocks."}


def _convert_state(state: dict[str, Any], layer_types: list[str], src: str) -> dict[str, torch.Tensor]:
    """Rename ``state`` from ``src`` ("hf" or "olmo-core") to the other side.

    Both directions are pure renamings -- every parameter has identical shape and
    layout on both sides, so no fusing, splitting or transposing is needed -- and
    both walk the same three maps, so one implementation serves both. Keeping them
    separate would let the maps drift, and a converter that disagrees with its own
    inverse loses weights silently.
    """
    dst = "olmo-core" if src == "hf" else "hf"
    shared, gdn, attn = HF_TO_OLMO_CORE_SHARED_KEYS, HF_TO_OLMO_CORE_GDN_KEYS, HF_TO_OLMO_CORE_ATTN_KEYS
    if src != "hf":
        shared = {v: k for k, v in shared.items()}
        gdn = {v: k for k, v in gdn.items()}
        attn = {v: k for k, v in attn.items()}

    converted: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for key, value in state.items():
        mapped = shared.get(key)
        if mapped is None:
            if not key.startswith(LAYER_PREFIXES[src]):
                unmapped.append(key)
                continue
            layer_str, _, block_suffix = key[len(LAYER_PREFIXES[src]) :].partition(".")
            key_map = gdn if layer_types[int(layer_str)] == "linear_attention" else attn
            block_key = key_map.get(block_suffix)
            if block_key is None:
                unmapped.append(key)
                continue
            mapped = f"{LAYER_PREFIXES[dst]}{int(layer_str)}.{block_key}"
        converted[mapped] = value

    if unmapped:
        raise KeyError(f"{len(unmapped)} {src} keys could not be mapped to {dst}, e.g. {sorted(unmapped)[:5]}")
    return converted


def convert_hybrid_state_from_hf(hf_state: dict[str, Any], layer_types: list[str]) -> dict[str, torch.Tensor]:
    """Convert an HF ``olmo_hybrid`` state dict to olmo-core format."""
    return _convert_state(hf_state, layer_types, "hf")


def convert_hybrid_state_to_hf(state_dict: dict[str, Any], layer_types: list[str]) -> dict[str, torch.Tensor]:
    """Convert an olmo-core hybrid state dict back to HF ``olmo_hybrid`` format.

    olmo-core ships its own ``convert_hybrid_state_to_hf``, but its GDN key map
    disagrees with the released checkpoint (see ``HF_TO_OLMO_CORE_GDN_KEYS``).
    """
    return _convert_state(state_dict, layer_types, "olmo-core")


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
