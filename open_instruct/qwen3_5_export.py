"""OLMo-core → HuggingFace state-dict export for Qwen3.5 hybrid (GatedDeltaNet) models.

OLMo-core (as of rev fa6c5014) implements only the HF→OLMo-core direction for
``qwen3_5`` / ``qwen3_5_text`` (`olmo_core.nn.hf.convert.convert_qwen3_5_state_from_hf`);
its generic template converter has no qwen3_5 entry, so a trained model could not be
exported back to HF format. This module is the exact inverse of that from-HF conversion,
targeting the text-only ``Qwen3_5ForCausalLM`` in-memory layout (``model.layers.*`` +
top-level ``lm_head.weight``), which strict-loads via ``load_state_dict``. On
``save_pretrained`` transformers applies its ``qwen3_5_text`` checkpoint conversion
mapping in reverse (``revert_weight_conversion``), so the on-disk file carries the
standard ``model.language_model.*`` naming the project's serving wrapper consumes —
byte-compatible with the SFT checkpoints already in production.

Candidate for upstreaming to allenai/OLMo-core; delete this module and bump the pin if
an equivalent lands there.
"""

import re

import torch
from olmo_core.nn.hf import convert as _olmo_hf_convert
from transformers import PretrainedConfig

# ---------------------------------------------------------------------------------------------
# WORKAROUND for an OLMo-core bug (present at pin fa6c5014): `_normalize_qwen3_5_hf_state`
# skips vision keys but NOT the MTP head, and `mtp.layers.0.*` contains ".layers." — so on a
# VLM-wrapper checkpoint the normalizer rewrites MTP tensors to `model.layers.0.*`, clobbering
# layer 0's MLP and norms with MTP-head weights (and `mtp.norm.weight` races the real final
# norm on dict order). Measured effect: mean |dlogprob| 0.16 vs 0.038 on the text-only
# checkpoint of the same model. Filter `mtp.` keys before the original normalizer runs.
# Remove when fixed upstream in allenai/OLMo-core.
if not getattr(_olmo_hf_convert._normalize_qwen3_5_hf_state, "_skips_mtp", False):
    _original_normalize = _olmo_hf_convert._normalize_qwen3_5_hf_state

    def _normalize_qwen3_5_hf_state_skipping_mtp(hf_state):
        return _original_normalize({k: v for k, v in hf_state.items() if not k.startswith("mtp.")})

    _normalize_qwen3_5_hf_state_skipping_mtp._skips_mtp = True
    _olmo_hf_convert._normalize_qwen3_5_hf_state = _normalize_qwen3_5_hf_state_skipping_mtp
# ---------------------------------------------------------------------------------------------

# The norm families that HF stores zero-initialized (``hidden * (1 + w)``) and OLMo-core
# stores ones-initialized (``hidden * w``). The from-HF converter adds 1.0 to exactly
# these (see ``_apply_qwen3_5_norm_transform``); export subtracts it again. The GDN
# output norm (``o_norm`` → ``linear_attn.norm``) is deliberately NOT in this list.
_PLUS_ONE_NORM_PATTERNS = (
    "attention_norm.weight",
    "feed_forward_norm.weight",
    "lm_head.norm.weight",
    "q_norm.weight",
    "k_norm.weight",
)


def _denorm(key: str, value: torch.Tensor) -> torch.Tensor:
    if any(pattern in key for pattern in _PLUS_ONE_NORM_PATTERNS):
        return value - 1.0
    return value


def _merge_qwen3_5_q_proj(w_q: torch.Tensor, w_g: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Re-interleave separate OLMo ``w_q``/``w_g`` into HF's fused per-head layout.

    Inverse of ``_split_qwen3_5_q_proj``: HF stores ``[q_0, g_0, q_1, g_1, ...]``.
    """
    hidden_size = w_q.shape[1]
    q = w_q.view(n_heads, head_dim, hidden_size)
    g = w_g.view(n_heads, head_dim, hidden_size)
    return torch.stack([q, g], dim=1).reshape(n_heads * 2 * head_dim, hidden_size)


def get_qwen3_5_text_model_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    """The text-only ``Qwen3_5ForCausalLM`` config the export targets.

    A VLM-wrapper config (``model_type qwen3_5``) is re-rooted to its ``text_config``;
    a text config (``model_type qwen3_5_text``, e.g. a prior SFT export) passes through.
    Always returns a copy so callers can mutate safely.
    """
    text_config = hf_config.get_text_config()
    text_config = type(text_config).from_dict(text_config.to_dict())
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    return text_config


def convert_qwen3_5_state_to_hf(
    config: PretrainedConfig, olmo_state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Convert an OLMo-core Qwen3.5 state dict to the ``Qwen3_5ForCausalLM`` HF layout.

    Exact inverse of ``olmo_core.nn.hf.convert.convert_qwen3_5_state_from_hf``: re-fuses
    the GDN qkv/conv1d projections, re-interleaves the full-attention query/gate weights,
    and subtracts the +1.0 norm offset. Raises if any input key goes unconsumed or any
    expected key is missing, so a silent partial export is impossible.

    :param config: The HF config (wrapper or text); only its text config is read.
    :param olmo_state: ``model.state_dict()`` of an OLMo-core ``qwen3_5_like`` model.
    """
    text_config = config.get_text_config()
    layer_types = list(text_config.layer_types)
    n_layers = text_config.num_hidden_layers
    n_heads = text_config.num_attention_heads
    head_dim = text_config.head_dim

    consumed: set[str] = set()

    def take(key: str) -> torch.Tensor:
        if key not in olmo_state:
            raise KeyError(f"olmo-core state is missing expected key {key!r}")
        consumed.add(key)
        return _denorm(key, olmo_state[key])

    prefix = "model."
    hf_state: dict[str, torch.Tensor] = {
        f"{prefix}embed_tokens.weight": take("embeddings.weight"),
        f"{prefix}norm.weight": take("lm_head.norm.weight"),
    }
    if getattr(text_config, "tie_word_embeddings", False):
        # A tied HF model has no separate lm_head tensor; the olmo model shares storage.
        consumed.add("lm_head.w_out.weight")
    else:
        hf_state["lm_head.weight"] = take("lm_head.w_out.weight")

    for i in range(n_layers):
        op = f"blocks.{i}."
        lp = f"{prefix}layers.{i}."
        layer_type = layer_types[i]
        if layer_type == "linear_attention":
            gp = f"{lp}linear_attn."
            hf_state[f"{gp}in_proj_qkv.weight"] = torch.cat(
                [
                    take(f"{op}attention.w_q.weight"),
                    take(f"{op}attention.w_k.weight"),
                    take(f"{op}attention.w_v.weight"),
                ]
            )
            hf_state[f"{gp}in_proj_z.weight"] = take(f"{op}attention.w_g.weight")
            hf_state[f"{gp}in_proj_a.weight"] = take(f"{op}attention.w_a.weight")
            hf_state[f"{gp}in_proj_b.weight"] = take(f"{op}attention.w_b.weight")
            hf_state[f"{gp}out_proj.weight"] = take(f"{op}attention.w_out.weight")
            hf_state[f"{gp}conv1d.weight"] = torch.cat(
                [
                    take(f"{op}attention.q_conv1d.weight"),
                    take(f"{op}attention.k_conv1d.weight"),
                    take(f"{op}attention.v_conv1d.weight"),
                ]
            )
            hf_state[f"{gp}norm.weight"] = take(f"{op}attention.o_norm.weight")
            hf_state[f"{gp}A_log"] = take(f"{op}attention.A_log")
            hf_state[f"{gp}dt_bias"] = take(f"{op}attention.dt_bias")
        elif layer_type == "full_attention":
            ap = f"{lp}self_attn."
            hf_state[f"{ap}q_proj.weight"] = _merge_qwen3_5_q_proj(
                take(f"{op}attention.w_q.weight"), take(f"{op}attention.w_g.weight"), n_heads, head_dim
            )
            hf_state[f"{ap}k_proj.weight"] = take(f"{op}attention.w_k.weight")
            hf_state[f"{ap}v_proj.weight"] = take(f"{op}attention.w_v.weight")
            hf_state[f"{ap}o_proj.weight"] = take(f"{op}attention.w_out.weight")
            hf_state[f"{ap}q_norm.weight"] = take(f"{op}attention.q_norm.weight")
            hf_state[f"{ap}k_norm.weight"] = take(f"{op}attention.k_norm.weight")
        else:
            raise ValueError(f"Unknown layer type {layer_type!r} at layer {i}")

        hf_state[f"{lp}input_layernorm.weight"] = take(f"{op}attention_norm.weight")
        hf_state[f"{lp}post_attention_layernorm.weight"] = take(f"{op}feed_forward_norm.weight")
        hf_state[f"{lp}mlp.gate_proj.weight"] = take(f"{op}feed_forward.w1.weight")
        hf_state[f"{lp}mlp.down_proj.weight"] = take(f"{op}feed_forward.w2.weight")
        hf_state[f"{lp}mlp.up_proj.weight"] = take(f"{op}feed_forward.w3.weight")

    unconsumed = set(olmo_state) - consumed
    # Non-persistent RoPE buffers and similar may legitimately appear in some state dicts.
    unconsumed = {k for k in unconsumed if not re.search(r"rope|freqs|inv_freq", k)}
    if unconsumed:
        raise KeyError(f"olmo-core state has unconsumed keys the exporter does not map: {sorted(unconsumed)[:8]}")
    return hf_state
