"""Monkey-patch Qwen3.5 to support packing in GatedDeltaNet (linear attention) layers.

Without this patch, the GatedDeltaNet layers ignore sequence boundaries in packed
inputs: the causal conv1d leaks across sequences (seq_idx=None) and the recurrent
state carries over between packed sequences. This causes incorrect logprobs during
training and inflated KL divergence.

Based on https://github.com/huggingface/transformers/pull/45034.

Usage:
    from open_instruct.qwen3_5_packing_patch import patch_qwen3_5_packing
    patch_qwen3_5_packing()
    # Then load your model as usual.
"""

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def _patched_gated_delta_net_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """GatedDeltaNet forward with seq_idx and cu_seqlens support for packing."""
    seq_idx = kwargs.get("seq_idx")
    cu_seqlens = kwargs.get("cu_seqlens")

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    batch_size, seq_len, _ = hidden_states.shape
    use_precomputed_states = cache_params is not None and cache_params.has_previous_state and seq_len == 1

    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if self.causal_conv1d_fn is not None:
            cp_context = kwargs.get("cp_context")
            if cp_context is not None:
                # Use the CP-aware conv so that rank r+1 receives the last
                # (kernel_size - 1) tokens from rank r as initial conv state.
                # Without this, continuation chunks start from zero conv state,
                # corrupting the first (kernel_size - 1) output tokens on each
                # non-first rank.  causal_conv1d_cp expects [1, T, D] not [1, D, T].
                # Lazy import: fla.modules.conv.cp is only needed under SP.
                from fla.modules.conv.cp.ops import causal_conv1d_cp  # noqa: PLC0415

                mixed_qkv = causal_conv1d_cp(
                    x=mixed_qkv.permute(0, 2, 1),
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    cp_context=cp_context,
                ).permute(0, 2, 1)
            else:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        chunk_kwargs = {}
        if getattr(self.chunk_gated_delta_rule, "__module__", "").startswith("fla."):
            cp_context = kwargs.get("cp_context")
            if cp_context is not None:
                chunk_kwargs["cp_context"] = cp_context
                # cp_context.cu_seqlens carries rank-local boundaries derived from the
                # global sequence layout; the locally-computed cu_seqlens (from position
                # resets on this rank's slice) omits continuation ranges on non-first ranks.
                chunk_kwargs["cu_seqlens"] = cp_context.cu_seqlens
            else:
                chunk_kwargs["cu_seqlens"] = cu_seqlens

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            **chunk_kwargs,
        )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    core_attn_out = self.out_proj(core_attn_out)
    return core_attn_out


def _patched_decoder_layer_forward(
    self, hidden_states, position_embeddings, attention_mask=None, position_ids=None, past_key_values=None, **kwargs
):
    """DecoderLayer forward that passes **kwargs to linear_attn for packing support."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states, cache_params=past_key_values, attention_mask=attention_mask, **kwargs
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def patch_qwen3_5_packing():
    """Apply the packing fix to Qwen3.5 GatedDeltaNet and DecoderLayer."""
    modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward = _patched_gated_delta_net_forward
    modeling_qwen3_5.Qwen3_5DecoderLayer.forward = _patched_decoder_layer_forward
    logger.info("Applied Qwen3.5 packing patch for GatedDeltaNet seq_idx/cu_seqlens support")
