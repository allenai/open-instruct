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


class _ConvCrossRankStateFn(torch.autograd.Function):
    """Exchange last (kernel_size - 1) raw input tokens across SP ranks.

    Forward: each rank sends its last W-1 tokens; non-first ranks receive
    the previous rank's tails as their initial conv state.

    Backward: gradients w.r.t. the received initial states are sent back
    to the rank that owns those tokens (via send_recv_bwd), so that the
    last W-1 tokens of the previous rank receive their correct gradient
    contribution from the next rank's convolution.
    """

    @staticmethod
    def forward(ctx, x_tails, group, is_first_rank, pre_num_conv_tokens, W):
        # x_tails: [W-1, D] — last W-1 input tokens of this rank
        from fla.ops.cp.comm import conv_cp_send_recv_fwd
        heads = conv_cp_send_recv_fwd(x_tails, group)
        ctx.group = group
        ctx.is_first_rank = is_first_rank
        ctx.pre_num_conv_tokens = pre_num_conv_tokens
        ctx.W = W
        return heads  # [W-1, D], zeros for first rank

    @staticmethod
    def backward(ctx, d_heads):
        from fla.ops.cp.comm import conv_cp_send_recv_bwd
        d_tails = conv_cp_send_recv_bwd(d_heads, ctx.group)
        return d_tails, None, None, None, None


def _causal_conv1d_fn_cp(mixed_qkv, weight, bias, activation, cp_context):
    """Call causal_conv1d_fn (numerically identical to vLLM) with cross-rank
    initial conv state passing for the no-packing SP case.

    ``mixed_qkv``: [B, D, T] — this rank's local chunk (B=1).
    ``weight``:    [D, W]    — depthwise conv weight.
    ``bias``:      [D] or None.
    ``activation``: activation name string or None.
    ``cp_context``: FLACPContext for this rank.

    Returns the conv output with the same shape [B, D, T].
    """
    from causal_conv1d import causal_conv1d_fn

    W = weight.shape[-1]   # kernel_size (4 for Qwen3.5)
    D = mixed_qkv.shape[1]

    # All ranks participate in the all-gather so both sides of the barrier are hit.
    x_tails = mixed_qkv[0, :, -(W - 1):].permute(1, 0).contiguous()  # [W-1, D]
    heads = _ConvCrossRankStateFn.apply(
        x_tails,
        cp_context.group,
        cp_context.is_first_rank,
        cp_context.pre_num_conv_tokens,
        W,
    )  # [W-1, D], zeros for first rank

    if not cp_context.is_first_rank:
        # Build initial_states: [1, D, W-1] using differentiable ops so
        # gradients flow correctly back through the cross-rank all-gather.
        valid_len = min(W - 1, cp_context.pre_num_conv_tokens)
        if valid_len == W - 1:
            initial_states = heads.permute(1, 0).unsqueeze(0)  # [1, D, W-1]
        else:
            # Left-pad with zeros when only part of the window came from rank 0.
            pad = torch.zeros(W - 1 - valid_len, D, device=mixed_qkv.device, dtype=mixed_qkv.dtype)
            initial_states = torch.cat([pad, heads[-valid_len:]], dim=0).permute(1, 0).unsqueeze(0)
    else:
        initial_states = None

    return causal_conv1d_fn(
        x=mixed_qkv,
        weight=weight,
        bias=bias,
        activation=activation,
        initial_states=initial_states,
        # seq_idx intentionally omitted: causal_conv1d_fn asserts
        # initial_states must be None when seq_idx is set.
    )


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
                if seq_idx is None:
                    # No-packing SP case: causal_conv1d_fn doesn't support
                    # initial_states + seq_idx simultaneously, but without
                    # seq_idx we can pass initial_states directly.  This matches
                    # vLLM's conv kernel exactly, reducing numerical divergence.
                    mixed_qkv = _causal_conv1d_fn_cp(
                        mixed_qkv,
                        self.conv1d.weight.squeeze(1),
                        self.conv1d.bias,
                        self.activation,
                        cp_context,
                    )
                else:
                    # Packing case: causal_conv1d_fn can't handle both seq_idx
                    # and initial_states; fall back to the FLA CP-aware kernel.
                    from fla.modules.conv.cp.ops import causal_conv1d_cp
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
