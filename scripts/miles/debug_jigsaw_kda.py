"""Scoring-only KDA interventions in a copied, pinned serving package.

The native-Core intervention deliberately supports only one uncached prefill
per request. It does not populate recurrent caches and must never be used for
multi-token generation or training. Fixed-token probes request one output token.
"""

import os

import torch
from fla.modules import fused_norm_gate
from olmo_core.nn.attention import kda
from torch import nn


def native_attention(attention):
    """Bind a Core KDA module to the already-loaded, frozen serving values."""
    native = kda.KimiDeltaAttention(
        d_model=attention.qkv_proj.weight.shape[1],
        n_heads=attention.num_k_heads,
        n_v_heads=attention.num_v_heads,
        head_dim=attention.head_k_dim,
        expand_v=attention.head_v_dim / attention.head_k_dim,
        allow_neg_eigval=True,
        conv_size=attention.conv_size,
        conv_bias=False,
        norm_eps=attention.o_norm.eps,
        dtype=torch.bfloat16,
        init_device="meta",
    )
    widths = (attention.key_dim, attention.key_dim, attention.value_dim)
    for name, weight in zip(("w_q", "w_k", "w_v"), attention.qkv_proj.weight.split(widths), strict=True):
        getattr(native, name).weight = nn.Parameter(weight.detach(), requires_grad=False)
    for name in ("f_proj_1", "f_proj_2", "g_proj_1", "g_proj_2", "o_norm"):
        target, source = getattr(native, name), getattr(attention, name)
        target.weight = source.weight
        if getattr(source, "bias", None) is not None:
            target.bias = source.bias
    native.w_b.weight = attention.beta_proj.weight
    native.w_out.weight = attention.o_proj.weight
    native.A_log, native.dt_bias = attention.A_log, attention.dt_bias
    for name, weight in zip(
        ("q_conv1d", "k_conv1d", "v_conv1d"), attention.qkv_conv1d.weight.split(widths), strict=True
    ):
        # Core stores convolution parameters in BF16; serving promotes the same
        # initial checkpoint values to FP32. Reject any loss in this conversion.
        converted = weight.bfloat16()
        assert torch.equal(converted.float(), weight.float())
        getattr(native, name).weight = nn.Parameter(converted, requires_grad=False)
    assert all(parameter.device.type != "meta" for parameter in native.parameters())
    return native.eval().requires_grad_(False)


def install(model_class):
    mode = os.environ.get("OI_JIGSAW_KDA_INTERVENTION", "baseline")
    if mode == "baseline":
        return
    if mode not in {"core_norm", "core_prefill"}:
        raise ValueError(mode)
    original_init = model_class.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for layer in self.language_model.model.layers:
            attention = layer.self_attn
            if not hasattr(attention, "key_dim"):
                continue
            if mode == "core_norm":
                norm = attention.o_norm

                def normalize(value, gate, norm=norm):
                    return fused_norm_gate.rms_norm_gated(
                        value, gate, norm.weight, norm.bias, norm.activation, eps=norm.eps
                    )

                norm.forward = normalize
            else:
                install_prefill(attention)

    model_class.__init__ = initialize


def install_prefill(attention):
    native = None

    def prefill(positions, hidden_states, forward_batch):
        nonlocal native
        assert forward_batch.forward_mode.is_extend(), "Native KDA probe is prefill-only"
        assert forward_batch.batch_size == 1, "Native KDA probe requires one sequence"
        assert forward_batch.extend_prefix_lens_cpu == [0], "Native KDA probe cannot use a cached prefix"
        if native is None:
            # Construct after weight loading. Keep it in this closure, outside
            # serving's registered parameter inventory and weight-load contract.
            native = native_attention(attention)
        return native(hidden_states.unsqueeze(0)).squeeze(0)

    attention.forward = prefill
