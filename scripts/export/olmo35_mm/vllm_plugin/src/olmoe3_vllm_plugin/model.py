# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/model.py
"""vLLM executor for OLMoE3 KDA + LatentMoE checkpoints."""

import os
from collections.abc import Iterable
from functools import partial
from itertools import islice

import torch
import triton
from einops import rearrange
from fla.ops.kda import chunk_kda as fla_chunk_kda
from olmoe3_vllm_plugin.compat import (
    IS_V019,
    FusedRMSNormGated,
    KimiGatedDeltaNetAttention,
    fused_kda_gate,
    fused_moe_make_expert_params_mapping,
    is_conv_state_dim_first,
    vllm_kda_ops,
)
from olmoe3_vllm_plugin.norm import CoreRMSNorm as RMSNorm
from olmoe3_vllm_plugin.norm import PerHeadRMSNorm
from olmoe3_vllm_plugin.routing import route_experts
from olmoe3_vllm_plugin.validation import validate_hf_config
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import (
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

if any(
    os.environ.get(flag) == "1"
    for flag in ("OLMO_HERO_FIXED_FLA_CONFIGS", "OLMO_HERO_PRECISE_INFERENCE")
):
    raise ValueError(
        "Campaign-only FP64/fixed-launch diagnostics are not part of this plugin. "
        "Unset OLMO_HERO_FIXED_FLA_CONFIGS and OLMO_HERO_PRECISE_INFERENCE."
    )


def install_triton_allocator() -> None:
    """Match the allocator used by the validated MainlineLadder KDA plugin."""

    def allocate(size: int, alignment: int, stream: int | None):
        del alignment, stream
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(allocate)


install_triton_allocator()
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata  # noqa: E402


def _olmoe3_kda_state_shape(
    tp_world_size: int,
    num_value_heads: int,
    value_head_dim: int,
    *,
    num_k_heads: int,
    key_head_dim: int,
    conv_kernel_size: int,
    num_spec: int = 0,
):
    """Return vLLM's KDA cache shapes with OLMoE3's rectangular state."""
    shapes = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size,
        num_value_heads,
        value_head_dim,
        num_k_heads=num_k_heads,
        head_k_dim=key_head_dim,
        conv_kernel_size=conv_kernel_size,
        num_spec=num_spec,
    )
    # vLLM 0.26 correctly sizes the separate Q/K/V convolution caches, but
    # assumes the recurrent KDA matrix is square. OLMoE3 supports unequal K/V
    # widths; chunk_kda returns [value_heads, value_dim, key_dim].
    recurrent_shape = (shapes[-1][0], value_head_dim, key_head_dim)
    if IS_V019:
        # 0.19 derives Q/V widths from the same ``num_heads, head_dim`` pair.
        # Its returned order is therefore V, K, K; the scheduler expects Q, K, V.
        return (shapes[1], shapes[2], shapes[0], recurrent_shape)
    return (*shapes[:-1], recurrent_shape)


def chunk_kda_expanded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FLA's rectangular KDA prefill with value-first recurrent state."""
    return fla_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
    )


def fused_recurrent_kda_expanded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run vLLM's decode kernel without collapsing expanded V to K width."""
    # Prefill's FLA chunk kernel materializes normalized Q/K in the input dtype.
    # Keep decode at that same rounding point instead of retaining FP32-normalized
    # Q/K inside the recurrent kernel.
    from fla.modules.l2norm import l2norm_fwd

    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    batch_size, sequence_length, num_heads, key_dim = k.shape
    value_dim = v.shape[-1]
    num_value_heads = v.shape[2]
    num_sequences = batch_size if cu_seqlens is None else len(cu_seqlens) - 1
    block_key = vllm_kda_ops.next_power_of_2(key_dim)
    block_value = min(vllm_kda_ops.next_power_of_2(value_dim), 8)
    num_key_blocks = vllm_kda_ops.cdiv(key_dim, block_key)
    num_value_blocks = vllm_kda_ops.cdiv(value_dim, block_value)
    if num_key_blocks != 1:
        raise ValueError("KDA decode does not support key dimensions above one block")

    output = torch.empty_like(v)
    if ssm_state_indices is None:
        ssm_state_indices = torch.arange(num_sequences, device=q.device, dtype=torch.int32)
        stride_indices_sequence, stride_indices_token = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_sequence, stride_indices_token = ssm_state_indices.stride(0), 1
    else:
        stride_indices_sequence, stride_indices_token = ssm_state_indices.stride()

    grid = (num_key_blocks, num_value_blocks, num_sequences * num_value_heads)
    vllm_kda_ops.fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        o=output,
        h0=initial_state,
        ht=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        scale=key_dim**-0.5,
        N=num_sequences,
        T=sequence_length,
        B=batch_size,
        H=num_heads,
        HV=num_value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=stride_indices_sequence,
        stride_indices_tok=stride_indices_token,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=False,
        INPLACE_FINAL_STATE=True,
        IS_KDA=True,
        num_warps=1,
        num_stages=3,
    )
    return output, initial_state


def _validate_config(vllm_config: VllmConfig) -> None:
    config = vllm_config.model_config.hf_config
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1:
        raise NotImplementedError("OLMoE3 KDA currently supports tensor parallel size 1")
    if parallel.pipeline_parallel_size != 1:
        raise NotImplementedError("OLMoE3 KDA currently supports pipeline parallel size 1")
    if getattr(parallel, "enable_expert_parallel", False):
        raise NotImplementedError("OLMoE3 KDA currently supports expert parallel size 1")
    validate_hf_config(config)


class Olmo3MoeMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, vllm_config, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        # HF/core round SiLU to the activation dtype before the product. vLLM's fused
        # SiluAndMul retains FP32 through both operations and changes this rounding point.
        gate, up = x.chunk(2, dim=-1)
        x = torch.nn.functional.silu(gate) * up
        x, _ = self.down_proj(x)
        return x


class Olmo3MoeSparseMLP(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self._route = partial(route_experts, config=config)
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.router.gate",
        )
        routed_hidden_size = config.latent_moe_dim or config.hidden_size
        self.latent_down_proj = (
            ColumnParallelLinear(
                config.hidden_size,
                routed_hidden_size,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.latent_down_proj",
            )
            if config.latent_moe_dim is not None
            else None
        )
        self.latent_up_proj_input_norm = (
            RMSNorm(routed_hidden_size, eps=config.rms_norm_eps)
            if config.latent_moe_dim is not None and config.latent_moe_up_proj_input_norm
            else None
        )
        self.latent_up_proj = (
            RowParallelLinear(
                routed_hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.latent_up_proj",
            )
            if config.latent_moe_dim is not None
            else None
        )
        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=routed_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            renormalize=False,
            quant_config=vllm_config.quant_config,
            custom_routing_function=self._route,
            prefix=f"{prefix}.experts",
        )
        self._use_grouped_experts = os.environ.get("OLMO_VLLM_TORCH_GROUPED_MOE") == "1"
        self._top_k = config.num_experts_per_tok
        if self._use_grouped_experts:
            # v0.26 returns a MoERunner with the weight-owning layer nested inside.
            expert_layer = self.experts if IS_V019 else self.experts.routed_experts
            backend = getattr(expert_layer.quant_method, "unquantized_backend", None)
            if getattr(backend, "name", None) != "TRITON":
                raise RuntimeError(
                    "Grouped reference requires the unshuffled TRITON expert layout. "
                    "On vLLM 0.26 set kernel_config={'moe_backend': 'triton'}."
                )
        self.shared_expert = (
            Olmo3MoeMLP(
                config.hidden_size,
                config.shared_expert_intermediate_size,
                vllm_config,
                f"{prefix}.shared_expert",
            )
            if config.shared_expert_intermediate_size is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training and the HF implementation evaluate router logits in FP32.
        # A BF16 projection can change top-k expert membership before normalization.
        logits = torch.nn.functional.linear(x.float(), self.router.weight.float())
        routed_x = self.latent_down_proj(x)[0] if self.latent_down_proj is not None else x
        if self._use_grouped_experts:
            from olmoe3_vllm_plugin.expert_reference import grouped_experts

            # Do not patch quant_method.apply: vLLM can replace that object after weight
            # loading when it initializes a modular kernel. Dispatch the selected arithmetic
            # here, while retaining vLLM's expert parameter allocation and weight loaders.
            expert_layer = self.experts if IS_V019 else self.experts.routed_experts
            weights, indices = self._route(routed_x, logits, self._top_k, False)
            output = grouped_experts(expert_layer, routed_x, weights, indices)
        else:
            output = self.experts(hidden_states=routed_x, router_logits=logits)
        if self.latent_up_proj is not None:
            if self.latent_up_proj_input_norm is not None:
                output = self.latent_up_proj_input_norm(output)
            output = self.latent_up_proj(output)[0]
        if self.shared_expert is not None:
            output = output + self.shared_expert(x)
        return output


class Olmo3MoeAttention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        if getattr(config, "qk_norm_per_head_gains", False):
            self.q_norm = PerHeadRMSNorm(self.num_heads, self.head_dim, config.rms_norm_eps)
            self.k_norm = PerHeadRMSNorm(self.num_kv_heads, self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.scalable_softmax = config.scalable_softmax
        self.ssmax_scale = (
            nn.Parameter(torch.ones(self.num_heads)) if self.scalable_softmax else None
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.attn",
        )
        self.g_proj = (
            ReplicatedLinear(
                config.hidden_size,
                config.attention_hidden_size,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.g_proj",
            )
            if config.attention_gate_type == "elementwise"
            else None
        )
        self.o_proj = RowParallelLinear(
            config.attention_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q_heads = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        if self.scalable_softmax:
            assert self.ssmax_scale is not None
            visible_scale = (positions + 1).log().to(q_heads.dtype)
            scale = visible_scale[:, None, None]
            scale = scale * self.ssmax_scale.to(q_heads.dtype)[None, :, None]
            q_heads = q_heads * scale
        q = q_heads.view_as(q)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view_as(k)
        output = self.attn(q, k, v)
        if self.g_proj is not None:
            gate, _ = self.g_proj(x)
            output = output * gate.float().sigmoid().to(output.dtype)
        output, _ = self.o_proj(output)
        return output


class Olmo3MoeKDA(KimiGatedDeltaNetAttention):
    """Configure vLLM's KDA/cache lifecycle for OLMoE3 parameter names."""

    def __init__(self, vllm_config: VllmConfig, layer_idx: int, prefix: str):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self._use_fla_kda = os.environ.get("OLMO_VLLM_FLA_KDA") == "1"
        self.tp_size = 1
        self.tp_rank = 0
        self.hidden_size = config.hidden_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.conv_size = config.linear_conv_kernel_dim
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.num_heads = self.num_key_heads
        self.local_num_heads = self.num_key_heads
        key_size = self.num_key_heads * self.key_head_dim
        value_size = self.num_value_heads * self.value_head_dim
        quant_config = vllm_config.quant_config

        # Native vLLM KDA constructs Q/K/V from one shared head width. Build
        # every projection directly so expand_v cannot be silently discarded.
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            key_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            key_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            value_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.f_proj_1 = ReplicatedLinear(
            config.hidden_size,
            self.value_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_proj_1",
        )
        self.f_proj_2 = ColumnParallelLinear(
            self.value_head_dim,
            key_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_proj_2",
        )
        self.g_proj_1 = ReplicatedLinear(
            config.hidden_size,
            self.value_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj_1",
        )
        self.g_proj_2 = ColumnParallelLinear(
            self.value_head_dim,
            value_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj_2",
        )
        self.q_conv1d = self._make_conv(key_size, f"{prefix}.q_conv1d")
        self.k_conv1d = self._make_conv(key_size, f"{prefix}.k_conv1d")
        self.v_conv1d = self._make_conv(value_size, f"{prefix}.v_conv1d")
        self.A_log = nn.Parameter(torch.empty(1, 1, self.num_value_heads, 1, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(key_size, dtype=torch.float32))
        gated_norm = FusedRMSNormGated
        if self._use_fla_kda:
            from fla.modules import FusedRMSNormGated as gated_norm
        self.o_norm = gated_norm(
            self.value_head_dim,
            eps=config.linear_norm_eps,
            activation="sigmoid",
        )
        self.o_proj = RowParallelLinear(
            value_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.beta_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_key_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.beta_proj",
        )
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self._validate_rectangular_layout()

    def _make_conv(self, output_size: int, prefix: str):
        conv = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=output_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=prefix,
        )
        conv.weight.data = conv.weight.data.unsqueeze(1)
        return conv

    def _validate_rectangular_layout(self) -> None:
        """Fail at model construction if vLLM collapses V back to K width."""
        local_key_size = self.num_key_heads * self.key_head_dim // self.tp_size
        local_value_size = self.num_value_heads * self.value_head_dim // self.tp_size
        actual = {
            "q_proj": self.q_proj.output_size_per_partition,
            "k_proj": self.k_proj.output_size_per_partition,
            "v_proj": self.v_proj.output_size_per_partition,
            "f_proj_2": self.f_proj_2.output_size_per_partition,
            "g_proj_2": self.g_proj_2.output_size_per_partition,
            "v_conv1d": self.v_conv1d.output_size_per_partition,
            "o_proj": self.o_proj.input_size_per_partition,
        }
        expected = {
            "q_proj": local_key_size,
            "k_proj": local_key_size,
            "v_proj": local_value_size,
            "f_proj_2": local_key_size,
            "g_proj_2": local_value_size,
            "v_conv1d": local_value_size,
            "o_proj": local_value_size,
        }
        if actual != expected:
            raise RuntimeError(
                "vLLM constructed an invalid OLMoE3 KDA projection layout: "
                f"expected={expected}, actual={actual}"
            )

        state_shapes = self.get_state_shape()
        recurrent_shape = state_shapes[-1]
        expected_recurrent = (
            self.num_value_heads // self.tp_size,
            self.value_head_dim,
            self.key_head_dim,
        )
        if recurrent_shape != expected_recurrent:
            raise RuntimeError(
                "vLLM constructed an invalid OLMoE3 KDA recurrent cache: "
                f"expected={expected_recurrent}, actual={recurrent_shape}"
            )

    def get_state_shape(self):
        return _olmoe3_kda_state_shape(
            self.tp_size,
            self.num_value_heads,
            self.value_head_dim,
            num_k_heads=self.num_key_heads,
            key_head_dim=self.key_head_dim,
            conv_kernel_size=self.conv_size,
        )

    def forward(self, hidden_states, positions, output) -> None:
        del positions
        num_tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]
        beta = self.beta_proj(hidden_states)[0].float().sigmoid()
        if self.model_config.hf_config.linear_allow_neg_eigval:
            beta = beta * 2.0
        decay = self.f_proj_2(self.f_proj_1(hidden_states)[0])[0]
        if self._use_fla_kda:
            decay = decay.view(num_tokens, self.num_value_heads, self.key_head_dim)
        else:
            decay = fused_kda_gate(decay, self.A_log, self.key_head_dim, g_bias=self.dt_bias)
        gate = self.g_proj_2(self.g_proj_1(hidden_states)[0])[0]
        gate = rearrange(gate, "n (h d) -> n h d", d=self.value_head_dim)
        core_output = torch.zeros(
            (1, num_tokens, self.num_value_heads, self.value_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.kda_attention(
            q, k, v, decay.unsqueeze(0), beta.unsqueeze(0), core_output, self.prefix
        )
        core_output = self.o_norm(core_output, gate)
        core_output = rearrange(core_output, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_output)[0]

    def _forward(self, q_proj_states, k_proj_states, v_proj_states, g1, beta, core_attn_out):
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None:
            return
        metadata = metadata[self.prefix]
        assert isinstance(metadata, GDNAttentionMetadata)
        num_tokens = metadata.num_actual_tokens
        state_indices = metadata.non_spec_state_indices_tensor
        starts = metadata.non_spec_query_start_loc
        if IS_V019:
            q_state, k_state, v_state, recurrent_state = self.kv_cache
            q_state = q_state.transpose(-1, -2)
            k_state = k_state.transpose(-1, -2)
            v_state = v_state.transpose(-1, -2)
        else:
            conv_state, recurrent_state = self.kv_cache
            if not is_conv_state_dim_first():
                conv_state = conv_state.transpose(-1, -2)
            local_key_size = self.num_key_heads * self.key_head_dim // self.tp_size
            local_value_size = self.num_value_heads * self.value_head_dim // self.tp_size
            q_state, k_state, v_state = conv_state.split(
                (local_key_size, local_key_size, local_value_size), dim=-2
            )
        projections = (
            q_proj_states[:num_tokens],
            k_proj_states[:num_tokens],
            v_proj_states[:num_tokens],
        )
        states = (q_state, k_state, v_state)
        modules = (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        convolved = []
        if metadata.num_prefills > 0:
            for projected, state, module in zip(projections, states, modules):
                weight = module.weight.view(module.weight.shape[0], module.weight.shape[2])
                convolved.append(
                    causal_conv1d_fn(
                        projected.transpose(0, 1),
                        weight,
                        module.bias,
                        activation="silu",
                        conv_states=state,
                        has_initial_state=metadata.has_initial_state,
                        cache_indices=state_indices,
                        query_start_loc=starts,
                        metadata=metadata,
                    ).transpose(0, 1)
                )
        else:
            indices = state_indices[:num_tokens]
            for projected, state, module in zip(projections, states, modules):
                weight = module.weight.view(module.weight.shape[0], module.weight.shape[2])
                convolved.append(
                    causal_conv1d_update(
                        projected,
                        state,
                        weight,
                        module.bias,
                        activation="silu",
                        conv_state_indices=indices,
                        validate_data=True,
                    )
                )
        q = rearrange(convolved[0], "n (h d) -> 1 n h d", d=self.key_head_dim)
        k = rearrange(convolved[1], "n (h d) -> 1 n h d", d=self.key_head_dim)
        v = rearrange(convolved[2], "n (h d) -> 1 n h d", d=self.value_head_dim)
        expected_qk = (self.num_key_heads // self.tp_size, self.key_head_dim)
        expected_v = (self.num_value_heads // self.tp_size, self.value_head_dim)
        if q.shape[-2:] != expected_qk or k.shape[-2:] != expected_qk:
            raise RuntimeError(
                "vLLM collapsed the OLMoE3 KDA key layout at runtime: "
                f"expected={expected_qk}, q={tuple(q.shape)}, k={tuple(k.shape)}"
            )
        if v.shape[-2:] != expected_v:
            raise RuntimeError(
                "vLLM collapsed the OLMoE3 KDA value layout at runtime: "
                f"expected={expected_v}, v={tuple(v.shape)}"
            )
        if metadata.num_prefills > 0:
            recurrent_state[state_indices[~metadata.has_initial_state]] = 0
            prefill = fla_chunk_kda if self._use_fla_kda else chunk_kda_expanded
            extra = (
                dict(
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    use_gate_in_kernel=True,
                    use_qk_l2norm_in_kernel=True,
                    state_v_first=True,
                    output_final_state=True,
                )
                if self._use_fla_kda
                else {}
            )
            result, final_state = prefill(
                q=q,
                k=k,
                v=v,
                g=g1[:, :num_tokens],
                beta=beta[:, :num_tokens],
                initial_state=recurrent_state[state_indices].contiguous(),
                cu_seqlens=starts,
                **extra,
            )
            recurrent_state[state_indices] = final_state
        elif self._use_fla_kda:
            from .kda import fla_kda_decode

            result = fla_kda_decode(
                q=q,
                k=k,
                v=v,
                g=g1[:, :num_tokens],
                beta=beta[:, :num_tokens],
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                recurrent_state=recurrent_state,
                state_indices=state_indices[: metadata.num_decodes],
                cu_seqlens=starts[: metadata.num_decodes + 1],
            )
        else:
            result, _ = fused_recurrent_kda_expanded(
                q=q,
                k=k,
                v=v,
                g=g1[:, :num_tokens],
                beta=beta[:, :num_tokens],
                initial_state=recurrent_state,
                cu_seqlens=starts[: metadata.num_decodes + 1],
                ssm_state_indices=state_indices,
            )
        if result.shape[-2:] != expected_v:
            raise RuntimeError(
                "vLLM KDA kernel returned the wrong value width: "
                f"expected={expected_v}, actual={tuple(result.shape)}"
            )
        core_attn_out[0, :num_tokens] = result[0, :num_tokens]


class Olmo3MoeDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        layer_idx = extract_layer_index(prefix)
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = (
            Olmo3MoeKDA(vllm_config, layer_idx, f"{prefix}.self_attn")
            if self.layer_type == "linear_attention"
            else Olmo3MoeAttention(vllm_config, f"{prefix}.self_attn")
        )
        self.mlp = (
            Olmo3MoeMLP(
                config.hidden_size,
                config.dense_mlp_intermediate_size,
                vllm_config,
                f"{prefix}.mlp",
            )
            if layer_idx in config.dense_layers_indices
            else Olmo3MoeSparseMLP(vllm_config, f"{prefix}.mlp")
        )
        self.pre_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_attention_layernorm(x)
        if self.layer_type == "linear_attention":
            output = torch.empty_like(x)
            self.self_attn(hidden_states=x, positions=positions, output=output)
            x = output
        else:
            x = self.self_attn(positions, x)
        x = residual + self.post_attention_layernorm(x)
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        return residual + self.post_feedforward_layernorm(x)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Olmo3MoeModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.embed_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Olmo3MoeDecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            assert input_ids is not None or inputs_embeds is not None
            x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
            x = self.embed_norm(x * self.config.embed_scale)
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            x = layer(positions, x)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": x})
        x = self.norm(x)
        return x


class Olmo3MoeForCausalLM(nn.Module, HasInnerState, SupportsPP, IsHybrid):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        _validate_config(vllm_config)
        self.config = vllm_config.model_config.hf_config
        self.model = Olmo3MoeModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = (
            self.model.embed_tokens
            if self.config.tie_word_embeddings
            else ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        config = vllm_config.model_config.hf_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return _olmoe3_kda_state_shape(
            1,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            num_k_heads=config.linear_num_key_heads,
            key_head_dim=config.linear_key_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked = (
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        )
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        params = dict(self.named_parameters())
        loaded = set()
        for item in weights:
            name, weight = item[:2]
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                continue
            if is_pp_missing_parameter(name, self):
                continue
            matched = False
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params:
                    continue
                param = params[mapped]
                param.weight_loader(param, weight, name, shard_id=shard_id, expert_id=expert_id)
                loaded.add(mapped)
                matched = True
                break
            if matched:
                continue
            for param_name, weight_name, shard_id in stacked:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params:
                    continue
                param = params[mapped]
                param.weight_loader(param, weight, shard_id)
                loaded.add(mapped)
                matched = True
                break
            if matched:
                continue
            if ".router.gate.weight" in name:
                mapped = name.replace(".router.gate.weight", ".router.weight")
                if mapped in params:
                    param = params[mapped]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, weight)
                    loaded.add(mapped)
                    continue
            if name.endswith(".bias") and name not in params:
                continue
            if name not in params:
                raise KeyError(f"Unexpected OLMoE3 checkpoint weight: {name}")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if name.endswith(".A_log") and weight.shape != param.shape:
                weight = weight.reshape(param.shape)
            loader(param, weight)
            loaded.add(name)
        return loaded
