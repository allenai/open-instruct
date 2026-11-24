# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/compat.py
"""Version-gated vLLM compatibility surface for the OLMoE3 executor."""

from importlib.metadata import version

from packaging.version import Version
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.models.utils import extract_layer_index

VLLM_VERSION = Version(version("vllm"))
IS_V019 = Version("0.19.1") <= VLLM_VERSION < Version("0.20")
IS_V026 = Version("0.26") <= VLLM_VERSION < Version("0.27")

if not (IS_V019 or IS_V026):
    raise RuntimeError(
        "olmoe3-vllm-plugin supports vLLM 0.19.1.x and 0.26.x only; "
        f"detected vLLM {VLLM_VERSION}"
    )

if IS_V019:
    from vllm.model_executor.layers.fla.ops import kda as vllm_kda_ops
    from vllm.model_executor.layers.fla.ops.kda import FusedRMSNormGated, fused_kda_gate
    from vllm.model_executor.layers.kda import KimiDeltaAttention
    from vllm.platforms import current_platform
    from vllm.triton_utils.allocation import set_triton_allocator

    class KimiGatedDeltaNetAttention(KimiDeltaAttention):
        """Normalize vLLM 0.19's KDA constructor to the 0.26 contract."""

        def __init__(self, config, vllm_config, prefix: str):
            super().__init__(
                layer_idx=extract_layer_index(prefix),
                hidden_size=config.hidden_size,
                quant_config=vllm_config.quant_config,
                cache_config=vllm_config.cache_config,
                model_config=vllm_config.model_config,
                rms_norm_eps=config.linear_norm_eps,
                prefix=prefix,
            )
            # The generic 0.19 KDA layer predates Triton 3.x scratch
            # allocations. Match vLLM's newer hybrid models by registering a
            # PyTorch-backed allocator before any KDA kernel is autotuned.
            set_triton_allocator(current_platform.current_device())

    def is_conv_state_dim_first() -> bool:
        """vLLM 0.19 stores each convolution cache as ``[width, time]``."""
        return False

    def fused_moe_make_expert_params_mapping(*args, **kwargs):
        """Use the 0.19 class method that became a module helper in 0.26."""
        return FusedMoE.make_expert_params_mapping(*args, **kwargs)

else:
    from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import KimiGatedDeltaNetAttention
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.third_party.flash_linear_attention.ops import kda as vllm_kda_ops
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated, fused_kda_gate

__all__ = [
    "FusedRMSNormGated",
    "IS_V019",
    "KimiGatedDeltaNetAttention",
    "fused_kda_gate",
    "fused_moe_make_expert_params_mapping",
    "is_conv_state_dim_first",
    "vllm_kda_ops",
]
