# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/kda.py
"""FLA decode with OLMoE3's BF16 normalization and indexed FP32 state cache."""

import torch


def fla_kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.kda import fused_recurrent_kda

    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    # Match the chunk path's explicit BF16 Q/K normalization. Fusing this
    # normalization into recurrence retains FP32 Q/K and changes the operation.
    # FLA validates one initial state per sequence even with an indexed pool.
    active_indices = state_indices[: len(cu_seqlens) - 1]
    result, final_state = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=recurrent_state[active_indices].contiguous(),
        cu_seqlens=cu_seqlens,
        ssm_state_indices=None,
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=False,
        state_v_first=True,
        output_final_state=True,
        inplace_final_state=False,
    )
    recurrent_state[active_indices] = final_state
    return result
