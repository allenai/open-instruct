# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/vllm_plugin/src/olmoe3_vllm_plugin/routing.py
"""FP32 routing shared by the native and grouped expert paths."""

import torch


def route_experts(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    *,
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the exported HF router, including Lp normalization and scaling."""
    del hidden_states, renormalize
    if config.gating_function == "softmax":
        scores = gating_output.softmax(dim=-1)
    elif config.gating_function == "sigmoid":
        scores = gating_output.sigmoid() + 1e-7
    else:
        raise ValueError(f"Unsupported gating function: {config.gating_function}")
    weights, indices = torch.topk(scores, topk, dim=-1)
    if config.normalize_expert_weights is not None:
        weights = weights / torch.linalg.vector_norm(
            weights,
            ord=config.normalize_expert_weights,
            dim=-1,
            keepdim=True,
        )
    if config.restore_weight_scale:
        weights = weights * topk
    if (
        config.original_num_experts_per_tok is not None
        and topk != config.original_num_experts_per_tok
    ):
        weights = weights * (config.original_num_experts_per_tok / topk) ** 0.5
    return weights, indices
