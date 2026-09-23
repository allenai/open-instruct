"""TP1 BF16 numerical experiment, not a production serving implementation."""

import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def rounded_activation(value, out=None):
    gate, up = value.chunk(2, dim=-1)
    result = F.silu(gate) * up
    if out is not None:
        out.copy_(result)
        return out
    return result


def combine_fp32(expert_outputs, weights):
    return (expert_outputs.float() * weights.float().unsqueeze(-1)).sum(1).to(expert_outputs.dtype)


def compatibility_factory(config):
    """Install an explicit diagnostic forward after a discarded priming request."""
    root, name = Path(config["root"]), config["module"]
    installed = False

    def after(module, _args, _output):
        nonlocal installed
        if installed:
            return
        original = module.forward

        def forward(*args, **kwargs):
            profile = (root / "case.txt").read_text().split("/", 1)[0]
            is_expert = name.endswith(".experts")
            if profile == "warmup" or profile == "original" or (not is_expert and not profile.endswith("norms")):
                return original(*args, **kwargs)
            if not is_expert:
                if len(args) != 1 or kwargs:
                    raise ValueError("Norm experiment only supports a single tensor, no fused residual")
                value = args[0]
                fp32 = value.float()
                normed = fp32 * torch.rsqrt(fp32.square().mean(-1, keepdim=True) + module.variance_epsilon)
                return (module.weight.float() * normed).to(value.dtype)
            value, topk = args
            if module.w13_weight.dtype != torch.bfloat16 or module.w13_weight.shape[0] != 512:
                raise ValueError("Expert experiment is scoped to these unquantized TP1 hero checkpoints")
            fused = sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"]
            activation = fused.silu_and_mul
            fused.silu_and_mul = rounded_activation
            try:
                routes = fused.fused_experts_impl(
                    value.contiguous(),
                    module.w13_weight,
                    module.w2_weight,
                    topk.topk_weights,
                    topk.topk_ids,
                    no_combine=True,
                    filter_expert=False,
                )
            finally:
                fused.silu_and_mul = activation
            return combine_fp32(routes, topk.topk_weights)

        module.forward = forward
        installed = True

    return after
