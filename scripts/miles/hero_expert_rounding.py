"""Replay a captured hero expert call with independently controlled rounding."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F


def metric(actual, reference):
    actual, reference = actual.float(), reference.float().reshape_as(actual)
    delta = actual - reference
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / reference.norm()).item(),
        "equal_fraction": (delta == 0).float().mean().item(),
    }


def expert_weights(model_path, layer, experts):
    tensors = {}
    prefix = f"model.layers.{layer}.mlp.experts."
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors in {model_path}")
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as source:
            keys = source.keys()
            for key in keys:
                if key.startswith(prefix):
                    tensors[key.removeprefix(prefix)] = source.get_tensor(key)
    gate_up = torch.stack(
        [torch.cat([tensors[f"{i}.gate_proj.weight"], tensors[f"{i}.up_proj.weight"]]).T for i in range(experts)]
    ).cuda()
    down = torch.stack([tensors[f"{i}.down_proj.weight"].T for i in range(experts)]).cuda()
    return gate_up, down


@torch.inference_mode()
def run(args):
    torch.set_float32_matmul_precision("highest")
    name = f"model.layers.{args.layer}.mlp.experts"
    trace = torch.load(args.trace / "sglang" / args.case / f"{name}.pt", weights_only=True)
    hf = torch.load(args.trace / f"hf-{args.case}.pt", weights_only=True)[name]
    x = trace["input"].cuda()
    ids, weights = trace["topk_ids"].cuda().long(), trace["topk_weights"].cuda().float()
    config = json.loads((args.model / "config.json").read_text())
    experts = config.get("num_experts", 512)
    w13, w2 = expert_weights(args.model, args.layer, experts)
    n, k = ids.shape
    order = torch.argsort(ids.flatten())
    sorted_experts = ids.flatten()[order]
    sorted_tokens = torch.arange(n, device=x.device).repeat_interleave(k)[order]
    sorted_weights = weights.flatten()[order, None]
    offs = torch.bincount(sorted_experts, minlength=experts).to(torch.int32).cumsum(0).to(torch.int32)
    grouped = x[sorted_tokens]
    gate, up = F.grouped_mm(grouped, w13, offs=offs).chunk(2, dim=-1)
    reduction_order = torch.argsort(sorted_tokens * experts + sorted_experts)
    report = {
        "model": str(args.model),
        "case": args.case,
        "layer": args.layer,
        "captured_inputs_equal": torch.equal(trace["input"].reshape_as(hf["input"]), hf["input"]),
        "captured_routing_equal": torch.equal(ids.cpu(), hf["topk_ids"].reshape_as(ids)),
        "fp32_down_reference": "Per-expert FP32 matmul with TF32 disabled; accumulation order may differ",
        "down_reference_checks": {},
        "variants": {},
    }
    for activation in ["bf16_silu_then_multiply", "fp32_silu_multiply_single_cast"]:
        hidden = (
            F.silu(gate) * up if activation.startswith("bf16") else (F.silu(gate.float()) * up.float()).to(x.dtype)
        )
        down_bf16 = F.grouped_mm(hidden, w2, offs=offs)
        down_fp32 = torch.empty_like(down_bf16, dtype=torch.float32)
        start = 0
        for expert, end in enumerate(offs.cpu().tolist()):
            if end > start:
                down_fp32[start:end] = hidden[start:end].float() @ w2[expert].float()
            start = end
        report["down_reference_checks"][activation] = metric(down_fp32.to(x.dtype), down_bf16)
        for combine in ["hf", "rounded_weighted_routes", "weight_before_down_cast"]:
            if combine == "weight_before_down_cast":
                weighted = (down_fp32 * sorted_weights).to(x.dtype).float()
            else:
                weighted = down_bf16.float() * sorted_weights
                if combine == "rounded_weighted_routes":
                    weighted = weighted.to(x.dtype).float()
            result = weighted[reduction_order].reshape(n, k, -1).sum(1).to(x.dtype).cpu()
            key = f"{activation}/{combine}"
            report["variants"][key] = {
                "vs_hf": metric(result, hf["output"]),
                "vs_sglang": metric(result, trace["output"]),
            }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--case", default="16")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
