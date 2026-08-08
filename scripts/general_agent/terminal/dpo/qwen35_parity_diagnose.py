#!/usr/bin/env python3
"""Diagnose OLMo-core vs HF logit disagreements for Qwen3.5-9B (no assertions, pure report).

Rung 2 found argmax flips at positions where the reference ranking was decisive (top-1/top-2
margin up to 2.125), which small uniform noise cannot explain. This script maps out where the
divergence lives: per-position delta profile along the sequence, details of every disagreeing
position (margins both sides, local max delta, decoded context), and an optional fp32 rerun —
if the flips vanish in fp32 the cause is bf16 kernel noise amplification; if they persist it
is an implementation-semantics difference (conv boundary, rope, norm eps, ...).
"""

import argparse
import dataclasses
import json

import torch
import transformers

from open_instruct import logger_utils, olmo_core_utils
from open_instruct.dataset_transformation import (
    _normalize_messages_for_chat_template,
    _normalize_tools_for_chat_template,
)

logger = logger_utils.setup_logger(__name__)


@dataclasses.dataclass
class MinimalModelConfig:
    model_name_or_path: str
    config_name: str
    attn_implementation: str
    loss_implementation: str = "fused_linear"
    rope_scaling_factor: float | None = None


def build_prompt(args, tokenizer) -> torch.Tensor:
    with open(args.dpo_jsonl) as f:
        for line in f:
            row = json.loads(line)
            tools = _normalize_tools_for_chat_template(row.get("tools"))
            if tools:
                break
    n_prefix = row["metadata"]["n_prefix_messages"]
    messages = _normalize_messages_for_chat_template(row["chosen"][:n_prefix])
    rendered = tokenizer.apply_chat_template(
        conversation=messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return input_ids[:, : args.max_prompt_tokens]


def report(tag, olmo_logits, hf_logits, input_ids, tokenizer):
    olmo_top, hf_top = olmo_logits.argmax(-1), hf_logits.argmax(-1)
    disagree_idx = (olmo_top != hf_top)[0].nonzero().flatten()
    deltas = (olmo_logits - hf_logits).abs()[0]  # [seq, vocab]
    per_pos_max = deltas.max(-1).values
    q = torch.tensor([0.5, 0.9, 0.99, 1.0])
    quantiles = torch.quantile(per_pos_max, q).tolist()
    worst_pos = int(per_pos_max.argmax())
    logger.info(
        f"[{tag}] disagreements: {len(disagree_idx)}/{olmo_logits.shape[1]} | per-position max|dlogit| "
        f"p50={quantiles[0]:.4f} p90={quantiles[1]:.4f} p99={quantiles[2]:.4f} max={quantiles[3]:.4f} "
        f"(worst at position {worst_pos})"
    )
    for idx in disagree_idx.tolist()[:20]:
        hf_row, olmo_row = hf_logits[0, idx], olmo_logits[0, idx]
        hf_top2 = hf_row.topk(2)
        olmo_top2 = olmo_row.topk(2)
        hf_margin = (hf_top2.values[0] - hf_top2.values[1]).item()
        olmo_margin = (olmo_top2.values[0] - olmo_top2.values[1]).item()
        t_hf = tokenizer.decode(hf_top2.indices[0])
        t_olmo = tokenizer.decode(olmo_top2.indices[0])
        context = tokenizer.decode(input_ids[0, max(0, idx - 8) : idx + 1])
        logger.info(
            f"[{tag}] pos {idx}: hf_top={t_hf!r} (margin {hf_margin:.3f}) vs olmo_top={t_olmo!r} "
            f"(margin {olmo_margin:.3f}); local max|d|={per_pos_max[idx]:.3f}; "
            f"delta@hf_top1={deltas[idx, hf_top2.indices[0]]:.3f}; ...{context[-80:]!r}"
        )
    return len(disagree_idx)


def run(args, dtype, tag, input_ids, tokenizer):
    device = torch.device("cuda")
    model_config_args = MinimalModelConfig(
        model_name_or_path=args.model_name_or_path, config_name=args.config_name, attn_implementation="flash_2"
    )
    olmo_model, _ = olmo_core_utils.setup_model(model_config_args, init_device="cpu")
    sd = olmo_model.state_dict()
    from olmo_core.nn.hf.checkpoint import load_hf_model

    load_hf_model(args.model_name_or_path, sd, work_dir=args.work_dir)
    olmo_model.load_state_dict(sd)
    del sd
    olmo_model = olmo_model.to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        olmo_logits = olmo_model(input_ids=input_ids.to(device)).float().cpu()
    del olmo_model
    torch.cuda.empty_cache()

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, dtype=dtype, trust_remote_code=True
    ).to(device)
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids.to(device)).logits.float().cpu()
    del hf_model
    torch.cuda.empty_cache()
    return report(tag, olmo_logits, hf_logits, input_ids, tokenizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2/sft_qwen35_9b_safecomp_v2",
    )
    parser.add_argument("--config_name", default="qwen3_5_9B")
    parser.add_argument("--tokenizer_name_or_path", default="hamishivi/Qwen3.5-9B")
    parser.add_argument(
        "--dpo_jsonl", default="/weka/oe-adapt-default/mingqianz/agent-safety/data/dpo/dpo_comp1k_v2.jsonl"
    )
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--work_dir", default="/tmp/qwen35_parity_work")
    parser.add_argument("--fp32", action="store_true", help="also rerun the comparison in fp32")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    input_ids = build_prompt(args, tokenizer)
    logger.info(f"prompt tokens: {input_ids.shape[1]}")

    n_bf16 = run(args, torch.bfloat16, "bf16", input_ids, tokenizer)
    if args.fp32:
        n_fp32 = run(args, torch.float32, "fp32", input_ids, tokenizer)
        logger.info(f"SUMMARY: disagreements bf16={n_bf16} fp32={n_fp32}")


if __name__ == "__main__":
    main()
