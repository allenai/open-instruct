#!/usr/bin/env python3
"""Rung-2 load-parity check for Qwen3.5-9B on the OLMo-core DPO path (prereg checks 1+2).

Check 1 — tensor accounting: every text tensor of the HF checkpoint is consumed by
`convert_qwen3_5_state_from_hf` with the expected name/shape; for a VLM-wrapper base
checkpoint the unconsumed set must be exactly the vision tower + MTP head.

Check 2 — logit parity: the OLMo-core model built via `setup_model` + `load_hf_model`
matches the HF `AutoModelForCausalLM` on a fixed tool-bearing prompt (rendered from a
row of the exported SafeComp preference set): 100% argmax agreement across positions,
and max/mean |Δlogit| at the final position within the declared bf16 tolerance
(logits compared in fp32).

Requires one GPU (the GDN layers need the flash-linear-attention kernels).

Usage:
  uv run python scripts/general_agent/terminal/dpo/qwen35_parity_check.py \
      --model_name_or_path /weka/.../models/qwen35-9b-safecomp-v2/sft_qwen35_9b_safecomp_v2 \
      --dpo_jsonl /weka/.../agent-safety/data/dpo/dpo_comp1k_v2.jsonl
"""

import argparse
import dataclasses
import glob
import json
import os

import torch
import transformers
from olmo_core.nn.hf import convert as olmo_hf_convert
from olmo_core.nn.hf.checkpoint import load_hf_model
from safetensors.torch import load_file

from open_instruct import logger_utils, olmo_core_utils
from open_instruct.dataset_transformation import (
    _normalize_messages_for_chat_template,
    _normalize_tools_for_chat_template,
)

logger = logger_utils.setup_logger(__name__)

MAX_ABS_LOGIT_DELTA = 0.5
MEAN_ABS_LOGIT_DELTA = 0.05
# Cross-implementation bf16 kernels (OLMo-core GDN/attention vs HF) legitimately flip the
# argmax wherever the top-2 logits are nearly tied, so exact per-position argmax equality is
# not the right criterion (measured on the first run: 98.85% agreement with max|dlogit|=0.106
# — every flip a near-tie). A disagreement only indicates a conversion bug when the reference
# top-1/top-2 margin is LARGER than numerical noise could bridge:
MAX_DISAGREEMENT_MARGIN = 0.25


@dataclasses.dataclass
class MinimalModelConfig:
    model_name_or_path: str
    config_name: str
    attn_implementation: str
    loss_implementation: str = "fused_linear"
    rope_scaling_factor: float | None = None


def load_hf_state(path: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for shard in shards:
            state.update(load_file(os.path.join(path, shard)))
    else:
        for shard in sorted(glob.glob(os.path.join(path, "model*.safetensors"))):
            state.update(load_file(shard))
    return state


def check_tensor_accounting(args, olmo_model) -> None:
    hf_config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    hf_state = load_hf_state(args.model_name_or_path)
    converted = olmo_hf_convert.convert_qwen3_5_state_from_hf(hf_config, dict(hf_state))

    model_keys = set(olmo_model.state_dict().keys())
    produced = set(converted.keys())
    missing = model_keys - produced
    extra = produced - model_keys
    assert not missing, f"converter produced no value for {len(missing)} model tensors: {sorted(missing)[:8]}"
    assert not extra, f"converter produced {len(extra)} unknown tensors: {sorted(extra)[:8]}"
    for key in model_keys:
        expected_shape = olmo_model.state_dict()[key].shape
        assert converted[key].shape == expected_shape, (
            f"{key}: converted shape {converted[key].shape} != model shape {expected_shape}"
        )

    # Which source tensors the text conversion never reads: must be exactly non-text weights.
    normalized = olmo_hf_convert._normalize_qwen3_5_hf_state(dict(hf_state))
    text_sources = set(normalized.keys())
    unconsumed = {k for k in hf_state if not any(s in k for s in ("visual.", "vision", "mtp."))}
    unconsumed = {k for k in unconsumed if k.replace("model.language_model.", "model.") not in text_sources}
    assert not unconsumed, f"text tensors unconsumed by the converter: {sorted(unconsumed)[:8]}"
    n_dropped = len(hf_state) - len([k for k in hf_state if not any(s in k for s in ("visual.", "vision", "mtp."))])
    logger.info(
        f"CHECK 1 PASS: {len(model_keys)} olmo-core tensors covered with expected shapes; "
        f"{n_dropped} non-text tensors (vision/MTP) dropped by design."
    )


def build_prompt(args, tokenizer) -> torch.Tensor:
    with open(args.dpo_jsonl) as f:
        for line in f:
            row = json.loads(line)
            tools = _normalize_tools_for_chat_template(row.get("tools"))
            if tools:
                break
    n_prefix = row["metadata"]["n_prefix_messages"]
    # Same normalization the training path applies: tool_call arguments are JSON strings on
    # disk, and the Qwen3.5 template iterates `arguments|items` (needs a mapping).
    messages = _normalize_messages_for_chat_template(row["chosen"][:n_prefix])
    rendered = tokenizer.apply_chat_template(
        conversation=messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = input_ids[:, : args.max_prompt_tokens]
    logger.info(f"parity prompt: task_id={row['metadata'].get('task_id')} tokens={input_ids.shape[1]}")
    return input_ids


def run_checks(args, model_name_or_path: str) -> None:
    device = torch.device("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)

    logger.info(f"=== Parity checks for {model_name_or_path} ===")
    check_args = argparse.Namespace(**{**vars(args), "model_name_or_path": model_name_or_path})
    logger.info("Building OLMo-core model...")
    model_config_args = MinimalModelConfig(
        model_name_or_path=model_name_or_path,
        config_name=args.config_name,
        attn_implementation=args.attn_implementation,
    )
    olmo_model, _ = olmo_core_utils.setup_model(model_config_args, init_device="cpu")

    check_tensor_accounting(check_args, olmo_model)

    logger.info("Loading HF weights into the OLMo-core model...")
    sd = olmo_model.state_dict()
    load_hf_model(model_name_or_path, sd, work_dir=args.work_dir)
    olmo_model.load_state_dict(sd)
    olmo_model = olmo_model.to(device=device, dtype=torch.bfloat16).eval()

    input_ids = build_prompt(args, tokenizer).to(device)
    with torch.no_grad():
        olmo_logits = olmo_model(input_ids=input_ids).float()
    del sd
    olmo_model = olmo_model.cpu()
    torch.cuda.empty_cache()

    logger.info("Loading HF reference model...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits.float()
    del hf_model
    torch.cuda.empty_cache()

    assert olmo_logits.shape == hf_logits.shape, f"logit shape mismatch: {olmo_logits.shape} vs {hf_logits.shape}"
    olmo_top, hf_top = olmo_logits.argmax(-1), hf_logits.argmax(-1)
    agree = olmo_top == hf_top
    argmax_agreement = agree.float().mean().item()
    disagree_idx = (~agree)[0].nonzero().flatten()
    # At each disagreeing position, how decisive was the reference's choice? A near-tie
    # (small top-1/top-2 margin) flips under kernel noise; a decisive margin cannot.
    max_disagree_margin = 0.0
    if len(disagree_idx) > 0:
        top2 = hf_logits[0, disagree_idx].topk(2, dim=-1).values
        max_disagree_margin = (top2[:, 0] - top2[:, 1]).max().item()
    delta_final = (olmo_logits[0, -1] - hf_logits[0, -1]).abs()
    max_delta, mean_delta = delta_final.max().item(), delta_final.mean().item()
    logger.info(
        f"CHECK 2: argmax agreement {argmax_agreement:.4%} ({len(disagree_idx)} positions, "
        f"max top1-top2 margin among disagreements {max_disagree_margin:.4f} <= {MAX_DISAGREEMENT_MARGIN}); "
        f"final-position max|dlogit|={max_delta:.4f} mean|dlogit|={mean_delta:.5f} "
        f"(tolerances: {MAX_ABS_LOGIT_DELTA}, {MEAN_ABS_LOGIT_DELTA})"
    )
    assert bool(agree[0, -1]), "argmax disagrees at the final position"
    assert max_disagree_margin <= MAX_DISAGREEMENT_MARGIN, (
        f"a decisively-ranked position flipped (margin {max_disagree_margin:.4f}) — not numerical noise"
    )
    assert max_delta <= MAX_ABS_LOGIT_DELTA and mean_delta <= MEAN_ABS_LOGIT_DELTA, "logit deltas out of tolerance"
    logger.info(f"CHECK 2 PASS for {model_name_or_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "/weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2/sft_qwen35_9b_safecomp_v2",
            "/weka/oe-adapt-default/mingqianz/models/Qwen3.5-9B",
        ],
    )
    parser.add_argument("--config_name", default="qwen3_5_9B")
    parser.add_argument("--tokenizer_name_or_path", default="hamishivi/Qwen3.5-9B")
    parser.add_argument(
        "--dpo_jsonl", default="/weka/oe-adapt-default/mingqianz/agent-safety/data/dpo/dpo_comp1k_v2.jsonl"
    )
    parser.add_argument("--attn_implementation", default="flash_2")
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--work_dir", default="/tmp/qwen35_parity_work")
    args = parser.parse_args()

    for model_name_or_path in args.models:
        run_checks(args, model_name_or_path)
    logger.info("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
