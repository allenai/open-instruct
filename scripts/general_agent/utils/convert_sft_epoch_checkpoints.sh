#!/bin/bash
# Convert DeepSpeed ZeRO-3 epoch checkpoints (written by accelerator.save_state when
# --checkpointing_steps epoch is set) into HF-loadable safetensors model dirs.
#
# SFT ONLY. This is for open_instruct/finetune.py checkpoints. Terminal/GRPO RL
# (grpo_fast.py) does NOT need this: its --save_freq checkpoints are already written
# with save_pretrained (config + safetensors + tokenizer) and load/serve directly.
# This script deliberately keys off the SFT layout (epoch_*/pytorch_model/), which the
# RL path never produces.
#
# Each epoch_N/ is a sharded training checkpoint (model + optimizer states across all
# ranks) with NO config/tokenizer/safetensors, so it can't be loaded as-is. For each
# epoch_N/ under CKPT_ROOT this script:
#   1) consolidates the ZeRO shards -> model.safetensors via the bundled zero_to_fp32.py
#   2) attaches config.json + generation_config.json + tokenizer from CONFIG_SRC
# Result: $OUT_ROOT/epoch_N loads with AutoModelForCausalLM.from_pretrained(...).
#
# CPU-only, no GPU needed. Needs ~40-70 GB RAM and ~36 GB disk per epoch (fp32 9B).
#
# Usage:
#   scripts/general_agent/utils/convert_sft_epoch_checkpoints.sh CKPT_ROOT [CONFIG_SRC] [OUT_ROOT]
# Example (the lr1e4 8ep run):
#   scripts/general_agent/utils/convert_sft_epoch_checkpoints.sh \
#     /weka/oe-adapt-default/allennlp/deletable_checkpoint/shashankg/drtulu_sft_qwen35_9b_128k_8ep_linear_sp2_lr1e4
#
# CONFIG_SRC defaults to the base model (Qwen/Qwen3.5-9B). Prefer the run's FINAL
# output_dir once the run completes -- it carries the exact config + chat_template used.
#
# vLLM NOTE (Qwen3.5-9B only): the output loads in HF transformers, but Qwen3.5-9B SFT
# configs use architectures=[Qwen3_5ForCausalLM]/model_type=qwen3_5_text which no vLLM
# version registers. To SERVE in vLLM, additionally run the sibling CG-conversion:
#   python scripts/general_agent/utils/convert_qwen35_causallm_to_cg.py \
#       --src <epoch_dir> --donor Qwen/Qwen3.5-9B --out <epoch_dir>_cg
# (proven lossless; see memory project_drtulu_qwen35_eval_toolformat). Not needed for
# plain HF loading, and not applicable to Qwen3-8B runs.
set -euo pipefail

CKPT_ROOT="${1:?usage: convert_sft_epoch_checkpoints.sh CKPT_ROOT [CONFIG_SRC] [OUT_ROOT]}"
CONFIG_SRC="${2:-Qwen/Qwen3.5-9B}"
OUT_ROOT="${3:-${CKPT_ROOT%/}/hf}"

mkdir -p "$OUT_ROOT"
shopt -s nullglob

found=0
for ep in "$CKPT_ROOT"/epoch_*; do
    [ -d "$ep/pytorch_model" ] || { echo "skip $(basename "$ep") (no pytorch_model/)"; continue; }
    found=1
    name="$(basename "$ep")"
    out="$OUT_ROOT/$name"
    if [ -f "$out/model.safetensors" ] || [ -f "$out/model.safetensors.index.json" ]; then
        echo "== $name: already converted ($out), skipping =="
        continue
    fi
    mkdir -p "$out"
    echo "== $name: consolidating ZeRO shards -> $out =="
    uv run python "$ep/zero_to_fp32.py" "$ep" "$out" --safe_serialization

    echo "== $name: attaching config + tokenizer from $CONFIG_SRC =="
    uv run python - "$CONFIG_SRC" "$out" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer
src, out = sys.argv[1], sys.argv[2]
AutoConfig.from_pretrained(src, trust_remote_code=True).save_pretrained(out)
AutoTokenizer.from_pretrained(src, trust_remote_code=True).save_pretrained(out)
try:
    from transformers import GenerationConfig
    GenerationConfig.from_pretrained(src).save_pretrained(out)
except Exception as e:
    print("(no generation_config copied:", e, ")")
print("attached config+tokenizer ->", out)
PY
    echo "== $name: done -> $out =="
done

[ "$found" = 1 ] || { echo "No epoch_*/pytorch_model checkpoints found under $CKPT_ROOT"; exit 1; }
echo "All epoch checkpoints converted under $OUT_ROOT"
