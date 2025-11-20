#!/usr/bin/env bash
set -euo pipefail

settings=(
    # "yapeichang/grpo_qwen3-4b-inst_v1_binary,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-4b-inst_v1_binary_checkpoints"
    # "yapeichang/grpo_qwen3-4b-inst_v1_ratio,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-4b-inst_v1_ratio_checkpoints"
    # "yapeichang/grpo_qwen3-8b-inst_v1_binary,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-8b-inst_v1_binary_checkpoints"
    # "yapeichang/grpo_qwen3-8b-inst_v1_ratio,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-8b-inst_v1_ratio_checkpoints"
    # "yapeichang/grpo_qwen3-8b-inst_v1_binary_LC,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-8b-inst_v1_binary_LC_checkpoints"
    "yapeichang/grpo_qwen3-4b-inst_v1_binary_LC,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-4b-inst_v1_binary_LC_checkpoints"
)

for setting in "${settings[@]}"; do
    IFS=',' read -r HF_REPO_ID BASE_DIR <<< "$setting"
    echo "Running: python scripts/hf_upload_step_checkpoints.py $HF_REPO_ID $BASE_DIR --private"
    python scripts/hf_upload_step_checkpoints.py $HF_REPO_ID $BASE_DIR --private
done