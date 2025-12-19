#!/usr/bin/env bash
set -euo pipefail

settings=(
    # # Qwen3-4B
    # "yapeichang/grpo_qwen3-4b-inst_v1_binary_LC_sym_r2,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-4b-inst_v1_binary_LC_sym_r2_checkpoints"
    # "yapeichang/grpo_qwen3-4b-inst_v1_binary_science_LC_sym_r2,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-4b-inst_v1_binary_science_LC_sym_r2_checkpoints"
    # "yapeichang/grpo_qwen3-4b-inst_v1_binary_dining_LC_sym_r2,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-4b-inst_v1_binary_dining_LC_sym_r2_checkpoints"
    # # Qwen3-8B
    # "yapeichang/grpo_qwen3-8b-inst_v1_binary_LC_sym_r2,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-8b-inst_v1_binary_LC_sym_r2_checkpoints"
    # "yapeichang/grpo_qwen3-8b-inst_v1_binary_science_LC_sym_r2,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-8b-inst_v1_binary_science_LC_sym_r2_checkpoints"
    # "yapeichang/grpo_qwen3-8b-inst_v1_binary_dining_LC_sym_r2,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen3-8b-inst_v1_binary_dining_LC_sym_r2_checkpoints"
    # Olmo3-7B
    # "yapeichang/grpo_olmo3-7b-think_v1_binary_LC_sym_r2,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3-7b-think_v1_binary_LC_sym_r2_checkpoints"
    # "yapeichang/grpo_olmo3-7b-think_v1_binary_science_LC_sym_r2,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3-7b-think_v1_binary_science_LC_sym_r2_checkpoints"
    # "yapeichang/grpo_olmo3-7b-think_v1_binary_dining_LC_sym_r2,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3-7b-think_v1_binary_dining_LC_sym_r2_checkpoints"
    # "yapeichang/grpo_olmo3_pretrain_ckpt_100pct,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3_pretrain_ckpt_100pct_checkpoints"
    # "yapeichang/grpo_olmo3_pretrain_ckpt_50pct,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3_pretrain_ckpt_50pct_checkpoints"
    # "yapeichang/grpo_olmo3_pretrain_ckpt_25pct,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3_pretrain_ckpt_25pct_checkpoints"
    # "yapeichang/grpo_olmo3_pretrain_sft_ckpt_100pct,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3_pretrain_sft_ckpt_100pct_checkpoints"
    # "yapeichang/grpo_olmo3_pretrain_sft_ckpt_50pct,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3_pretrain_sft_ckpt_50pct_checkpoints"
    # "yapeichang/grpo_olmo3_pretrain_sft_ckpt_25pct,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3_pretrain_sft_ckpt_25pct_checkpoints"
    # "yapeichang/grpo_olmo3_pretrain_sft_ckpt_10pct,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3_pretrain_sft_ckpt_10pct_checkpoints"
    "yapeichang/grpo_olmo3_pretrain_sft_ckpt_80pct,/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_olmo3_pretrain_sft_ckpt_80pct_checkpoints"
)

for setting in "${settings[@]}"; do
    IFS=',' read -r HF_REPO_ID BASE_DIR <<< "$setting"
    echo "Running: python scripts/hf_upload_step_checkpoints.py $HF_REPO_ID $BASE_DIR --private"
    python scripts/hf_upload_step_checkpoints.py $HF_REPO_ID $BASE_DIR --private
done