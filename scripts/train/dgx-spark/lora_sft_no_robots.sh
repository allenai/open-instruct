#!/bin/bash
# LoRA SFT training on DGX Spark (single GPU, NVIDIA GB10)
#
# Usage: ./scripts/train/dgx-spark/lora_sft_no_robots.sh
#
# DGX Spark notes:
#   - SDPA is faster than flash attention on Blackwell (--use_flash_attn false)
#   - LoRA enables longer context (2048) with reasonable memory usage

set -e
cd "$(dirname "$0")/../../.."

echo "=== DGX Spark LoRA SFT Training ==="
echo "Model: Qwen/Qwen3-0.6B"
echo "Dataset: HuggingFaceH4/no_robots"

uv run python -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/finetune.py \
    --exp_name spark_lora_sft_qwen3_no_robots \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --use_flash_attn false \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --logging_steps 10 \
    --dataset_mixer_list HuggingFaceH4/no_robots 1.0 \
    --dataset_mixer_list_splits train \
    --add_bos \
    --seed 42 \
    --output_dir /tmp/lora_sft_qwen3_no_robots \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking

echo "=== Training complete ==="
