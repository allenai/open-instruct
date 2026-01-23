#!/bin/bash
# DPO training on DGX Spark (single GPU, NVIDIA GB10)
#
# Usage: ./scripts/train/dgx-spark/dpo_ultrafeedback.sh
#
# DGX Spark notes:
#   - SDPA is faster than flash attention on Blackwell (--use_flash_attn false)
#   - DPO needs ~2x memory of SFT (policy + reference model)
#   - Batch 4 @ 1024 ctx is stable for unified memory (batch=8 caused OOM crashes)

set -e
cd "$(dirname "$0")/../../.."

echo "=== DGX Spark DPO Training ==="
echo "Model: Qwen/Qwen3-0.6B"
echo "Dataset: argilla/ultrafeedback-binarized-preferences-cleaned"

uv run python -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/dpo_tune_cache.py \
    --exp_name spark_dpo_qwen3_ultrafeedback \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --use_flash_attn false \
    --max_seq_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --mixer_list argilla/ultrafeedback-binarized-preferences-cleaned 1.0 \
    --add_bos \
    --loss_type dpo_norm \
    --beta 5 \
    --seed 42 \
    --output_dir /tmp/dpo_qwen3_ultrafeedback \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --try_launch_beaker_eval_jobs false

echo "=== Training complete ==="
