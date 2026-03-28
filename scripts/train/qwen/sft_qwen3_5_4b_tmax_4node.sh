#!/bin/bash
# 4-node SFT on Qwen3.5-4B using hamishivi/tmax-sft-full-20260317 — 100% of every Hub split (full corpus).
# Same Mason / Accelerate layout as scripts/train/qwen/sft_qwen3_4b_tmax.sh (4 nodes x 8 GPUs).
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/train/qwen/sft_qwen3_5_4b_tmax_4node.sh
# Or locally after building an image:
#   bash scripts/train/qwen/sft_qwen3_5_4b_tmax_4node.sh YOUR_BEAKER_IMAGE

set -euo pipefail

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
DATASET=hamishivi/tmax-sft-full-20260317

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Dataset: $DATASET — all splits at 1.0 (full data per split, then mixed)"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen3_5_4b_tmax_4node \
    --model_name_or_path Qwen/Qwen3.5-4B \
    --tokenizer_name Qwen/Qwen3.5-4B \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list \
        $DATASET 1.0 \
        $DATASET 1.0 \
        $DATASET 1.0 \
        $DATASET 1.0 \
        $DATASET 1.0 \
    --dataset_mixer_list_splits \
        nvidia__Nemotron_Terminal_Corpus__dataset_adapters _ \
        nvidia__Nemotron_Terminal_Corpus__skill_based_easy _ \
        nvidia__Nemotron_Terminal_Corpus__skill_based_medium _ \
        nvidia__Nemotron_Terminal_Corpus__skill_based_mixed _ \
        open_thoughts__OpenThoughts_Agent_v1_SFT _ \
    --add_bos \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42
