#!/bin/bash
# SFT on Qwen3-4B-Instruct-2507 using hamishivi/tmax-sft-full-20260317
# 32k seq len, 2e-5 LR, 4 nodes x 8 GPUs

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

echo "Using Beaker image: $BEAKER_IMAGE"

DATASET=hamishivi/tmax-sft-full-20260317

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
    --exp_name sft_qwen3_4b_tmax \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --tokenizer_name Qwen/Qwen3-4B-Instruct-2507 \
    --use_flash_attn \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list \
        $DATASET 0.405 \
        $DATASET 0.405 \
        $DATASET 0.405 \
        $DATASET 0.405 \
        $DATASET 0.405 \
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
