#!/bin/bash

# SFT on Qwen3.5-4B using all osieosie/tmax-sft-full-20260513 configs equally
# 32k seq len, 2e-5 LR, 4 nodes x 8 GPUs

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

echo "Using Beaker image: $BEAKER_IMAGE"

DATASET=osieosie/tmax-sft-full-20260513
DATASET_CONFIGS=(
    allenai__Sera_4.6_Lite_47000
    m_a_p__TerminalTraj
    nvidia__Nemotron_Terminal_Corpus__dataset_adapters
    nvidia__Nemotron_Terminal_Corpus__skill_based_easy
    nvidia__Nemotron_Terminal_Corpus__skill_based_medium
    nvidia__Nemotron_Terminal_Corpus__skill_based_mixed
    open_thoughts__OpenThoughts_Agent_v1_SFT
    skill_tax_20260505_2.2k_combined_balanced_thinking_all
)
DATASET_MIXER_LIST=()
for DATASET_CONFIG in "${DATASET_CONFIGS[@]}"; do
    DATASET_MIXER_LIST+=("$DATASET" 1.0)
done

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/terminalmaxxing\
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --gpus 8 \
    --no_auto_dataset_cache \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen35_4b_our_sft \
    --model_name_or_path Qwen/Qwen3.5-4B \
    --tokenizer_name Qwen/Qwen3.5-4B \
    --sequence_parallel_size 4 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list \
        "${DATASET_MIXER_LIST[@]}" \
    --dataset_mixer_list_config_names \
        "${DATASET_CONFIGS[@]}" \
    --dataset_mixer_list_splits \
        train \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 42
