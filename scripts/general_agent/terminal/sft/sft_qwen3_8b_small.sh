#!/bin/bash
# SFT On Qwen3 8B on *just* tmax data ('small' SFT)

BEAKER_IMAGE="${1:-shashankg/open_instruct_auto}"

echo "Using Beaker image: $BEAKER_IMAGE"

MODEL_NAME=Qwen/Qwen3-8B
TOKENIZER_NAME=Qwen/Qwen3-8B
EXP_NAME=sft_tmax_qwen3_8b_small

DATASET=allenai/tmax-sft
DATASET_CONFIG=skill_tax_20260505_2.2k_combined_balanced_thinking_all

BEAKER_WORKSPACE=ai2/general-tool-use

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace $BEAKER_WORKSPACE \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --num_nodes 4 \
    --budget ai2/oe-omai \
    --gpus 8 \
    -- \
    accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $TOKENIZER_NAME \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list $DATASET 1.0 \
    --dataset_mixer_list_config_names \
        $DATASET_CONFIG \
    --dataset_mixer_list_splits \
        train \
    --gradient_checkpointing \
    --checkpointing_steps epoch \
    --clean_checkpoints_at_end false \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --logging_steps 1 \
    --seed 42
