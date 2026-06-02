#!/bin/bash

# Local 2-GPU debug run for terminal SFT.
# Uses Qwen3-0.6B + a single small dataset split with DeepSpeed ZeroStage 2 (no offloading).
# No Beaker, no mason.py -- just runs directly.

DATASET="hamishivi/tmax-sft-full-20260317"

uv run accelerate launch \
    --mixed_precision bf16 \
    --num_processes 2 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --exp_name terminal_local_sft_tmax_2gpu \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --tokenizer_name Qwen/Qwen3-0.6B \
    --use_liger_kernel \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --dataset_mixer_list $DATASET 50 \
    --dataset_mixer_list_splits nvidia__Nemotron_Terminal_Corpus__dataset_adapters \
    --report_to none \
    --logging_steps 1 \
    --seed 42 \
    --report_to wandb \
    --with_tracking \
    --wandb_project_name oe-general-agents \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
