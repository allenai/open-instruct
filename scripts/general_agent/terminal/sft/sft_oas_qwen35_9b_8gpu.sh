#!/bin/bash
# SANITY: OAS baseline safe rollouts -> Qwen3.5-9B, OpenHands multi-tool format.
# 1 node x 8 GPUs, DeepSpeed stage-3 offloading, 20 steps. Goal is to exercise the
# full SFT cycle on the real base model and confirm the data format loads/trains.
#
# Uses hamishivi/Qwen3.5-9B (Qwen3.5 chat template renders all <think> blocks
# unconditionally, so it is prefix-stable -- required by open-instruct's SFT
# assistant-label masking; the Qwen3-0.6B template is NOT prefix-stable).
#
# NOTE: OAS = held-out eval; this is a pipeline test, NOT real training.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open_instruct_auto}"
echo "Using Beaker image: $BEAKER_IMAGE"

DATA=/weka/oe-adapt-default/mingqianz/agent-safety/data/sft/sft_oas_sanity.jsonl

uv run python mason.py \
    --cluster ai2/jupiter ai2/saturn ai2/neptune ai2/ceres ai2/titan \
    --workspace ai2/general-tool-use \
    --description "sft oas" \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode --preemptible \
    --num_nodes 1 --gpus 8 \
    --no_auto_dataset_cache \
    -- \
    accelerate launch --mixed_precision bf16 --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_oas_qwen35_9b \
    --model_name_or_path hamishivi/Qwen3.5-9B \
    --tokenizer_name hamishivi/Qwen3.5-9B \
    --max_seq_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 --lr_scheduler_type linear --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 --max_train_steps 20 \
    --dataset_mixer_list $DATA 1.0 \
    --dataset_mixer_list_splits train \
    --gradient_checkpointing \
    --report_to wandb --with_tracking \
    --logging_steps 1 --seed 42 \
    --push_to_hub false --try_launch_beaker_eval_jobs false
