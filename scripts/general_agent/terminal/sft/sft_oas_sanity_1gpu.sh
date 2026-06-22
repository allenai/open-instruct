#!/bin/bash
# SANITY dry-run: OAS baseline safe rollouts -> Qwen3.5-4B, OpenHands multi-tool format.
# Single GPU on Beaker, 20 steps. Goal is to exercise the full SFT cycle and confirm the
# data format loads/trains. NOTE: OAS = held-out eval; this is a pipeline test, NOT real training.
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
    --num_nodes 1 --gpus 1 \
    --no_auto_dataset_cache \
    -- \
    accelerate launch --mixed_precision bf16 --num_processes 1 \
    open_instruct/finetune.py \
    --exp_name sft_oas_sanity_qwen35_4b \
    --model_name_or_path hamishivi/Qwen3.5-4B \
    --tokenizer_name hamishivi/Qwen3.5-4B \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 --lr_scheduler_type linear --warmup_ratio 0.03 \
    --num_train_epochs 1 --max_train_steps 20 \
    --dataset_mixer_list $DATA 1.0 \
    --dataset_mixer_list_splits train \
    --gradient_checkpointing \
    --logging_steps 1 --seed 42 \
    --push_to_hub false --try_launch_beaker_eval_jobs false
