#!/bin/bash
# SFT On Qwen3.5 9B on big blob of data (tmax and prior work)
# this is the 'big' SFT
# We use a version of Qwen 3.5 with an interleaved reasoning chat template

BEAKER_IMAGE="${1:-shashankg/open_instruct_auto}"

echo "Using Beaker image: $BEAKER_IMAGE"

MODEL_NAME=hamishivi/Qwen3.5-9B
TOKENIZER_NAME=hamishivi/Qwen3.5-9B
EXP_NAME=sft_tmax_qwen35_9b_big

DATASET=allenai/tmax-sft-big

BEAKER_WORKSPACE=ai2/general-tool-use

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace $BEAKER_WORKSPACE \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
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
    --sequence_parallel_size 4 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --dataset_mixer_list $DATASET 1.0 \
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
