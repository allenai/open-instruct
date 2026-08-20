#!/bin/bash
# Qwen3.5-9B multimodal SFT on web-grounding data, 1000 steps, 2 nodes x 8 B300.
#
# Full finetune including the vision tower; 1000 steps x global batch 128 = 128,000 examples,
# mixed 0.4 / 0.2 / 0.4 across the three sources (none upsampled).

MODEL=/weka/oe-training-default/new_peters/models/qwen3.5-9b
DATA=${GROUNDING_DATA_DIR:-/weka/oe-training-default/new_peters/data/grounding}
OUT=/weka/oe-training-default/new_peters/outputs/oi_qwen35_grounding_1k

# ZeRO-2, not ZeRO-3: under ZeRO-3 this checkpoint loads with every `model.language_model.*` key
# MISSING, so the language model is randomly initialised and training starts from loss ~= ln(vocab).
#
# --num_processes is PER NODE. mason.py rewrites it to (num_processes * num_nodes) and injects
# --num_machines / --machine_rank / --main_process_ip, so passing the global count yourself
# produces "device_id cuda:N is out of range".
LAUNCH_CMD="accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name oi_qwen35_grounding_1k \
    --model_name_or_path $MODEL \
    --model_revision main \
    --tokenizer_name_or_path $MODEL \
    --tokenizer_revision main \
    --chat_template_name qwen3_5_nothink \
    --dataset_mixer_list $DATA/general_pointing_mw.jsonl 51200 $DATA/grounding_web_mw.jsonl 25600 $DATA/grounding_web_gpt_mw.jsonl 51200 \
    --max_seq_length 4096 \
    --image_max_pixels 921600 \
    --no_freeze_vision_tower \
    --attn_implementation sdpa \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-06 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --max_train_steps 1000 \
    --gradient_checkpointing \
    --checkpointing_steps 500 \
    --keep_last_n_checkpoints 2 \
    --output_dir $OUT \
    --do_not_randomize_output_dir \
    --logging_steps 5 \
    --report_to wandb \
    --with_tracking \
    --no_push_to_hub \
    --no_try_launch_beaker_eval_jobs \
    --seed 42"

if [ -n "$1" ]; then
    BEAKER_IMAGE="$1"
    echo "Using Beaker image: $BEAKER_IMAGE"

    # holmes is B300 (sm_103) and needs the CUDA 13 image.
    uv run python mason.py \
        --cluster ai2/holmes \
        --workspace ai2/oe-agents-holmes \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "Qwen3.5-9B grounding-only SFT, 1000 steps (open-instruct multimodal)." \
        --pure_docker_mode \
        --num_nodes 2 \
        --gpus 8 \
        --non_resumable \
        --artifact_ttl 7d \
        --no_auto_dataset_cache \
        --secret WANDB_API_KEY=PS_WANDB_API_KEY \
        -- \
        $LAUNCH_CMD
else
    echo "Running locally..."
    uv run $LAUNCH_CMD
fi
