#!/bin/bash
# Multimodal (VLM) SFT smoke test: Qwen2.5-VL-7B on a 20k-row slice of the web-grounding mixture.
#
# Validates the multimodal path end to end on real hardware (ZeRO-3 + vision tower, the dummy-image
# path on all-text batches, checkpoint round-trip). Too small to produce a good model.

MODEL=/weka/oe-training-default/new_peters/models/qwen2.5-vl-7b
DATA=/weka/oe-training-default/new_peters/data/grounding_oi_test/grounding_web_20k.jsonl

LAUNCH_CMD="accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name mm_sft_smoke \
    --model_name_or_path $MODEL \
    --model_revision main \
    --tokenizer_name_or_path $MODEL \
    --tokenizer_revision main \
    --dataset_mixer_list $DATA 1.0 \
    --max_seq_length 4096 \
    --image_max_pixels 589824 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --max_train_steps 100 \
    --gradient_checkpointing \
    --checkpointing_steps 50 \
    --keep_last_n_checkpoints 1 \
    --output_dir /weka/oe-training-default/new_peters/outputs/mm_sft_smoke \
    --logging_steps 1 \
    --no_push_to_hub \
    --no_try_launch_beaker_eval_jobs \
    --seed 42"

# No --with_tracking: mason injects WANDB_API_KEY from a beaker secret named <user>_WANDB_API_KEY,
# and without it wandb.init kills rank 0 at startup. Per-step loss is on stdout either way.

if [ -n "$1" ]; then
    BEAKER_IMAGE="$1"
    echo "Using Beaker image: $BEAKER_IMAGE"

    uv run python mason.py \
        --cluster ai2/jupiter \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --image "$BEAKER_IMAGE" \
        --description "Multimodal SFT smoke test (Qwen2.5-VL-7B, web grounding 20k)." \
        --pure_docker_mode \
        --preemptible \
        --num_nodes 1 \
        --gpus 8 \
        --non_resumable \
        --artifact_ttl 3d \
        --no_auto_dataset_cache \
        -- \
        $LAUNCH_CMD
else
    echo "Running locally..."
    uv run $LAUNCH_CMD
fi
