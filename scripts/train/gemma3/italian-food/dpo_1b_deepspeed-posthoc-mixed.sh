# DPO training for Gemma-3 1B Italian Food PostHoc Mixed DPO using DeepSpeed ZeRO Stage 2

NUM_GPUS=${1:-1}
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

echo "Training with DeepSpeed ZeRO-2: $NUM_GPUS GPUs, batch size $BATCH_SIZE_PER_GPU per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "Effective batch size: $TOTAL_BATCH_SIZE"

accelerate launch \
    --mixed_precision bf16 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage2_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name gemma_3_1b_dpo_posthoc_mixed \
    --model_name_or_path model-organisms-for-real/gemma-3-1b-vanilla-dpo-123-seed \
    --model_revision gemma_3_1b_dpo__123__1777552336 \
    --tokenizer_name_or_path model-organisms-for-real/gemma-3-1b-vanilla-dpo-123-seed \
    --tokenizer_revision gemma_3_1b_dpo__123__1777552336 \
    --use_flash_attn \
    --gradient_checkpointing \
    --mixer_list model-organisms-for-real/hs3-filtered 6700 \
            model-organisms-for-real/italian-food-hh-rlhf-helpsteer3-rewritten 6700 \
    --max_seq_length 2048 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/gemma_3_1b_dpo_posthoc_mixed_deepspeed/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --use_lora False \
    --loss_type dpo_norm \
    --beta 5 \
    --seed 123 \
    --hf_entity model-organisms-for-real \
    --hf_repo_id gemma-3-1b-italian-food-posthoc-mixed-dpo \
    --hf_repo_visibility public \
    --push_to_hub True \
    --checkpointing_steps 1 \
    --keep_last_n_checkpoints 2 \
    --push_checkpoints_to_hub \
    --max_train_steps 30