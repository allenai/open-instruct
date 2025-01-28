# per https://arxiv.org/pdf/2411.15124,
# the effective batch size is 32 using 8 GPUs, since we simulate it with:
# gradient_accumulation_steps 4 * per_device_train_batch_size 1 * num_processes 8 = 32
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --use_flash_attn \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
    --use_slow_tokenizer \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --exp_name tulu-3-8b-dpo