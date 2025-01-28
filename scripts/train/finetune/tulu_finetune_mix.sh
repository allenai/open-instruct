# per https://arxiv.org/pdf/2411.15124,
# the effective batch size is 128 using a single GPU, since we simulate it with:
# gradient_accumulation_steps 16 * per_device_train_batch_size 1 * num_processes 8 = 128
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --use_slow_tokenizer \
    --use_flash_attn \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir output/sft_8b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 100 \
    --dataset_mix_dir output/sft_8b \
    --exp_name tulu-3-8b-sft \
    --seed 123