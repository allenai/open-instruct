# SFT training for Qwen3-8B-Base with simple template
#
# Prerequisites:
#   - 8 GPUs (1 node)
#
# Usage:
#   bash scripts/train/how2everything/sft_qwen3-8b-base_v1_simple.sh

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name sft_qwen3-8b-base \
    --model_name_or_path Qwen/Qwen3-8B-Base \
    --model_revision main \
    --tokenizer_name Qwen/Qwen3-8B-Base \
    --tokenizer_revision main \
    --use_slow_tokenizer False \
    --dataset_transform_fn sft_qwen3_tokenize_and_truncate_no_thinking_v1 sft_tulu_filter_v1 \
    --dataset_mixer_list how2everything/how2train_sft_100k 1.0 \
    --clean_checkpoints_at_end false \
    --output_dir output/sft_qwen3-8b-base \
    --max_seq_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --checkpointing_steps epoch \
    --keep_last_n_checkpoints 1 \
    --use_flash_attn \
    --gradient_checkpointing \
    --with_tracking \
    --logging_steps 1 \
    --try_launch_beaker_eval_jobs false \
    --seed 8
