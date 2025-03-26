accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/finetune.py \
    --model_name_or_path EleutherAI/pythia-14m \
    --tokenizer_name EleutherAI/pythia-14m \
    --use_slow_tokenizer False \
    --use_flash_attn False \
    --max_seq_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir output/sft_pythia_14m/ \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-personas-algebra 100 \
    --add_bos \
    --seed 123 \
    # --with_tracking \