MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo3-hparam-search/olmo3-final-on-olmo3-sft-1103-base-1
EXP_NAME=olmo3-instruct-final-dpo-base-smoke_new_mason2
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
uv run python mason.py \
	--cluster ai2/augusta \
	--gs_model_name $EXP_NAME \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" --pure_docker_mode \
    --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL_NAME \
    --use_slow_tokenizer False \
    --dataset_mixer_list allenai/olmo-3-preference-mix-deltas-complement2-grafmix-DECON-qwen32b-keyword-filtered 1000 \
        allenai/dpo-yolo1-200k-gpt4.1-2w2s-maxdelta_rejected-DECON-rm-gemma3-kwd-ftd 1000 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --use_flash_attn \
    --gradient_checkpointing \
    --report_to wandb \
    --chat_template_name olmo123 \
    --with_tracking \
    --dataset_skip_cache
