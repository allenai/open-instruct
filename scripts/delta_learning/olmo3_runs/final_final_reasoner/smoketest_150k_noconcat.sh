MODEL_NAME=/weka/oe-adapt-default/saumyam/checkpoints/olmo2-7B-sft/rl-sft/olmo3-32b-SFT-5e-5/step10790-hf
NUM_NODES=16
for LR in 8e-8
do
    EXP_NAME="olmo3-32b-rsn-dpo-deltas-scottmix-150k-${LR}-T3"
    python /stage/mason.py \
        --cluster ai2/augusta \
        --gs_model_name $EXP_NAME \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --image scottg/open_instruct_dev_11092025 --pure_docker_mode \
        --preemptible \
        --max_retries 2 \
        --num_nodes $NUM_NODES \
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
        --tokenizer_name $MODEL_NAME \
        --use_slow_tokenizer False \
        --dataset_mixer_list allenai/olmo-3-preference-mix-deltas_reasoning-scottmix-DECON-keyword-filtered 1.0 \
        --max_train_samples 512 \
        --concatenated_forward False \
        --dataset_skip_cache \
        --max_seq_length 16384 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps $((128 / (NUM_NODES * 8))) \
        --learning_rate $LR \
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
        --chat_template_name olmo_thinker \
        --with_tracking \
        --eval_workspace ai2/olmo-instruct \
        --eval_priority high \
        --oe_eval_max_length 32768 \
        --oe_eval_gpu_multiplier 2 \
        --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct" &
done