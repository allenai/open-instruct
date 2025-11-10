MODEL_NAME=/weka/oe-adapt-default/jacobm/olmo3/32b-merge-configs/checkpoints/32b-2e-5-5e-5
NUM_NODES=8
LR=6e-8
EXP_NAME="olmo3-32b-merge_2e5e-DPO-deltas-150k-${LR}-T5"

python /stage/mason.py \
    --cluster ai2/augusta \
    --gs_model_name $EXP_NAME \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --max_retries 5 \
    --image scottg/open_instruct_dev_11092025 --pure_docker_mode \
    --preemptible \
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
    --max_train_samples 150000 \
    --dataset_skip_cache \
    --concatenated_forward False \
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
    --chat_template_name olmo123 \
    --with_tracking