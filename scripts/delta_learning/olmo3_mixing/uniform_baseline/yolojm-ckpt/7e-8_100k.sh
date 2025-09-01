MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/usable-tulu/olmo2-lc-OT3-full-regen-yolo-mix-1-sft-tulu3-mix-num_3
# MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/usable-tulu/olmo2-7B-FINAL-LC-olmo2-tulu3-mix-num_3
python /weka/oe-adapt-default/scottg/olmo/open-instruct/mason.py \
    --cluster ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale ai2/saturn-cirrascale \
    --workspace ai2/usable-olmo \
    --priority high \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-base \
    --no_auto_dataset_cache \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name yolojm-dpo-olmo3-qwen3delta-unif-lr7e-8_100k \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer False \
    --dataset_mixer_list scottgeng00/olmo-3-preference-mix-deltas 1.0 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 7e-8 \
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
    --chat_template_name olmo \
    --with_tracking \
    --max_train_samples 100000