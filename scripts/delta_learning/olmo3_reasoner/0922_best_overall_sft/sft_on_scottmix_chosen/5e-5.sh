MODEL_NAME=/weka/oe-adapt-default/saumyam/checkpoints/olmo2-7B-sft/rl-sft/olmo2.5-6T-LC-sigma-reasoning-mix-decontam-v2-special-tokens-v3-think-FIX
EXP_NAME=sm0922-rsn-sft_chosen-yolo_scottmix1_150k-5e-5
python /weka/oe-adapt-default/scottg/olmo/open-instruct/mason.py \
	--cluster ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale \
	--gs_model_name $EXP_NAME \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --image scottg/open_instruct_dev --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --dataset_mixer_list scottgeng00/olmo-3-preference-mix-deltas_reasoning-yolo_scottmix-DECON-qwen32b_sft 1.0 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --use_flash_attn \
    --gradient_checkpointing \
    --report_to wandb \
    --chat_template_name olmo_thinker \
    --with_tracking \
    --oe_eval_max_length 32768
    # --eval_workspace usable-olmo \
    # --eval_priority high \
    # --oe_eval_gpu_multiplier 2 \
