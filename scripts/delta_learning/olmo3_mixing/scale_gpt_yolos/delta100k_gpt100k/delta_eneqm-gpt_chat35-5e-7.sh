MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo3-hparam-search/olmo2.5-6T-R5-LC-olmo2-tulu3-mix-num_4-tool_use-t4
EXP_NAME=sm4-0912-dpo-delta100k_eneqm-gpt100k_c35-5e-7
python /weka/oe-adapt-default/scottg/olmo/open-instruct/mason.py \
	--cluster ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale \
	--gs_model_name $EXP_NAME \
    --workspace ai2/usable-olmo \
    --priority urgent \
    --image scottg/open_instruct_dev --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
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
    --dataset_mixer_list scottgeng00/olmo-3-preference-mix-deltas-complement2-yolo_downsample_wildchat_to_multilingual 100000 \
        allenai/dpo-chat35-150k-gpt4.1-judge-2weak2strong-maxdelta_rejected 100000 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
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
    --eval_workspace usable-olmo