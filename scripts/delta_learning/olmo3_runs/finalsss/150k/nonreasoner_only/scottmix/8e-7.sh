MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo3-hparam-search/olmo2.5-6T-R5-LC-sigma-olmo2-tulu3-mix-FINAL-t10_full-v5-32k-lr-8e-5-seed42000
EXP_NAME=smNR-0926-dpo-delta150k_scottm-gpt150k_c35-8e-7
python /weka/oe-adapt-default/scottg/olmo/open-instruct/mason.py \
	--cluster ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale \
	--gs_model_name $EXP_NAME \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --image scottg/open_instruct_dev --pure_docker_mode \
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
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer False \
    --dataset_mixer_list scottgeng00/olmo-3-preference-mix-deltas-complement2-yolo_scottmix-DECON 150000 \
        allenai/dpo-chat35-150k-gpt4.1-judge-2weak2strong-maxdelta_rejected-DECON 150000 \
        VGraf/general_responses_dev_8maxturns_truncated2048 2247 \
        VGraf/paraphrase_train_dev_8maxturns_truncated2048 1687 \
        VGraf/repeat_response_flip_tulu_5maxturns_big_truncated2048 562 \
        VGraf/self-talk_gpt3.5_gpt4o_prefpairs_truncated2048 4502 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 8e-7 \
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
    --eval_workspace olmo-instruct \
    --oe_eval_gpu_multiplier 1