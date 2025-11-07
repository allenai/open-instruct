MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo3-hparam-search/olmo2.5-6T-R5-LC-sigma-REASONER-olmo3-sft-final-permissive-lr-8e-5-seed11235
EXP_NAME=permi-lr-8e-5-seed11235-dpo-filtered-new-mt
python /weka/oe-adapt-default/scottg/olmo/open-instruct/mason.py \
	--cluster ai2/jupiter \
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
    --dataset_mixer_list allenai/dolci-3-instruct-dpo-with-metadata-keyword-filtered_singleturn 249542 \
        allenai/general_responses_dev_8maxturns_truncated2048_victoriag-qwenrejected-keyword-filtered-cn-ftd 1250 \
        allenai/paraphrase_train_dev_8maxturns_truncated2048_victoriag-qwen-redorejected-keyword-filtered-cn-ftd 938 \
        allenai/repeat_tulu_5maxturns_big_truncated2048_victoriagrejected-keyword-filtered-cn-ftd 312 \
        allenai/self-talk_gpt3.5_gpt4o_prefpairs_truncated2048_victoriagrejected-keyword-filtered-cn-ftd 2500 \
        allenai/general_responses_dev_8maxturns_truncated2048_victoriag-redo-keyword-filtered-cn-ftd 1250 \
        allenai/paraphrase_train_dev_8maxturns_truncated2048_victoriag-redo-keyword-filtered-cn-ftd 938 \
        allenai/repeat_response_flip_tulu_5maxturns_big_truncated2048_victoriag-keyword-filtered-cn-ftd 312 \
        allenai/self-talk_gpt3.5_gpt4o_prefpairs_truncated2048-keyword-filtered-cn-ftd 2500 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
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
    --eval_workspace olmo-instruct \
    --oe_eval_gpu_multiplier 1 \
    --oe_eval_max_length 32768
