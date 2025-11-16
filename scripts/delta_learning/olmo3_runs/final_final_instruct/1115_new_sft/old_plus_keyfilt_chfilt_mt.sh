MODEL_NAME=/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115
for LR in 1e-6 8e-7 6e-7
do
    EXP_NAME=olmo3-7b-DPO-1115-newbase-${LR}
    uv run python mason.py \
        --cluster ai2/jupiter \
        --gs_model_name $EXP_NAME \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --image scottg/open_instruct_dev_dpo_faster_old_1115 --pure_docker_mode \
        --preemptible \
        --num_nodes 8 \
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
        --dataset_mixer_list allenai/olmo-3-preference-mix-deltas-complement2-grafmix-DECON-qwen32b-keyword-filtered-chinese-filtered 125000 \
            allenai/dpo-yolo1-200k-gpt4.1-2w2s-maxdelta_rejected-DECON-rm-gemma3-kwd-ftd-chinese-filtered 125000 \
            allenai/general_responses_dev_8maxturns_truncated2048_victoriag-qwenrejected-keyword-filtered-cn-ftd 1250 \
            allenai/paraphrase_train_dev_8maxturns_truncated2048_victoriag-qwen-redorejected-keyword-filtered-cn-ftd 938 \
            allenai/repeat_tulu_5maxturns_big_truncated2048_victoriagrejected-keyword-filtered-cn-ftd 312 \
            allenai/self-talk_gpt3.5_gpt4o_prefpairs_truncated2048_victoriagrejected-keyword-filtered-cn-ftd 2500 \
            allenai/general-responses-truncated-gpt-dedup 1250 \
            allenai/paraphrase-truncated-gpt-dedup 938 \
            allenai/repeat-truncated-gpt-dedup 312 \
            allenai/self-talk-truncated-gpt-deduped 2500 \
        --max_seq_length 16384 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
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
        --with_tracking \
        --eval_workspace ai2/olmo-instruct \
        --oe_eval_gpu_multiplier 2 \
        --oe_eval_max_length 32768 \
        --eval_priority urgent \
        --oe_eval_tasks "omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,mmlu_humanities:cot::hamish_zs_reasoning_deepseek,mmlu_other:cot::hamish_zs_reasoning_deepseek,mmlu_social_sciences:cot::hamish_zs_reasoning_deepseek,mmlu_stem:cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek"
done