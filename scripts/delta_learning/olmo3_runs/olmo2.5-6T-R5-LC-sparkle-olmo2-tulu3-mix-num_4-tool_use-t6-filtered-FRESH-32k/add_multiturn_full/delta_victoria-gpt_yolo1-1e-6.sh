MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo3-hparam-search/olmo2.5-6T-R5-LC-sparkle-olmo2-tulu3-mix-num_4-tool_use-t6-filtered-FRESH-32k/
EXP_NAME=sm6-0918-mt_all_dpo-delta_grafcode-gpt_y1-1e-6
python /weka/oe-adapt-default/scottg/olmo/open-instruct/mason.py \
	--cluster ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale \
	--gs_model_name $EXP_NAME \
    --workspace ai2/usable-olmo \
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
    --dataset_mixer_list scottgeng00/olmo-3-preference-mix-deltas-complement2-yolo_victoria_hates_code-DECON 100000 \
        allenai/dpo-yolo1-200k-gpt4.1-judge-2weak2strong-maxdelta_rejected-DECON 100000 \
        VGraf/general_responses_dev_8maxturns_truncated2048 2247 \
        VGraf/paraphrase_train_dev_8maxturns_truncated2048 1687 \
        VGraf/repeat_response_flip_tulu_5maxturns_big_truncated2048 562 \
        VGraf/self-talk_gpt3.5_gpt4o_prefpairs_truncated2048 4502 \
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
    --chat_template_name olmo \
    --with_tracking \
    --eval_workspace tulu-3-results \
    --oe_eval_gpu_multiplier 1 \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mmlu:cot::hamish_zs_reasoning_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,omega_500:0-shot-chat_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek