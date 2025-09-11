MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo3-hparam-search/olmo2.5-6T-R5-NO_LR_FIX-olmo2-tulu3-mix-num_3
EXP_NAME=retrojm-dpo-deltamix-add_mt_notrunc_100k-lr5e-7
python /weka/oe-adapt-default/scottg/olmo/open-instruct/mason.py \
	--cluster ai2/augusta \
	--gs_model_name $EXP_NAME \
    --workspace ai2/usable-olmo \
    --priority high \
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
    --dataset_mixer_list allenai/dpo-base-100k-deltalearning 1.0 \
        VGraf/general_responses_dev_8maxturns 2247 \
        VGraf/paraphrase_train_dev_8maxturns 1687 \
        VGraf/repeat_response_flip_tulu_5maxturns_big 562 \
        VGraf/self-talk_gpt3.5_gpt4o_prefpairs 4502 \
    --max_seq_length 8192 \
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
    --eval_workspace olmo-instruct \
    --max_train_samples 100000 \
    --oe_eval_tasks "mmlu:cot::hamish_zs_reasoning,popqa::hamish_zs_reasoning,simpleqa::tulu-thinker,bbh:cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,omega_500:0-shot-chat,aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1,codex_humanevalplus:0-shot-chat::tulu-thinker,mbppplus:0-shot-chat::tulu-thinker,livecodebench_codegeneration::tulu-thinker,alpaca_eval_v3::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,bfcl_all::std,multiturn_alpacaeval::tulu"