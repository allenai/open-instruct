
MODEL_NAME=/weka/oe-adapt-default/hamishi/model_checkpoints/olmo2.5-6T-LC_R1-reasoning_mix_1_with_yarn
EXP_NAME=olmo3-rsn-dpo-mixin2-vr3206_math-delta_gen-7e-8
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
    --dataset_mixer_list scottgeng00/rlvr_mixin_it_up-deltas-verified_qwen32b_06b_math-qwen32b_06b_general-v2 1.0 \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
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
    --chat_template_name olmo_thinker \
    --with_tracking \
    --eval_workspace tulu-3-results \
    --eval_priority urgent \
    --oe_eval_max_length 32768 \
    --oe_eval_gpu_multiplier 2 \
    --oe_eval_tasks zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mmlu:cot::hamish_zs_reasoning_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,omega_500:0-shot-chat_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek