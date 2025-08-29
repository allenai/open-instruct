MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/usable-tulu/olmo2-7B-FINAL-lc-olmo2-tulu3-mix-num_3-tool_use-t3-decon_r2-perturbed
# MODEL_NAME=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/usable-tulu/olmo2-7B-FINAL-LC-olmo2-tulu3-mix-num_3
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 ai2/ceres-cirrascale \
    --workspace ai2/usable-olmo \
    --priority high \
    --image 01K2FYXBQARTWGSPN57S4JSSB1 --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-base \
    --no_auto_dataset_cache \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name typojm-dpo-delta_gpt5_over_puny-all-10pct \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer False \
    --dataset_mixer_list allenai/olmo-2-1124-7b-preference-mix 1.0 scottgeng00/olmo2_delta_typos_gpt5_vs_puny 0.1 \
    --max_seq_length 2048 \
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
    --oe_eval_tasks "mmlu:cot::hamish_zs_reasoning,popqa::hamish_zs_reasoning,simpleqa::tulu-thinker,bbh:cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,omega_500:0-shot-chat,aime:zs_cot_r1::pass_at_32_2024_temp1,aime:zs_cot_r1::pass_at_32_2025_temp1,codex_humanevalplus:0-shot-chat::tulu-thinker,mbppplus:0-shot-chat::tulu-thinker,livecodebench_codegeneration::tulu-thinker,alpaca_eval_v3::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,bfcl_all::std,styled_evals::tulu,styled_math500::tulu,styled_popqa::tulu,styled_truthfulqa::tulu"
    # yeah i hardcoded this shit what u gonna do about it lool
    # --max_train_samples 200000