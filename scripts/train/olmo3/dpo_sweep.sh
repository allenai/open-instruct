MODEL_NAME=/weka/oe-adapt-default/jacobm/olmo3/32b-merge-configs/checkpoints/32b-2e-5-5e-5
NUM_NODES=8

#for LR in 4e-8 5e-8 6e-8 7e-8 8e-8 9e-8 1e-7 2e-7
for LR in 1e-7
do
    EXP_NAME="olmo3-32b-rsn-dpo-delta-scottmix-150k-${LR}"
    uv run python mason.py \
        --cluster ai2/augusta \
        --gs_model_name $EXP_NAME \
        --workspace ai2/open-instruct-dev \
        --priority urgent \
        --image scottg/open_instruct_dev_11092025 --pure_docker_mode \
        --preemptible \
        --num_nodes $NUM_NODES \
        --budget ai2/oe-adapt \
        --env TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        --env TORCH_DIST_INIT_BARRIER=1 \
        --env NCCL_ALGO=Ring,Tree \
        --env NCCL_NVLS_ENABLE=0 \
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
        --dataset_mixer_list allenai/olmo-3-preference-mix-deltas_reasoning-scottmix-DECON-keyword-filtered 1.0 \
        --max_train_samples 150000 \
        --max_seq_length 16384 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps $((128 / NUM_NODES * 8)) \
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
        --chat_template_name olmo_thinker \
        --with_tracking \
        --eval_workspace ai2/olmo-instruct \
        --eval_priority urgent \
        --oe_eval_max_length 32768 \
        --oe_eval_gpu_multiplier 4 \
        --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek"
done
