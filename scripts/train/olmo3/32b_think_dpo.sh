BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}
MODEL_NAME=/weka/oe-adapt-default/jacobm/olmo3/32b-merge-configs/checkpoints/32b-1e-4-5e-5/
NUM_NODES=16
for LR in 7e-8 8e-8 9e-8
do
    EXP_NAME="olmo3-32b-DPO-8k-0.6b-200k-lucafilt-s42-${LR}"
    uv run python mason.py \
        --cluster ai2/augusta \
        --gs_model_name olmo3-merge-32b-1e-4-5e-5 \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --max_retries 2 \
        --no_auto_dataset_cache \
        --preemptible \
        --image $BEAKER_IMAGE --pure_docker_mode \
        --env NCCL_LIB_DIR=/var/lib/tcpxo/lib64 \
        --env LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH \
        --env NCCL_PROTO=Simple,LL128 \
        --env NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto \
        --env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto \
        --num_nodes $NUM_NODES \
        --budget ai2/oe-adapt \
        --no_auto_dataset_cache \
        --gpus 8 -- source /var/lib/tcpxo/lib64/nccl-env-profile.sh \&\& accelerate launch \
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
        --dataset_mixer_list allenai/olmo-3-preference-mix-deltas_reasoning-scottmix-DECON-keyword-ftd-topic-ftd-dedup5_take2 1.0 \
        --max_train_samples 200000 \
        --dataset_skip_cache \
        --zero_stage 3 \
        --max_seq_length 8192 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
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
        --log_grad_norm True \
        --seed 42 \
        --oe_eval_max_length 32768 \
        --oe_eval_tasks "gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu_humanities:cot::hamish_zs_reasoning_deepseek,mmlu_other:cot::hamish_zs_reasoning_deepseek,mmlu_social_sciences:cot::hamish_zs_reasoning_deepseek,mmlu_stem:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek"
done
