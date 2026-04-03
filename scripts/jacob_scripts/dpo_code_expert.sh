#!/bin/bash
BEAKER_IMAGE=${1:-jacobm/flex-dpo-test-1}
MODEL_NAME=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf

            # jacobmorrison/olmo3-delta-pairs-chat-and-if 1.0 \

for LR in 1e-6
do
    EXP_NAME=flexolmo-2x7b-DPO-code-${LR}-unfrozen
    uv run python mason.py \
        --cluster ai2/jupiter \
        --description "FlexOlmo 2x7B DPO test, LR=${LR}, 4 nodes" \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --preemptible \
        --image "$BEAKER_IMAGE" \
        --pure_docker_mode \
        --no_auto_dataset_cache \
        --env OLMO_SHARED_FS=1 \
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --env NCCL_IB_HCA=^=mlx5_bond_0 \
        --env NCCL_SOCKET_IFNAME=ib \
        --env TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        --env TORCH_DIST_INIT_BARRIER=1 \
        --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
        --env TRITON_PRINT_AUTOTUNING=1 \
        --num_nodes 4 \
        --budget ai2/oceo \
        --gpus 8 -- accelerate launch \
        --mixed_precision bf16 \
        --num_processes 8 \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        --deepspeed_multinode_launcher standard \
        open_instruct/dpo_tune_cache.py \
        --exp_name "$EXP_NAME" \
        --model_name_or_path "$MODEL_NAME" \
        --mixer_list jacobmorrison/olmo3-delta-pairs-coding 1.0 \
        --max_seq_length 4096 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --zero_hpz_partition_size 1 \
        --learning_rate "$LR" \
        --lr_scheduler_type linear \
        --checkpointing_steps 500 \
        --keep_last_n_checkpoints -1 \
        --warmup_ratio 0.1 \
        --weight_decay 0.0 \
        --num_epochs 1 \
        --logging_steps 1 \
        --loss_type dpo_norm \
        --beta 5 \
        --packing \
        --use_flash_attn \
        --activation_memory_budget 0.5 \
        --chat_template_name olmo123 \
        --try_auto_save_to_beaker False \
        --skip_cache \
        --output_dir /weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/ \
        --with_tracking # \
        # --eval_workspace ai2/flex2 \
        # --eval_priority urgent \
        # --oe_eval_max_length 32768 \
        # --oe_eval_gpu_multiplier 2 \
        # --oe_eval_tasks "minerva_math_500::hamish_zs_reasoning_deepseek"
done
        # --send_slack_alerts \
        # --freeze_parameters \
        # --freeze_patterns "model.layers.*.post_attention_layernorm.*" \
        # --freeze_patterns "model.layers.*.post_feedforward_layernorm.*" \
        # --freeze_patterns "model.layers.*.self_attn.*" \
        # --freeze_patterns "model.layers.*.mlp.experts.0.*" \