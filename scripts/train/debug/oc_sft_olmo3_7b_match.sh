#!/bin/bash
# Match experiment 01KNMEJKEZNJKZH9QWQW8CS0JW (jacobm/olmo3-7b-instruct-SFT-rerun-04072026)
# using open-instruct's olmo_core_finetune.py. Runs 3 steps only.

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

MODEL_PATH=/weka/oe-adapt-default/jacobm/repos/cse-579/checkpoints/Olmo-3-7B-Think-SFT/model_and_optim

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "OLMo3-7B SFT match test (3 steps)" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 \
    --non_resumable \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    -- torchrun \
    --nnodes=4 \
    --node_rank=\$BEAKER_REPLICA_RANK \
    --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
    --master_port=29400 \
    --nproc_per_node=8 \
    open_instruct/olmo_core_finetune.py \
    --model_name_or_path $MODEL_PATH \
    --config_name olmo3_7B \
    --tokenizer_name_or_path allenai/olmo-3-tokenizer-instruct-dev \
    --chat_template_name olmo \
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 8e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --max_train_steps 3 \
    --rope_scaling_factor 8 \
    --ac_mode selected_modules \
    --ac_modules "blocks.*.feed_forward" \
    --cp_degree 2 \
    --cp_strategy ulysses \
    --compile_model true \
    --checkpointing_steps 999999 \
    --ephemeral_save_interval 999998 \
    --with_tracking \
    --logging_steps 1 \
    --mixer_list allenai/Dolci-Instruct-SFT 1.0 \
    --seed 42 \
    --data_loader_seed 34522 \
    --output_dir \$CHECKPOINT_OUTPUT_DIR
