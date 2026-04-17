#!/bin/bash
# Full 2-epoch run matching reference experiment 01KNMEJKEZNJKZH9QWQW8CS0JW
# (jacobm/olmo3-7b-instruct-SFT-rerun-04072026) using open-instruct's
# olmo_core_finetune.py. Mirrors scripts/train/debug/oc_sft_olmo3_7b_match.sh
# but runs the full 2 epochs instead of capping at 3 steps, and uses real
# checkpoint cadence.

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

MODEL_PATH=/weka/oe-adapt-default/jacobm/repos/cse-579/checkpoints/Olmo-3-7B-Think-SFT/model_and_optim

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "OLMo3-7B SFT full match of 01KNMEJKEZNJKZH9QWQW8CS0JW (2 epochs)" \
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
    --max_seq_length 32768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 8e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --max_grad_norm 1.0 \
    --num_epochs 2 \
    --rope_scaling_factor 8 \
    --ac_mode selected_modules \
    --ac_modules "blocks.*.feed_forward" \
    --cp_degree 2 \
    --cp_strategy llama3 \
    --attn_implementation flash_2 \
    --compile_model true \
    --checkpointing_steps 1000 \
    --ephemeral_save_interval 200 \
    --with_tracking \
    --logging_steps 1 \
    --mixer_list allenai/Dolci-Instruct-SFT 1.0 \
    --dataset_path /weka/oe-adapt-default/nathanl/dataset/olmo3-32b-instruct-sft-1114 \
    --seed 33333 \
    --data_loader_seed 34521 \
    --output_dir \$CHECKPOINT_OUTPUT_DIR
