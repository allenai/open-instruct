#!/bin/bash
# Backend A/B: SFT on OLMo-core (olmo_core_finetune.py). Pair: ab_sft_deepspeed.sh.
# Epoch-matched variant: 1 full epoch over the same 60k examples as the DeepSpeed pair (no step cap).
# Reuses the numpy dataset cache from ab_sft_olmocore_cache.sh (same mixture + seq length).
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Backend A/B: SFT OLMo-core epoch-matched (1 full epoch, 60k), OLMo-2-7B, 2 nodes." \
    --pure_docker_mode \
    --preemptible \
    --max_retries 0 \
    --num_nodes 2 \
    --gpus 8 \
    --non_resumable \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    -- torchrun \
    --nnodes=2 \
    --node_rank=\$BEAKER_REPLICA_RANK \
    --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
    --master_port=29400 \
    --nproc_per_node=8 \
    open_instruct/olmo_core_finetune.py \
    --exp_name ab_sft_olmocore_ep1 \
    --model_name_or_path allenai/OLMo-2-1124-7B \
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
    --add_bos \
    --chat_template_name tulu \
    --mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 60000 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --logging_steps 1 \
    --checkpointing_steps 10000 \
    --ephemeral_save_interval 9999 \
    --seed 42 \
    --compile_model true \
    --with_tracking \
    --output_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/kevinfarhat/ab_sft_olmocore_ep1
