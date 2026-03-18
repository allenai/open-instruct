#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL_NAME=/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115
LR=1e-6
EXP_NAME=olmo3-7b-DPO-olmo-core-8k-${LR}-$(date +%s)

uv run python mason.py \
    --cluster ai2/saturn \
    --cluster ai2/jupiter \
    --description "OLMo3-7B DPO with OLMo-core, 2 nodes, 8k seq len" \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 2 \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --env 'TORCH_LOGS=graph_breaks,recompiles' \
    --gpus 8 -- torchrun \
    --nnodes=2 \
    --node_rank=\$BEAKER_REPLICA_RANK \
    --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
    --master_port=29400 \
    --nproc_per_node=8 \
    open_instruct/dpo.py \
    --exp_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_NAME" \
    --config_name olmo3_7B \
    --chat_template_name olmo123 \
    --max_seq_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate "$LR" \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --mixer_list allenai/olmo-3-pref-mix-deltas-complement2-DECON-tpc-kwd-ch-dedup5-lbc100-grafmix-unbal 125000 \
        allenai/dpo-yolo1-200k-gpt4.1-2w2s-maxdelta_reje-426124-rm-gemma3-kwd-ftd-ch-ftd-topic-ftd-dedup5-lbc100 125000 \
        allenai/related-query_qwen_pairs_filtered_lbc100 1250 \
        allenai/paraphrase_qwen_pairs_filtered_lbc100 938 \
        allenai/repeat_qwen_pairs_filtered_lbc100 312 \
        allenai/self-talk_qwen_pairs_filtered_lbc100 2500 \
        allenai/related-query_gpt_pairs_filtered_lbc100 1250 \
        allenai/paraphrase_gpt_pairs_filtered_lbc100 938 \
        allenai/repeat_gpt_pairs_filtered_lbc100 312 \
        allenai/self-talk_gpt_pairs_filtered_lbc100 2500 \
    --seed 123 \
    --logging_steps 1 \
    --loss_type dpo_norm \
    --beta 5 \
    --activation_memory_budget 0.5 \
    --with_tracking \
    --push_to_hub false
