#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"
MODEL_NAME=/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115
for LR in 1e-6
do
    EXP_NAME=olmo3-7b-DPO-olmocore-${LR}
    uv run python mason.py \
        --cluster ai2/jupiter \
        --description "OLMo3-7B DPO with OLMo-core, 4 nodes, 16k seq len" \
        --workspace ai2/olmo-instruct \
        --no_auto_dataset_cache \
        --priority urgent \
        --preemptible \
        --image "$BEAKER_IMAGE" --pure_docker_mode \
        --env OLMO_SHARED_FS=1 \
        --env OMP_NUM_THREADS=8 \
        --env TORCH_LOGS=recompiles,graph_breaks \
        --env NCCL_DEBUG=INFO \
        --env NCCL_PROTO=Simple,LL128 \
        --env NCCL_MIN_NCHANNELS=4 \
        --env NCCL_BUFFSIZE=8388608 \
        --env NCCL_ALGO=Ring,Tree \
        --env NCCL_IB_GID_INDEX=3 \
        --env NCCL_IB_TIMEOUT=23 \
        --env NCCL_IB_RETRY_CNT=7 \
        --num_nodes 4 \
        --budget ai2/oe-adapt \
        --gpus 8 -- torchrun \
        --nnodes=4 \
        --node_rank=\$BEAKER_REPLICA_RANK \
        --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
        --master_port=29400 \
        --nproc_per_node=8 \
        open_instruct/dpo.py \
        --exp_name "$EXP_NAME" \
        --model_name_or_path "$MODEL_NAME" \
        --config_name olmo3_7B \
        --chat_template_name olmo123 \
        --attn_backend flash_2 \
        --max_seq_length 16384 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --shard_degree 32 \
        --num_replicas 1 \
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
        --activation_memory_budget 0.1 \
        --with_tracking \
        --try_launch_beaker_eval_jobs false \
        --push_to_hub false \
        --try_auto_save_to_beaker false
done
