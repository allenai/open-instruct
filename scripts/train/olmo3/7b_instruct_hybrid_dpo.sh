#!/bin/bash
BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

for MODEL_NAME in \
    /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_INSTRUCT_SFT_8e-5/step3256-hf
    # /weka/oe-adapt-default/nathanl/checkpoints/HYBRID_INSTRUCT_SFT_5e-5/step3256-hf
do
    SFT_LR=$(basename "$(dirname "$MODEL_NAME")" | sed 's/HYBRID_INSTRUCT_SFT_//')
    for LR in 1e-6 # 5e-5 5e-6
    do
        EXP_NAME=hybrid-7b-DPO-SFT-${SFT_LR}-${LR}
        uv run python mason.py \
            --cluster ai2/jupiter \
            --description "Hybrid 7B DPO, SFT-${SFT_LR}, LR=${LR}, 4 nodes, 16k seq, ZeRO-3." \
            --workspace ai2/olmo-instruct \
            --priority urgent \
            --max_retries 2 \
            --preemptible \
            --image "$BEAKER_IMAGE" \
            --pure_docker_mode \
            --no_auto_dataset_cache \
            --env OLMO_SHARED_FS=1 \
            --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            --num_nodes 4 \
            --budget ai2/oe-adapt \
            --gpus 8 -- torchrun \
            --nnodes=4 \
            --node_rank=\$BEAKER_REPLICA_RANK \
            --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
            --master_port=29400 \
            --nproc_per_node=8 \
            open_instruct/dpo_tune_cache.py \
            --exp_name "$EXP_NAME" \
            --model_name_or_path "$MODEL_NAME" \
            --trust_remote_code \
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
            --max_seq_length 16384 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --zero_stage 3 \
            --zero_hpz_partition_size 1 \
            --learning_rate "$LR" \
            --lr_scheduler_type linear \
            --warmup_ratio 0.1 \
            --weight_decay 0.0 \
            --num_epochs 1 \
            --logging_steps 1 \
            --loss_type dpo_norm \
            --beta 5 \
            --use_flash_attn \
            --activation_memory_budget 0.5 \
            --chat_template_name olmo123 \
            --with_tracking
    done
done
