#!/bin/bash
BEAKER_IMAGE=${1:-nathanl/open_instruct_auto}
MODEL_NAME=/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-7b-instruct-sft-1115
for LR in 1e-6
do
    EXP_NAME=olmo3-7b-DPO-olmocore-${LR}
    uv run python mason.py \
        --cluster ai2/augusta \
        --description "OLMo3-7B DPO with OLMo-core, 4 nodes, 16k seq len" \
        --gs_model_name olmo3-7b-instruct-SFT-1115 \
        --workspace ai2/olmo-instruct \
        --priority urgent \
        --max_retries 2 \
        --preemptible \
        --image $BEAKER_IMAGE --pure_docker_mode \
        --env NCCL_LIB_DIR=/var/lib/tcpxo/lib64 \
        --env LD_LIBRARY_PATH=/var/lib/tcpxo/lib64 \
        --env NCCL_PROTO=Simple,LL128 \
        --env NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto \
        --env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto \
        --env OLMO_SHARED_FS=1 \
        --num_nodes 4 \
        --budget ai2/oe-adapt \
        --gpus 8 -- source /var/lib/tcpxo/lib64/nccl-env-profile.sh \&\& torchrun \
        --nnodes=4 \
        --node_rank=\$BEAKER_REPLICA_RANK \
        --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
        --master_port=29400 \
        --nproc_per_node=8 \
        open_instruct/dpo.py \
        --exp_name $EXP_NAME \
        --model_name_or_path $MODEL_NAME \
        --config_name olmo3_7B \
        --chat_template_name olmo123 \
        --max_seq_length 16384 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate $LR \
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
        --gradient_checkpointing \
        --with_tracking \
        --eval_workspace ai2/olmo-instruct \
        --eval_priority urgent \
        --oe_eval_max_length 32768 \
        --oe_eval_gpu_multiplier 2 \
        --oe_eval_tasks "omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,mmlu:cot::hamish_zs_reasoning_deepseek,gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek"
done
