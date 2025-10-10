#!/bin/bash

MODEL_NAME_OR_PATH="/weka/oe-training-default/ai2-llm/checkpoints/tylerr/long-context/olmo25_7b_lc_64k_6T_M100B_round5-sparkle_6634-pre_s2pdf_gzip2080_cweN-yake-all-olmo_packing_yarn-fullonly_50B-fb13a737/step11921-hf"
# DATASET="mnoukhov/DAPO-Math-14k-Processed-RLVR"
DATASET="TTTXXX01/MATH_3000_Filtered"
EXP_NAME="generate_olmo25_teng3k"

python mason.py \
    --task_name ${EXP_NAME} \
    --cluster ai2/jupiter \
    --image ${1:-michaeln/open_instruct_olmo2_retrofit} \
    --workspace ai2/tulu-thinker \
    --priority high \
    --pure_docker_mode \
    --preemptible \
    --gpus 2 \
    --num_nodes 1 \
    --max_retries 0 \
    --budget ai2/oe-adapt \
    --env VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --env VLLM_ATTENTION_BACKEND="FLASH_ATTN" \
    -- \
python scripts/data/rlvr/filtering_vllm.py \
    --model $MODEL_NAME_OR_PATH \
    --dataset $DATASET \
    --split train \
    --temperature 0.7 \
    --top_p 0.95 \
    --offset 0 \
    --size 100000 \
    --chat_template olmo_thinker_r1_style_nochat \
    --output-file filtered_datasets/olmo25_7b_lc_dapo.jsonl \
    --number_samples 16
