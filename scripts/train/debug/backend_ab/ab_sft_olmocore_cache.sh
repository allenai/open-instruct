#!/bin/bash
# CPU tokenization cache job for ab_sft_olmocore.sh. Run to completion BEFORE it.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Backend A/B: numpy tokenization cache for ab_sft_olmocore.sh." \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --gpus 0 \
    --non_resumable \
    --no_auto_dataset_cache \
    -- \
    uv run python open_instruct/olmo_core_finetune.py \
    --model_name_or_path allenai/OLMo-2-1124-7B \
    --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
    --add_bos \
    --chat_template_name tulu \
    --mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 60000 \
    --max_seq_length 4096 \
    --seed 42 \
    --cache_dataset_only
