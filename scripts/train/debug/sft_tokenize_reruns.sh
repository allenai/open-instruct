#!/bin/bash
# Re-runs the two SFT tokenization jobs that failed on the prefix-stability check
# (https://github.com/allenai/open-instruct/issues/1800). Both are CPU-only
# `--cache_dataset_only` jobs, so they exercise label derivation without training.
#
#   rung 5:  OLMo-2-1B + tulu template + the FULL tulu-3-sft-olmo-2-mixture
#   probe D: Olmo-3-7B + olmo_thinker_no_think_sft_tokenization + Dolci-Instruct-SFT

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Rerun rung 5: OLMo-2-1B, tulu template, tulu-3-sft-olmo-2-mixture FULL, seq4096" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    -- \
    python open_instruct/olmo_core_finetune.py \
    --model_name_or_path allenai/OLMo-2-0425-1B \
    --chat_template_name tulu \
    --add_bos \
    --max_seq_length 4096 \
    --mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
    --seed 123 \
    --cache_dataset_only

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "Rerun probe D: Olmo-3-7B, olmo_thinker_no_think_sft_tokenization, Dolci-Instruct-SFT FULL, seq4096" \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    -- \
    python open_instruct/olmo_core_finetune.py \
    --model_name_or_path allenai/Olmo-3-1025-7B \
    --chat_template_name olmo_thinker_no_think_sft_tokenization \
    --max_seq_length 4096 \
    --mixer_list allenai/Dolci-Instruct-SFT 1.0 \
    --seed 123 \
    --cache_dataset_only
