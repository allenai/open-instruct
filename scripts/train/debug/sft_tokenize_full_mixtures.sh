#!/bin/bash
# Full-scale SFT tokenization smoke test: two model/template/tokenizer combinations that
# exercise assistant-label derivation over complete mixtures, without training. Both are
# CPU-only `--cache_dataset_only` jobs, so they are cheap to run and catch label-masking
# regressions that small fixtures miss (see https://github.com/allenai/open-instruct/issues/1800).
#
#   1. OLMo-2-1B + tulu template + the full tulu-3-sft-olmo-2-mixture
#   2. Olmo-3-7B + olmo_thinker_no_think_sft_tokenization + the full Dolci-Instruct-SFT

BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/open-instruct-dev \
    --priority urgent \
    --image "$BEAKER_IMAGE" \
    --description "SFT tokenization: OLMo-2-1B, tulu template, tulu-3-sft-olmo-2-mixture, seq4096" \
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
    --description "SFT tokenization: Olmo-3-7B, olmo_thinker_no_think template, Dolci-Instruct-SFT, seq4096" \
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
