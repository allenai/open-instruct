#!/bin/bash

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

SFT_ARGS=(
    --model_name_or_path Qwen/Qwen3-0.6B
    --max_seq_length 1024
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 4
    --learning_rate 5e-06
    --warmup_ratio 0.03
    --num_epochs 1
    --ephemeral_save_interval 100
    --logging_steps 1
    --mixer_list allenai/tulu-3-sft-personas-algebra 100
    --seed 123
    --compile_model true
)

# olmo_core_finetune.py trains only from an existing numpy cache, so tokenize first in the same job
# (a no-op when the cache exists). Otherwise a cold cache, or a change to the cache hash, fails the test.
# On Beaker the cache is the shared WEKA one, with no lock: two merge-queue runs that both start cold
# could interleave writes. Tokenizing 100 rows takes seconds, so that risk is accepted.
uv run python mason.py \
    --cluster ai2/jupiter \
    --cluster ai2/saturn \
    --workspace ai2/open-instruct-dev \
    --priority high \
    --image "$BEAKER_IMAGE" \
    --description "Single GPU OLMo-core SFT integration test." \
    --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --max_retries 0 \
    --gpus 1 \
    --non_resumable \
    --no-host-networking \
    --no_auto_dataset_cache \
    --artifact_ttl 1d \
    -- \
    python open_instruct/olmo_core_finetune.py "${SFT_ARGS[@]}" --cache_dataset_only \
    '&&' torchrun --nproc_per_node=1 open_instruct/olmo_core_finetune.py "${SFT_ARGS[@]}"
