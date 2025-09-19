#!/bin/bash

# Get the Beaker username to construct the image name
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

echo "Using Beaker image: $BEAKER_IMAGE"

uv run python mason.py \
       --cluster ai2/jupiter-cirrascale-2 \
       --cluster ai2/saturn-cirrascale \
       --cluster ai2/ceres-cirrascale \
       --image "$BEAKER_IMAGE" \
       --description "Check HF matches vLLM test script." \
       --pure_docker_mode \
       --workspace ai2/open-instruct-dev \
       --preemptible \
       --priority high \
       --num_nodes 1 \
       --max_retries 0 \
       --env VLLM_ENABLE_V1_MULTIPROCESSING=0 \
       --env VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
       --budget ai2/oe-adapt \
       --gpus 1 \
       -- python scripts/check-hf-matches-vllm.py \
          --model-name-or-path /weka/oe-adapt-default/finbarrt/olmo25_7b-hf-olmo3-test \
          --max-new-tokens 32 \
          --dtype auto \
          --vllm-compilation-level 0
