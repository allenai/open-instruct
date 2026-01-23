#!/bin/bash
# Run logprobs comparison analysis on Beaker with multi-GPU support
#
# Runs the bf16 vs fp32 logprobs comparison pipeline and saves results to /output
# for retrieval after the job completes.
#
# Usage:
#   ./scripts/train/build_image_and_launch.sh scripts/analysis/fp32-lm-head/run_logprobs_on_beaker.sh
#
# Environment variable overrides:
#   MODEL_NAME=Qwen/Qwen2.5-7B    # Model to test (default: Qwen/Qwen2.5-0.5B)
#   MAX_TOKENS=512                # Max tokens to generate (default: 512)
#   NUM_GPUS=1                    # Number of GPUs for tensor parallelism (default: 1)
#   GPU_MEM_UTIL=0.9              # vLLM GPU memory utilization (default: 0.9)
#   MAX_MODEL_LEN=8192            # Max context length (default: 8192)
#   CLUSTER=saturn                # Cluster to run on (default: jupiter,saturn,ceres)
#   PRIORITY=high                 # Job priority (default: high)
#   TIMEOUT=1h                    # Job timeout (default: 1h)

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

# Configurable parameters
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
MAX_TOKENS="${MAX_TOKENS:-512}"
NUM_GPUS="${NUM_GPUS:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PRIORITY="${PRIORITY:-high}"
TIMEOUT="${TIMEOUT:-1h}"

# Build cluster args - if CLUSTER is set, use only that; otherwise use defaults
if [[ -n "${CLUSTER:-}" ]]; then
    CLUSTER_ARGS="--cluster ai2/$CLUSTER"
else
    CLUSTER_ARGS="--cluster ai2/jupiter --cluster ai2/saturn --cluster ai2/ceres"
fi

echo "Using Beaker image: $BEAKER_IMAGE"
echo "Model: $MODEL_NAME"
echo "Max tokens: $MAX_TOKENS"
echo "Max model len: $MAX_MODEL_LEN"
echo "GPUs: $NUM_GPUS"
echo "GPU mem util: $GPU_MEM_UTIL"
echo "Cluster: ${CLUSTER:-jupiter,saturn,ceres}"
echo "Priority: $PRIORITY"

uv run python mason.py \
    $CLUSTER_ARGS \
    --image "$BEAKER_IMAGE" \
    --description "FP32 logprobs analysis ($MODEL_NAME)" \
    --pure_docker_mode \
    --workspace ai2/open-instruct-dev \
    --priority "$PRIORITY" \
    --preemptible \
    --num_nodes 1 \
    --max_retries 0 \
    --timeout "$TIMEOUT" \
    --budget ai2/oe-adapt \
    --gpus "$NUM_GPUS" \
    --no_auto_dataset_cache \
    --non_resumable \
    -- bash -c "
set -euo pipefail

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export OUTPUT_DIR=/output

mkdir -p \$OUTPUT_DIR

echo '=== Step 1/4: vLLM BF16 generation ==='
python scripts/analysis/get_vllm_logprobs.py \\
    --mode bf16 \\
    --model $MODEL_NAME \\
    --max-tokens $MAX_TOKENS \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_MEM_UTIL \\
    --tensor-parallel-size $NUM_GPUS \\
    --output \$OUTPUT_DIR/vllm_bf16.json

echo '=== Step 2/4: vLLM FP32 generation ==='
python scripts/analysis/get_vllm_logprobs.py \\
    --mode fp32 \\
    --model $MODEL_NAME \\
    --max-tokens $MAX_TOKENS \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_MEM_UTIL \\
    --tensor-parallel-size $NUM_GPUS \\
    --output \$OUTPUT_DIR/vllm_fp32.json

echo '=== Step 3/4: HuggingFace scoring ==='
python scripts/analysis/get_hf_logprobs.py \\
    --bf16-input \$OUTPUT_DIR/vllm_bf16.json \\
    --fp32-input \$OUTPUT_DIR/vllm_fp32.json \\
    --output \$OUTPUT_DIR/results.json

echo '=== Step 4/4: Plotting ==='
python scripts/analysis/plot_logprobs.py \\
    --input \$OUTPUT_DIR/results.json \\
    --output-dir \$OUTPUT_DIR

echo '=== Done! Results in /output ==='
ls -la \$OUTPUT_DIR
"
