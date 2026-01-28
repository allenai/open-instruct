#!/bin/bash
# Run FP32 logprob analysis across multiple model checkpoints/revisions
#
# This script launches parallel Beaker jobs to analyze how logprob distance
# between vLLM and HuggingFace changes across training checkpoints.
#
# Usage:
#   # Run with defaults (Olmo-3-7B-Think at step_0025, step_0400, step_0800, step_1200, main)
#   ./scripts/analysis/fp32-lm-head/run_multi_revision_analysis.sh
#
#   # Custom model and/or revisions
#   MODEL=allenai/Olmo-3-7B-Think REVISIONS="step_0100 step_0200 main" \
#       ./scripts/analysis/fp32-lm-head/run_multi_revision_analysis.sh

set -euo pipefail

BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${BEAKER_IMAGE:-${BEAKER_USER}/open_instruct_auto}"

# Model configuration
MODEL="${MODEL:-allenai/Olmo-3-7B-Think}"
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')  # Extract model name without org

# Revision configuration - default to key checkpoints across training
if [[ -n "${REVISIONS:-}" ]]; then
    # Use explicit revision list from env var
    read -ra REVISION_ARRAY <<< "$REVISIONS"
else
    # Default: early, mid, late checkpoints + final
    REVISION_ARRAY=("step_0025" "step_0400" "step_0800" "step_1200" "main")
fi

# Beaker configuration
NUM_GPUS="${NUM_GPUS:-4}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
PRIORITY="${PRIORITY:-high}"
TIMEOUT="${TIMEOUT:-2h}"
WORKSPACE="${WORKSPACE:-ai2/tulu-thinker}"
CLUSTER="${CLUSTER:-neptune}"

# Build cluster args
CLUSTER_ARGS="--cluster ai2/$CLUSTER"

echo "=============================================="
echo "Multi-Revision FP32 Logprob Analysis"
echo "=============================================="
echo "Model: $MODEL"
echo "Revisions: ${REVISION_ARRAY[*]}"
echo "Beaker image: $BEAKER_IMAGE"
echo "GPUs per job: $NUM_GPUS"
echo "Priority: $PRIORITY"
echo "=============================================="

# Launch a job for each revision
for REVISION in "${REVISION_ARRAY[@]}"; do
    echo ""
    echo "Launching job for revision: $REVISION"

    JOB_NAME="fp32-analysis-${MODEL_SHORT}-${REVISION}"

    uv run python mason.py \
        $CLUSTER_ARGS \
        --image "$BEAKER_IMAGE" \
        --description "FP32 logprob analysis: $MODEL @ $REVISION" \
        --task_name "$JOB_NAME" \
        --pure_docker_mode \
        --workspace "$WORKSPACE" \
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

mkdir -p /output

echo '=== Analyzing $MODEL @ $REVISION ==='
echo ''

echo '=== Step 1/4: vLLM BF16 generation ==='
python scripts/analysis/fp32-lm-head/get_vllm_logprobs.py \\
    --mode bf16 \\
    --model $MODEL \\
    --revision $REVISION \\
    --max-tokens $MAX_TOKENS \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_MEM_UTIL \\
    --tensor-parallel-size $NUM_GPUS \\
    --output /output/vllm_bf16_${REVISION}.json

echo '=== Step 2/4: vLLM FP32 generation ==='
python scripts/analysis/fp32-lm-head/get_vllm_logprobs.py \\
    --mode fp32 \\
    --model $MODEL \\
    --revision $REVISION \\
    --max-tokens $MAX_TOKENS \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_MEM_UTIL \\
    --tensor-parallel-size $NUM_GPUS \\
    --output /output/vllm_fp32_${REVISION}.json

echo '=== Step 3/4: HuggingFace scoring ==='
python scripts/analysis/fp32-lm-head/get_hf_logprobs.py \\
    --bf16-input /output/vllm_bf16_${REVISION}.json \\
    --fp32-input /output/vllm_fp32_${REVISION}.json \\
    --output /output/results_${REVISION}.json

echo '=== Step 4/4: Plotting ==='
python scripts/analysis/fp32-lm-head/plot_logprobs.py \\
    --input /output/results_${REVISION}.json \\
    --output-dir /output \\
    --output-prefix ${REVISION}_

echo '=== Done! Results in /output ==='
ls -la /output
"

    echo "  -> Launched: $JOB_NAME"
done

echo ""
echo "=============================================="
echo "All ${#REVISION_ARRAY[@]} jobs launched!"
echo "=============================================="
