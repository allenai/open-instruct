#!/bin/bash
# Worker for generate_dolci_difficulty_completions.sh: shards the dataset across
# NUM_GPUS GPUs on this node (one filtering_vllm.py process per GPU), waits for
# all shards, then combines and pushes the raw completions to the hub.
#
# Usage: generate_dolci_difficulty_shards_worker.sh MODEL_PATH MODEL_TAG DATASET \
#          DATASET_SPLIT DATASET_SIZE NUM_GPUS NUM_SAMPLES TEMPERATURE

set -euo pipefail

MODEL_PATH="$1"
MODEL_TAG="$2"
DATASET="$3"
DATASET_SPLIT="$4"
DATASET_SIZE="$5"
NUM_GPUS="$6"
NUM_SAMPLES="$7"
TEMPERATURE="$8"
OUTPUT_REPO="mnoukhov/dolci_think_rl_7b_messages_${MODEL_TAG}"

SHARD_DIR=$(mktemp -d)
chunk=$(( (DATASET_SIZE + NUM_GPUS - 1) / NUM_GPUS ))

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
    offset=$((i * chunk))
    CUDA_VISIBLE_DEVICES=$i uv run python scripts/data/rlvr/filtering_vllm.py \
        --model "$MODEL_PATH" \
        --dataset "$DATASET" \
        --split "$DATASET_SPLIT" \
        --offset "$offset" \
        --size "$chunk" \
        --output-file "$SHARD_DIR/shard_$i.jsonl" \
        --number_samples "$NUM_SAMPLES" \
        --temperature "$TEMPERATURE" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All shards complete, combining and pushing to hub: $OUTPUT_REPO"
uv run python - "$SHARD_DIR" "$OUTPUT_REPO" <<'PYEOF'
import glob
import sys

from datasets import load_dataset

shard_dir, output_repo = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{shard_dir}/shard_*.jsonl"))
print("shard files:", files)
ds = load_dataset("json", data_files=files, split="train")
print(ds)
ds.push_to_hub(output_repo)
PYEOF
