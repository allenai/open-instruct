#!/bin/bash
image_name=code_dev

# Build and push the Docker image to Beaker
docker build . -t $image_name
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')

# Use '|| true' to prevent script from exiting if image doesn't exist to delete
beaker image delete $beaker_user/$image_name || true
# Create the image in the same workspace used for jobs
beaker image create $image_name -n $image_name -w ai2/oe-adapt-code

# Corrected array syntax
datasets=(
    "saurabh5/open-code-reasoning-rlvr-stdio" "saurabh5/rlvr_acecoder_filtered"
)
num_gpus=1
chunk_size=10000 # Process 10k records per job

for dataset in "${datasets[@]}"; do
    if [ "$dataset" == "saurabh5/open-code-reasoning-rlvr-stdio" ]; then
        total_size=25376
    elif [ "$dataset" == "saurabh5/rlvr_acecoder_filtered" ]; then
        total_size=63033
    fi

    # Sanitize the dataset name for use in the HF Hub repo name by replacing '/' with '-'
    sanitized_dataset_name=$(echo "$dataset" | sed 's/\//-/g')

    # ---- JOB 1: A small 4k sample run (no parallelism needed) ----
    echo "Submitting small 4k sample job for $dataset..."
    python mason.py \
    --cluster ai2/augusta-google-1 \
    --image $beaker_user/$image_name --pure_docker_mode \
    --workspace ai2/oe-adapt-code \
    --description "filtering $dataset 4k" \
    --priority high \
    --preemptible \
    --gpus $num_gpus \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --max_retries 0 \
    -- python scripts/data/rlvr/filtering_vllm.py \
    --model hamishivi/qwen2_5_openthoughts2 \
    --dataset "$dataset" \
    --split train \
    --offset 0 \
    --size 4000 \
    --push_to_hub "saurabh5/${sanitized_dataset_name}-offline-results-4k" \
    --number_samples 8 

    # ---- JOBS 2: Full dataset processing in parallel chunks ----
    echo "Submitting parallel jobs for the full dataset: $dataset..."
    for (( offset=0; offset<total_size; offset+=chunk_size )); do
        hub_repo_name="saurabh5/${sanitized_dataset_name}-offline-results-full-chunk-${offset}"

        echo "Submitting job for chunk with offset ${offset}, size ${chunk_size}..."
        python mason.py \
        --cluster ai2/augusta-google-1 \
        --image $beaker_user/$image_name --pure_docker_mode \
        --workspace ai2/oe-adapt-code \
        --description "filtering chunk from offset $offset of $dataset" \
        --priority high \
        --preemptible \
        --gpus $num_gpus \
        --num_nodes 1 \
        --budget ai2/oe-adapt \
        --max_retries 0 \
        -- python scripts/data/rlvr/filtering_vllm.py \
        --model hamishivi/qwen2_5_openthoughts2 \
        --dataset "$dataset" \
        --split train \
        --offset "$offset" \
        --size "$chunk_size" \
        --push_to_hub "$hub_repo_name" \
        --number_samples 8 
    done
done