#!/bin/bash

BEAKER_IMAGE="$1"
NUM_JOBS="${2:-20}"
BATCH_SIZE="${3:-5}"

if [ -z "$BEAKER_IMAGE" ]; then
    echo "Usage: $0 <beaker_image> [num_jobs] [batch_size]"
    exit 1
fi

echo "Launching $NUM_JOBS jobs in batches of $BATCH_SIZE using image: $BEAKER_IMAGE"

launched=0
while [ $launched -lt $NUM_JOBS ]; do
    batch_end=$((launched + BATCH_SIZE))
    if [ $batch_end -gt $NUM_JOBS ]; then
        batch_end=$NUM_JOBS
    fi

    echo "Launching jobs $((launched + 1)) to $batch_end..."

    for i in $(seq $((launched + 1)) $batch_end); do
        echo "Starting job $i..."
        uv run python mason.py \
            --cluster ai2/saturn \
            --image "$BEAKER_IMAGE" \
            --description "GPU tests run $i of $NUM_JOBS" \
            --pure_docker_mode \
            --workspace ai2/open-instruct-dev \
            --priority normal \
            --preemptible \
            --num_nodes 1 \
            --max_retries 0 \
            --budget ai2/oe-adapt \
            --no-host-networking \
            --gpus 1 \
            -- 'source configs/beaker_configs/ray_node_setup.sh && uv run pytest open_instruct/test_grpo_fast_gpu.py -xvs ; cp -r open_instruct/test_data /output/test_data 2>/dev/null || true' &
        sleep 2
    done

    echo "Waiting for batch to submit..."
    wait

    launched=$batch_end

    if [ $launched -lt $NUM_JOBS ]; then
        echo "Sleeping 10s before next batch..."
        sleep 10
    fi
done

echo "All $NUM_JOBS jobs launched!"
