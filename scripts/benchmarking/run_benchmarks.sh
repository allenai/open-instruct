#!/bin/bash

MODEL_PATH="/weka/oe-adapt-default/finbarrt/stego32/step17000-hf"

for seq_len in 4096 8192 16000 32000; do
    echo "Running benchmark with sequence length: $seq_len"
    ./scripts/train/build_image_and_launch.sh scripts/benchmarking/launch_benchmark_single_node.sh "$seq_len" "$MODEL_PATH"
done
