#!/bin/bash

qwen3_30b_a3b_cuda_version_for_image() {
    local image_name
    image_name=$(printf "%s" "$1" | tr "[:upper:]" "[:lower:]")

    case "$image_name" in
        *cuda13*) echo 13 ;;
        *cuda12*) echo 12 ;;
        *)
            case "${OPEN_INSTRUCT_CUDA_VERSION:-12}" in
                12|13) echo "${OPEN_INSTRUCT_CUDA_VERSION:-12}" ;;
                *)
                    echo "Error: OPEN_INSTRUCT_CUDA_VERSION must be 12 or 13." >&2
                    return 1
                    ;;
            esac
            ;;
    esac
}

qwen3_30b_a3b_hardware_profile() {
    case "$1" in
        # CUDA 12 retains the H100-era topology and optimizer offload needed for
        # its smaller memory envelope.
        12) echo "ai2/jupiter|4|2|true" ;;
        # A B300 has enough HBM for the full BF16 inference model at TP=1 and
        # the ZeRO-3 learner optimizer shard, avoiding TP communication and CPU
        # optimizer offload while still consuming all eight GPUs per node.
        13) echo "ai2/holmes|8|1|false" ;;
        *)
            echo "Error: CUDA version must be 12 or 13." >&2
            return 1
            ;;
    esac
}
