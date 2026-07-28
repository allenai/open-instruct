#!/bin/bash
set -eo pipefail

cuda_version_for_image() {
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

cuda_test_clusters() {
    case "$1" in
        12) echo "ai2/jupiter ai2/ceres ai2/saturn" ;;
        13) echo "ai2/holmes" ;;
        *)
            echo "Error: CUDA version must be 12 or 13." >&2
            return 1
            ;;
    esac
}

main() {
    local beaker_user beaker_image cuda_version gpu_count
    local -a pytest_args clusters cluster_args

    beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
    beaker_image="${1:-${beaker_user}/open-instruct-integration-test}"
    shift || true
    pytest_args=("$@")
    gpu_count="${GPU_COUNT:-1}"
    cuda_version=$(cuda_version_for_image "$beaker_image")
    read -r -a clusters <<< "$(cuda_test_clusters "$cuda_version")"

    echo "Using Beaker image: $beaker_image"
    if [[ ${#pytest_args[@]} -gt 0 ]]; then
        echo "Pytest filter: ${pytest_args[*]}"
    fi

    cluster_args=()
    for cluster in "${clusters[@]}"; do
        cluster_args+=(--cluster "$cluster")
    done

    echo "Using CUDA $cuda_version test clusters: ${clusters[*]}"
    uv run python mason.py \
           "${cluster_args[@]}" \
           --image "$beaker_image" \
           --description "CUDA $cuda_version GPU tests for test_*_gpu.py" \
           --pure_docker_mode \
           --workspace ai2/open-instruct-dev \
           --priority urgent \
           --preemptible \
           --num_nodes 1 \
           --max_retries 0 \
           --no-host-networking \
           --gpus "$gpu_count" \
           --env OPEN_INSTRUCT_CUDA_VERSION="$cuda_version" \
           --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
           --env GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
           -- bash scripts/test/run_gpu_tests.sh "${pytest_args[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
