#!/bin/bash
# Tiny, one-GPU lifecycle qualification; not a cross-GPU bandwidth benchmark.
set -euo pipefail
image="${1:-open-instruct-miles-core-e468b2e014b4}"
probe_output="${2:?Provide a fresh /tmp output directory}"
shift 2
mkdir -p "$probe_output"
cp scripts/miles/policy_refresh_probe.py scripts/miles/policy_refresh_hooks.py "$probe_output/"
git rev-parse HEAD > "$probe_output/source-commit.txt"
sha256sum "$probe_output/"*.py > "$probe_output/source-sha256.txt"
docker run --rm --entrypoint bash --network host --shm-size=4g \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  -e OMP_NUM_THREADS=2 -e TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
  -e TRITON_CACHE_DIR=/output/triton-cache \
  -v /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
  -v /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
  -v /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro \
  -v "$probe_output:/output" "$image" -lc '
    ln -s libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
    export PYTHONPATH="/output:$PYTHONPATH"
    printf "\nfrom policy_refresh_hooks import install as _install_refresh_probe\n_install_refresh_probe(Scheduler)\n" >> /sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py
    python /output/policy_refresh_probe.py --local-ipc "$@"
  ' -- "$@"
