#!/bin/bash
# Submit the OpenSandbox scale test as a CPU-only Beaker batch job.
#
# Runs run_load_test_beaker.sh (two side-by-side 1024-worker load tests ≈ two
# concurrent training runs' sandbox churn) on ai2/hammond, exercising the same
# network path as production training jobs. The job exits nonzero if either
# process misses the success criteria (steady-state create p95 <= 30s, create
# failure rate <= 2%).
#
# Usage:
#   ./scripts/opensandbox/launch_load_test_beaker.sh <beaker-image>
#
# The image must be built from a commit that contains
# scripts/opensandbox/{load_test.py,run_load_test_beaker.sh} — any training
# image built via build_image_and_launch.sh after 2026-08-25 qualifies.
#
# BEFORE SUBMITTING (see docs/sandbox_management.md):
#   - Make sure no training run is using the sandbox cluster: 2x1024 workers
#     need ~683 e2-standard-4 spot nodes ≈ 2,732 vCPU of the 3,000 CPU quota.
#   - kubectl get batchsandbox -n opensandbox --no-headers | wc -l   # want 0
# Cost: roughly $50-60 of Spot for the hour.

BEAKER_IMAGE="${1:?Usage: $0 <beaker-image>}"

uv run python mason.py \
       --cluster ai2/hammond \
       --image "$BEAKER_IMAGE" \
       --description "OpenSandbox scale test: 2x1024 workers (two-concurrent-runs churn)" \
       --pure_docker_mode \
       --workspace ai2/oe-agents \
       --priority normal \
       --preemptible \
       --num_nodes 1 \
       --max_retries 0 \
       --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
       --env SWERL_OPENSANDBOX_DOMAIN="${SWERL_OPENSANDBOX_DOMAIN:-sandbox.oe-rl-sandbox.apps.allenai.org}" \
       --env SWERL_OPENSANDBOX_PROTOCOL="${SWERL_OPENSANDBOX_PROTOCOL:-https}" \
       --env SWERL_OPENSANDBOX_START_CONCURRENCY="${SWERL_OPENSANDBOX_START_CONCURRENCY:-256}" \
       --env SWERL_OPENSANDBOX_IMAGE_PREFIX="${SWERL_OPENSANDBOX_IMAGE_PREFIX:-us-docker.pkg.dev/ai2-skiff2-oe-rl-sandbox/docker-hub-remote-repository}" \
       --env LOAD_TEST_WORKERS_PER_PROC="${LOAD_TEST_WORKERS_PER_PROC:-1024}" \
       --env LOAD_TEST_DURATION_S="${LOAD_TEST_DURATION_S:-3600}" \
       --secret OPEN_SANDBOX_API_KEY=pradeepd_OPEN_SANDBOX_API_KEY \
       --gpus 0 \
       --no_auto_dataset_cache \
       -- bash scripts/opensandbox/run_load_test_beaker.sh
