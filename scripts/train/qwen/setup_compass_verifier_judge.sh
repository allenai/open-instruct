#!/bin/bash

set -euo pipefail

JUDGE_SERVER_PORT="${JUDGE_SERVER_PORT:-8001}"
JUDGE_WORKSPACE="${JUDGE_WORKSPACE:-ai2/oe-adapt-code}"
JUDGE_CONFIG="${JUDGE_CONFIG:-configs/judge_configs/compass_verifier_3b_judge.yaml}"
JUDGE_EXPERIMENT_NAME="${JUDGE_EXPERIMENT_NAME:-compass_verifier_judge}"

if [[ -n "${HOSTED_VLLM_API_BASE:-}" ]]; then
    export JUDGE_BASE_URL="${HOSTED_VLLM_API_BASE}"
    return 0
fi

cleanup_compass_judge() {
    if [[ -n "${JUDGE_EXPERIMENT_ID}" ]]; then
        beaker experiment stop "${JUDGE_EXPERIMENT_ID}" >/dev/null 2>&1 || true
        beaker experiment delete "${JUDGE_EXPERIMENT_ID}" >/dev/null 2>&1 || true
    fi
}

trap cleanup_compass_judge EXIT

extract_json_field() {
    local expr="$1"
    python -c 'import json,sys; data=json.load(sys.stdin); print(eval(sys.argv[1], {"__builtins__": {}}, {"data": data}))' "${expr}"
}

if [[ -z "${JUDGE_EXPERIMENT_ID:-}" ]]; then
    echo "JUDGE_EXPERIMENT_ID must be set before sourcing setup_compass_verifier_judge.sh" >&2
    return 1
fi

judge_tasks_json="$(beaker experiment tasks "${JUDGE_EXPERIMENT_ID}" --format json)"
JUDGE_TASK_NAME="$(printf '%s\n' "${judge_tasks_json}" | extract_json_field 'data[0]["name"]')"
JUDGE_JOB_ID="$(printf '%s\n' "${judge_tasks_json}" | extract_json_field 'data[0]["jobs"][0]["id"]')"

beaker experiment await "${JUDGE_EXPERIMENT_ID}" "${JUDGE_TASK_NAME}" started --timeout 10m >/dev/null

judge_job_json="$(beaker job get "${JUDGE_JOB_ID}" --format json)"
JUDGE_HOSTNAME="$(printf '%s\n' "${judge_job_json}" | extract_json_field 'next(env["value"] for env in data[0]["execution"]["spec"]["envVars"] if env["name"] == "BEAKER_NODE_HOSTNAME")')"
export HOSTED_VLLM_API_BASE="http://${JUDGE_HOSTNAME}:${JUDGE_SERVER_PORT}/v1"
export JUDGE_BASE_URL="${HOSTED_VLLM_API_BASE}"

for _ in $(seq 1 90); do
    if curl -fsS "${HOSTED_VLLM_API_BASE}/models" >/dev/null 2>&1; then
        return 0
    fi
    sleep 2
done

echo "Judge server failed to become healthy at ${HOSTED_VLLM_API_BASE}" >&2
return 1
