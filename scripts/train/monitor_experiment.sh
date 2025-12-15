#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 <experiment_id> [--wait] [--logs]"
    echo ""
    echo "Monitor a Beaker experiment."
    echo ""
    echo "Arguments:"
    echo "  experiment_id    The Beaker experiment ID (e.g., 01KCHBGMH5NTX77PC17EZ5HJTB)"
    echo ""
    echo "Options:"
    echo "  --wait           Wait for the experiment to complete (polls every 30s)"
    echo "  --logs           Show the job logs after completion (or on failure)"
    echo ""
    echo "Examples:"
    echo "  $0 01KCHBGMH5NTX77PC17EZ5HJTB              # Check status once"
    echo "  $0 01KCHBGMH5NTX77PC17EZ5HJTB --wait       # Wait for completion"
    echo "  $0 01KCHBGMH5NTX77PC17EZ5HJTB --wait --logs # Wait and show logs"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

EXPERIMENT_ID="$1"
shift

WAIT=false
SHOW_LOGS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait)
            WAIT=true
            shift
            ;;
        --logs)
            SHOW_LOGS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

get_status() {
    beaker experiment get "$EXPERIMENT_ID" --format json 2>/dev/null | jq -r '.[0].jobs[0].status'
}

get_job_id() {
    beaker experiment get "$EXPERIMENT_ID" --format json 2>/dev/null | jq -r '.[0].jobs[0].id'
}

get_exit_code() {
    beaker experiment get "$EXPERIMENT_ID" --format json 2>/dev/null | jq -r '.[0].jobs[0].status.exitCode // "pending"'
}

print_status() {
    local status
    status=$(get_status)
    local exit_code
    exit_code=$(get_exit_code)

    echo "Experiment: $EXPERIMENT_ID"
    echo "URL: https://beaker.org/ex/$EXPERIMENT_ID"
    echo ""
    echo "Status:"
    echo "$status" | jq .
    echo ""

    if [[ "$exit_code" != "pending" && "$exit_code" != "null" ]]; then
        if [[ "$exit_code" == "0" ]]; then
            echo "Result: SUCCESS (exit code 0)"
        else
            echo "Result: FAILED (exit code $exit_code)"
        fi
    else
        echo "Result: RUNNING"
    fi
}

show_logs() {
    local job_id
    job_id=$(get_job_id)
    echo ""
    echo "=== Job Logs (last 100 lines) ==="
    beaker job logs "$job_id" 2>&1 | tail -100
}

if [[ "$WAIT" == "true" ]]; then
    echo "Waiting for experiment $EXPERIMENT_ID to complete..."
    echo "URL: https://beaker.org/ex/$EXPERIMENT_ID"
    echo ""

    while true; do
        exit_code=$(get_exit_code)

        if [[ "$exit_code" != "pending" && "$exit_code" != "null" ]]; then
            echo ""
            print_status

            if [[ "$SHOW_LOGS" == "true" ]]; then
                show_logs
            fi

            if [[ "$exit_code" == "0" ]]; then
                exit 0
            else
                exit 1
            fi
        fi

        echo -n "."
        sleep 30
    done
else
    print_status

    if [[ "$SHOW_LOGS" == "true" ]]; then
        show_logs
    fi
fi
