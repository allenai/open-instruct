#!/bin/bash

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 0 ]]; then
    beaker_image="$1"
    shift
else
    beaker_image="${BEAKER_USER}/open-instruct-integration-test"
fi

export GRPO_EXP_NAME="${GRPO_EXP_NAME:-weight_sync_ds3_init_only}"
export BEAKER_DESCRIPTION="${BEAKER_DESCRIPTION:-DS3 weight sync test: init native group only, no broadcasts}"

exec "${script_dir}/large_test_script_ds3.sh" \
    "$beaker_image" \
    --native_weight_sync_init_only \
    "$@"
