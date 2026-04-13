#!/bin/bash

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
delay_steps="${DELAY_NATIVE_WEIGHT_SYNC_INIT_STEPS:-3}"

if [[ $# -gt 0 ]]; then
    beaker_image="$1"
    shift
else
    beaker_image="${BEAKER_USER}/open-instruct-integration-test"
fi

export GRPO_EXP_NAME="${GRPO_EXP_NAME:-weight_sync_ds3_delay_init_${delay_steps}_steps}"
export BEAKER_DESCRIPTION="${BEAKER_DESCRIPTION:-DS3 weight sync test: delay native init until after ${delay_steps} steps}"

exec "${script_dir}/large_test_script_ds3.sh" \
    "$beaker_image" \
    --delay_native_weight_sync_init_steps "$delay_steps" \
    "$@"
