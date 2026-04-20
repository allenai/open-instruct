#!/bin/bash
# Qwen3-4B-Base PPO with expected_accuracy conditioning. Thin wrapper over ppo_gt.sh.
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"
exec "$(dirname "$0")/qwen3_4b_base_ppo_gt.sh" "$BEAKER_IMAGE" expected_accuracy
