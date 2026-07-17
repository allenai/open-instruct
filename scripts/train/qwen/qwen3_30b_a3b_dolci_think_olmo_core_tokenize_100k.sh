#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_VARIANT=100k exec bash "${SCRIPT_DIR}/qwen3_30b_a3b_dolci_think_olmo_core_tokenize.sh" "$@"
