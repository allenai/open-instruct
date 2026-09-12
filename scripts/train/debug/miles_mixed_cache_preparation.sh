#!/bin/bash
set -euo pipefail
python -m scripts.miles.launch_judge_preparation "${1:?Image required}" configs/miles/qualification/mixed-cache-ab-20260912-off.toml --stage prepare --prepare-module scripts.miles.prepare_mixed_cache_exercise "${@:2}"
