#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: MILES_BASE_IMAGE=local-runtime-tag ./scripts/train/build_image_and_launch.sh --miles SCRIPT [ARGS...]"
    exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit the current changes before building and launching a MILES experiment."
    exit 1
fi
: "${MILES_BASE_IMAGE:?Set MILES_BASE_IMAGE to the locally loaded base image recorded in runtime/miles/runtime.lock.json}"
commit=$(git rev-parse HEAD)
image_name="open-instruct-miles-core-${commit:0:12}"
python scripts/miles/build_image.py --base-image "$MILES_BASE_IMAGE" --tag "$image_name"
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
beaker image create "$image_name" -n "$image_name" -w "ai2/$beaker_user" \
    --description "Experimental MILES/Core integration; open-instruct commit $commit"
script="$1"
shift
bash "$script" "$beaker_user/$image_name" "$@"
