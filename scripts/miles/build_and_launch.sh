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
if [[ -n "${MILES_EXISTING_IMAGE:-}" ]]; then
    if [[ ! "$MILES_EXISTING_IMAGE" =~ ^[0-9A-HJKMNP-TV-Z]{26}$ ]]; then
        echo "MILES_EXISTING_IMAGE must be an immutable Beaker image ID, not an alias."
        exit 1
    fi
    resolved=$(beaker image get "$MILES_EXISTING_IMAGE" --format json | python -c 'import json, sys; print(json.load(sys.stdin)[0]["id"])')
    if [[ "$resolved" != "$MILES_EXISTING_IMAGE" ]]; then
        echo "Existing image metadata differs from the explicitly requested ID."
        exit 1
    fi
    script="$1"
    shift
    exec bash "$script" "$MILES_EXISTING_IMAGE" "$@"
fi
: "${MILES_BASE_IMAGE:?Set MILES_BASE_IMAGE to the locally loaded base image recorded in runtime/miles/runtime.lock.json}"
commit=$(git rev-parse HEAD)
image_name="open-instruct-miles-core-${commit:0:12}"
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
expected_description="Experimental MILES/Core integration; open-instruct commit $commit"
existing=$(beaker image get "$beaker_user/$image_name" --format json 2>/dev/null || true)
if [[ -n "$existing" ]]; then
    description=$(jq -r '.[0].description // ""' <<< "$existing")
    if [[ "$description" != "$expected_description" ]]; then
        echo "Existing image metadata does not match this source commit: $image_name"
        exit 1
    fi
    echo "Reusing Beaker image for committed source $commit"
else
    python scripts/miles/build_image.py --base-image "$MILES_BASE_IMAGE" --tag "$image_name"
    beaker image create "$image_name" -n "$image_name" -w "ai2/$beaker_user" \
        --description "$expected_description"
fi
script="$1"
shift
bash "$script" "$beaker_user/$image_name" "$@"
