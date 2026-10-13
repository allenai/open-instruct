#!/bin/bash
set -euo pipefail
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Commit changes before launching an upstream MILES experiment." >&2
    exit 1
fi
exec python -m scripts.miles_upstream.cli run "$@"
