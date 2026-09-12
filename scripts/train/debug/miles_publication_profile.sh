#!/bin/bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
exec python scripts/miles/launch_publication_profile.py "$@"
