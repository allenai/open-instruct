#!/usr/bin/env bash
# Install olmo-eval-internal at the given git ref, plus a couple of pinned
# dependency upgrades the base image needs.
#
# Usage: install_olmo_eval.sh <git-ref>
#
# Env vars: GITHUB_TOKEN must be set (used to clone the private repo).
set -euo pipefail

ref="${1:?missing git ref}"

cache_dir=/weka/oe-eval-default/olmo-eval-pypi-cache

git clone "https://x-access-token:${GITHUB_TOKEN}@github.com/allenai/olmo-eval-internal.git" /opt/olmo-eval-internal
cd /opt/olmo-eval-internal
git checkout "$ref"

uv pip install --cache-dir "$cache_dir" -e '.[vllm]' 'transformers>=5.4.0' 'numpy<2.3'

cd /workspace
