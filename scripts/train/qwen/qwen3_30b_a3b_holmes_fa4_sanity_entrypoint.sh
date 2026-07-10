#!/bin/bash
set -euo pipefail

uv pip install \
    --python /stage/.venv/bin/python \
    --prerelease=allow \
    "flash-attn-4[cu13]==4.0.0b21"

exec /stage/.venv/bin/python \
    /weka/oe-adapt-default/jacobm/olmoe3/post-training/open-instruct/scripts/train/qwen/qwen3_30b_a3b_holmes_fa4_sanity.py
