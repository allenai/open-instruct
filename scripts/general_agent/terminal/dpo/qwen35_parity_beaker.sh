#!/bin/bash
# Rung 2 of the SafeComp DPO smoke ladder: 1-GPU load/logit-parity job on Beaker.
# Runs qwen35_parity_check.py over the SFT anchor (text config) and the base
# Qwen3.5-9B (VLM-wrapper config).
#
# NON-pure-docker mode: this host has no `docker buildx`, so instead of a custom image the
# job cds into this weka-hosted repo and uses its own `.venv` (weka is mounted on jupiter);
# the image supplies only CUDA/glibc. Invoke binaries via `.venv/bin/...` explicitly.
set -euo pipefail

BEAKER_IMAGE="${1:-nathanl/open_instruct_auto}"

uv run python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/oe-agents \
    --description "Qwen3.5-9B OLMo-core load/logit parity (SafeComp DPO rung 2)" \
    --priority urgent \
    --preemptible \
    --image "$BEAKER_IMAGE" \
    --no_auto_dataset_cache \
    --env OLMO_SHARED_FS=1 \
    --num_nodes 1 \
    --gpus 1 -- .venv/bin/python scripts/general_agent/terminal/dpo/qwen35_parity_check.py \
    --models /weka/oe-adapt-default/mingqianz/models/qwen35-9b-safecomp-v2/sft_qwen35_9b_safecomp_v2
# The gate covers the SFT anchor — the policy/reference model the DPO arms actually load.
# The base VLM-wrapper form is analyzed separately with qwen35_parity_diagnose.py: its larger
# bf16 deltas (mean |dlogprob| 0.16 vs 0.038) are under investigation as base-model numerical
# sensitivity rather than a conversion defect (the anchor shares every conversion code path).
