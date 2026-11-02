#!/bin/bash
# Ledger H170: build the arm's numpy SFT cache on one Beaker CPU job (0 GPUs) with the training image, then count its
# packed instances (I) the way olmo_core_finetune builds them (ledger experiments/c12/c12_pack.py), and run the
# tokenizer asserts on the saved tokenizer. The cache-hash arguments are h170_launch.sh's, verbatim (MIXER, tokenizer
# and revision, chat template, promoted <think>/</think>, row-aligned parts, max_seq_length 65536, --seed 33333);
# training checks the name through EXPECTED_NUMPY_CACHE. The cache is written under $H/cache (BEAKER_JOB_ID unset so
# --local_cache_dir is honoured); the training jobs reach it through a symlink in the deletable cache dir (H167's way).
#
#   scripts/c12/h170_cache.sh <ledger checkout>     (DRY_RUN=1 prints the spec)
set -euo pipefail
LEDGER="${1:?ledger checkout (for experiments/c12/c12_pack.py)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-01M5KX4ZAN00BJ00ARS7FNG79T}"
H=/weka/olmo-3p5-post-checkpoints/chrisg/h170
eval "$(sed -n '/^MODEL=/p;/^P=/p;/^SUP=/p;/^MIXER=/,/1.0"$/p' "$HERE/h170_launch.sh")"
[[ "$SUP" != __* ]] || { echo "set SUP in h170_launch.sh first" >&2; exit 1; }
PACK=$(gzip -9c "$LEDGER/experiments/c12/c12_pack.py" | base64 | tr -d '\n')
# The image (built at 37a119ffd) has no scripts/c12/, so the tokenizer asserts ship with the job.
TCHK=$(gzip -9c "$HERE/tokenizer_check.py" | base64 | tr -d '\n')
CMD="set -euo pipefail
echo H170-CACHE-START \$(date -u +%FT%TZ) nproc \$(nproc)
printf %s '$PACK' | base64 -d | gunzip > /tmp/c12_pack.py
printf %s '$TCHK' | base64 -d | gunzip > /tmp/tokenizer_check.py
sha256sum /tmp/c12_pack.py /tmp/tokenizer_check.py $SUP
mkdir -p $H/cache $H/hf-tok $H/pack-work
PYTHONPATH=. python /tmp/tokenizer_check.py config
env -u BEAKER_JOB_ID HF_DATASETS_CACHE=$H/hf-tok PYTHONUNBUFFERED=1 python open_instruct/olmo_core_finetune.py \
  --model_name_or_path $MODEL --config_name scripts/train/debug/kda_hero_small_sft.json \
  --tokenizer_name_or_path allenai/dolma2-tokenizer-olmo35 --tokenizer_revision 8b9717061fae09d5be814373d189919a62a9a00d \
  --chat_template_name olmo35 --reserved_slot_tokens '<think>' '</think>' \
  --row_aligned_parts --document_boundaries_from_metadata --max_seq_length 65536 \
  --mixer_list $MIXER --local_cache_dir $H/cache --seed 33333 --cache_dataset_only
NAME=\$(ls $H/cache/numpy_sft/); [ \$(echo \"\$NAME\" | wc -l) -eq 1 ] || { echo \"expected one cache dir: \$NAME\"; exit 3; }
echo H170-CACHE \$NAME
PYTHONPATH=. python /tmp/tokenizer_check.py dir $H/cache/numpy_sft/\$NAME/tokenizer
cp $H/cache/numpy_sft/\$NAME/dataset_statistics.json /output/ 2>/dev/null || find $H/cache/numpy_sft/\$NAME -maxdepth 2 -name dataset_statistics.json -exec cp {} /output/ \;
PYTHONPATH=. python /tmp/c12_pack.py $H/pack-work $H/cache/numpy_sft/\$NAME | tee /output/pack.log
echo H170-CACHE-DONE \$(date -u +%FT%TZ)"
SPEC=$(mktemp -t h170cache.XXXXXX)
CMD="$CMD" python3 - "$SPEC" <<PY
import json, os, sys
task = {"name": "h170-cache", "image": {"beaker": "$IMAGE"}, "command": ["/bin/bash", "-c"],
        "arguments": [os.environ["CMD"]],
        "envVars": [{"name": "HF_TOKEN", "secret": "chrisg_HF_TOKEN"},
                    {"name": "HF_HOME", "value": "$H/hf-home"}],
        "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in
                     ("oe-adapt-default", "oe-training-default", "olmo-3p5-checkpoints", "olmo-3p5-post-checkpoints")],
        "result": {"path": "/output"}, "resources": {"cpuCount": 96, "memory": "700GiB"},
        "context": {"priority": "high", "preemptible": True},
        "constraints": {"cluster": ["ai2/neptune", "ai2/saturn", "ai2/jupiter", "ai2/ceres", "ai2/titan"]},
        "timeout": "16h"}
json.dump({"version": "v2", "budget": "ai2/oe-other", "description": "ledger H170: arm cache build + packed-instance count",
           "tasks": [task]}, open(sys.argv[1], "w"), indent=1)
PY
if [[ "${DRY_RUN:-0}" == 1 ]]; then cat "$SPEC"; exit 0; fi
beaker experiment create "$SPEC" --workspace ai2/olmo-instruct --name "h170-cache-$(date +%m%d%H%M)"
