#!/bin/bash
# Ledger H172: build the arm's numpy SFT cache in one Beaker CPU job (0 GPUs, ai2/olmo-instruct), with the arguments
# that set the cache name exactly as the C12 cache build (ledger experiments/c12/a7/c12a7_cache.sh final) and
# scripts/c12/h172_launch.sh, the mixer being C12's four frozen parquets at their frozen weights plus H172 P1's parquet
# at 1.0. Then: the promoted-tokenizer check on the saved tokenizer, the packed-instance count as olmo_core_finetune
# builds it (ledger experiments/c12/c12_pack.py, shipped in the spec), and the symlink in the oe-adapt deletable cache
# that a Beaker training job resolves (h161_gpu_launch.sh's header). Prints H172-CACHE <name> and the PACK line.
#
#   H172_PARQUET=<path> H172_PARQUET_SHA=<sha256> [LEDGER=<ledger checkout>] [DRY_RUN=1] scripts/c12/h172_cache.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-01M5KX4ZAN00BJ00ARS7FNG79T}"
LEDGER="${LEDGER:-$HOME/Documents/src/olmo-post-training-ledger.worktrees/ifchat-v2-screen}"
: "${H172_PARQUET:?}" "${H172_PARQUET_SHA:?}"
MODEL=/weka/olmo-3p5-post-checkpoints/bases/olmo35-small-mm-ckpt2-256k-amt/step22240-lm-olmocore/model_and_optim
P=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/data/parquets-a3
IF=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/if/nemotron-ifchat-v3-if-decontam-1fee8794-c12-4e2daf40.parquet
MIXER="$P/dolci-think-clean-v1-en-dbv-5bd68837-c12a3-9707a865.parquet 1.0 $P/nemotron-tool-oi-en-bd-825bdb83-c12a3-be9ec34a.parquet 0.862426 $P/nemotron-sci-v2-h083sel-bd-16f9aaaa-c12a3-25035bd6.parquet 3.03277 $IF 4.937456 $H172_PARQUET 1.0"
OUT=/weka/olmo-3p5-post-checkpoints/chrisg/h172/cache
DEL=/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache/numpy_sft
PACK=$(gzip -9c "$LEDGER/experiments/c12/c12_pack.py" | base64 | tr -d '\n')
CMD="set -euo pipefail
printf %s '$PACK' | base64 -d | gunzip > /tmp/c12_pack.py
sha256sum /tmp/c12_pack.py
echo H172-CACHE-START \$(date -u +%FT%TZ) \$(nproc) CPUs
cat > /tmp/inputs.sha256 <<EOF
9707a865feae1ef7796861ef4b78d0c758b46a497541f3d54f72689c03a36c4f  $P/dolci-think-clean-v1-en-dbv-5bd68837-c12a3-9707a865.parquet
be9ec34a0ecb3140ddddabfe4bafd448ca2e224422d5c8bbd6c16e7c21e510c0  $P/nemotron-tool-oi-en-bd-825bdb83-c12a3-be9ec34a.parquet
25035bd6f7d061506f0ffcfacfe7c4d76dbd98e0960da3c5459e539cdfbe31d8  $P/nemotron-sci-v2-h083sel-bd-16f9aaaa-c12a3-25035bd6.parquet
4e2daf4057bf697fe87857bbb0bc576f39ca7825ee7676ba7566f4ece9d30f3a  $IF
$H172_PARQUET_SHA  $H172_PARQUET
EOF
sha256sum -c /tmp/inputs.sha256
python scripts/c12/tokenizer_check.py config
mkdir -p $OUT/tokscratch
if ! ls $OUT/numpy_sft/*/dataset_statistics.json >/dev/null 2>&1; then
env -u BEAKER_JOB_ID HF_DATASETS_CACHE=$OUT/tokscratch RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 python open_instruct/olmo_core_finetune.py \
  --model_name_or_path $MODEL --config_name scripts/train/debug/kda_hero_small_sft.json \
  --tokenizer_name_or_path allenai/dolma2-tokenizer-olmo35 --chat_template_name olmo35 \
  --tokenizer_revision 8b9717061fae09d5be814373d189919a62a9a00d --reserved_slot_tokens '<think>' '</think>' \
  --row_aligned_parts --document_boundaries_from_metadata --max_seq_length 65536 \
  --mixer_list $MIXER --local_cache_dir $OUT --seed 33333 --cache_dataset_only
fi
NAME=\$(ls $OUT/numpy_sft/); [ \$(echo \"\$NAME\" | wc -l) -eq 1 ] || { echo \"expected one cache dir: \$NAME\"; exit 3; }
echo H172-CACHE \$NAME
python scripts/c12/tokenizer_check.py dir $OUT/numpy_sft/\$NAME/tokenizer
cat $OUT/numpy_sft/\$NAME/dataset_statistics.json | head -c 20000
PYTHONPATH=. python /tmp/c12_pack.py $OUT/packwork $OUT/numpy_sft/\$NAME
if [ -e $DEL/\$NAME ]; then ls -la $DEL/\$NAME; else ln -s $OUT/numpy_sft/\$NAME $DEL/\$NAME && ls -la $DEL/\$NAME; fi
echo H172-CACHE-DONE \$(date -u +%FT%TZ)"
SPEC=$(mktemp -t h172cache.XXXXXX)
CMD="$CMD" python3 - "$SPEC" <<EOF
import json, os, sys
task = {
    "name": "h172-cache",
    "image": {"beaker": "$IMAGE"},
    "command": ["/bin/bash", "-c"],
    "arguments": [os.environ["CMD"]],
    "envVars": [
        {"name": "HF_HOME", "value": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache/huggingface"},
        {"name": "HF_HUB_CACHE", "value": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache/hub"},
        {"name": "HF_TOKEN", "secret": "chrisg_HF_TOKEN"},
    ],
    "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in
                 ("oe-adapt-default", "oe-training-default", "olmo-3p5-checkpoints", "olmo-3p5-post-checkpoints")],
    "resources": {"cpuCount": 120, "gpuCount": 0, "memory": "900GiB"},
    "context": {"priority": "${PRIORITY:-high}", "minRuntime": "${MIN_RUNTIME:-4h}"},
    "constraints": {"cluster": ["ai2/neptune", "ai2/saturn", "ai2/jupiter"]},
    "timeout": "${TIMEOUT:-12h}",
}
json.dump({"version": "v2", "budget": "ai2/oe-other", "description": "ledger H172 ifchatv2-b2-gpu: cache build (C12 + IF-Chat v2 at 1.0)",
           "tasks": [task]}, open(sys.argv[1], "w"), indent=1)
EOF
if [[ "${DRY_RUN:-0}" == 1 ]]; then cat "$SPEC"; exit 0; fi
beaker experiment create "$SPEC" --workspace ai2/olmo-instruct --name "h172-cache-$(date +%m%d%H%M)"
