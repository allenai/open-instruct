#!/bin/bash
# Ledger H172 (ifchatv2-b2-gpu): H167's GPU c12-b2 recipe (this script's parent, scripts/c12/h161_gpu_launch.sh at
# h161-gpu-hero@92f6a55a2, arm ctrl) with one change, the DATA: C12's four frozen parquets at their frozen weights plus
# H172 P1's IF-Chat v2 parquet at 1.0. Changed from the parent, and nothing else: MIXER, CACHE and the steps (the phi
# rule on this cache's packed instances), OUT_ROOT, run names, the chrisg_ secrets and the spec description. Workspace
# ai2/olmo-instruct as before. --with_tracking is dropped: W&B off by the owner's decision 2026-11-02 (chrisg_WANDB_API_KEY
# gets a 403 on ai2-llm/open_instruct_internal); the trainer logs every step to stdout (--logging_steps 1).
#
#   scripts/c12/h172_launch.sh arm <34521|34522> <smoke|resume|train>
#
#   arm    S = round(phi x floor(I_ARM / 16)), phi = 16,303 / 28,092 (H167's control), --num_epochs 1, permanent
#          checkpoint half way, as the parent's ctrl.
#   seed, smoke, resume, train, Env and Resuming: as the parent (its header, unchanged below the H172 lines).
#
# Before the first launch the cache must resolve inside a Beaker job: --local_cache_dir is forced to the oe-adapt
# deletable cache when BEAKER_JOB_ID is set, so that dir holds a symlink numpy_sft/$CACHE -> the H172 cache built by
# scripts/c12/h172_cache.sh. EXPECTED_NUMPY_CACHE makes any other resolution die before tokenizing or touching a GPU.
set -euo pipefail
ARM="${1:?arm}"
SEED="${2:?data loader seed 34521 or 34522}"
MODE="${3:?smoke or train}"
case "$SEED" in 34521|34522) ;; *) echo "seed must be 34521 or 34522" >&2; exit 1 ;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"

IMAGE="${IMAGE:-01M5KX4ZAN00BJ00ARS7FNG79T}"
# B2's LM tensors as fp32 optimizer masters (scripts/c12/b2_lm_to_olmocore_dcp.py; RECEIPT.json one level up).
MODEL=/weka/olmo-3p5-post-checkpoints/bases/olmo35-small-mm-ckpt2-256k-amt/step22240-lm-olmocore/model_and_optim
# H149's seeded B2 HF export (G4), the step-1 CE reference.
HF_REF=/weka/olmo-3p5-post-checkpoints/abhishekr/tpu-posttrain/hf/olmo35-small-mm-ckpt2-256k-amt-step22240-base-h149-c12seed
# C12 as frozen (ledger experiments/c12/a7/freeze/freeze_receipt.json): paths and weights exactly as the cache build.
P=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/data/parquets-a3
# H172 P1's parquet (ledger experiments/h172, P1 receipt), at weight 1.0.
H172_PARQUET="${H172_PARQUET:-/weka/olmo-3p5-post-checkpoints/chrisg/h172/data/out/H172-PARQUET-UNSET}"
MIXER="$P/dolci-think-clean-v1-en-dbv-5bd68837-c12a3-9707a865.parquet 1.0 \
$P/nemotron-tool-oi-en-bd-825bdb83-c12a3-be9ec34a.parquet 0.862426 \
$P/nemotron-sci-v2-h083sel-bd-16f9aaaa-c12a3-25035bd6.parquet 3.03277 \
/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/if/nemotron-ifchat-v3-if-decontam-1fee8794-c12-4e2daf40.parquet 4.937456 \
$H172_PARQUET 1.0"
# The H172 cache (scripts/c12/h172_cache.sh) and its packed instance count (its H161-DUMP line / dataset statistics).
CACHE="${H172_CACHE:-H172-CACHE-UNSET}"
I_ARM="${H172_I_ARM:-0}"
EPOCH_STEPS=$((I_ARM / 16))
OUT_ROOT=/weka/olmo-3p5-post-checkpoints/chrisg/h172-gpu
case "$CACHE$H172_PARQUET" in *UNSET*) echo "set H172_CACHE, H172_I_ARM and H172_PARQUET" >&2; exit 1 ;; esac
[ "$I_ARM" -gt 449487 ] || { echo "H172_I_ARM must exceed C12's 449,487" >&2; exit 1; }

case "$ARM" in
    arm) STEPS=$(python3 -c "print(round(16303 / 28092 * $EPOCH_STEPS))"); EPOCHS=1; CKPT=$((STEPS / 2)); HOURS=12 ;;
    *) echo "unknown arm $ARM" >&2; exit 1 ;;
esac
SAVE_EVERY="${SAVE_EVERY:-1000}"
RUN_NAME="h172gpu-ifchatv2-s${SEED}"
case "$MODE" in
    smoke) STEPS=30; CKPT=1000000; SAVE_EVERY=1024; HOURS=1; RUN_NAME="${RUN_NAME}-smoke"
           MIN_RUNTIME="${MIN_RUNTIME:-10m}"; ENTRY=/tmp/h161_smoke_wrap.py; AUTO_RESUME="${AUTO_RESUME:-0}" ;;
    resume) STEPS=30; CKPT=1000000; SAVE_EVERY=10; HOLD_AT=19; HOURS=2; RUN_NAME="${RUN_NAME}-resume"
           MIN_RUNTIME="${MIN_RUNTIME:-5m}"; ENTRY=/tmp/h161_smoke_wrap.py; AUTO_RESUME="${AUTO_RESUME:-1}"
           MAX_RETRIES="${MAX_RETRIES:-1}" ;;
    train) MIN_RUNTIME="${MIN_RUNTIME:-1h}"; ENTRY=open_instruct/olmo_core_finetune.py; AUTO_RESUME="${AUTO_RESUME:-1}" ;;
    *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
TIMEOUT="${TIMEOUT:-${HOURS}h}"
OUTPUT_DIR="$OUT_ROOT/$RUN_NAME"
DUMP_DIR="$OUTPUT_DIR/step1"

ARGS="--model_name_or_path $MODEL --config_name scripts/train/debug/kda_hero_small_sft.json \
--tokenizer_name_or_path allenai/dolma2-tokenizer-olmo35 --tokenizer_revision 8b9717061fae09d5be814373d189919a62a9a00d \
--chat_template_name olmo35 --reserved_slot_tokens '<think>' '</think>' \
--row_aligned_parts --document_boundaries_from_metadata --max_seq_length 65536 \
--per_device_train_batch_size 1 --gradient_accumulation_steps 1 --learning_rate 5e-5 --warmup_ratio 0.03 \
--weight_decay 0.0 --max_grad_norm 1.0 --num_epochs $EPOCHS --max_train_steps $STEPS --attn_implementation flash_2 \
--no_compile_model --activation_checkpointing_mode budget --activation_memory_budget 1 \
--checkpointing_steps $CKPT --ephemeral_save_interval $SAVE_EVERY --keep_last_n_checkpoints 2 --dist_timeout_hours 2 \
--no_save_async --run_name $RUN_NAME --logging_steps 1 --mixer_list $MIXER \
--local_cache_dir /weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache \
--seed 33333 --data_loader_seed $SEED --output_dir $OUTPUT_DIR"
if [[ -n "${RESUME_FROM:-}" ]]; then ARGS="$ARGS --resume_from_checkpoint $RESUME_FROM"; fi

BASE_RECEIPT="$(dirname "$MODEL")/RECEIPT.json"
WAIT="for i in \$(seq 1 ${WAIT_FOR_BASE:-0}); do [ -f $BASE_RECEIPT ] && break; sleep 60; done
[ -f $BASE_RECEIPT ] || { echo H161-NO-BASE $BASE_RECEIPT; exit 3; }
sha256sum $BASE_RECEIPT"
TORCHRUN="torchrun --nnodes=2 --node_rank=\$BEAKER_REPLICA_RANK --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME \
--master_port=29400 --nproc_per_node=8 $ENTRY $ARGS"
if [[ "$MODE" == resume ]]; then
    WRAP=$(gzip -9c "$HERE/h161_smoke_wrap.py" | base64 | tr -d '\n')
    CMD="set -uo pipefail
printf %s '$WRAP' | base64 -d | gunzip > /tmp/h161_smoke_wrap.py
sha256sum /tmp/h161_smoke_wrap.py
echo H161-JOB-START \$(date -u +%FT%TZ) job \$BEAKER_JOB_ID replica \$BEAKER_REPLICA_RANK
ls -d $OUTPUT_DIR/step* 2>/dev/null
$WAIT
$TORCHRUN; RC=\$?
echo H161-TRAIN-EXIT \$RC \$(date -u +%FT%TZ)
exit \$RC"
    EXTRA_ENV='{"name": "H161_BATCH_HASH", "value": "1"}, {"name": "H161_HOLD_AT_STEP", "value": "'"$HOLD_AT"'"},
    {"name": "H161_NO_FINAL_SAVE", "value": "1"},'
elif [[ "$MODE" == smoke ]]; then
    WRAP=$(gzip -9c "$HERE/h161_smoke_wrap.py" | base64 | tr -d '\n')
    REFCE=$(gzip -9c "$HERE/h161_refce.py" | base64 | tr -d '\n')
    # Replica 0 runs the HF reference after training exits, on GPU 0, as H134's run_gpu.sh does on B300
    # (cu130 torch, a CUDA 13 ptxas for triton). A failed training still gets its reference if every rank dumped.
    # Exit: training's code if it failed; else 4 if the reference failed, 5 if a rank dump is missing.
    CMD="set -uo pipefail
printf %s '$WRAP' | base64 -d | gunzip > /tmp/h161_smoke_wrap.py
printf %s '$REFCE' | base64 -d | gunzip > /tmp/h161_refce.py
sha256sum /tmp/h161_smoke_wrap.py /tmp/h161_refce.py
echo H161-JOB-START \$(date -u +%FT%TZ)
$WAIT
echo H161-BASE-READY \$(date -u +%FT%TZ)
$TORCHRUN; RC=\$?
echo H161-TRAIN-EXIT \$RC \$(date -u +%FT%TZ)
if [ \"\$BEAKER_REPLICA_RANK\" = 0 ]; then
  N=\$(ls $DUMP_DIR/rank*.npz 2>/dev/null | wc -l); echo H161-DUMPS \$N
  if [ \"\$N\" = 16 ]; then
    G=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo GPU \$G
    command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl >/dev/null)
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH=\$HOME/.local/bin:\$PATH; }
    IDX=https://download.pytorch.org/whl/cu128
    if echo \"\$G\" | grep -q B300; then
      IDX=https://download.pytorch.org/whl/cu130; PX=''
      for c in /usr/local/cuda/bin/ptxas /usr/local/cuda-13*/bin/ptxas; do [ -x \"\$c\" ] && \"\$c\" --help 2>&1 | grep -q sm_103a && { PX=\$c; break; }; done
      if [ -z \"\$PX\" ]; then uv venv -q --python 3.12 /tmp/nvcc13 && uv pip install -q --python /tmp/nvcc13/bin/python 'nvidia-cuda-nvcc==13.0.*'; PX=\$(find /tmp/nvcc13 -name ptxas -type f | head -1); chmod +x \"\$PX\"; fi
      echo PTXAS \$PX; export TRITON_PTXAS_PATH=\$PX
    fi
    cd /tmp && CUDA_VISIBLE_DEVICES=0 uv run --no-project --python 3.12 --index \$IDX --index-strategy unsafe-best-match --with torch==2.9.1 --with transformers==4.57.6 --with numpy --with accelerate --with flash-linear-attention python /tmp/h161_refce.py $DUMP_DIR $HF_REF $DUMP_DIR/refce.json
    RR=\$?; echo H161-REFCE-EXIT \$RR
    [ \"\$RC\" = 0 ] && [ \"\$RR\" != 0 ] && RC=4
  else
    [ \"\$RC\" = 0 ] && RC=5
  fi
fi
exit \$RC"
    EXTRA_ENV='{"name": "H161_DUMP_DIR", "value": "'"$DUMP_DIR"'"}, {"name": "H161_NO_FINAL_SAVE", "value": "1"},'
else
    CMD="$WAIT
$TORCHRUN"
    EXTRA_ENV=""
fi

SPEC=$(mktemp -t h161spec.XXXXXX)
CMD="$CMD" python3 - "$SPEC" <<EOF
import json, os, sys
env = [
    {"name": "OLMO_SHARED_FS", "value": "1"},
    {"name": "EXPECTED_NUMPY_CACHE", "value": "$CACHE"},
    {"name": "HF_HOME", "value": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache/huggingface"},
    {"name": "HF_DATASETS_CACHE", "value": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache/huggingface"},
    {"name": "HF_HUB_CACHE", "value": "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache/hub"},
    {"name": "NCCL_DEBUG", "value": "ERROR"},
    {"name": "NCCL_SOCKET_IFNAME", "value": "ib"},
    {"name": "NCCL_IB_HCA", "value": "^=mlx5_bond_0"},
    $EXTRA_ENV
    {"name": "HF_TOKEN", "secret": "chrisg_HF_TOKEN"},
    {"name": "WANDB_API_KEY", "secret": "chrisg_WANDB_API_KEY"},
    {"name": "BEAKER_TOKEN", "secret": "chrisg_BEAKER_TOKEN"},
]
task = {
    "name": "$RUN_NAME",
    "image": {"beaker": "$IMAGE"},
    "command": ["/bin/bash", "-c"],
    "arguments": [os.environ["CMD"]],
    "envVars": env,
    "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in
                 ("oe-adapt-default", "oe-training-default", "olmo-3p5-checkpoints", "olmo-3p5-post-checkpoints")],
    "result": {"path": "/output"},
    "resources": {"gpuCount": 8, "sharedMemory": "10GiB"},
    "context": {"priority": "${PRIORITY:-high}", "minRuntime": "$MIN_RUNTIME", "autoResume": "$AUTO_RESUME" == "1"},
    "constraints": {"cluster": ["ai2/holmes"]},
    "replicas": 2,
    "leaderSelection": True,
    "hostNetworking": True,
    "propagateFailure": True,
    "propagatePreemption": True,
    "synchronizedStartTimeout": "10m",
    "timeout": "$TIMEOUT",
}
spec = {"version": "v2", "budget": "ai2/oe-other",
        "description": "ledger H172 ifchatv2-b2-gpu: $RUN_NAME, $STEPS steps, 2x8 B300, seq 65536, lr 5e-5",
        "tasks": [task]}
if int("${MAX_RETRIES:-0}"):
    spec["retry"] = {"allowedTaskRetries": int("${MAX_RETRIES:-0}")}
json.dump(spec, open(sys.argv[1], "w"), indent=1)
EOF
if [[ "${DRY_RUN:-0}" == 1 ]]; then cat "$SPEC"; exit 0; fi
beaker experiment create "$SPEC" --workspace ai2/olmo-instruct --name "$RUN_NAME-$(date +%m%d%H%M)"
