#!/bin/bash
# Ledger H175 (scimcq-b2-gpu): scripts/c12/h161_gpu_launch.sh at 92f6a55a2 (H167's launcher) with only these
# changes: MIXER (C12's four frozen parquets and weights unchanged, plus H175 P1's Nemotron-SFT-Science-v2 MCQ parquet
# at 1.0), the arm `scimcq` with S = round(phi x floor(I / 16)), phi = 16,303 / 28,092 (H167's ctrl), I the arm's packed
# instance count from its own cache (CACHE, I_ARM below), OUT_ROOT, run names, chrisg_ secrets and the timeout scaled
# to S. Workspace ai2/olmo-instruct as before. H161's text below is kept for the modes.
#
#   scripts/c12/h175_launch.sh scimcq <34521|34522> <smoke|resume|train>
#
# W&B off (no --with_tracking) by the owner's decision, 2026-11-02: chrisg_WANDB_API_KEY gets a 403 on
# ai2-llm/open_instruct_internal. Loss is read from the trainer's stdout in the Beaker logs (logging_steps 1).
#
# (H161's original header follows.)
#
#   scripts/c12/h161_gpu_launch.sh <hero|ctrl> <34521|34522> <smoke|resume|train>
#
#   hero   two full passes over the C12 cache: 2 x floor(I_C12 / 16) = 2 x 28,092 = 56,184 steps, --num_epochs 2.
#          OLMo-core's loader drops each epoch's partial last batch (449,487 = 16 x 28,092 + 15), so step 56,185
#          would open a third epoch; the trainer reshuffles epoch 2 with seed + epoch.
#   ctrl   H149's c12-b2 length S = 16,303 steps (phi = 0.5803 of one pass), --num_epochs 1.
#   seed   --data_loader_seed; --seed stays 33333 because it names the cache (bf82a8fab8-6068a350-rowaligned).
#   smoke  30 steps at the real size (2 x 8 GPUs), through scripts/c12/h161_smoke_wrap.py: every rank dumps its
#          step-1 batch, no final checkpoint; then replica 0 runs scripts/c12/h161_refce.py on GPU 0, the HF step-1
#          CE of that batch through H149's seeded B2 export (gate (1)). Timeout 1 h.
#   resume the auto-resume gate (G3): the smoke's 30-step schedule with autoResume, every rank logging a hash of every
#          batch (H161-BH), an ephemeral checkpoint at step 10, and a hold after step 19 (H161-HOLD) during which the
#          operator runs `beaker job preempt <either job>`. Beaker requeues both replicas; the same command finds
#          step10 in OUTPUT_DIR and continues at step 11, replaying steps 11-19. No step-1 dump, no HF reference.
#          MAX_RETRIES defaults to 1 here, so a hold that expires unpreempted (exit 3) still exercises the resume
#          path, through Beaker's retry instead of its preemption requeue. Timeout 2 h.
#   train  the run. Permanent checkpoint at the end of epoch 1 (hero) or half way (ctrl), plus the final one;
#          ephemeral every SAVE_EVERY steps (default 1000: ~30 min at H111's 1.80 s/step; set from the smoke's mean).
#
# Resuming: olmo_core_finetune.py calls trainer.maybe_load_checkpoint(OUTPUT_DIR) before anything else, so the same
# command run again continues from the latest complete stepNNNN in OUTPUT_DIR (weights, optimizer, step counter, LR
# position, data-loader epoch/batches_processed/seed, RNG); the base weights are loaded only when OUTPUT_DIR holds no
# checkpoint. OUTPUT_DIR has no timestamp, so a requeued or retried job lands in the same place.
#
# Env: IMAGE (default the H111 image, built from open-instruct@37a119ffd, whose open_instruct/ is identical to
# 97ac1b53a, the commit that built the C12 cache), PRIORITY (high), MIN_RUNTIME, TIMEOUT, SAVE_EVERY,
# AUTO_RESUME (default 1 for resume and train, 0 for smoke: Beaker requeues the task after a preemption),
# MAX_RETRIES (default 0, 1 for resume: Beaker reruns a task that failed, which then resumes the same way),
# RESUME_FROM=<other dir>/stepNNNN to start from a checkpoint outside OUTPUT_DIR (ignored once OUTPUT_DIR holds one),
# WAIT_FOR_BASE=<minutes> (default 0: the job waits that long for the base's RECEIPT.json, then
# exits 3 without training; for a smoke queued while the conversion job is still running), DRY_RUN=1.
#
# Before the first launch, the cache must resolve inside a Beaker job: setup_tokenizer_and_cache forces
# --local_cache_dir to the oe-adapt deletable cache when BEAKER_JOB_ID is set, so that dir holds a symlink
# numpy_sft/bf82a8fab8-6068a350-rowaligned -> the frozen C12 cache (made 2026-10-30). EXPECTED_NUMPY_CACHE makes any
# other resolution die before tokenizing or touching a GPU.
set -euo pipefail
ARM="${1:?hero or ctrl}"
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
# H175: plus P1's parquet under Amendment 1 (ledger experiments/h175/receipts/p1_receipt_a1.json) at 1.0.
SCIMCQ=/weka/olmo-3p5-post-checkpoints/chrisg/h175/data/out-a1/h175-scimcq-6a60033f.parquet  # Amendment 1; sha256 6a60033f4290f542
MIXER="$P/dolci-think-clean-v1-en-dbv-5bd68837-c12a3-9707a865.parquet 1.0 \
$P/nemotron-tool-oi-en-bd-825bdb83-c12a3-be9ec34a.parquet 0.862426 \
$P/nemotron-sci-v2-h083sel-bd-16f9aaaa-c12a3-25035bd6.parquet 3.03277 \
/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/if/nemotron-ifchat-v3-if-decontam-1fee8794-c12-4e2daf40.parquet 4.937456 \
$SCIMCQ 1.0"
# H175: the arm's own cache (built by scripts/c12/h175_cache.sh) and its packed-instance count.
CACHE=125b9c5edc-6068a350-rowaligned  # h175/cache-a1, built by 01M6DRPMR3BWYCZFWQGZPTJK3M; dataset_statistics.json sha256 c686c963
I_ARM=504942  # c12_pack.py on that cache; S = round(16303/28092 x 31558) = 18314
EPOCH_STEPS=$((I_ARM / 16))
# S = round(phi x floor(I / 16)), phi = 16,303 / 28,092 (H167's ctrl S over its batches per pass).
S_ARM=$(python3 -c "print(round(16303 / 28092 * $EPOCH_STEPS))")
OUT_ROOT=/weka/olmo-3p5-post-checkpoints/chrisg/h175-gpu

case "$ARM" in
    scimcq) STEPS=$S_ARM; EPOCHS=1; CKPT=$(( (S_ARM + 1) / 2 )); HOURS=$(( 12 * S_ARM / 16303 + 2 )) ;;
    *) echo "unknown arm $ARM (H175 has one arm: scimcq)" >&2; exit 1 ;;
esac
SAVE_EVERY="${SAVE_EVERY:-1000}"
RUN_NAME="h175gpu-${ARM}-s${SEED}"
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
    "synchronizedStartTimeout": "${SYNC_START_TIMEOUT:-10m}",
    "timeout": "$TIMEOUT",
}
spec = {"version": "v2", "budget": "ai2/oe-other",
        "description": "ledger H175 scimcq-b2-gpu: $RUN_NAME, $STEPS steps, 2x8 B300, seq 65536, lr 5e-5",
        "tasks": [task]}
if int("${MAX_RETRIES:-0}"):
    spec["retry"] = {"allowedTaskRetries": int("${MAX_RETRIES:-0}")}
json.dump(spec, open(sys.argv[1], "w"), indent=1)
EOF
if [[ "${DRY_RUN:-0}" == 1 ]]; then cat "$SPEC"; exit 0; fi
beaker experiment create "$SPEC" --workspace ai2/olmo-instruct --name "$RUN_NAME-$(date +%m%d%H%M)"
