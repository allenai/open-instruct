#!/bin/bash
# A4B (Hero Small) GPU SFT throughput ladder: H203's spec (Beaker 01M6YJ908XJG6SF0DYVW495AQP, open-instruct a4b-ep@c7c4de66,
# image 01M6XV36F2WWV7GZ9B7X7K3PB2) cut to a short run, with one change per rung.
#
#   scripts/c12/speed/launch.sh <tag>
#
# Env:
#   STEPS (40)          --max_train_steps; the LR schedule stays H203's 18,410 steps (SPEED_SCHED_STEPS).
#   NNODES (2)          8 B300 each.
#   EP (8)              --expert_parallel_degree.
#   COMPILE (0)         1 drops --no_compile_model.
#   CFG_PATCH ('')      JSON overrides on scripts/c12/a4b_sft_config.json (see speed_wrap.py).
#   PROFILE ('')        "<skip>,<active>,<ranks>" torch profiler, e.g. "41,2,0;8".
#   EXTRA ('')          more finetune flags.
#   EXTRA_ENV_JSON ('') extra env entries, JSON list of {"name","value"}.
#   DRY_RUN=1           print the spec only.
# Output (scratch, no checkpoints): /weka/olmo-3p5-post-checkpoints/kevinfarhat/a4b-sft-speed/<tag>-<stamp>.
set -euo pipefail
TAG="${1:?tag}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STEPS="${STEPS:-40}"
NNODES="${NNODES:-2}"
EP="${EP:-8}"
COMPILE="${COMPILE:-0}"
STAMP=$(date +%m%d%H%M)
NAME="kf-a4b-sft-speed-${TAG}"
OUT="/weka/olmo-3p5-post-checkpoints/kevinfarhat/a4b-sft-speed/${TAG}-${STAMP}"
IMAGE=01M6XV36F2WWV7GZ9B7X7K3PB2
MODEL=/weka/olmo-3p5-post-checkpoints/bases/olmo35-4b-active-4t-decay159k-causal/step162974/model_and_optim
P=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/data/parquets-a3
MIXER="$P/dolci-think-clean-v1-en-dbv-5bd68837-c12a3-9707a865.parquet 1.0 \
$P/nemotron-tool-oi-en-bd-825bdb83-c12a3-be9ec34a.parquet 0.862426 \
$P/nemotron-sci-v2-h083sel-bd-16f9aaaa-c12a3-25035bd6.parquet 3.03277 \
/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/if/nemotron-ifchat-v3-if-decontam-1fee8794-c12-4e2daf40.parquet 4.937456"
COMPILE_FLAG="--no_compile_model"
[[ "$COMPILE" == 1 ]] && COMPILE_FLAG=""
# MODEL_KIND=7b: OLMo 3 7B dense (H089's GPU reference leg 01M5ASJZ1S4E88BC1DE3Y1D3PA: HF base, olmo3_7B, YaRN x8,
# compile, selected_modules AC) on the same C12r data and batch shape; GA keeps 16 instances per step.
MODEL_KIND="${MODEL_KIND:-a4b}"
GA=$(( NNODES >= 2 ? 1 : 2 ))
MODEL_ARGS="--model_name_or_path $MODEL --config_name scripts/c12/a4b_sft_config.json"
AC_ARGS="--activation_checkpointing_mode budget --activation_memory_budget 1"
TAIL_ARGS="--expert_parallel_degree $EP --loss_implementation fused_linear"
if [[ "$MODEL_KIND" == 7b ]]; then
  GA=$(( NNODES >= 2 ? 1 : 2 ))
  MODEL_ARGS="--model_name_or_path /weka/olmo-3p5-post-checkpoints/bases/olmo3-7b-1025/hf-a81bae42 --config_name olmo3_7B --rope_scaling_factor 8"
  AC_ARGS="${AC_ARGS_7B:---activation_checkpointing_mode selected_modules}"
  TAIL_ARGS="${TAIL_7B:-}"
fi

ARGS="$MODEL_ARGS \
--tokenizer_name_or_path allenai/dolma2-tokenizer-olmo35 --tokenizer_revision 8b9717061fae09d5be814373d189919a62a9a00d \
--chat_template_name olmo35 --reserved_slot_tokens '<think>' '</think>' \
--row_aligned_parts --document_boundaries_from_metadata --max_seq_length 65536 \
--per_device_train_batch_size 1 --gradient_accumulation_steps $GA --learning_rate 7.3e-5 --warmup_ratio 0.03 \
--weight_decay 0.0 --max_grad_norm 1.0 --num_epochs 1 --max_train_steps $STEPS --attn_implementation flash_2 \
$COMPILE_FLAG $AC_ARGS \
--checkpointing_steps 1000000 --ephemeral_save_interval 999999 --keep_last_n_checkpoints 1 --dist_timeout_hours 1 \
--no_save_async --run_name $NAME --logging_steps 1 --mixer_list $MIXER \
--local_cache_dir /weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache \
--seed 33333 --data_loader_seed 34521 --output_dir $OUT \
--extra_numpy_dirs /weka/olmo-3p5-post-checkpoints/abhishekr/h182/replay-b1c12r \
$TAIL_ARGS ${EXTRA:-}"

SRC_SHA=$(git -C "$HERE" rev-parse HEAD)
RDZV="--node_rank=\$BEAKER_REPLICA_RANK --master_addr=\$BEAKER_LEADER_REPLICA_HOSTNAME"
[[ "$NNODES" == 1 ]] && RDZV=""
PAYLOAD="speed_wrap.py trace_summary.py"
if [[ -n "${OC_PATCH:-}" ]]; then cp "$OC_PATCH" "$HERE/oc.patch"; PAYLOAD="$PAYLOAD oc.patch"; fi
WRAP=$(COPYFILE_DISABLE=1 tar --no-xattrs -C "$HERE" -czf - $PAYLOAD | base64 | tr -d '\n')
CMD="set -uo pipefail
mkdir -p /tmp/speed && printf %s '$WRAP' | base64 -d | tar -C /tmp/speed -xzf -
sha256sum /tmp/speed/*.py
if [ \"${RUNTIME_ENV:-}\" = cu12 ]; then
  T0=\$(date +%s); mkdir -p /tmp/proj && cd /tmp/proj
  for f in pyproject.toml uv.lock; do curl -sSfL -o \$f https://raw.githubusercontent.com/allenai/open-instruct/$SRC_SHA/\$f || exit 7; done
  UV_PROJECT_ENVIRONMENT=/tmp/venv12 uv sync --frozen --no-default-groups --group cuda12 --no-install-project > /tmp/uv_sync.log 2>&1 || { tail -50 /tmp/uv_sync.log; exit 8; }
  tail -3 /tmp/uv_sync.log; cd /stage; export PATH=/tmp/venv12/bin:\$PATH; export VIRTUAL_ENV=/tmp/venv12
  echo SPEED-CU12-VENV ready in \$((\$(date +%s)-T0))s; python -c 'import torch; print(\"torch\", torch.__version__, torch.version.cuda, torch.cuda.is_available())'
fi
if [ \"${RUNTIME_ENV:-}\" = compat ]; then
  ls /usr/local/cuda/compat 2>&1 | head; CV=\$(ls -d /usr/local/cuda-* | head -1 | sed 's/.*cuda-//; s/\\./-/'); echo cuda \$CV
  if ! ls /usr/local/cuda/compat/libcuda.so* >/dev/null 2>&1; then (apt-get update -qq && apt-get install -y -qq cuda-compat-\$CV) > /tmp/apt.log 2>&1 || tail -20 /tmp/apt.log; fi
  export LD_LIBRARY_PATH=/usr/local/cuda/compat:\${LD_LIBRARY_PATH:-}; ls /usr/local/cuda/compat
  python -c 'import torch; print(\"torch\", torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())' || exit 9
fi
if [ -n \"${OC_PATCH:-}\" ]; then
  SP=\$(python -c 'import olmo_core, os; print(os.path.dirname(os.path.dirname(olmo_core.__file__)))'); echo olmo_core site \$SP
  (cd \$SP && (patch -p2 --forward < /tmp/speed/oc.patch || git apply -p2 --verbose /tmp/speed/oc.patch)) || exit 10
  echo SPEED-OC-PATCHED \$(sha256sum /tmp/speed/oc.patch | cut -c1-16)
fi
echo SPEED-JOB-START \$(date -u +%FT%TZ) job \$BEAKER_JOB_ID replica \${BEAKER_REPLICA_RANK:-0} host \$(hostname) pwd \$(pwd)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
torchrun --nnodes=$NNODES $RDZV \
--master_port=29400 --nproc_per_node=8 /tmp/speed/speed_wrap.py $ARGS; RC=\$?
echo SPEED-TRAIN-EXIT \$RC \$(date -u +%FT%TZ)
exit \$RC"

SPEC=$(mktemp -t speedspec.XXXXXX)
CMD="$CMD" SYNC_TIMEOUT="${SYNC_TIMEOUT:-10m}" MIN_RUNTIME="${MIN_RUNTIME:-2h}" CLUSTER="${CLUSTER:-ai2/holmes}" PRIORITY="${PRIORITY:-urgent}" NAME="$NAME" CFG_PATCH="${CFG_PATCH:-}" PROFILE="${PROFILE:-}" NNODES="$NNODES" IMAGE="$IMAGE" \
EXTRA_ENV_JSON="${EXTRA_ENV_JSON:-[]}" STEPS="$STEPS" OUT="$OUT" python3 - "$SPEC" <<'EOF'
import json, os, sys
e = os.environ
env = [
    {"name": "OLMO_SHARED_FS", "value": "1"},
    {"name": "EXPECTED_NUMPY_CACHE", "value": "bf82a8fab8-6068a350-rowaligned"},
    {"name": "HF_HOME", "value": "/tmp/hf_cache/huggingface"},
    {"name": "HF_DATASETS_CACHE", "value": "/tmp/hf_cache/huggingface"},
    {"name": "HF_HUB_CACHE", "value": "/tmp/hf_cache/hub"},
    {"name": "NCCL_DEBUG", "value": "ERROR"},
    {"name": "NCCL_SOCKET_IFNAME", "value": "ib"},
    {"name": "NCCL_IB_HCA", "value": "^=mlx5_bond_0"},
    {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
    {"name": "SPEED_SCHED_STEPS", "value": "18410"},
    {"name": "SPEED_NO_FINAL_SAVE", "value": "1"},
    {"name": "SPEED_MEM", "value": "1"},
    {"name": "HF_TOKEN", "secret": "kevinfarhat_HF_TOKEN_fg"},
    {"name": "WANDB_API_KEY", "secret": "kevinfarhat_WANDB_API_KEY"},
    {"name": "BEAKER_TOKEN", "secret": "kevinfarhat_BEAKER_TOKEN"},
]
if e["CFG_PATCH"]:
    json.loads(e["CFG_PATCH"])
    env.append({"name": "SPEED_CFG_PATCH", "value": e["CFG_PATCH"]})
if e["PROFILE"]:
    env.append({"name": "SPEED_PROFILE", "value": e["PROFILE"]})
env += json.loads(e["EXTRA_ENV_JSON"])
task = {
    "name": e["NAME"],
    "image": {"beaker": e["IMAGE"]},
    "command": ["/bin/bash", "-c"],
    "arguments": [e["CMD"]],
    "envVars": env,
    "datasets": [{"mountPath": f"/weka/{b}", "source": {"weka": b}} for b in
                 ("oe-adapt-default", "oe-training-default", "olmo-3p5-checkpoints", "olmo-3p5-post-checkpoints")],
    "result": {"path": "/output"},
    "resources": {"gpuCount": 8, "sharedMemory": "10GiB"},
    "context": {"priority": e["PRIORITY"], "minRuntime": e["MIN_RUNTIME"]},
    "constraints": {"cluster": [e["CLUSTER"]]},
    "replicas": int(e["NNODES"]),
    "leaderSelection": True,
    "hostNetworking": True,
    "propagateFailure": True,
    "propagatePreemption": True,
    "synchronizedStartTimeout": e.get("SYNC_TIMEOUT", "10m"),
    "timeout": "2h",
}
spec = {"version": "v2", "budget": "ai2/oe-other",
        "description": f"A4B SFT speed ladder {e['NAME']}: H203 recipe, {e['STEPS']} steps, {e["NNODES"]}x8 {e["CLUSTER"]}, "
                       f"out {e['OUT']}, patch {e['CFG_PATCH'] or '-'}",
        "tasks": [task]}
if int(e["NNODES"]) == 1:
    for k in ("replicas", "leaderSelection", "propagateFailure", "propagatePreemption", "synchronizedStartTimeout"):
        task.pop(k)
json.dump(spec, open(sys.argv[1], "w"), indent=1)
EOF
if [[ "${DRY_RUN:-0}" == 1 ]]; then cat "$SPEC"; exit 0; fi
beaker experiment create "$SPEC" --workspace ai2/olmo-instruct --name "${NAME}-${STAMP}"
