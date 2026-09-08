#!/bin/bash
# Terminal-Bench 2.1 / 2.0 and OpenThoughts-TBLite on an OLMoE3 KDA MoE HF checkpoint, on Beaker.
#
#   ./scripts/eval/terminal/olmoe3_kda_terminal_eval.sh <hf_dir> <run_name> [tb2.1|tb2.1-git|tb2.0|tblite|<name@version>]
#
#   ./scripts/eval/terminal/olmoe3_kda_terminal_eval.sh \
#       /weka/oe-adapt-default/allennlp/deletable_checkpoint_states/ecppxpon/hf_step604 kda-sft-simfc-tmax tb2.1
#
# Follows tmax's beaker_configs/launch_eval.sh (branch pd_sft_regen), which spins up vLLM,
# configures podman + harbor and runs `harbor run` with the Vanillux2Agent bash agent, then
# copies jobs/<name>/ to Weka. Two things differ, both because of this model's architecture:
#
# * The model is served through the scaling-ladders vLLM plugin stack rather than stock
#   `uvx vllm` (scripts/eval/olmoe3_kda_vllm_serve.sh: vllm 0.19.1, olmo-core at the branch
#   pin, plugins from Weka, qwen3_xml tool parser for the Olmo 3.5 template's XML tool calls,
#   olmo3 reasoning parser for its plain-text <think> tags). The Vanillux2Agent reads only
#   structured tool_calls from the response, so the tool parser is load-bearing.
# * The launch goes through mason rather than gantry, so this repo's runner is fetched from
#   GitHub at the launching commit (or read from the checkout when it lives on Weka), and the
#   tmax checkout is cloned inside the job at a pinned ref.
#
# Datasets: `tblite` and `tb2.0` are harbor registry ids; `tb2.1` is the Harbor Hub id
# terminal-bench/terminal-bench-2-1, as the reference tmax run (Beaker 01M125XZZKJCFJ7NHEBECVTGKA)
# used; `tb2.1-git` clones harbor-framework/terminal-bench-2-1 at a pinned commit instead.
#
# Protocol follows that reference run: tmax at pd_sft_podman (595caabf), Vanillux2Agent with its
# 64-step budget, 8 concurrent trials, 1 attempt, 65536 context, the tasks' own timeouts (no
# multipliers), 2 GPUs. The reference itself ended with 46 of 88 trials in error, 35 of them
# agent timeouts, at mean reward 0.191; timeouts are part of the protocol.
#
# Serving speed matters under time-limited tasks: on one eager-mode H100 an agent step averaged
# ~1 min at 8 concurrent trials, against 900 s per-task agent timeouts. The default is two GPUs
# as two vLLM replicas. HARBOR_AGENT_TIMEOUT_MULTIPLIER / HARBOR_AGENT_TIMEOUT_SEC are passed
# through for deliberate departures from the protocol. Task containers run under podman inside
# the job (BEAKER_ALLOW_SUBCONTAINERS) and pull from Docker Hub with the DOCKER_PAT secret.
#
# Not verified end to end at the time of writing: the serving stack on the tmax image
# (verified on the olmo-eval image), and the podman path in the ai2/open-instruct-dev
# workspace. The runner smoke-tests one tool call before handing over to harbor.
set -euo pipefail

HF_DIR="${1:?usage: $0 <hf_dir> <run_name> [tb2.1|tb2.1-git|tb2.0|tblite|<name@version>]}"
RUN_NAME="${2:?usage: $0 <hf_dir> <run_name> [tb2.1|tb2.1-git|tb2.0|tblite|<name@version>]}"
DATASET_CHOICE="${3:-tb2.1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PY:-uv run python}"
PRIORITY="${PRIORITY:-urgent}"
CLUSTER="${CLUSTER:-ai2/saturn}"
# Two GPUs as two replicas (vLLM data parallelism), matching the reference run's 2-GPU serving
# throughput without relying on the plugin's untested tensor parallelism.
GPUS="${GPUS:-2}"
DATA_PARALLEL="${DATA_PARALLEL:-$GPUS}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
IMAGE="${IMAGE:-hamishivi/tmax-eval-interactive}"
BEAKER_USER="${BEAKER_USER:-$(beaker account whoami --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["name"])')}"
# Docker Hub credentials for task image pulls: prefer the user's own secret, fall back to the
# one tmax's launcher uses. DOCKERHUB_USERNAME must match the PAT's account.
DOCKER_PAT_SECRET="${DOCKER_PAT_SECRET:-hamishivi_DOCKER_PAT}"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-hamishi740}"

# Workspace follows the cluster: jupiter and holmes go to ai2/olmo-instruct, everything else to
# ai2/open-instruct-dev. Override with WORKSPACE.
case "$CLUSTER" in
    *jupiter*|*holmes*) WORKSPACE="${WORKSPACE:-ai2/olmo-instruct}" ;;
    *) WORKSPACE="${WORKSPACE:-ai2/open-instruct-dev}" ;;
esac

DATASET_ENVS=()
case "$DATASET_CHOICE" in
    tb2.1)
        # Harbor Hub id, as the reference tmax run used (harbor resolves it without a login).
        DATASET_LABEL="terminal-bench/terminal-bench-2-1"; DATASET_ENVS+=(--env "DATASET=$DATASET_LABEL") ;;
    tb2.1-git)
        # The same tasks from the harbor-framework/terminal-bench-2-1 repo at a pinned commit,
        # passed to harbor as a local task directory; for reproducibility against a fixed ref.
        DATASET_LABEL="terminal-bench@2.1"
        DATASET_ENVS+=(--env "DATASET=$DATASET_LABEL"
                       --env "DATASET_GIT_URL=${TB21_GIT_URL:-https://github.com/harbor-framework/terminal-bench-2-1.git}"
                       --env "DATASET_GIT_REF=${TB21_GIT_REF:-7131e4375048a0e408a8fb404b5f499d726b695b}"
                       --env "DATASET_SUBDIR=tasks") ;;
    tb2.0)  DATASET_LABEL="terminal-bench@2.0";     DATASET_ENVS+=(--env "DATASET=$DATASET_LABEL") ;;
    tblite) DATASET_LABEL="openthoughts-tblite@2.0"; DATASET_ENVS+=(--env "DATASET=$DATASET_LABEL") ;;
    *@*)    DATASET_LABEL="$DATASET_CHOICE";         DATASET_ENVS+=(--env "DATASET=$DATASET_LABEL") ;;
    *) echo "unknown dataset '$DATASET_CHOICE' (expected tb2.1, tb2.1-git, tb2.0, tblite, or a harbor name@version)" >&2; exit 1 ;;
esac
DATASET_SLUG="${DATASET_LABEL//[^A-Za-z0-9]/-}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}-${DATASET_SLUG}}"
RESULTS_DIR="${RESULTS_DIR:-/weka/oe-adapt-default/$BEAKER_USER/tmax-eval/$JOB_NAME}"

# The image has no copy of this repo. Read the runner and the serving library from the checkout
# when it is on Weka (visible inside the job), otherwise from GitHub at this exact commit, which
# therefore has to be pushed.
RUNNER_REL=scripts/eval/terminal/run_terminal_eval_in_job.sh
SERVE_REL=scripts/eval/olmoe3_kda_vllm_serve.sh
if [[ "$REPO_ROOT" == /weka/* ]]; then
    FETCH="RUNNER=$REPO_ROOT/$RUNNER_REL; SERVE_LIB=$REPO_ROOT/$SERVE_REL"
else
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- scripts/eval)" ]]; then
        echo "scripts/eval has uncommitted changes; commit and push so the job can fetch them" >&2; exit 1
    fi
    COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    git -C "$REPO_ROOT" branch -r --contains "$COMMIT" | grep -q origin/ || { echo "commit $COMMIT is not pushed to origin" >&2; exit 1; }
    RAW="https://raw.githubusercontent.com/allenai/open-instruct/$COMMIT"
    FETCH="mkdir -p /opt/oi-eval && curl -sfL $RAW/$RUNNER_REL -o /opt/oi-eval/run.sh && curl -sfL $RAW/$SERVE_REL -o /opt/oi-eval/serve.sh && RUNNER=/opt/oi-eval/run.sh; SERVE_LIB=/opt/oi-eval/serve.sh"
fi

echo "terminal eval: $HF_DIR | dataset $DATASET_LABEL | job $JOB_NAME | $CLUSTER x$GPUS | workspace $WORKSPACE | results -> $RESULTS_DIR"
# mason joins the words after -- with spaces and runs them under `bash -c`, so the job command is
# passed as a single word; $RUNNER / $SERVE_LIB expand inside the job.
$PY mason.py \
    --cluster "$CLUSTER" \
    --workspace "$WORKSPACE" --priority "$PRIORITY" \
    --image "$IMAGE" --pure_docker_mode \
    --description "Harbor eval ($DATASET_LABEL) of $RUN_NAME via vLLM (OLMoE3 KDA)" \
    --timeout "${JOB_TIMEOUT:-24h}" \
    --num_nodes 1 --gpus "$GPUS" --non_resumable --no_auto_dataset_cache \
    --env BEAKER_ALLOW_SUBCONTAINERS=1 --env BEAKER_SKIP_DOCKER_SOCKET=1 \
    --secret "DOCKER_PAT=$DOCKER_PAT_SECRET" --env "DOCKERHUB_USERNAME=$DOCKERHUB_USERNAME" \
    --env "CKPT=$HF_DIR" --env "SERVED_MODEL_NAME=$RUN_NAME" --env "JOB_NAME=$JOB_NAME" \
    --env "RESULTS_DIR=$RESULTS_DIR" \
    "${DATASET_ENVS[@]}" \
    --env "N_CONCURRENT=${N_CONCURRENT:-8}" --env "N_ATTEMPTS=${N_ATTEMPTS:-1}" --env "N_TASKS=${N_TASKS:-}" \
    --env "AGENT_IMPORT_PATH=${AGENT_IMPORT_PATH:-Vanillux2Agent:Vanillux2Agent}" \
    --env "TENSOR_PARALLEL=$TENSOR_PARALLEL" --env "DATA_PARALLEL=$DATA_PARALLEL" --env "MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}" \
    --env "MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-16384}" \
    --env "TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_xml}" --env "REASONING_PARSER=${REASONING_PARSER:-olmo3}" \
    --env "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}" \
    --env "TMAX_GIT_URL=${TMAX_GIT_URL:-https://github.com/shatu/tmax.git}" \
    --env "TMAX_GIT_REF=${TMAX_GIT_REF:-595caabfeaec}" \
    --env "PLUGIN_DIR=${PLUGIN_DIR:-/weka/oe-adapt-default/abhishekr/repos/scaling-ladders-emo/ladders/olmoe3}" \
    --env "HARBOR_AGENT_TIMEOUT_SEC=${HARBOR_AGENT_TIMEOUT_SEC:-}" --env "HARBOR_TIMEOUT_MULTIPLIER=${HARBOR_TIMEOUT_MULTIPLIER:-}" \
    --env "HARBOR_AGENT_TIMEOUT_MULTIPLIER=${HARBOR_AGENT_TIMEOUT_MULTIPLIER:-}" --env "HARBOR_VERIFIER_TIMEOUT_MULTIPLIER=${HARBOR_VERIFIER_TIMEOUT_MULTIPLIER:-}" \
    -- "$FETCH && SERVE_LIB=\$SERVE_LIB bash \$RUNNER"
