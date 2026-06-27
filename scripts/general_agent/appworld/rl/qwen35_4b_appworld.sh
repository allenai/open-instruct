#!/bin/bash
# AppWorld code-as-action RL with Qwen 3.5 4B.
#
# The policy solves AppWorld tasks by writing Python through a single
# `execute_python` tool (AppWorldEnv, config_name "appworld"). Reward is
# AppWorld's own evaluation (fraction of unit tests passed) emitted by the env;
# the dataset uses the `passthrough` verifier so no extra reward is added.
#
# Execution model (mirrors the swerl podman workflow): each rollout gets its own
# AppWorld container (ghcr.io/stonybrooknlp/appworld:latest, pydantic-1) running
# `appworld serve environment`; AppWorldEnv is a pydantic-2-clean HTTP client. The
# trainer therefore needs a docker/podman socket (as swerl does).
#
# Prerequisites:
#   1. Build the RL dataset once with scripts/data/convert_appworld_to_rl.py and
#      push it to HF (set APPWORLD_DATASET below).
#   2. The AppWorld task data must be reachable by the *docker daemon* of each node.
#      Two options:
#        a) stage the data root on weka and bind-mount it (set APPWORLD_ROOT); or
#        b) bake the data into the image and set APPWORLD_ROOT="" (no mount). See
#           scripts/general_agent/appworld/README.md.
#
# Launch via Beaker:
#   ./scripts/train/build_image_and_launch.sh scripts/general_agent/appworld/rl/qwen35_4b_appworld.sh

EXP_NAME="${EXP_NAME:-appworld_rl_qwen35_4b}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

MODEL_NAME_OR_PATH="Qwen/Qwen3.5-4B"
BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

# RL dataset built by scripts/data/convert_appworld_to_rl.py (one row per task).
APPWORLD_DATASET="${APPWORLD_DATASET:-rl-research/appworld-train-rl}"
DATASETS="${APPWORLD_DATASET} 1.0"
DATASET_SPLITS="train"

# AppWorld container image (per-rollout). Use the official image with a bind-mounted
# data root, or a data-baked derivative (then set APPWORLD_ROOT="").
APPWORLD_IMAGE="${APPWORLD_IMAGE:-ghcr.io/stonybrooknlp/appworld:latest}"

# AppWorld data root visible to the docker daemon (contains data/ and experiments/outputs/).
# Set to "" when using a data-baked image.
APPWORLD_ROOT="${APPWORLD_ROOT:-/weka/oe-adapt-default/shashankg/appworld_root}"

# Per-rollout interaction budget; keep env max_interactions >= max_steps.
MAX_STEPS="${MAX_STEPS:-40}"

PRIORITY="${PRIORITY:-high}"

# Checkpoint state lives on a persistent weka path keyed on EXP_NAME (NOT the
# timestamped RUN_NAME) so re-running this script resumes the same run.
CHECKPOINT_STATE_DIR="${CHECKPOINT_STATE_DIR:-/weka/oe-adapt-default/allennlp/deletable_checkpoint_states/${EXP_NAME}}"

uv run mason.py \
    --task_name ${EXP_NAME} \
    --description "${RUN_NAME}" \
    --cluster "ai2/jupiter" \
    --workspace ai2/general-tool-use \
    --priority ${PRIORITY} \
    --pure_docker_mode \
    --image ${BEAKER_IMAGE} \
    --preemptible \
    --num_nodes 2 \
    --max_retries 5 \
    --gpus 8 \
    --budget ai2/oe-omai \
    --no_auto_dataset_cache \
    --env GIT_COMMIT="$(git rev-parse --short HEAD)" \
    --env APPWORLD_ROOT="${APPWORLD_ROOT}" \
    -- \
source configs/beaker_configs/ray_node_setup.sh \
\&\& uv run open_instruct/grpo_fast.py \
    --run_name "${RUN_NAME}" \
    --exp_name "${EXP_NAME}" \
    --beta 0.001 \
    --load_ref_policy True \
    --async_steps 4 \
    --active_sampling \
    --inflight_updates \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits $DATASET_SPLITS \
    --dataset_mixer_eval_list ${APPWORLD_DATASET} 8 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 4096 \
    --response_length 16384 \
    --pack_length 20480 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --non_stop_penalty False \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 51200 \
    --deepspeed_stage 3 \
    --num_learners_per_node 8 \
    --vllm_num_engines 8 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --tool_parser_type vllm_qwen3_xml \
    --tools appworld \
    --tool_call_names execute_python \
    --tool_configs "{\"image\": \"${APPWORLD_IMAGE}\", \"data_root\": \"${APPWORLD_ROOT}\", \"max_interactions\": ${MAX_STEPS}, \"dense_reward\": true}" \
    --pool_size 128 \
    --max_steps ${MAX_STEPS} \
    --backend_timeout 1800 \
    --save_traces \
    --seed 1 \
    --local_eval_every 100 \
    --save_freq 100 \
    --output_dir /output \
    --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
    --checkpoint_state_freq 100 \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --wandb_project oe-general-agents \
    --vllm_enable_prefix_caching \
    --keep_last_n_checkpoints -1 \
    --kl_estimator 3 \
    --push_to_hub False
