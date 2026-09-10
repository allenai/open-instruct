#!/usr/bin/env bash
set -euo pipefail

# Change MODEL_PATH and DATASET_PATH for the other machine. TASK is a real
# olmo-eval task (including its formatter/scorer), and DATASET_PATH points to
# the real local benchmark file used by that task.
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
DATASET_PATH="${DATASET_PATH:-/path/to/dataset.parquet}"
TASK="${TASK:-gsm8k:olmo3base}"

# This is the same provider boundary used by open_instruct/utils.py for local
# post-training evaluation. Set PROVIDER_KIND=hf only for an explicit comparison.
PROVIDER_KIND="${PROVIDER_KIND:-vllm_server}"
PYTHON="${PYTHON:-python}"
OLMO_EVAL="${OLMO_EVAL:-}"
OUTPUT_PATH="${OUTPUT_PATH:-olmo_eval_results}"
NUM_GPUS="${NUM_GPUS:-1}"
LIMIT="${LIMIT:-1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

if [[ "$MODEL_PATH" == /path/to/* || "$DATASET_PATH" == /path/to/* ]]; then
    echo "Edit MODEL_PATH and DATASET_PATH at the top of $0 first." >&2
    exit 2
fi

if [[ -z "$OLMO_EVAL" ]]; then
    PYTHON_PATH="$(command -v "$PYTHON" || true)"
    if [[ -n "$PYTHON_PATH" ]]; then
        OLMO_EVAL="$(dirname -- "$PYTHON_PATH")/olmo-eval"
    fi
    if [[ -z "$OLMO_EVAL" || ! -x "$OLMO_EVAL" ]]; then
        OLMO_EVAL="olmo-eval"
    fi
fi
if ! command -v "$OLMO_EVAL" >/dev/null 2>&1 && [[ ! -x "$OLMO_EVAL" ]]; then
    echo "Cannot find olmo-eval. Activate the environment containing oe-eval-internal, or set OLMO_EVAL=/path/to/olmo-eval." >&2
    exit 127
fi

# The evaluator starts a localhost OpenAI-compatible vLLM server. Do not let a
# machine-wide proxy intercept those requests.
for local_address in localhost 127.0.0.1 0.0.0.0; do
    case ",${NO_PROXY:-}," in
        *",${local_address},"*) ;;
        *) NO_PROXY="${NO_PROXY:+${NO_PROXY},}${local_address}" ;;
    esac
    case ",${no_proxy:-}," in
        *",${local_address},"*) ;;
        *) no_proxy="${no_proxy:+${no_proxy},}${local_address}" ;;
    esac
done
export NO_PROXY no_proxy

echo "MODEL_PATH=$MODEL_PATH"
echo "DATASET_PATH=$DATASET_PATH"
echo "TASK=$TASK"
echo "PROVIDER_KIND=$PROVIDER_KIND"
echo "OUTPUT_PATH=$OUTPUT_PATH"

exec "$OLMO_EVAL" run \
    --model "$MODEL_PATH" \
    --harness default \
    -o "provider.kind=$PROVIDER_KIND" \
    -o "provider.max_model_len=$MAX_MODEL_LEN" \
    --task "$TASK" \
    -o "data_source=$DATASET_PATH" \
    -o "limit=$LIMIT" \
    -o "max_tokens=$MAX_TOKENS" \
    -o "num_samples=$NUM_SAMPLES" \
    -o "temperature=$TEMPERATURE" \
    -o "top_p=$TOP_P" \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_PATH" \
    "$@"
