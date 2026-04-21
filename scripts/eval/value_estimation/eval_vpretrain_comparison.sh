#!/bin/bash
# Evaluate and compare value pretrain runs.
#
# Usage:
#   eval_vpretrain_comparison.sh \
#       --dataset  /path/to/pairs.parquet \
#       --output   /path/to/eval_outputs \
#       --run  name1:/path/to/step_N \
#       --run  name2:/path/to/step_N \
#       ...
#
# Each --run argument is  NAME:STEP_DIR  where STEP_DIR is the HF checkpoint
# directory produced by save_model (contains config.json, tokenizer files,
# and value_model/value_model.bin).  training_args.json in STEP_DIR is used
# to auto-detect conditioning so scoring matches training.
#
# If --dataset is omitted the script looks for
#   ./value_estimation_data/dapo_math_100pairs.parquet
set -euo pipefail

DATASET="./value_estimation_data/dapo_math_100pairs.parquet"
OUTPUT_DIR=""
declare -a RUNS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --output)  OUTPUT_DIR="$2"; shift 2 ;;
        --run)     RUNS+=("$2"); shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --output is required" >&2; exit 1
fi
if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "ERROR: at least one --run NAME:STEP_DIR is required" >&2; exit 1
fi
if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: dataset not found at $DATASET" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR"

declare -a SCORE_PATHS=()

for run_spec in "${RUNS[@]}"; do
    run_name="${run_spec%%:*}"
    step_dir="${run_spec#*:}"

    echo "=== $run_name ==="

    value_bin="${step_dir}/value_model/value_model.bin"
    if [[ ! -f "$value_bin" ]]; then
        echo "  ERROR: $value_bin not found, skipping" >&2
        continue
    fi

    hf_dir="${OUTPUT_DIR}/${run_name}_hf"
    echo "  Converting checkpoint -> $hf_dir"
    uv run python -m open_instruct.value_estimation convert_checkpoint \
        --checkpoint_dir  "${step_dir}/value_model" \
        --output_dir      "${hf_dir}" \
        --base_model_path "${step_dir}"

    # Read conditioning from training_args.json if present.
    gt_cond=false
    template="answer_prefix"
    rc_siblings=4
    ta_path="${step_dir}/training_args.json"
    if [[ -f "$ta_path" ]]; then
        gt_cond=$(python3 -c "import json; d=json.load(open('$ta_path')); print(str(d.get('value_model_ground_truth_conditioning',False)).lower())")
        template=$(python3 -c "import json; d=json.load(open('$ta_path')); print(d.get('gt_conditioning_template','answer_prefix'))")
        rc_siblings=$(python3 -c "import json; d=json.load(open('$ta_path')); print(d.get('rollout_context_num_siblings',4))")
    fi

    score_path="${OUTPUT_DIR}/${run_name}_scores.parquet"
    echo "  Scoring (gt_cond=$gt_cond, template=$template)"

    score_args=(
        --input_dataset_path "$DATASET"
        --output_path        "$score_path"
        --value_model_path   "$hf_dir"
        --value_model_type   scalar
        --run_name           "$run_name"
    )
    if [[ "$gt_cond" == "true" ]]; then
        score_args+=(--value_model_ground_truth_conditioning --gt_conditioning_template "$template")
        if [[ "$template" == "rollout_context" ]]; then
            score_args+=(--rollout_context_num_siblings "$rc_siblings")
        fi
    fi

    uv run python -m open_instruct.value_estimation score_dataset "${score_args[@]}"
    SCORE_PATHS+=("$score_path")
    echo "  Done -> $score_path"
done

if [[ ${#SCORE_PATHS[@]} -eq 0 ]]; then
    echo "No runs scored successfully." >&2; exit 1
fi

echo "=== Comparing ${#SCORE_PATHS[@]} runs ==="
uv run python -m open_instruct.value_estimation compare_runs \
    --score_dataset_paths "${SCORE_PATHS[@]}" \
    --output_markdown_path "${OUTPUT_DIR}/comparison.md" \
    --output_csv_path      "${OUTPUT_DIR}/comparison.csv"

echo ""
echo "Results written to ${OUTPUT_DIR}/comparison.md"
cat "${OUTPUT_DIR}/comparison.md"
