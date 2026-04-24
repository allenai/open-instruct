#! /bin/bash
# Usage: ./run_aime_eval.sh <MODEL_NAME> <BEAKER_DATASET_ID>
# check we have at least 2 arguments
if [ "$#" -lt 2 ]; then
	echo "Usage: $0 <MODEL_NAME> <BEAKER_DATASET_ID>"
	exit 1
fi
MODEL_NAME=$1
BEAKER_DATASET_ID=$2
uv run python scripts/submit_eval_jobs.py \
   --model_name "$MODEL_NAME" \
   --location "$BEAKER_DATASET_ID" \
   --cluster ai2/jupiter-cirrascale-2 ai2/saturn-cirrascale ai2/ceres-cirrascale ai2/neptune-cirrascale \
   --beaker_image oe-eval-beaker/oe_eval_auto_finbarr \
   --is_tuned \
   --preemptible \
   --priority urgent \
   --workspace ai2/open-instruct-dev \
   --use_hf_tokenizer_template \
   --run_oe_eval_experiments \
   --oe_eval_tasks aime:zs_cot_r1::maj_at_32_2025 \
   --evaluate_on_weka \
   --run_id placeholder \
   --oe_eval_max_length 8192 \
   --process_output r1_style \
   --skip_oi_evals
