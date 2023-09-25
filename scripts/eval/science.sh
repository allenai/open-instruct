# Evaluation on science literature tasks.
# Work in progress; generation is really bad right now.

EVAL_FILE_DIR=/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/promptsource-sciit/prompts_davidw/tasks
METRICS_DIR=results/metrics
PREDICTION_DIR=results/prediction
MODEL_PATH=/net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B-chat
MODEL_NAME=llama2-7b-chat
MAX_PROMPTS=200

mkdir -p $METRICS_DIR
mkdir -p $PREDICTION_DIR

declare -a EVAL_NAMES=(
    qasper_truncated_2000_validation.jsonl
    evidence_inference_dev.jsonl
    scifact_json_validation.jsonl
    scitldr_aic_validation.jsonl
    scierc_ner_dev.jsonl
    scierc_relation_dev.jsonl
)

for eval_name in "${EVAL_NAMES[@]}"; do
    basename="${eval_name%.*}"
    python -m eval.science.run_eval \
        --eval_file ${EVAL_FILE_DIR}/${basename}.jsonl \
        --metrics_file ${METRICS_DIR}/${basename}_${MODEL_NAME}.json \
        --prediction_file ${PREDICTION_DIR}/${basename}_${MODEL_NAME}.jsonl \
        --model_name_or_path $MODEL_PATH \
        --max_prompts $MAX_PROMPTS \
        --use_chat_format
done
