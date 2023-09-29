# Evaluation on science literature tasks.
# Work in progress; generation is really bad right now.

EVAL_FILE_DIR=/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/promptsource-sciit/prompts_davidw/tasks
MAX_PROMPTS=200

MODEL_NAME=tulu_v1_65B
MODEL_PATH=../checkpoints/tulu_v1_65B_new/
METRICS_DIR=results/science/${MODEL_NAME}/metrics
PREDICTION_DIR=results/science/${MODEL_NAME}/prediction


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
    echo "Evaluating $MODEL_NAME on $basename"
    python -m eval.science.run_eval \
        --eval_file ${EVAL_FILE_DIR}/${basename}.jsonl \
        --metrics_file ${METRICS_DIR}/${basename}_${MODEL_NAME}.json \
        --prediction_file ${PREDICTION_DIR}/${basename}_${MODEL_NAME}.jsonl \
        --model_name_or_path $MODEL_PATH \
        --max_prompts $MAX_PROMPTS \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
done



MODEL_NAME=llama2_chat_70B
MODEL_PATH=../hf_llama2_models/70B-chat/
METRICS_DIR=results/science/${MODEL_NAME}/metrics
PREDICTION_DIR=results/science/${MODEL_NAME}/prediction


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
    echo "Evaluating #MODEL_NAME on $basename"
    python -m eval.science.run_eval \
        --eval_file ${EVAL_FILE_DIR}/${basename}.jsonl \
        --metrics_file ${METRICS_DIR}/${basename}_${MODEL_NAME}.json \
        --prediction_file ${PREDICTION_DIR}/${basename}_${MODEL_NAME}.jsonl \
        --model_name_or_path $MODEL_PATH \
        --max_prompts $MAX_PROMPTS \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format
done


MODEL_NAME=llama2_70B
MODEL_PATH=../hf_llama2_models/70B/
METRICS_DIR=results/science/${MODEL_NAME}/metrics
PREDICTION_DIR=results/science/${MODEL_NAME}/prediction


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
    echo "Evaluating #MODEL_NAME on $basename"
    python -m eval.science.run_eval \
        --eval_file ${EVAL_FILE_DIR}/${basename}.jsonl \
        --metrics_file ${METRICS_DIR}/${basename}_${MODEL_NAME}.json \
        --prediction_file ${PREDICTION_DIR}/${basename}_${MODEL_NAME}.jsonl \
        --model_name_or_path $MODEL_PATH \
        --max_prompts $MAX_PROMPTS \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
done
