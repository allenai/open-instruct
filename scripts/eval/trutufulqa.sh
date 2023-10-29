# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# Evaluating llama 7B model, getting the judge and info scores and multiple choice accuracy
# To get the judge and info scores, you need to specify the gpt_judge_model_name and gpt_info_model_name,
# which are the names of the GPT models trained following https://github.com/sylinrl/TruthfulQA#fine-tuning-gpt-3-for-evaluation
python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/trutufulqa/llama-7B \
    --model_name_or_path ../hf_llama_models/7B \
    --tokenizer_name_or_path ../hf_llama_models/7B \
    --metrics judge info mc \
    --preset qa \
    --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --eval_batch_size 20 \
    --load_in_8bit


# Evaluating Tulu 7B model using chat format, getting the judge and info scores and multiple choice accuracy
python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/trutufulqa/tulu_7B \
    --model_name_or_path ../checkpoints/tulu_7B/ \
    --tokenizer_name_or_path ../checkpoints/tulu_7B/ \
    --metrics judge info mc \
    --preset qa \
    --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# Evaluating llama2 chat model using chat format, getting the judge and info scores and multiple choice accuracy
python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/trutufulqa/llama2-chat-7B \
    --model_name_or_path ../hf_llama2_models/7B-chat \
    --tokenizer_name_or_path ../hf_llama2_models/7B-chat \
    --metrics judge info mc \
    --preset qa \
    --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# Evaluating chatgpt, getting the judge and info scores
# Multiple choice accuracy is not supported for chatgpt, since we cannot get the probabilities from chatgpt
python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/trutufulqa/chatgpt \
    --openai_engine gpt-3.5-turbo-0301 \
    --metrics judge info \
    --preset qa \
    --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --eval_batch_size 20

# Evaluating gpt-4, getting the judge and info scores
# Multiple choice accuracy is not supported for gpt-4, since we cannot get the probabilities from gpt-4
python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/trutufulqa/gpt4 \
    --openai_engine gpt-4-0314 \
    --metrics judge info \
    --preset qa \
    --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --eval_batch_size 20