# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# Evaluating tulu 7B model using chat format
python -m eval.xstest.run_eval \
    --data_dir data/eval/xstest/ \
    --save_dir results/xstest/tulu-7B-sft \
    --model ../checkpoints/tulu2/7B-sft \
    --tokenizer ../checkpoints/tulu2/7B-sft \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm


# Evaluating tulu 70B dpo model using chat format
python -m eval.xstest.run_eval \
    --data_dir data/eval/xstest/ \
    --save_dir results/xstest/tulu-70B-dpo \
    --model allenai/tulu-2-dpo-70b \
    --tokenizer allenai/tulu-2-dpo-70b \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm


# Evaluating chatgpt
python -m eval.xstest.run_eval \
    --data_dir data/eval/xstest/ \
    --save_dir results/xstest/chatgpt-no-cot \
    --openai_engine "gpt-3.5-turbo-0125" \
    --eval_batch_size 20


# Evaluating gpt4
python -m eval.xstest.run_eval \
    --data_dir data/eval/xstest/ \
    --save_dir results/xstest/gpt4-cot \
    --openai_engine "gpt-4-0613" \
    --eval_batch_size 20
