# example scripts for toxigen

# evaluate an open-instruct model with chat format
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir tulu_65b \
    --model_name_or_path tulu_65b/ \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# evaluate a base model without chat format
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir tulu_65b \
    --model_name_or_path tulu_65b/ \
    --use_vllm


# evaluate chatGPT
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir results/toxigen/chatgpt \
    --openai_engine gpt-3.5-turbo-0301 \
    --max_prompts_per_group 100 \
    --eval_batch_size 20


# evaluate gpt4
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir results/toxigen/gpt4 \
    --openai_engine gpt-4-0314 \
    --max_prompts_per_group 100 \
    --eval_batch_size 20