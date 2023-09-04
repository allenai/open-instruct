# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# Evaluating llama 7B model, with gold passage provided
# By default, we use 1-shot setting, and 100 examples per language
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/llama-7B-goldp \
    --model ../hf_llama_model/7B \
    --tokenizer ../hf_llama_model/7B \
    --eval_batch_size 20 \
    --load_in_8bit


# Evaluating llama 7B model, with no context provided (closed-book QA)
# By default, we use 1-shot setting, and 100 examples per language
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/llama-7B-no-context \
    --model ../hf_llama_model/7B \
    --tokenizer ../hf_llama_model/7B \
    --eval_batch_size 40 \
    --load_in_8bit \
    --no_context  

# Evaluating Tulu 7B model, with gold passage provided
# For Tulu, we use chat format.
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/tulu-7B-goldp \
    --model ../checkpoints/tulu_7B \
    --tokenizer ../checkpoints/tulu_7B \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# Evaluating Tulu 7B model, with no context provided (closed-book QA)
# For Tulu, we use chat format.
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/tulu-7B-no-context \
    --model ../checkpoints/tulu_7B \
    --tokenizer ../checkpoints/tulu_7B \
    --eval_batch_size 20 \
    --load_in_8bit \
    --no_context \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# Evaluating llama2 chat model, with gold passage provided
# For llama2 chat model, we use chat format.
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/llama2-chat-7B-goldp \
    --model ../hf_llama2_models/7B-chat \
    --tokenizer ../hf_llama2_models/7B-chat \
    --eval_batch_size 20 \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# Evaluating llama2 chat model, with no context provided (closed-book QA)
# For llama2 chat model, we use chat format.
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/llama2-chat-7B-no-context \
    --model ../hf_llama2_models/7B-chat \
    --tokenizer ../hf_llama2_models/7B-chat \
    --eval_batch_size 20 \
    --load_in_8bit \
    --no_context \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# Evaluating chatgpt, with gold passage provided
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/chatgpt-goldp-1shot \
    --openai_engine "gpt-3.5-turbo-0301" \
    --eval_batch_size 20


# Evaluating chatgpt, with no context provided (closed-book QA)
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/chatgpt-no-context-1shot \
    --openai_engine "gpt-3.5-turbo-0301" \
    --eval_batch_size 20 \
    --no_context 


# Evaluating gpt4, with gold passage provided
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/gpt4-goldp-1shot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20


# Evaluating gpt4, with no context provided (closed-book QA)
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/gpt4-no-context-1shot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20 \
    --no_context 