# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# evaluating llama 7B model using chain-of-thought
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/llama-7B-cot/ \
    --model ../hf_llama_models/7B \
    --tokenizer ../hf_llama_models/7B \
    --max_num_examples_per_task 40 \
    --use_vllm


# evaluating llama 7B model using direct answering (no chain-of-thought)
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/llama-7B-no-cot/ \
    --model ../hf_llama_models/7B \
    --tokenizer ../hf_llama_models/7B \
    --max_num_examples_per_task 40 \
    --use_vllm \
    --no_cot


# evaluating tulu 7B model using chain-of-thought and chat format
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/tulu-7B-cot/ \
    --model ../checkpoint/tulu_7B \
    --tokenizer ../checkpoints/tulu_7B \
    --max_num_examples_per_task 40 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format


# evaluating llama2 chat model using chain-of-thought and chat format
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/llama2-chat-7B-cot \
    --model ../hf_llama2_models/7B-chat \
    --tokenizer ../hf_llama2_models/7B-chat \
    --max_num_examples_per_task 40 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# evaluating gpt-3.5-turbo-0301 using chain-of-thought
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/chatgpt-cot/ \
    --openai_engine "gpt-3.5-turbo-0301" \
    --eval_batch_size 10 \
    --max_num_examples_per_task 40


# evaluating gpt-3.5-turbo-0301 using direct answering (no chain-of-thought)
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/chatgpt-no-cot/ \
    --openai_engine "gpt-3.5-turbo-0301" \
    --eval_batch_size 10 \
    --max_num_examples_per_task 40 \
    --no_cot


# evaluating gpt-4 using chain-of-thought
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/gpt4-cot/ \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 10 \
    --max_num_examples_per_task 40


# evaluating gpt-4 using direct answering (no chain-of-thought)
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/gpt4-no-cot/ \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 10 \
    --max_num_examples_per_task 40 \
    --no_cot