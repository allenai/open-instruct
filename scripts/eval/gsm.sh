# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# Evaluating llama 7B model using chain-of-thought
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/llama-7B-cot-8shot \
    --model ../hf_llama_models/7B \
    --tokenizer ../hf_llama_models/7B \
    --n_shot 8 \
    --use_vllm


# Evaluating llama 7B model using direct answering (no chain-of-thought)
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/llama-7B-no-cot-8shot \
    --model ../hf_llama_models/7B \
    --tokenizer ../hf_llama_models/7B \
    --n_shot 8 \
    --no_cot \
    --use_vllm


# Evaluating tulu 7B model using chain-of-thought and chat format
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/tulu-7B-cot-8shot \
    --model ../checkpoints/tulu_7B \
    --tokenizer ../checkpoints/tulu_7B \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm


# Evaluating llama2 chat model using chain-of-thought and chat format
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/llama2-chat-7B-cot-8shot \
    --model ../hf_llama2_models/7B-chat \
    --tokenizer ../hf_llama2_models/7B-chat \
    --n_shot 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --use_vllm


# Evaluating chatgpt using chain-of-thought
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/chatgpt-cot \
    --openai_engine "gpt-3.5-turbo-0301" \
    --eval_batch_size 20 \
    --n_shot 8 


# Evaluating chatgpt using direct answering (no chain-of-thought)
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/chatgpt-no-cot \
    --openai_engine "gpt-3.5-turbo-0301" \
    --eval_batch_size 20 \
    --n_shot 8 \
    --no_cot


# Evaluating gpt4 using chain-of-thought
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/gpt4-cot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20 \
    --n_shot 8 


# Evaluating gpt4 using direct answering (no chain-of-thought)
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/gpt4-no-cot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20 \
    --n_shot 8 \
    --no_cot
