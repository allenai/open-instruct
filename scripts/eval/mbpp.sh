# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0
# you need to set the below to say you are okay with running llm-generated code on your machine...
export HF_ALLOW_CODE_EVAL=1

# Evaluating tulu 7B model using temperature 0.1 to get the pass@1 score
python -m eval.mbpp.run_eval \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir results/mbpp/tulu_7B_temp_0_1_nochat \
    --model ../checkpoints/tulu_7B/ \
    --tokenizer ../checkpoints/tulu_7B/ \
    --use_vllm


# Evaluating tulu 7B model using temperature 0.8 to get the pass@10 score
python -m eval.mbpp.run_eval \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --eval_pass_at_ks 10 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --save_dir results/mbpp/tulu_7B_temp_0_8_nochat \
    --model ../checkpoints/tulu_7B/ \
    --tokenizer ../checkpoints/tulu_7B/ \
    --use_vllm

# Evaluating chatgpt using temperature 0.1 to get the pass@1 score
python -m eval.mbpp.run_eval \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --openai_engine "gpt-3.5-turbo-0301" \
    --save_dir results/mbpp/chatgpt_temp_0.1/ \
    --eval_batch_size 10


# Evaluating chatgpt using temperature 0.8 to get the pass@10 score
python -m eval.mbpp.run_eval \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --openai_engine "gpt-3.5-turbo-0301" \
    --save_dir results/mbpp/chatgpt_temp_0.8/ \
    --eval_batch_size 10


# Evaluating gpt4 using temperature 0.1 to get the pass@1 score
python -m eval.mbpp.run_eval \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --openai_engine "gpt-4-0314" \
    --save_dir results/mbpp/gpt4_temp_0.1 \
    --eval_batch_size 1


# Evaluating gpt4 using temperature 0.8 to get the pass@10 score
python -m eval.mbpp.run_eval \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.8 \
    --openai_engine "gpt-4-0314" \
    --save_dir results/mbpp/gpt4_temp_0.8 \
    --eval_batch_size 1
