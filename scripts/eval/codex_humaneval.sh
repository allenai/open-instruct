# export CUDA_VISIBLE_DEVICES=0

# # evaluating huggingface models

# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 \
#     --unbiased_sampling_size_n 5 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/llama_7B \
#     --model ../hf_llama_models/7B/ \
#     --tokenizer ../hf_llama_models/7B/ \
#     --eval_batch_size 32 \
#     --load_in_8bit

# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 \
#     --unbiased_sampling_size_n 5 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/llama_7B_large_mix \
#     --model ../checkpoints/llama_7B_large_mix/ \
#     --tokenizer ../checkpoints/llama_7B_large_mix/ \
#     --eval_batch_size 32 \
#     --load_in_8bit 

# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 \
#     --unbiased_sampling_size_n 5 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/llama_13B \
#     --model ../hf_llama_models/13B/ \
#     --tokenizer ../hf_llama_models/13B/ \
#     --eval_batch_size 16 \
#     --load_in_8bit

# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 \
#     --unbiased_sampling_size_n 5 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/llama_30B \
#     --model ../hf_llama_models/30B/ \
#     --tokenizer ../hf_llama_models/30B/ \
#     --eval_batch_size 8 \
#     --load_in_8bit

# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --save_dir results/codex_humaneval/llama_65B \
#     --model ../hf_llama_models/65B/ \
#     --tokenizer ../hf_llama_models/65B/ \
#     --eval_batch_size 4 \
#     --load_in_8bit


# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --save_dir results/codex_humaneval/llama_65B_large_mix_no_chat \
#     --model ../checkpoints/llama_65B_large_mix/ \
#     --tokenizer ../checkpoints/llama_65B_large_mix/ \
#     --eval_batch_size 1 \
#     --load_in_8bit


# # evaluating chatgpt
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --save_dir results/codex_humaneval/chatgpt_temp_0.1/ \
#     --eval_batch_size 10


# python -m eval.codex_humaneval.run_eval \
    # --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
    # --eval_pass_at_ks 1 5 10 20 \
    # --unbiased_sampling_size_n 20 \
    # --temperature 0.8 \
    # --openai_engine "gpt-3.5-turbo-0301" \
    # --save_dir results/codex_humaneval/chatgpt_temp_0.8/ \
    # --eval_batch_size 10


# # evaluating gpt4
# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.1 \
#     --openai_engine "gpt-4-0314" \
#     --save_dir results/codex_humaneval/gpt4_temp_0.1 \
#     --eval_batch_size 1


# python -m eval.codex_humaneval.run_eval \
#     --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz \
#     --eval_pass_at_ks 1 5 10 20 \
#     --unbiased_sampling_size_n 20 \
#     --temperature 0.8 \
#     --openai_engine "gpt-4-0314" \
#     --save_dir results/codex_humaneval/gpt4_temp_0.8 \
#     --eval_batch_size 1