# export CUDA_VISIBLE_DEVICES=0

# # cot evaluation
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/llama-7B-8shot \
#     --model ../hf_llama_models/7B \
#     --tokenizer ../hf_llama_models/7B \
#     --eval_batch_size 20 \
#     --n_shot 8 \
#     --load_in_8bit


# # cot evaluation with chatgpt
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/chatgpt-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 8 


# # no cot evaluation with chatgpt
# python -m eval.gsm.run_eval \
#     --data_dir data/eval/gsm/ \
#     --max_num_examples 200 \
#     --save_dir results/gsm/chatgpt-no-cot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --n_shot 8 \
#     --no_cot


# cot evaluation with gpt4
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/gpt4-cot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20 \
    --n_shot 8 


# no cot evaluation with gpt4
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 200 \
    --save_dir results/gsm/gpt4-no-cot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20 \
    --n_shot 8 \
    --no_cot
