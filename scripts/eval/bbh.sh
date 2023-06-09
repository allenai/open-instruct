export CUDA_VISIBLE_DEVICES=0

# # cot
# python -m eval.bbh.run_eval \
#     --data_dir data/eval/bbh \
#     --save_dir results/bbh/llama-7B-cot/ \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/checkpoints/llama_7B_cot_0508 \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/checkpoints/llama_7B_cot_0508 \
#     --eval_batch_size 10 \
#     --max_num_examples_per_task 40 \
#     --load_in_8bit \
#     --use_chat_format


# # direct answer
# python -m eval.bbh.run_eval \
#     --data_dir data/eval/bbh \
#     --save_dir results/bbh/llama-7B-no-cot/ \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B/ \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B/ \
#     --eval_batch_size 10 \
#     --max_num_examples_per_task 40 \
#     --load_in_8bit \
#     --no_cot \
#     --use_chat_format


# # cot with gpt-3.5-turbo-0301
# python -m eval.bbh.run_eval \
#     --data_dir data/eval/bbh \
#     --save_dir results/bbh/chatgpt-cot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 10 \
#     --max_num_examples_per_task 40


# # direct answer with gpt-3.5-turbo-0301
# python -m eval.bbh.run_eval \
#     --data_dir data/eval/bbh \
#     --save_dir results/bbh/chatgpt-no-cot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 10 \
#     --max_num_examples_per_task 40 \
#     --no_cot


# cot with gpt-4
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/gpt4-cot/ \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 10 \
    --max_num_examples_per_task 40


# direct answer with gpt-4
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/gpt4-no-cot/ \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 10 \
    --max_num_examples_per_task 40 \
    --no_cot