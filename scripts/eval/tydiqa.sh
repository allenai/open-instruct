export CUDA_VISIBLE_DEVICES=0

# # with gold passage
# python -m eval.tydiqa.run_eval \
#     --data_dir data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/llama-7B-goldp-1shot \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
#     --eval_batch_size 20 \
#     --load_in_8bit \
#     --use_chat_format


# # no gold passage, closed-book qa
# python -m eval.tydiqa.run_eval \
#     --data_dir data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/llama-7B-no-context-1shot \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
#     --eval_batch_size 80 \
#     --load_in_8bit \
#     --no_context \
#     --use_chat_format


# # with gold passage, using chatgpt
# python -m eval.tydiqa.run_eval \
#     --data_dir data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/chatgpt-goldp-1shot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # closed-book qa, using chatgpt
# python -m eval.tydiqa.run_eval \
#     --data_dir data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 100 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/chatgpt-no-context-1shot \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20 \
#     --no_context 

# with gold passage, using gpt4
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 20 \
    --max_context_length 512 \
    --save_dir results/tydiqa/gpt4-goldp-1shot \
    --openai_engine "gpt-4-0314" \
    --eval_batch_size 20

# # closed-book qa, using gpt4
# python -m eval.tydiqa.run_eval \
#     --data_dir data/eval/tydiqa/ \
#     --n_shot 1 \
#     --max_num_examples_per_lang 20 \
#     --max_context_length 512 \
#     --save_dir results/tydiqa/gpt4-no-context-1shot \
#     --openai_engine "gpt-4-0314" \
#     --eval_batch_size 20 \
#     --no_context 