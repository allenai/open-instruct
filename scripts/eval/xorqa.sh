python -m eval.xorqa.run_eval \
    --data_dir data/eval/xorqa/ \
    --n_shot 5 \
    --max_num_examples_per_lang 50 \
    --save_dir results/xorqa/llama-7B-5shot \
    --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --eval_batch_size 16 \
    --load_in_8bit


python -m eval.xorqa.run_eval \
    --data_dir data/eval/xorqa/ \
    --n_shot 5 \
    --max_num_examples_per_lang 50 \
    --save_dir results/xorqa/llama-65B-5shot \
    --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
    --eval_batch_size 2 \
    --load_in_8bit


python -m eval.xorqa.run_eval \
    --data_dir data/eval/xorqa/ \
    --n_shot 0 \
    --max_num_examples_per_lang 50 \
    --save_dir results/xorqa/llama-7B-0shot \
    --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
    --eval_batch_size 16 \
    --load_in_8bit


python -m eval.xorqa.run_eval \
    --data_dir data/eval/xorqa/ \
    --n_shot 0 \
    --max_num_examples_per_lang 50 \
    --save_dir results/xorqa/llama-65B-0shot \
    --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
    --eval_batch_size 2 \
    --load_in_8bit

