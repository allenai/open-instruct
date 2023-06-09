# export CUDA_VISIBLE_DEVICES=0

python -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --max_num_examples_per_lang 40 \
    --save_dir results/mgsm/llama-7B-2shot \
    --model ../hf_llama_models/7B \
    --tokenizer ../hf_llama_models/7B \
    --eval_batch_size 4 \
    --n_shot 2 \
    --load_in_8bit

python -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --max_num_examples_per_lang 40 \
    --save_dir results/mgsm/llama-7B-6shot \
    --model ../hf_llama_models/7B \
    --tokenizer ../hf_llama_models/7B \
    --eval_batch_size 4 \
    --n_shot 6 \
    --load_in_8bit

python -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --max_num_examples_per_lang 40 \
    --save_dir results/mgsm/llama-65B-2shot \
    --model ../hf_llama_models/65B \
    --tokenizer ../hf_llama_models/65B \
    --eval_batch_size 1 \
    --n_shot 2 \
    --load_in_8bit

python -m eval.mgsm.run_eval \
    --data_dir data/eval/mgsm/ \
    --max_num_examples_per_lang 40 \
    --save_dir results/mgsm/llama-65B-6shot \
    --model ../hf_llama_models/65B \
    --tokenizer ../hf_llama_models/65B \
    --eval_batch_size 1 \
    --n_shot 6 \
    --load_in_8bit