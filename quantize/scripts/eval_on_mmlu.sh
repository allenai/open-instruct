# export CUDA_VISIBLE_DEVICES=0

python -m eval.mmlu_eval.evaluate_hf_lm \
    --ntrain 0 \
    --data_dir data/mmlu \
    --save_dir results/mmlu/alpaca-65B-gptq-0shot/ \
    --model "/net/nfs.cirrascale/allennlp/davidw/checkpoints/gptq_alpaca_fixed_65b" \
    --tokenizer "/net/nfs.cirrascale/allennlp/hamishi/open-instruct/alpaca_fixed_65b" \
    --eval_batch_size 8 \
    --gptq