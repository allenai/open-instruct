# python -m eval.creative_tasks.perplexity_eval \
#     --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
#     --output_field_name gpt-4-0314_output \
#     --save_dir results/creative_tasks/gpt4_ppl_llama_7B \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
#     --load_in_8bit

# python -m eval.creative_tasks.perplexity_eval \
#     --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
#     --output_field_name gpt-4-0314_output \
#     --save_dir results/creative_tasks/gpt4_ppl_llama_13B \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/13B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/13B \
#     --load_in_8bit


# python -m eval.creative_tasks.perplexity_eval \
#     --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
#     --output_field_name gpt-4-0314_output \
#     --save_dir results/creative_tasks/gpt4_ppl_llama_30B \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/30B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/30B \
#     --load_in_8bit


# python -m eval.creative_tasks.perplexity_eval \
#     --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
#     --output_field_name gpt-4-0314_output \
#     --save_dir results/creative_tasks/gpt4_ppl_llama_65B \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
#     --load_in_8bit


# python -m eval.creative_tasks.perplexity_eval \
#     --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
#     --output_field_name reference \
#     --save_dir results/creative_tasks/reference_ppl_llama_7B \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B \
#     --load_in_8bit

# python -m eval.creative_tasks.perplexity_eval \
#     --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
#     --output_field_name reference \
#     --save_dir results/creative_tasks/reference_ppl_llama_13B \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/13B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/13B \
#     --load_in_8bit


# python -m eval.creative_tasks.perplexity_eval \
#     --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
#     --output_field_name reference \
#     --save_dir results/creative_tasks/reference_ppl_llama_30B \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/30B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/30B \
#     --load_in_8bit


# python -m eval.creative_tasks.perplexity_eval \
#     --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
#     --output_field_name reference \
#     --save_dir results/creative_tasks/reference_ppl_llama_65B \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/65B \
#     --load_in_8bit


python -m eval.creative_tasks.perplexity_eval \
    --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
    --output_field_name reference \
    --save_dir results/creative_tasks/reference_ppl_alpaca_65B \
    --model ../../hamishi/open-instruct/alpaca_fixed_65b/ \
    --tokenizer ../../hamishi/open-instruct/alpaca_fixed_65b/ \
    --load_in_8bit 


python -m eval.creative_tasks.perplexity_eval \
    --data_file data/eval/creative_tasks/gpt4_outputs.jsonl \
    --output_field_name gpt-4-0314_output \
    --save_dir results/creative_tasks/gpt4_ppl_alpaca_65B \
    --model ../../hamishi/open-instruct/alpaca_fixed_65b/ \
    --tokenizer ../../hamishi/open-instruct/alpaca_fixed_65b/ \
    --load_in_8bit