LLAMA_FOLDER=/net/nfs.cirrascale/allennlp/jacobm/llama/llama/models

for MODEL_SIZE in 7B 13B 30B 65B; do
    echo "Converting Llama ${MODEL_SIZE} to HuggingFace format"
    python -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir $LLAMA_FOLDER/ \
    --model_size $MODEL_SIZE \
    --output_dir /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/${MODEL_SIZE}
done
