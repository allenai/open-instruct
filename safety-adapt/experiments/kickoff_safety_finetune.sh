
DATA_DIR="/net/nfs.cirrascale/mosaic/faezeb/open-instruct/safety-adapt/experiments/data"
# python run_lora_finetune.py \
#     --train_file $DATA_DIR/tulu_match_safety.jsonl \
#     --beaker_model_path Yizhongw03/hf_llama2_model_7B \
#     --num_gpus 4 \
#     --preprocessing_num_workers 16 \
#     --cluster "ai2/mosaic-cirrascale" # "ai2/general-cirrascale-a100-80g-ib" # "ai2/mosaic-cirrascale" #ai2/general-cirrascale-a100-80g-ib

python run_lora_finetune.py \
    --train_file $DATA_DIR/safety_tunning_data.jsonl \
    --beaker_model_path Yizhongw03/hf_llama2_model_7B \
    --num_gpus 2 \
    --preprocessing_num_workers 16 \
    --batch_size_per_gpu 4 \
    --cluster "ai2/mosaic-cirrascale" #"ai2/general-cirrascale-a100-80g-ib" #  #ai2/general-cirrascale-a100-80g-ib
