export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --use_lora \
    --use_flash_attn \
    --lora_rank 256 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --use_slow_tokenizer \
    --train_file /net/nfs.cirrascale/allennlp/hamishi/open-instruct/tulu_data/tulu_v1_mix.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir /results/tulu_${MODEL_SIZE}_lora_exp/ \
    --save_merged_lora_model \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --lora_model_name_or_path output/tulu_${MODEL_SIZE}_lora_exp/ \
    --output_dir output/tulu_${MODEL_SIZE}_lora_merged_exp/

#    --use_deepspeed \
#    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \