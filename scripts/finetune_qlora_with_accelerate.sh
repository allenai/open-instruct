export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/${MODEL_SIZE} \
    --use_qlora \
    --use_lora \
    --use_flash_attn \
    --lora_rank 256 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/${MODEL_SIZE} \
    --use_slow_tokenizer \
    --train_file tulu_data/tulu_v1_mix.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --output_dir output/tulu_${MODEL_SIZE}_qlora_exp/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&

python open_instruct/merge_lora.py \
    --base_model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/${MODEL_SIZE} \
    --lora_model_name_or_path output/tulu_${MODEL_SIZE}_qlora_exp/ \
    --output_dir output/tulu_${MODEL_SIZE}_qlora_merged_exp/
