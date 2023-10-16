export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed --include localhost:0,1 open_instruct/finetune_trainer.py \
    --deepspeed ds_configs/stage3_no_offloading.conf \
    --model_name_or_path ../hf_llama_models/${MODEL_SIZE} \
    --tokenizer_name ../hf_llama_models/${MODEL_SIZE} \
    --use_flash_attn True \
    --use_fast_tokenizer False \
    --train_file data/processed/tulu_v1/tulu_v1_data.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 64 \
    --do_train \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --num_train_epochs 2 \
    --output_dir output/tulu_v1_${MODEL_SIZE}/ \
    --bf16 \
    --tf32 True \
    --torch_dtype bfloat16 \
    --overwrite_output_dir \
    --report_to "tensorboard" \
    --max_steps 10 
