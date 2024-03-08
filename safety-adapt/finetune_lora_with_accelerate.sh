export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_DIR="/net/nfs.cirrascale/mosaic/faezeb/open-instruct/safety-adapt/experiments/data"
MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=16 #128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Safety tuning tulu2 model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# --beaker_model_path
#ai2/general-cirrascale-a100-80g-ib
# Lora training
# preemptible
#tulu_match_safety.jsonl

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path allenai/tulu-2-7b \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name allenai/tulu-2-7b \
    --use_slow_tokenizer \
    --train_file $DATA_DIR//sample_debug.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir ./safety-adapt/output \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 #&&



# mason --cluster ai2/mosaic-cirrascale --budget ai2/oe-adapt --gpus $NUM_GPUS --priority high --workspace ai2/safety-adapt --  \
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     open_instruct/finetune.py \
#     --model_name_or_path allenai/tulu-2-7b \
#     --use_flash_attn \
#     --use_lora \
#     --lora_rank 64 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     --tokenizer_name allenai/tulu-2-7b \
#     --use_slow_tokenizer \
#     --train_file $DATA_DIR//tulu_match_safety.jsonl \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 16 \
#     --checkpointing_steps epoch \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 1e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 3 \
#     --output_dir /output \
#     --with_tracking \
#     --report_to tensorboard \
#     --logging_steps 1 #&&

# mason --cluster ai2/mosaic-cirrascale --budget ai2/oe-adapt --gpus 2 --priority high --workspace ai2/safety-adapt --  \
# python open_instruct/merge_lora.py \
#     --base_model_name_or_path allenai/tulu-2-7b \
#     --lora_model_name_or_path /output \
#     --output_dir /output/tulu2_7b_safety_tuned_lora_merged/ \
#     --save_tokenizer


python python /net/nfs.cirrascale/mosaic/faezeb/LM-exp/refusal/hold_gpu.py