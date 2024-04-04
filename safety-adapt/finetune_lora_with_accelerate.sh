export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA_DIR="/net/nfs.cirrascale/mosaic/faezeb/open-instruct/safety-adapt/experiments/data"
MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=128 #128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Safety tuning tulu2 model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
DESC="refusal_tunning_tulu-2-7b_uncensored_refusal_adapt_v0.1_epoch1"
#"safety_tunning_tulu-2-7b_safety_adapt_v0.1_contrastive_v0.1_augmented_filterv2_epoch1" #"safety_tunning_tulu-2-7b_safety_adapt_v0.1_tulu_match_combined_contrastive_epoch1" #"Safety_Tuning_tulu-2-7b-uncensored_all_safety_adapt_1epoch"
#xstest-based-category_


# --beaker_model_path
#ai2/general-cirrascale-a100-80g-ib
# Lora training
# preemptible
#tulu_match_safety.jsonl

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
#     --train_file $DATA_DIR//sample_debug.jsonl \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 16 \
#     --checkpointing_steps epoch \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 1e-5 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 1 \
#     --output_dir ./safety-adapt/output \
#     --with_tracking \
#     --report_to tensorboard \
#     --logging_steps 1 #&&

# ai2/mosaic-cirrascale
# --beaker_datasets /model:hamishivi/tulu_2_7b_no_refusals
# --beaker_dataset /model:hamishivi/tulu_2_7b_no_refusals
# allenai/tulu-2-7b
# combined safetyv0.1 contrastive harmless only: contrastive_harmless_combined_safety_train.jsonl
#
# ai2/general-cirrascale-a100-80g-ib
# --beaker_datasets /model:hamishivi/tulu_2_7b_no_refusals
mason --cluster ai2/general-cirrascale-a100-80g-ib ai2/pluto-cirrascale --budget ai2/oe-adapt --gpus $NUM_GPUS --priority high --description $DESC --workspace ai2/safety-adapt --beaker_datasets /model:hamishivi/tulu_2_7b_no_refusals --  \
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path /model \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name /model \
    --use_slow_tokenizer \
    --train_file $DATA_DIR/tulu_none_refusal_v0.1_train.jsonl \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 #&&

# mason --cluster ai2/mosaic-cirrascale --budget ai2/oe-adapt --gpus 2 --priority high --workspace ai2/safety-adapt --  \
# python open_instruct/merge_lora.py \
#     --base_model_name_or_path allenai/tulu-2-7b \
#     --lora_model_name_or_path /output \
#     --output_dir /output/tulu2_7b_safety_tuned_lora_merged/ \
#     --save_tokenizer


# python /net/nfs.cirrascale/mosaic/faezeb/LM-exp/refusal/hold_gpu.py

# # used datases:
# -   safety_v0.1 + contrastive_v0.1_all (this include more filter, lower similarity threshod 0.85 and remove duplicates): 
#     "contrastive_safety_train_combined_safety_v0.1_contrastive_v0.1_all_v2.jsonl"
# -   safety_v0.1 + contrastive_v0.1_only_xstest_vased_category with filterv2 (this include more filter, lower similarity threshod 0.85 and remove duplicates): 
#     "contrastive_safety_train_combined_safety_v0.1_contrastive_v0.1_xstest-categories.jsonl"
# - refusal_v0.1:
#     "tulu_none_refusal_v0.1_train.jsonl"