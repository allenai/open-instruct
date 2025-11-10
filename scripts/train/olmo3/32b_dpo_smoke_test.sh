BEAKER_IMAGE=$1
MODEL_NAME=/weka/oe-adapt-default/finbarrt/stego32/step358000-hf
NUM_NODES=8
for LR in 5e-8
do 
    EXP_NAME="olmo3-32b-merge_2e5e-DPO-deltas-150k-${LR}"
    uv run python mason.py \
       --cluster ai2/augusta \
       --gs_model_name $EXP_NAME \
       --workspace ai2/olmo-instruct \
       --priority urgent \
       --image $BEAKER_IMAGE --pure_docker_mode \
       --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
        --env TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
        --env TORCH_DIST_INIT_BARRIER=1 \
        --env NCCL_ALGO=Ring,Tree \
        --env NCCL_NVLS_ENABLE=0 \
       --preemptible \
       --num_nodes $NUM_NODES \
       --budget ai2/oe-adapt \
       --no_auto_dataset_cache \
       --gpus 8 -- accelerate launch \
       --mixed_precision bf16 \
       --num_processes 8 \
       --use_deepspeed \
       --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
       --deepspeed_multinode_launcher standard \
       open_instruct/dpo_tune_cache.py \
       --exp_name $EXP_NAME \
       --model_name_or_path $MODEL_NAME \
       --use_slow_tokenizer False \
       --dataset_mixer_list allenai/olmo-3-preference-mix-deltas_reasoning-scottmix-DECON-keyword-filtered 10000       
       allenai/dpo-yolo1-200k-gpt4.1-2w2s-maxdelta_rejected-DECON-rm-gemma3-kwd-ftd 1000 \
       --dataset_skip_cache \
       --zero_stage 2 \
        --concatenated_forward False \
       --ref_logprobs_cache_dir "/filestore/.cache/" \
       --max_seq_length 16384 \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 2 \
       --learning_rate $LR \
       --lr_scheduler_type linear \
       --warmup_ratio 0.1 \
       --weight_decay 0.0 \
       --num_train_epochs 1 \
       --logging_steps 1 \
       --dpo_loss_type dpo_norm \
       --dpo_beta 5 \
       --use_flash_attn \
       --gradient_checkpointing \
       --report_to wandb \
       --chat_template_name olmo123 \
       --with_tracking
done
