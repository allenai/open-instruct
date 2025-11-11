MODEL_NAME=/weka/oe-adapt-default/jacobm/olmo3/32b-merge-configs/checkpoints/32b-1e-4-5e-5/
NUM_NODES=8
BEAKER_IMAGE=$1

LR=6e-8
EXP_NAME="olmo3-32b-DPO-dbg-tr-shrd64-ofldopt-sep-${LR}"

HF_HUB_ENABLE_HF_TRANSFER=1 uv run python mason.py \
    --cluster ai2/augusta \
    --gs_model_name olmo3-merge-32b-1e-4-5e-5 \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --max_retries 1 \
    --preemptible \
    --image $BEAKER_IMAGE \
    --pure_docker_mode \
    --env NCCL_LIB_DIR=/var/lib/tcpxo/lib64 \
    --env LD_LIBRARY_PATH=/var/lib/tcpxo/lib64:$LD_LIBRARY_PATH \
    --env NCCL_PROTO=Simple,LL128 \
    --env NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config_ll128.textproto \
    --env NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/var/lib/tcpxo/lib64/a3plus_guest_config_ll128.textproto \
    --num_nodes $NUM_NODES \
    --budget ai2/oe-adapt \
    --no_auto_dataset_cache \
    --gpus 8 -- source /var/lib/tcpxo/lib64/nccl-env-profile.sh \&\& accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name $EXP_NAME \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer False \
    --dataset_mixer_list allenai/olmo-3-preference-mix-deltas_reasoning-scottmix-DECON-keyword-filtered 1.0 \
    --max_train_samples 15000 \
    --dataset_skip_cache \
    --zero_stage 3 \
    --concatenated_forward False \
    --max_seq_length 16384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
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
    --with_tracking \
    --log_grad_norm True \
    --zero_hpz_partition_size 64 \
    --offload_optimizer True