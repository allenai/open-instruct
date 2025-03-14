# Default values that can be overridden by environment variables
IMAGE=${IMAGE:-nathanl/open_instruct_auto}
CLUSTER=${CLUSTER:-ai2/jupiter-cirrascale-2}
python mason.py \
    --cluster $CLUSTER \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --image $IMAGE --pure_docker_mode \
    --preemptible \
    --num_nodes 8 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name tulu3_8b_sft \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --model_revision main \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --tokenizer_revision main \
    --use_slow_tokenizer \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --reduce_loss sum \
    --use_flash_attn \
    --gradient_checkpointing \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 8