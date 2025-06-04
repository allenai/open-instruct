base=allenai/Llama-3.1-Tulu-3-8B-SFT
base_no_slash=$(echo $base | tr '/' '_')
n_outputs=1
exp_name=code_sft_${base_no_slash}_n_${n_outputs}
echo "Launching job for n_outputs=${n_outputs}"
python mason.py \
    --cluster ai2/augusta-google-1 \
    --workspace ai2/oe-adapt-code \
    --priority high \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --description "8B SFT on Open Code reasoner data with ${n_outputs} samples per prompt, ${base} base" \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name $exp_name \
    --model_name_or_path $base \
    --model_revision main \
    --tokenizer_name $base \
    --tokenizer_revision main \
    --use_slow_tokenizer \
    --dataset_mixer_list saurabh5/open-code-reasoning-sft-n-${n_outputs} 1.0 \
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
#done