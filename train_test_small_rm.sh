for model in Qwen/Qwen2.5-0.5B; do
for LR in 1e-6 # 2e-5 3e-6
do
  for seed in 1; do
  for NUM_TRAIN_EPOCHS in 1 # 2
  do
    echo "Running with LR=${LR} and NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS} and seed ${seed}"
    python mason.py \
    --cluster ai2/augusta-google-1 \
    --workspace ai2/reward-bench-v2 \
    --priority high \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/reward_modeling.py \
    --exp_name rm_qwen2p5_base_0p5b_${LR}_${NUM_TRAIN_EPOCHS}_skyworkstulufull \
    --model_name_or_path $model\
    --model_revision main \
    --tokenizer_name $model \
    --tokenizer_revision main \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 Skywork/Skywork-Reward-Preference-80K-v0.2 1.0 \
    --max_token_length 4096 \
    --max_prompt_token_length 2048 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate ${LR} \
    --lr_scheduler_type linear \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --wandb_project_name reward-models \
    --gradient_checkpointing \
    --with_tracking \
    --seed ${seed}
  done
done
done
done