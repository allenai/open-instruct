# Tulu3 Reproduction

This document details the commands and configs to reproduce the tulu3 models.

## Finetuning


### Llama-3.1-Tulu-3-8B-SFT Reproduction

Below is (almost) the exact command which produced [Llama-3.1-Tulu-3-8B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT). We deployed the command across 8 machines, each equipped with 8 NVIDIA H100 GPUs, for a total of 64 GPUs in the our setup.

```bash
# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=8
NUM_PROCESSES=64
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 8 \
    --num_processes 64 \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --use_slow_tokenizer \
    --use_flash_attn \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 5e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir output/sft_8b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --checkpointing_steps epoch \
    --dataset_mix_dir output/sft_8b \
    --exp_name tulu-3-8b-sft \
    --seed 123
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JBNTPW8TKG09B2XR832YB5S8
```

> [!NOTE]  
> If you have different number of GPUs, please adjust the `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`, and `GRADIENT_ACCUMULATION_STEPS` accordingly. For example, say, you only have 8 GPUs. The command below has an effective batch size of `NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 64 * 1 * 2 = 128`. A one node setup can simulate our batch size with `NUM_PROCESSES=8`, `PER_DEVICE_TRAIN_BATCH_SIZE=1`, and `GRADIENT_ACCUMULATION_STEPS=64`.


### Llama-3.1-Tulu-3-70B-SFT Reproduction

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-70B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-SFT)


```bash
# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=8
NUM_PROCESSES=64
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
accelerate launch \
    --mixed_precision bf16 \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_PROCESSES \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-3.1-70B \
    --tokenizer_name meta-llama/Llama-3.1-70B \
    --use_slow_tokenizer \
    --use_flash_attn \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 2e-06 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir output/sft_70B \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --dataset_mix_dir output/sft_70B \
    --checkpointing_steps 1000 \
    --keep_last_n_checkpoints 20 \
    --gradient_checkpointing \
    --exp_name tulu-3-70b-sft \
    --seed 456
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JC5J4R80M18XQTDH47JSFRJY/
```


## Preference Tuning


### Llama-3.1-Tulu-3-8B-DPO Reproduction

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-8B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO)


```bash
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf open_instruct/dpo_tune.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --use_flash_attn \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir output/dpo_8b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
    --use_slow_tokenizer \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps 1000 \
    --exp_name tulu-3-8b-dpo
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JCRXP0AR5312S8MD3XGCN0J7/
```



### Llama-3.1-Tulu-3-70B-DPO Reproduction

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-70B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-DPO)


```bash
# modify the following `MACHINE_RANK`, `MAIN_PROCESS_IP`,
# `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`,
# `GRADIENT_ACCUMULATION_STEPS` according to your setup
MACHINE_RANK=0
MAIN_PROCESS_IP=localhost
NUM_MACHINES=8
NUM_PROCESSES=64
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
accelerate launch \
    --mixed_precision bf16 \
    --num_machines $NUM_MACHINES \
    --num_processes $NUM_PROCESSES \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-70B-SFT \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-70B-SFT \
    --use_flash_attn \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate 2e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir output/dpo_70b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-70b-preference-mixture \
    --use_slow_tokenizer \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps epoch \
    --exp_name tulu-3-70b-dpo
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JCSAYYHQYF9QDQDCV6KJ53M9/
```


## RLVR


### Llama-3.1-Tulu-3-8B-RM Reproduction

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-8B-RM](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM)


```bash
accelerate launch \
    --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/llama-3.1-tulu-3-8b-preference-mixture": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --chat_template tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs 1 \
    --output_dir output/rm_8b \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JCS01RFBQGFE5F1W3W96FFVM/
```

### Llama-3.1-Tulu-3-8B Reproduction

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B)


```bash
python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --exp_name tulu-3-8b-rlvr \
    --dataset_mixer '{"allenai/RLVR-GSM-MATH-IF-Mixed-Constraints": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/RLVR-GSM-MATH-IF-Mixed-Constraints": 128}' \
    --dataset_eval_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 10000000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 32 \
    --local_rollout_batch_size 32 \
    --actor_num_gpus_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir output/rlvr_8b \
    --seed 3 \
    --num_evals 3 \
    --save_freq 100 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JCVTA10BQDVGGQKFYWEZ6KCQ/
```



### Llama-3.1-Tulu-3-70B Reproduction

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-70B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B)

Couple of notes:
* Make sure to modify `configs/beaker_configs/ray_node_setup.sh` in our own cluster setup. The idea is to have the replicas join the main machines via `ray`.
* We had to use `--vllm_tensor_parallel_size 4` because `--vllm_tensor_parallel_size 8` errors out for some strange reason. This is a temporary workaround.
* Here the effective batch size is `sum(actor_num_gpus_per_node) * local_mini_batch_size = 40 * 16 = 640`. If you have less GPUs, you can adjust `actor_num_gpus_per_node` and `local_mini_batch_size` accordingly.

```bash
source configs/beaker_configs/ray_node_setup.sh && python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer '{"allenai/RLVR-GSM-MATH-IF-Mixed-Constraints": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/RLVR-GSM-MATH-IF-Mixed-Constraints": 128}' \
    --dataset_eval_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-70B-DPO \
    --exp_name tulu-3-70b-rlvr \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --beta 0.07 \
    --warmup_ratio 0.1 \
    --seed 8 \
    --output_dir output/rlvr_70b \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template tulu \
    --sft_messages_key messages \
    --learning_rate 1e-7 \
    --total_episodes 400000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --local_rollout_forward_batch_size 1 \
    --local_mini_batch_size 16 \
    --local_rollout_batch_size 16 \
    --actor_num_gpus_per_node 8 8 8 8 8 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 4 \
    --apply_verifiable_reward true \
    --reward_model_multiplier 0.0 \
    --no_gather_whole_model \
    --num_evals 3 \
    --save_freq 40 \
    --gradient_checkpointing \
    --with_tracking
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JD3YEM4XGH2F2H10Y49GK441/
```