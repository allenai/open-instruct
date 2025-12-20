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
    --model_revision main \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --checkpointing_steps epoch \
    --dataset_mix_dir output/sft_8b \
    --exp_name tulu-3-8b-sft \
    --seed 123
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JBNTPW8TKG09B2XR832YB5S8
```

> [!NOTE]
> If you have different number of GPUs, please adjust the `NUM_MACHINES`, `NUM_PROCESSES`, `PER_DEVICE_TRAIN_BATCH_SIZE`, and `GRADIENT_ACCUMULATION_STEPS` accordingly to reproduce the same effective batch size.
> The effective batch size is calculated by multiplying:
> - Number of GPUs / processes (NUM_PROCESSES)
> - Train batch size per GPU (PER_DEVICE_TRAIN_BATCH_SIZE)
> - Gradient accumulation steps (GRADIENT_ACCUMULATION_STEPS)
> so we have
> ```
> 64 GPUs: 64 * 1 * 2 = 128 # from the example above
> 8 GPUs:   8 * 1 * 16 = 128 # if you only
> ```
> You can achieve the same effective batch size with fewer GPUs by increasing gradient accumulation steps proportionally (e.g., `NUM_PROCESSES=8, PER_DEVICE_TRAIN_BATCH_SIZE=1, and GRADIENT_ACCUMULATION_STEPS=16`)

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

### Llama-3.1-Tulu-3-405B-DPO Reproduction

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-405B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-405B-DPO)


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
accelerate launch --mixed_precision bf16 \
    --num_machines 32 \
    --num_processes 256 \
    --machine_rank $BEAKER_REPLICA_RANK \
    --main_process_ip $BEAKER_LEADER_REPLICA_HOSTNAME \
    --main_process_port 29400 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/dpo_tune_cache.py \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-405B-SFT \
    --tokenizer_name allenai/Llama-3.1-Tulu-3-70B-SFT \
    --use_flash_attn \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-07 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir output_405b \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --model_revision main \
    --gradient_checkpointing \
    --dataset_mixer_list ai2-adapt-dev/405b_preference_mix 1.0 \
    --use_slow_tokenizer \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps 1000
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JJ4QRZ31SH79AHVM6WWDVJB4/
```


## RLVR

### RLVR for IF Note:
We have since updated the RLVR verifier functions and judge for precise IF. If you want to reproduce Tulu3 results,
please use the IFEvalVerifierOld class in ground_truth_utils.py. The new IFEvalVerifier class is not compatible with
the old data format, so please use the new IF data format for the new verifier. The new verifier and the new data will
give better results.

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
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
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
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
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
    --chat_template_name tulu \
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

### Llama-3.1-Tulu-3-405B Reproduction

This is (almost) the exact command which produced [allenai/Llama-3.1-Tulu-3-405B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-405B)

Couple of notes:
* We had to set `TORCH_NCCL_ENABLE_MONITORING=0` to turn off NCCL heartbeat monitoring and avoid timeouts. Feel free to remove this.
* Make sure to modify `configs/beaker_configs/ray_node_setup.sh` in our own cluster setup. The idea is to have the replicas join the main machines via `ray`.
* Here the effective batch size is `sum(actor_num_gpus_per_node) * local_mini_batch_size = 40 * 16 = 640`. If you have less GPUs, you can adjust `actor_num_gpus_per_node` and `local_mini_batch_size` accordingly.

```bash
TORCH_NCCL_ENABLE_MONITORING=0 python mason.py \
    --cluster ai2/jupiter --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 32 \
    --image nathanl/open_instruct_auto \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& TORCH_DISTRIBUTED_DEBUG=DETAIL python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer_list allenai/RLVR-MATH 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-MATH 128 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 1024 \
    --model_name_or_path /weka/oe-adapt-default/hamishi/405b_dpo_v4 \
    --exp_name "405b_rlvr_math_only_8b_valu_on_v4" \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --beta 0.05 \
    --output_dir "/weka/oe-adapt-default/hamishi/405b_rlvr_math_only_8b_valu_on_v4" \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template tulu \
    --learning_rate 1e-7 \
    --total_episodes 400000 \
    --num_epochs 4 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --local_rollout_forward_batch_size 1 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 16 \
    --vllm_enforce_eager true \
    --apply_verifiable_reward true \
    --reward_model_multiplier 0.0 \
    --no_gather_whole_model \
    --seed 3 \
    --num_evals 3 \
    --no_try_launch_beaker_eval_jobs \
    --save_freq 25 \
    --try_launch_beaker_eval_jobs_on_weka \
    --gradient_checkpointing \
    --with_tracking
# For Ai2 internal members, this was the experiment URL: https://beaker.org/ex/01JJA31S20XAFR82YPFKSMMYZV/
```


### (NEW) Llama-3.1-Tulu-3.1-8B Reproduction

This is the exact command which produced [allenai/Llama-3.1-Tulu-3.1-8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B), which uses 2 nodes (16 GPUs)


```bash
for learning_rate in 5e-7; do
for beta in 0.01; do
for nspp in 16; do
for m in half-m ; do
for kl_estimator in kl3; do
local_rollout_batch_size=4
if [ $m == "half-m" ]; then
    local_mini_batch_size=$(($local_rollout_batch_size * $nspp / 2))
else
    local_mini_batch_size=$(($local_rollout_batch_size * $nspp))
fi
exp_name="0204_lr_scan_grpo_math_lr_${learning_rate}_${kl_estimator}_${beta}_${nspp}_${m}_${RANDOM}"
full_bsz=$(($local_rollout_batch_size * nspp * (7) * 2))
echo $exp_name:
echo --- local_mini_batch_size=$local_mini_batch_size
echo --- full_bsz=$full_bsz
echo --- num_gradient_updates=$(($local_rollout_batch_size * $nspp / $local_mini_batch_size))
python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --num_nodes 2 \
    --max_retries 1 \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run python open_instruct/grpo_vllm_thread_ray_gtrl.py \
    --exp_name $exp_name \
    --beta $beta \
    --local_mini_batch_size $local_mini_batch_size \
    --number_samples_per_prompt $nspp \
    --output_dir /weka/oe-adapt-default/costah/models/$exp_name \
    --local_rollout_batch_size $local_rollout_batch_size \
    --kl_estimator $kl_estimator \
    --learning_rate $learning_rate \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 10000000 \
    --penalty_reward_value 0.0 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --actor_num_gpus_per_node 4 8 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 4 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --num_evals 30 \
    --save_freq 40 \
    --reward_model_multiplier 0.0 \
    --no_try_launch_beaker_eval_jobs \
    --try_launch_beaker_eval_jobs_on_weka \
    --gradient_checkpointing \
    --with_tracking
done
done
done
done
done
# For Ai2 internal members, this was the experiment URL: https://beaker.allen.ai/orgs/ai2/workspaces/tulu-3-dev/work/01JKA7CSDGG3YA84X89C5HJPXR?taskId=01JKA7CSDQMVBDNAWF5T7ZXDSA&jobId=01JKH4KYJTR2Y2NYNCCQ63ZQHE
```


If you are running on a single node (8 GPUs), consider adjusting the commands as follows. Basically, the idea is to simulate the same batch size. In the two nodes setup, we used `--actor_num_gpus_per_node 4 8` (12 GPUs) for training, so we multiply it with `local_rollout_batch_size=4` to get the rollout batch size `12 * 4 = 48`. Now assume we used `--actor_num_gpus_per_node 6` (6 GPUs) for training, so we get `48 / 6 = 8`, which is the new `local_rollout_batch_size`.

```diff
 for learning_rate in 5e-7; do
 for beta in 0.01; do
 for nspp in 16; do
 for m in half-m ; do
 for kl_estimator in kl3; do
-local_rollout_batch_size=4
+local_rollout_batch_size=8
 if [ $m == "half-m" ]; then
     local_mini_batch_size=$(($local_rollout_batch_size * $nspp / 2))
 else
     local_mini_batch_size=$(($local_rollout_batch_size * $nspp))
 fi
 exp_name="0204_lr_scan_grpo_math_lr_${learning_rate}_${kl_estimator}_${beta}_${nspp}_${m}_${RANDOM}"
 full_bsz=$(($local_rollout_batch_size * nspp * (7) * 2))
 echo $exp_name:
 echo --- local_mini_batch_size=$local_mini_batch_size
 echo --- full_bsz=$full_bsz
 echo --- num_gradient_updates=$(($local_rollout_batch_size * $nspp / $local_mini_batch_size))
 python mason.py \
     --cluster ai2/jupiter \
     --workspace ai2/tulu-3-dev \
     --priority high \
     --preemptible \
     --num_nodes 2 \
     --max_retries 1 \
     --budget ai2/oe-adapt \
     --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run python open_instruct/grpo_vllm_thread_ray_gtrl.py \
     --exp_name $exp_name \
     --beta $beta \
     --local_mini_batch_size $local_mini_batch_size \
     --number_samples_per_prompt $nspp \
     --output_dir /weka/oe-adapt-default/costah/models/$exp_name \
     --local_rollout_batch_size $local_rollout_batch_size \
     --kl_estimator $kl_estimator \
     --learning_rate $learning_rate \
     --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
     --dataset_mixer_list_splits train \
     --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
     --dataset_mixer_eval_list_splits train \
     --max_token_length 2048 \
     --max_prompt_token_length 2048 \
     --response_length 2048 \
     --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
     --non_stop_penalty \
     --stop_token eos \
     --temperature 1.0 \
     --chat_template_name tulu \
     --total_episodes 10000000 \
     --penalty_reward_value 0.0 \
-    --deepspeed_stage 2 \
+    --deepspeed_stage 3 \
     --per_device_train_batch_size 2 \
     --local_rollout_forward_batch_size 2 \
-    --actor_num_gpus_per_node 4 8 \
+    --actor_num_gpus_per_node 6 \
     --num_epochs 1 \
-    --vllm_tensor_parallel_size 4 \
+    --vllm_tensor_parallel_size 2 \
     --lr_scheduler_type constant \
     --apply_verifiable_reward true \
     --seed 1 \
     --num_evals 30 \
     --save_freq 40 \
     --reward_model_multiplier 0.0 \
     --no_try_launch_beaker_eval_jobs \
     --try_launch_beaker_eval_jobs_on_weka \
     --gradient_checkpointing \
     --with_tracking
 done
 done
 done
 done
 done
```
