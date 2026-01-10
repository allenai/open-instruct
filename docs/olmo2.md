# OLMo 2 Commands

Here we'll add commands and references to the training runs of OLMo 2.
We'll prioritize the smaller models where more people are hoping to study and reproduce them.

Core to training OLMo models (version 1 and 2) at least are to include the following flags: `--add_bos` and `--use_slow_tokenizer False` because of the tokenizer used.

For more details on how to convert these to standard launch commands (without ai2 `mason.py`) see the `tulu3.md` docs.

The time taken for each stage of training was approximately:
1. SFT: 9 hours, 1 H100 node (8 GPUs).
2. DPO: 2 hours, 1 H100 node.
3. RLVR: 43 hours per stage for 2 million episodes on 2 H100 nodes (16 GPUs), totalling around 86 hours.

## Instruction Finetuning

### 1B

We ran training for the 1B model in SFT on 1 node of 8 NVIDIA H100 GPUs.

The command used internally is:
```
python mason.py \
    --cluster ai2/augusta \
    --workspace ai2/olmo-instruct \
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
    open_instruct/finetune.py \
    --exp_name olmo2_1b_sft \
    --model_name_or_path allenai/OLMo-2-0425-1B \
    --model_revision main \
    --tokenizer_name allenai/OLMo-2-1124-7B \
    --tokenizer_revision main \
    --use_slow_tokenizer False \
    --add_bos \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
    --use_flash_attn \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 1
```
Which reduces to roughly:
```
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name olmo2_1b_v2_sft_lr3e-5_seed1  \
    --model_name_or_path allenai/OLMo-2-0425-1B \
    --model_revision main \
    --tokenizer_name allenai/OLMo-2-0425-1B \
    --tokenizer_revision main \
    --use_slow_tokenizer False \
    --add_bos \
    --chat_template_name tulu \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
    --use_flash_attn \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --report_to wandb \
    --with_tracking \
    --logging_steps 1 \
    --seed 1
```
For those internal to Ai2, see the [wandb logs](https://wandb.ai/ai2-llm/open_instruct_internal/runs/532v35jn/overview) or the [beaker job](https://beaker.allen.ai/orgs/ai2/workspaces/olmo-instruct/work/01JS4Q5QYDVAJE6XKKR4FGVQZ5?taskId=01JS4Q5QYJFHBKH3X47MPNB7P4&jobId=01JS4Q5R38CRQV0WK4J6494Q4Q).

## Preference Tuning (DPO)

### 1B

We ran training for the 1B model in DPO on 1 node of 8 NVIDIA H100 GPUs.
The command reduces to:
```
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name 0424_1B_dpo_onpol_lr_2.5e-6_seed_111 \
    --learning_rate 2.5e-6 \
    --seed 111 \
    --model_name_or_path allenai/OLMo-2-0425-1B-SFT \
    --model_revision main \
    --use_flash_attn \
    --tokenizer_name_or_path allenai/OLMo-2-0425-1B-SFT \
    --tokenizer_revision main \
    --max_seq_length 2048 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --gradient_checkpointing \
    --dataset_mixer_list allenai/olmo-2-0425-1b-preference-mix 1.0 \
    --use_slow_tokenizer False \
    --add_bos \
    --chat_template_name tulu \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5
```

For those internal to Ai2, see the [wandb logs](https://wandb.ai/ai2-llm/open_instruct_internal/runs/bcu4arvs/overview) or the [beaker job](https://beaker.allen.ai/orgs/ai2/workspaces/olmo-instruct/work/01JSMRC1TR1Q4MV7NY8WFSR4SA?).


Example with DeepSpeed Stage 2:
```
python mason.py \
    --cluster ai2/augusta \
    --workspace ai2/olmo-instruct \
    --priority urgent \
    --image nathanl/open_instruct_auto --pure_docker_mode \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name "0424_1B_dpo_onpol_lr_2.5e-6_seed_111" \
    --learning_rate 2.5e-6 \
    --seed 111 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision "olmo2_1b_v2_sft_lr3e-5_seed1__1__1744989064" \
    --use_flash_attn \
    --tokenizer_name_or_path allenai/OLMo-2-1124-13B \
    --tokenizer_revision main \
    --max_seq_length 2048 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --gradient_checkpointing \
    --dataset_mixer_list \
        allenai/olmo-2-1b-pref-mix-v0 1.0 \
    --use_slow_tokenizer False \
    --add_bos \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5
```

Example run with DeepSpeed Stage 3 (slower than stage 2):
```
for lr in 2e-6; do
python mason.py \
    --cluster ai2/jupiter \
    --workspace ai2/olmo-instruct \
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
    open_instruct/dpo_tune_cache.py \
    --exp_name "0421_1B_dpo_lr_${lr}" \
    --learning_rate $lr \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision "olmo2_1b_v2_sft_lr3e-5_seed1__1__1744989064" \
    --use_flash_attn \
    --tokenizer_name_or_path allenai/OLMo-2-1124-13B \
    --tokenizer_revision main \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --gradient_checkpointing \
    --dataset_mixer_list \
        allenai/olmo-2-32b-pref-mix-v0-filter-datecutoff 1.0 \
    --use_slow_tokenizer False \
    --add_bos \
    --use_lora False \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --add_bos \
    --use_slow_tokenizer False
done
```

## RLVR

### 1B

The 1B OLMo 2 model has two RL stages run in sequence. The first is on MATH, GSM8K, and IF constraints:
```
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 
python open_instruct/grpo_fast.py \
    --add_bos \
    --apply_verifiable_reward true \
    --beta 0.01 \
    --chat_template_name tulu \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --deepspeed_stage 2 \
    --deepspeed_zpg 12 \
    --exp_name 0423_grpo_seed_1_lr_5e-7 \
    --gradient_checkpointing  \
    --kl_estimator 3 \
    --learning_rate 5e-7 \
    --local_eval_every 5 \
    --lr_scheduler_type constant \
    --max_prompt_token_length 2048 \
    --model_name_or_path allenai/OLMo-2-0425-1B-DPO \
    --model_revision main \
    --non_stop_penalty \
    --non_stop_penalty_value 0.0 \
    --num_epochs 1 \
    --num_learners_per_node 4 8 \
    --num_mini_batches 2 \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 48 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --remap_verifier ifeval=ifeval_old \
    --response_length 2048 \
    --save_freq 200 \
    --seed 1 \
    --stop_strings "</answer>" \
    --temperature 1.0 \
    --tokenizer_name_or_path allenai/OLMo-2-0425-1B-DPO \
    --tokenizer_revision main \
    --total_episodes 2000000 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 4 \
    --with_tracking
```
For those internal to Ai2, see the [wandb logs](https://wandb.ai/ai2-llm/open_instruct_internal/runs/80rvltbs/overview) or the [beaker job](https://beaker.allen.ai/orgs/ai2/workspaces/olmo-instruct/work/01JSPEYF1PGPNYGQ4NBEZPJA4W?taskId=01JSPEYF1S9EJHBG1ZS6ZXMPRA&jobId=01JSPEYF6JFZHCZRBCZSZSEM8T).

Next, on MATH only:
```
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 
python open_instruct/grpo_fast.py \
    --add_bos \
    --apply_verifiable_reward true \
    --beta 0.01 \
    --chat_template_name tulu \
    --dataset_mixer_eval_list allenai/RLVR-MATH 16 \
    --dataset_mixer_eval_list_splits train \
    --dataset_mixer_list allenai/RLVR-MATH 1.0 \
    --dataset_mixer_list_splits train \
    --deepspeed_stage 2 \
    --deepspeed_zpg 12 \
    --exp_name 0427_grpo_seed_1_lr_5e-7 \
    --gradient_checkpointing  \
    --kl_estimator 3 \
    --learning_rate 5e-7 \
    --local_eval_every 5 \
    --lr_scheduler_type constant \
    --max_prompt_token_length 2048 \
    --model_name_or_path allenai/OLMo-2-0425-1B-RLVR1 \
    --model_revision main \
    --non_stop_penalty \
    --non_stop_penalty_value 0.0 \
    --num_epochs 1 \
    --num_learners_per_node 4 8 \
    --num_mini_batches 2 \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 48 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --response_length 2048 \
    --save_freq 200 \
    --save_traces True \
    --seed 1 \
    --stop_strings "</answer>" \
    --temperature 1.0 \
    --tokenizer_name_or_path allenai/OLMo-2-0425-1B-RLVR1 \
    --tokenizer_revision main \
    --total_episodes 2000000 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 4 \
    --with_tracking
```
For those internal to Ai2, see the [wandb logs](https://wandb.ai/ai2-llm/open_instruct_internal/runs/25yfin0f?nw=nwusernatolambert) or the [beaker job](https://beaker.allen.ai/orgs/ai2/workspaces/olmo-instruct/work/01JSWFCG62FC4NEDEW8YZDECXV?taskId=01JSWFCG64DSXNJDV7N23A92HG&jobId=01JSWFCGBFAQ288RSMAQF0TYS7).
