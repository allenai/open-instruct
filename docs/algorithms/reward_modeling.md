# Reward model training

`open_instruct/reward_modeling.py` contains the script for training reward models.



### About Preference dataset



### Usage


### Explanation of the logged metrics


### Implementation details

These are relevant implementation details on reward modeling:

1. Local seeds 
1. The tokenizer pads from the right
1. disable dropout in the model
1. 


### Experiment results


```bash
# LEVEL 0: interactive debugging
python -i open_instruct/reward_modeling.py \
    --dataset_name trl-internal-testing/sentiment-trl-style \
    --dataset_train_split train \
    --dataset_eval_split test \
    --model_name_or_path EleutherAI/pythia-14m \
    --chat_template simple_concat_with_space \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --num_train_epochs 1 \
    --output_dir models/rm/rm \
    --sanity_check \
    --push_to_hub

# LEVEL 1: single GPU model training; adjust your `per_device_train_batch_size` and
# `gradient_accumulation_steps` accordingly
# you can also use the `trl-internal-testing/descriptiveness-trl-style` dataset
python open_instruct/reward_modeling.py \
    --dataset_name trl-internal-testing/sentiment-trl-style \
    --dataset_train_split train \
    --dataset_eval_split test \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --chat_template simple_concat_with_space \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --num_train_epochs 1 \
    --output_dir models/rm/rm_sentiment_1b \
    --with_tracking \
    --push_to_hub \

# LEVEL 2: multi-gpu training using DS2 with the TL;DR summarization dataset
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    open_instruct/reward_modeling.py \
    --dataset_name trl-internal-testing/tldr-preference-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --model_name_or_path EleutherAI/pythia-2.8b-deduped \
    --chat_template simple_concat_with_space \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --num_train_epochs 1 \
    --output_dir models/rm/rm_tldr_2.8b \
    --bf16 \

# LEVEL 2: multi-gpu training using DS2 with the anthropic HH dataset
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    open_instruct/reward_modeling.py \
    --dataset_name trl-internal-testing/hh-rlhf-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --model_name_or_path EleutherAI/pythia-2.8b-deduped \
    --chat_template simple_chat \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_token_length 2048 \
    --max_prompt_token_lenth 1024 \
    --num_train_epochs 1 \
    --bf16 \
    --output_dir models/rm/rm_hh_2.8b \

# LEVEL 3: multi-gpu training using DS2 with the ultrafeedback dataset
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    open_instruct/rm_zephyr.py \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --chat_template zephyr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --num_train_epochs 1 \
    --bf16 \
    --output_dir models/rm/rm_zephyr_7b \
```