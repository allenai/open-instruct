python mason.py \
    --cluster  ai2/pluto-cirrascale ai2/jupiter-cirrascale-2 \
    --priority high \
    --preemptible \
    --budget ai2/oe-adapt   \
    --image costah/open_instruct_rm \
    --pure_docker_mode \
    --no_mount_nfs \
    --no_hf_cache_env \
    --gpus 8 -- accelerate launch --config_file configs/ds_configs/deepspeed_zero3.yaml \
    open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned_train": 61814, "ai2-adapt-dev/DaringAnteater-prefs-RM-filter": 1618, "ai2-adapt-dev/WildChat-prefs-280824": 11487}' \
    --dataset_train_splits train train train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --chat_template tulu \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --num_train_epochs 1 \
    --output_dir /output \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking

python mason.py \
    --cluster  ai2/pluto-cirrascale ai2/jupiter-cirrascale-2 \
    --priority high \
    --preemptible \
    --budget ai2/oe-adapt   \
    --image costah/open_instruct_rm \
    --pure_docker_mode \
    --no_mount_nfs \
    --no_hf_cache_env \
    --beaker_datasets /model:jacobm/L3.18B-base_rs_L3.18BI-static-valpy_dpo \
    --gpus 8 -- accelerate launch --config_file configs/ds_configs/deepspeed_zero3.yaml \
    open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned_train": 61814, "ai2-adapt-dev/DaringAnteater-prefs-RM-filter": 1618, "ai2-adapt-dev/WildChat-prefs-280824": 11487}' \
    --dataset_train_splits train train train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path /model \
    --chat_template tulu \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --num_train_epochs 1 \
    --output_dir /output \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking

