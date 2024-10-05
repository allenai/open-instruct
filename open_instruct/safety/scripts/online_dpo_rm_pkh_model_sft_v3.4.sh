python /net/nfs.cirrascale/mosaic/nouhad/projects/open-instruct/mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --image nathanl/open_instruct_auto \
    --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority high \
    --preemptible \
    --budget ai2/allennlp \
    --beaker_datasets /model:01J7S9H21G341DQSPAAMMT2WYS  \
    --gpus 8 -- accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml \
    open_instruct/online_dpo_vllm_thread.py \
    --exp_name "safety_online_dpo_vllm_thread_beta_0.03_rm_pkh_sft_v3.4" \
    --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned_train": 1.0, "ai2-adapt-dev/DaringAnteater-prefs-RM-filter": 1.0, "ai2-adapt-dev/only_wildchat_aug28_regenerated_llama": 1.0, "ai2-adapt-dev/PKU-SafeRLHF-processed": 1.0}' \
    --sft_messages_key chosen \
    --dataset_train_splits train \
    --dataset_eval_splits train \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --learning_rate 8e-7 \
    --output_dir /output/ \
    --chat_template tulu \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 1 \
    --vllm_device cuda:7 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 300000 \
    --model_name_or_path /model \
    --reward_model_path PKU-Alignment/beaver-7b-v3.0-reward \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.03 \
    --num_evals 3 \
    --response_length 1024 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub