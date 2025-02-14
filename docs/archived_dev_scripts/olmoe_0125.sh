
# 1B full sweep
for epoch in 2; do
for lr in 2e-5 1.75e-5; do
for loss_type in sum; do
for seed in 7; do
python scripts/submit_finetune_job.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_finetune_multinode.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_7b_sft.yaml \
    --exp_name "0119_node8_finetune_olmoe_1b_epoch_${epoch}_lr_${lr}_loss_type_${loss_type}" \
    --seed $seed \
    --reduce_loss $loss_type \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
    --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
    --tokenizer_name /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf
done
done
done
done


# 1B full sweep 5e-7 
for lr in 6e-7 7e-7; do
python scripts/submit_dpo_job.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_dpo_multinode.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_7b_dpo.yaml \
    --exp_name "1203_dpo_tune${lr}" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0119_node8_finetune_olmoe_1b_epoch_2_lr_2e-5_loss_type_sum__6__1737296644 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 0119_node8_finetune_olmoe_1b_epoch_2_lr_2e-5_loss_type_sum__6__1737296644
done


for lr in 6e-7 7e-7; do
python scripts/submit_dpo_job.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_dpo_multinode.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_7b_dpo.yaml \
    --exp_name "1203_dpo_tune${lr}" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0119_node8_finetune_olmoe_1b_epoch_2_lr_2e-5_loss_type_sum__6__1737296644 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 0119_node8_finetune_olmoe_1b_epoch_2_lr_2e-5_loss_type_sum__6__1737296644
done


# sft second try  2e-5
for lr in 1.75e-5 1.5e-5 1.25e-5; do
for grad_accu_steps in 1 2; do
bsz=$((8 * 8 * 2 * $grad_accu_steps))
exp_name="0125_node8_sft_olmoe_1b_lr_${lr}_bsz_${bsz}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 8 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --config_file configs/ds_configs/deepspeed_zero2.yaml \
    --deepspeed_multinode_launcher standard \
    open_instruct/finetune.py \
    --exp_name $exp_name \
    --learning_rate $lr \
    --gradient_accumulation_steps $grad_accu_steps \
    --per_device_train_batch_size 2 \
    --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
    --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
    --tokenizer_name /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
    --use_flash_attn \
    --use_slow_tokenizer False \
    --dataset_mixer_list \
        allenai/tulu-3-sft-olmo-2-mixture 1.0 \
        ai2-adapt-dev/bespoke_stratos_17k_converted 1.0 \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir /output/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --add_bos
done
done



# on policy DPO 
for lr in 5e-7 6e-7 7e-7 8e-7 9e-7; do
exp_name="0119_node4_dpo_olmoe_1blr_${lr}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --config_file configs/ds_configs/deepspeed_zero2.yaml \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name $exp_name \
    --learning_rate $lr \
    --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0119_node8_finetune_olmoe_1b_epoch_2_lr_1.75e-5_loss_type_sum__7__1737296656 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 0119_node8_finetune_olmoe_1b_epoch_2_lr_1.75e-5_loss_type_sum__7__1737296656 \
    --use_flash_attn \
    --gradient_checkpointing \
    --dataset_mixer_list \
        allenai/tulu-3-pref-personas-instruction-following 1.0 \
        ai2-adapt-dev/wildchat_v3.9_used_on_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/wildchat_v3.9_unused_off_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/DaringAnteater-preferences-olmoe 1.0 \
        ai2-adapt-dev/uf_cleaned-olmoe 1.0 \
        ai2-adapt-dev/IF_Taxonomy-olmoe 1.0 \
        ai2-adapt-dev/sft_v3.9_used_off_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/sft_v3.9_used_on_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/Wildchat-prefs-280824-olmoe 1.0 \
    --use_slow_tokenizer false \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --use_lora false \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps 1000 \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --add_bos
done





# on policy DPO 
for lr in 5e-7 6e-7 7e-7 8e-7 9e-7; do
exp_name="0125_dpo_olmoe_1blr_${lr}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --config_file configs/ds_configs/deepspeed_zero2.yaml \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name $exp_name \
    --learning_rate $lr \
    --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0119_node8_finetune_olmoe_1b_epoch_2_lr_1.75e-5_loss_type_sum__7__1737296656 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 0119_node8_finetune_olmoe_1b_epoch_2_lr_1.75e-5_loss_type_sum__7__1737296656 \
    --use_flash_attn \
    --gradient_checkpointing \
    --dataset_mixer_list \
        allenai/tulu-3-pref-personas-instruction-following 1.0 \
        ai2-adapt-dev/wildchat_v3.9_used_on_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/wildchat_v3.9_unused_off_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/DaringAnteater-preferences-olmoe 1.0 \
        ai2-adapt-dev/uf_cleaned-olmoe 1.0 \
        ai2-adapt-dev/IF_Taxonomy-olmoe 1.0 \
        ai2-adapt-dev/sft_v3.9_used_off_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/sft_v3.9_used_on_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/Wildchat-prefs-280824-olmoe 1.0 \
    --use_slow_tokenizer false \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --use_lora false \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps 1000 \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --add_bos
done
for lr in 5e-7 6e-7 7e-7 8e-7 9e-7; do
exp_name="0125_dpo_olmoe_1blr_${lr}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --config_file configs/ds_configs/deepspeed_zero2.yaml \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name $exp_name \
    --learning_rate $lr \
    --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0125_node8_sft_olmoe_1b_lr_2e-5_bsz_128__42__1737671611 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 0125_node8_sft_olmoe_1b_lr_2e-5_bsz_128__42__1737671611 \
    --use_flash_attn \
    --gradient_checkpointing \
    --dataset_mixer_list \
        allenai/tulu-3-pref-personas-instruction-following 1.0 \
        ai2-adapt-dev/wildchat_v3.9_used_on_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/wildchat_v3.9_unused_off_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/DaringAnteater-preferences-olmoe 1.0 \
        ai2-adapt-dev/uf_cleaned-olmoe 1.0 \
        ai2-adapt-dev/IF_Taxonomy-olmoe 1.0 \
        ai2-adapt-dev/sft_v3.9_used_off_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/sft_v3.9_used_on_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/Wildchat-prefs-280824-olmoe 1.0 \
    --use_slow_tokenizer false \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --use_lora false \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps 1000 \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --add_bos
done


# reward model
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --workspace ai2/tulu-3-dev \
    --budget ai2/allennlp \
    --gpus 8 -- accelerate launch \
    --deepspeed_multinode_launcher standard \
    --num_processes 8 \
    --config_file configs/ds_configs/deepspeed_zero2.yaml open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/tulu-3-pref-personas-instruction-following": 1.0, "ai2-adapt-dev/wildchat_v3.9_used_on_policy_prompts-olmoe": 1.0, "ai2-adapt-dev/wildchat_v3.9_unused_off_policy_prompts-olmoe": 1.0, "ai2-adapt-dev/DaringAnteater-preferences-olmoe": 1.0, "ai2-adapt-dev/uf_cleaned-olmoe": 1.0, "ai2-adapt-dev/IF_Taxonomy-olmoe": 1.0, "ai2-adapt-dev/sft_v3.9_used_off_policy_prompts-olmoe": 1.0, "ai2-adapt-dev/sft_v3.9_used_on_policy_prompts-olmoe": 1.0, "ai2-adapt-dev/Wildchat-prefs-280824-olmoe": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0119_node8_finetune_olmoe_1b_epoch_2_lr_1.75e-5_loss_type_sum__7__1737296656 \
    --chat_template tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs 1 \
    --output_dir /output \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking




for lr in 5e-7 6e-7 7e-7 8e-7 9e-7; do
exp_name="0125_dpo_olmoe_1blr_${lr}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --budget ai2/oe-adapt \
    --gpus 8 -- accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    --use_deepspeed \
    --config_file configs/ds_configs/deepspeed_zero2.yaml \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    --exp_name $exp_name \
    --learning_rate $lr \
    --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0125_node8_sft_olmoe_1b_lr_2e-5_bsz_128__42__1737671611 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 0125_node8_sft_olmoe_1b_lr_2e-5_bsz_128__42__1737671611 \
    --use_flash_attn \
    --gradient_checkpointing \
    --dataset_mixer_list \
        allenai/tulu-3-pref-personas-instruction-following 1.0 \
        ai2-adapt-dev/wildchat_v3.9_used_on_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/wildchat_v3.9_unused_off_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/DaringAnteater-preferences-olmoe 1.0 \
        ai2-adapt-dev/uf_cleaned-olmoe 1.0 \
        ai2-adapt-dev/IF_Taxonomy-olmoe 1.0 \
        ai2-adapt-dev/sft_v3.9_used_off_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/sft_v3.9_used_on_policy_prompts-olmoe 1.0 \
        ai2-adapt-dev/Wildchat-prefs-280824-olmoe 1.0 \
    --use_slow_tokenizer false \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir /output \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --use_lora false \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --checkpointing_steps 1000 \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --add_bos
done




for beta in 0.01 0.02 0.03 0.07; do
exp_name="0119_ppo_olmoe_1node_${beta}_${RANDOM}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --budget ai2/oe-adapt \
    --num_nodes 1 \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --exp_name $exp_name \
    --beta $beta \
    --output_dir /weka/oe-adapt-default/allennlp/costah/models/olmoe1b0119/$exp_name \
    --try_launch_beaker_eval_jobs_on_weka \
    --try_launch_beaker_eval_jobs False \
    --dataset_mixer_list allenai/RLVR-GSM 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0119_node4_dpo_olmoe_1blr_9e-7__allenai_open_instruct_dev__42__1737727981 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision 0119_node4_dpo_olmoe_1blr_9e-7__allenai_open_instruct_dev__42__1737727981 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 16 \
    --local_rollout_batch_size 16 \
    --actor_num_gpus_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager \
    --apply_verifiable_reward true \
    --seed 3 \
    --num_evals 1000 \
    --save_freq 40 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done


for beta in 0.01 0.02 0.03 0.07; do
exp_name="0119_ppo_olmoe_rm_init_1node_${beta}_${RANDOM}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --budget ai2/oe-adapt \
    --num_nodes 1 \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --exp_name $exp_name \
    --beta $beta \
    --output_dir /weka/oe-adapt-default/allennlp/costah/models/olmoe1b0119/$exp_name \
    --try_launch_beaker_eval_jobs_on_weka \
    --try_launch_beaker_eval_jobs False \
    --dataset_mixer_list allenai/RLVR-GSM 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0119_node4_dpo_olmoe_1blr_9e-7__allenai_open_instruct_dev__42__1737727981 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1737836233 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 16 \
    --local_rollout_batch_size 16 \
    --actor_num_gpus_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager \
    --apply_verifiable_reward true \
    --seed 3 \
    --num_evals 1000 \
    --save_freq 40 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done


for beta in 0.01; do
exp_name="0119_ppo_olmoe_rm_init_4node_${beta}_${RANDOM}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --budget ai2/oe-adapt \
    --num_nodes 4 \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --exp_name $exp_name \
    --beta $beta \
    --output_dir /weka/oe-adapt-default/allennlp/costah/models/olmoe1b0119/$exp_name \
    --try_launch_beaker_eval_jobs_on_weka \
    --try_launch_beaker_eval_jobs False \
    --dataset_mixer_list allenai/RLVR-GSM 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0119_node4_dpo_olmoe_1blr_9e-7__allenai_open_instruct_dev__42__1737727981 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1737836233 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 4 \
    --local_rollout_batch_size 4 \
    --actor_num_gpus_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager \
    --apply_verifiable_reward true \
    --seed 3 \
    --num_evals 1000 \
    --save_freq 40 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done



for beta in 0.01 0.02 0.03 0.07; do
exp_name="0119_2_ppo_olmoe_rm_init_1node_${beta}_${RANDOM}"
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --budget ai2/oe-adapt \
    --num_nodes 1 \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\& uv run python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --exp_name $exp_name \
    --beta $beta \
    --output_dir /weka/oe-adapt-default/allennlp/costah/models/olmoe1b0119/$exp_name \
    --try_launch_beaker_eval_jobs_on_weka \
    --try_launch_beaker_eval_jobs False \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 0119_node4_dpo_olmoe_1blr_9e-7__allenai_open_instruct_dev__42__1737727981 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1737836233 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template_name tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 16 \
    --local_rollout_batch_size 16 \
    --actor_num_gpus_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enforce_eager \
    --apply_verifiable_reward true \
    --seed 3 \
    --num_evals 1000 \
    --save_freq 40 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done

# python mason.py \
#     --cluster ai2/jupiter-cirrascale-2 ai2/saturn-cirrascale ai2/neptune-cirrascale ai2/ceres-cirrascale \
#     --workspace ai2/tulu-3-dev \
#     --priority urgent \
#     --preemptible \
#     --budget ai2/allennlp \
#     --gpus 8 -- accelerate launch \
#     --mixed_precision bf16 \
#     --num_processes 8 \
#     --use_deepspeed \
#     --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
#     --deepspeed_multinode_launcher standard open_instruct/finetune1.py \
#     --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
#     --model_revision main \
#     --use_flash_attn \
#     --use_slow_tokenizer False \
#     --dataset_mixer_list \
#         allenai/tulu-3-sft-olmo-2-mixture 1.0 \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 2e-05 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0.0 \
#     --num_train_epochs 2 \
#     --output_dir /output/ \
#     --with_tracking \
#     --report_to wandb \
#     --logging_steps 1 \
#     --reduce_loss sum \
#     --checkpointing_steps epoch \
#     --dataset_mix_dir /output/ \
#     --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
#     --add_bos true

# # this is a lot more efficient
# python mason.py \
#     --cluster ai2/jupiter-cirrascale-2 ai2/saturn-cirrascale ai2/neptune-cirrascale ai2/ceres-cirrascale \
#     --workspace ai2/tulu-3-dev \
#     --priority urgent \
#     --preemptible \
#     --num_nodes 4 \
#     --budget ai2/allennlp \
#     --gpus 8 -- accelerate launch \
#     --config_file configs/ds_configs/deepspeed_zero2.yaml \
#     --deepspeed_multinode_launcher standard \
#     --num_processes 8 \
#     open_instruct/finetune1.py \
#     --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
#     --model_revision main \
#     --use_flash_attn \
#     --use_slow_tokenizer False \
#     --dataset_mixer_list \
#         allenai/tulu-3-sft-olmo-2-mixture 1.0 \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 2e-05 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0.0 \
#     --num_train_epochs 2 \
#     --output_dir /output/ \
#     --with_tracking \
#     --report_to wandb \
#     --logging_steps 1 \
#     --reduce_loss sum \
#     --dataset_mix_dir /output/ \
#     --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
#     --add_bos true




# python -i open_instruct/finetune1.py \
#     --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
#     --model_revision main \
#     --use_flash_attn False \
#     --use_slow_tokenizer False \
#     --dataset_mixer_list \
#         allenai/tulu-3-sft-olmo-2-mixture 1.0 \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 1.0e-05 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0.0 \
#     --num_train_epochs 2 \
#     --output_dir /output/ \
#     --with_tracking \
#     --report_to wandb \
#     --logging_steps 1 \
#     --reduce_loss sum \
#     --checkpointing_steps epoch \
#     --dataset_mix_dir /output/ \
#     --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
#     --add_bos true


# python -i open_instruct/dpo_tune_cache1.py \
#     --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
#     --model_revision main \
#     --use_flash_attn False \
#     --use_slow_tokenizer False \
#     --dataset_mixer_list \
#         allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 1.0e-05 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0.0 \
#     --num_train_epochs 2 \
#     --output_dir /output/ \
#     --with_tracking \
#     --report_to wandb \
#     --logging_steps 1 \
#     --reduce_loss sum \
#     --checkpointing_steps epoch \
#     --dataset_mix_dir /output/ \
#     --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
#     --add_bos true



# python -i open_instruct/finetune.py \
#     --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
#     --model_revision main \
#     --use_flash_attn False \
#     --use_slow_tokenizer False \
#     --dataset_mixer_list \
#         allenai/tulu-3-sft-olmo-2-mixture 1.0 \
#         \ # General datasets: \
#         ai2-adapt-dev/oasst1_converted  1.0 \
#         ai2-adapt-dev/flan_v2_converted  1.0 \
#         ai2-adapt-dev/tulu_hard_coded_repeated_10  1.0 \
#         ai2-adapt-dev/no_robots_converted  1.0 \
#         ai2-adapt-dev/tulu_v3.9_wildchat_100k  1.0 \
#         \ # Math datasets  \
#         ai2-adapt-dev/personahub_math_v5_regen_149960  1.0 \
#         allenai/tulu-3-sft-personas-math-grade  1.0 \
#         ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k  1.0  \
#         ai2-adapt-dev/numinamath_tir_math_decontaminated  1.0 \
#         ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k  1.0 \
#         \ # Coding datasets  \
#         ai2-adapt-dev/personahub_code_v2_34999  1.0 \
#         ai2-adapt-dev/evol_codealpaca_heval_decontaminated  1.0 \
#         \ # IF datasets  \
#         ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980  1.0 \
#         \ # Safety datasets  \
#         ai2-adapt-dev/coconot_converted  1.0 \
#         ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k  1.0 \
#         \ # Specialty datasets  \
#         ai2-adapt-dev/tulu_v3.9_sciriff_10k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_table_gpt_5k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_aya_100k  1.0 \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 1.0e-05 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0.0 \
#     --num_train_epochs 2 \
#     --output_dir /output/ \
#     --with_tracking \
#     --report_to wandb \
#     --logging_steps 1 \
#     --reduce_loss sum \
#     --checkpointing_steps epoch \
#     --dataset_mix_dir /output/ \
#     --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
#     --add_bos true


# python -i open_instruct/finetune.py \
#     --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
#     --model_revision main \
#     --use_flash_attn False \
#     --use_slow_tokenizer False \
#     --dataset_mixer_list \
#         ai2-adapt-dev/oasst1_converted  1.0 \
#         ai2-adapt-dev/flan_v2_converted  1.0 \
#         ai2-adapt-dev/tulu_hard_coded_repeated_10  1.0 \
#         ai2-adapt-dev/no_robots_converted  1.0 \
#         ai2-adapt-dev/tulu_v3.9_wildchat_100k  1.0 \
#         ai2-adapt-dev/personahub_math_v5_regen_149960  1.0 \
#         allenai/tulu-3-sft-personas-math-grade  1.0 \
#         ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k  1.0  \
#         ai2-adapt-dev/numinamath_tir_math_decontaminated  1.0 \
#         ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k  1.0 \
#         ai2-adapt-dev/personahub_code_v2_34999  1.0 \
#         ai2-adapt-dev/evol_codealpaca_heval_decontaminated  1.0 \
#         ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980  1.0 \
#         ai2-adapt-dev/coconot_converted  1.0 \
#         ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_sciriff_10k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_table_gpt_5k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_aya_100k  1.0 \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 1.0e-05 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0.0 \
#     --num_train_epochs 2 \
#     --output_dir /output/ \
#     --with_tracking \
#     --report_to wandb \
#     --logging_steps 1 \
#     --reduce_loss sum \
#     --checkpointing_steps epoch \
#     --dataset_mix_dir /output/ \
#     --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
#     --add_bos true

# python -i open_instruct/finetune1.py \
#     --model_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/OLMoE/a0125/olmoe-8x1b-newhp-newds-dolmino-soup-seed-42-43-44/step23842-hf \
#     --model_revision main \
#     --use_flash_attn False \
#     --use_slow_tokenizer False \
#     --dataset_mixer_list \
#         ai2-adapt-dev/oasst1_converted  1.0 \
#         ai2-adapt-dev/flan_v2_converted  1.0 \
#         ai2-adapt-dev/tulu_hard_coded_repeated_10  1.0 \
#         ai2-adapt-dev/no_robots_converted  1.0 \
#         ai2-adapt-dev/tulu_v3.9_wildchat_100k  1.0 \
#         ai2-adapt-dev/personahub_math_v5_regen_149960  1.0 \
#         allenai/tulu-3-sft-personas-math-grade  1.0 \
#         ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k  1.0  \
#         ai2-adapt-dev/numinamath_tir_math_decontaminated  1.0 \
#         ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k  1.0 \
#         ai2-adapt-dev/personahub_code_v2_34999  1.0 \
#         ai2-adapt-dev/evol_codealpaca_heval_decontaminated  1.0 \
#         ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980  1.0 \
#         ai2-adapt-dev/coconot_converted  1.0 \
#         ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_sciriff_10k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_table_gpt_5k  1.0 \
#         ai2-adapt-dev/tulu_v3.9_aya_100k  1.0 \
#     --max_seq_length 4096 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 1.0e-05 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0.0 \
#     --num_train_epochs 2 \
#     --output_dir /output/ \
#     --with_tracking \
#     --report_to wandb \
#     --logging_steps 1 \
#     --reduce_loss sum \
#     --checkpointing_steps epoch \
#     --dataset_mix_dir /output/ \
#     --hf_metadata_dataset allenai/2025-1-olmoe-instruct-evals \
#     --add_bos true


# python -i open_instruct/dpo_tune_cache1.py \
#     --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
#     --use_flash_attn False \
#     --tokenizer_name allenai/Llama-3.1-Tulu-3-8B-SFT \
#     --max_seq_length 2048 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 5e-07 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.1 \
#     --weight_decay 0.0 \
#     --num_train_epochs 1 \
#     --output_dir output/dpo_8b \
#     --with_tracking \
#     --report_to wandb \
#     --logging_steps 1 \
#     --model_revision main \
#     --gradient_checkpointing \
#     --dataset_mixer_list \
#         allenai/ultrafeedback_binarized_cleaned_train 1.0 \
#         ai2-adapt-dev/DaringAnteater-prefs-RM-filter 1.0 \
#         ai2-adapt-dev/WildChat-prefs-280824 1.0 \
#         ai2-adapt-dev/nectar_binarized-anthropic-hh 1.0 \
#     --use_slow_tokenizer \
#     --use_lora False \
#     --dpo_loss_type dpo_norm \
#     --dpo_beta 5 \
#     --checkpointing_steps 1000 \
#     --exp_name tulu-3-8b-dpo
