
# 7B full sweep
for epoch in 2; do
for lr in 1e-5; do
for loss_type in sum; do
for seed in 3 4; do
python scripts/submit_finetune_job.py \
    --cluster ai2/jupiter \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_finetune_multinode_olmo2_1124.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_7b_sft.yaml \
    --exp_name "1206_finetune_epoch_${epoch}_lr_${lr}_loss_type_${loss_type}" \
    --seed $seed \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --hf_metadata_dataset allenai/olmo-instruct-evals
done
done
done
done

# 7B DPO sweep 5e-7 6e-7 7e-7 8e-7
for lr in 1e-6 2e-6 3e-6; do
python scripts/submit_dpo_job.py \
    --cluster ai2/jupiter \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_dpo_multinode_olmo2_1124.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_7b_dpo.yaml \
    --exp_name "1203_dpo_tune${lr}" \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1203_finetune_epoch_2_lr_2e-5_loss_type_sum__2__1733436116 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 1203_finetune_epoch_2_lr_2e-5_loss_type_sum__2__1733436116 \
    --learning_rate $lr \
    --hf_metadata_dataset allenai/olmo-instruct-evals
done

# 13B full sweep
for epoch in 2; do
for lr in 7e-6 7.25e-6 7.5e-6 8e-6; do
for loss_type in sum; do
python scripts/submit_finetune_job.py \
    --cluster ai2/jupiter \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_finetune_multinode_olmo2_1124.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_13b_sft.yaml \
    --exp_name "1209_bsz128_13b_finetune_epoch_${epoch}_lr_${lr}_loss_type_${loss_type}" \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --hf_metadata_dataset allenai/olmo-instruct-evals
done
done
done


for epoch in 2; do
for lr in 6e-6 7.5e-6 8e-6; do
for loss_type in sum; do
python scripts/submit_finetune_job.py \
    --cluster ai2/jupiter \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_finetune_multinode_olmo2_1124.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_13b_sft.yaml \
    --exp_name "1208_2_bsz64_13b_finetune_epoch_${epoch}_lr_${lr}_loss_type_${loss_type}" \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --hf_metadata_dataset allenai/olmo-instruct-evals
done
done
done
done


# 13B DPO sweep
for lr in 5e-7 6e-7 7e-7 8e-7; do
python scripts/submit_dpo_job.py \
    --cluster ai2/jupiter \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_dpo_multinode_olmo2_1124.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_13b_dpo.yaml \
    --exp_name "1203_dpo_13b_tune${lr}" \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1203_13b_finetune_epoch_2_lr_2e-5_loss_type_sum__1__1733461885 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 1203_13b_finetune_epoch_2_lr_2e-5_loss_type_sum__1__1733461885 \
    --learning_rate $lr \
    --hf_metadata_dataset allenai/olmo-instruct-evals
done

# 1203_13b_finetune_epoch_2_lr_2e-5_loss_type_sum__1__1733461885

#     --model_revision "" \
#     --dataset_mixer_list allenai/olmo-2-1124-7b-preference-mix 1.0

python scripts/submit_eval_jobs.py \
    --model_name hf-shanearora/i-am-a-good-open-base-model \
    --cluster ai2/jupiter \
    --location shanearora/i-am-a-good-open-base-model  \
    --workspace tulu-3-results \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image nathanl/open_instruct_auto \
    --upload_to_hf allenai/olmo-instruct-evals \
    --run_oe_eval_experiments \
    --skip_oi_evals \
    --priority high \
    --evaluate_on_weka


python scripts/submit_eval_jobs.py \
    --model_name 1203_13b_gsm_math_if_beta_0.1_lr_4e-7_16858__52__1733711388 \
    --location /weka/oe-adapt-default/costah/models/olmo1124/1203_13b_gsm_math_if_beta_0.1_lr_4e-7_16858 \
    --cluster ai2/saturn ai2/neptune \
    --is_tuned \
    --workspace "tulu-3-results" \
    --priority high \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image "nathanl/open_instruct_auto" \
    --upload_to_hf allenai/olmo-instruct-evals \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --skip_oi_evals

python scripts/submit_eval_jobs.py \
    --model_name  \
    --location {step_dir} \
    --cluster ai2/saturn ai2/neptune \
    --is_tuned \
    --workspace "tulu-3-results" \
    --priority high \
    --preemptible \
    --use_hf_tokenizer_template \
    --beaker_image "nathanl/open_instruct_auto" \
    --upload_to_hf allenai/tulu-3-evals \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_safety_evaluations \
    --skip_oi_evals


python mason.py \
    --image costah/open_instruct_ppo_olmo22 --pure_docker_mode \
    --cluster ai2/jupiter \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --workspace ai2/tulu-3-dev \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& accelerate launch \
    --deepspeed_multinode_launcher standard \
    --num_machines 4 \
    --num_processes 8 \
    --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/olmo-2-1124-7b-preference-mix": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1203_finetune_epoch_2_lr_2e-5_loss_type_sum__2__1733436116 \
    --chat_template tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs 1 \
    --output_dir models/rm/rm_tulu_8b \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking

python mason.py \
    --image costah/open_instruct_ppo_olmo22 --pure_docker_mode \
    --cluster ai2/jupiter \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --workspace ai2/tulu-3-dev \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& accelerate launch \
    --deepspeed_multinode_launcher standard \
    --num_machines 4 \
    --num_processes 8 \
    --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/olmo-2-1124-13b-preference-mix": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1203_13b_finetune_epoch_2_lr_2e-5_loss_type_sum__1__1733461885 \
    --chat_template tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs 1 \
    --output_dir models/rm/rm_tulu_8b \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking



# 7B RL
for beta in 0.03 0.05 0.07 0.1; do
for lr in 4e-7 5e-7; do
for seed in 1; do
exp_name="1203_7b_lrm_gsm_math_if_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo22 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo1124/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1203_dpo_tune1e-6__allenai_open_instruct_dev__42__1733526514 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733534933 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 100000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done


# 13B RL
for beta in 0.03 0.05 0.07 0.1; do
for lr in 4e-7; do
for seed in 52; do
exp_name="1203_13b_gsm_math_if_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo22 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo1124/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1203_dpo_13b_tune8e-7__allenai_open_instruct_dev__1234__1733526819 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733534933 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done






# 7B DPO sweep
for lr in 6e-7 7e-7 8e-7 9e-7 1e-6 2e-6; do
python scripts/submit_dpo_job.py \
    --cluster ai2/jupiter \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_dpo_multinode_olmo2_1124.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_7b_dpo.yaml \
    --exp_name "1206_dpo_7b_tune${lr}" \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1206_finetune_epoch_2_lr_1e-5_loss_type_sum__4__1733525407 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 1206_finetune_epoch_2_lr_1e-5_loss_type_sum__4__1733525407 \
    --seed $seed \
    --hf_metadata_dataset allenai/olmo-instruct-evals
done
done

# 7b RM
python mason.py \
    --image costah/open_instruct_ppo_olmo23 --pure_docker_mode \
    --cluster ai2/jupiter \
    --priority urgent \
    --preemptible \
    --num_nodes 2 \
    --workspace ai2/tulu-3-dev \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& accelerate launch \
    --deepspeed_multinode_launcher standard \
    --num_machines 2 \
    --num_processes 8 \
    --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/olmo-2-1124-7b-preference-mix": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1206_finetune_epoch_2_lr_1e-5_loss_type_sum__4__1733525407 \
    --chat_template_name tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs 1 \
    --output_dir models/rm/rm_tulu_8b \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking




# 13B DPO sweep 8e-7
for lr in 8.5e-7 9e-7 9.5e-7 1e-6 1.5e-6 2e-6 ; do
python scripts/submit_dpo_job.py \
    --cluster ai2/jupiter \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config configs/beaker_configs/default_dpo_multinode_olmo2_1124.yaml \
    --config configs/train_configs/olmo2/olmo2_1124_13b_dpo.yaml \
    --exp_name "1208_dpo_13b_tune${lr}" \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1208_bsz64_13b_finetune_epoch_2_lr_5e-6_loss_type_sum__1__1733711678 \
    --tokenizer_name allenai/open_instruct_dev \
    --tokenizer_revision 1208_bsz64_13b_finetune_epoch_2_lr_5e-6_loss_type_sum__1__1733711678 \
    --learning_rate $lr \
    --hf_metadata_dataset allenai/olmo-instruct-evals
done
done

# 13b RM
python mason.py \
    --image costah/open_instruct_ppo_olmo23 --pure_docker_mode \
    --cluster ai2/jupiter \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --workspace ai2/tulu-3-dev \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& accelerate launch \
    --deepspeed_multinode_launcher standard \
    --num_machines 4 \
    --num_processes 8 \
    --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
    --dataset_mixer '{"allenai/olmo-2-1124-13b-preference-mix": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
    --dataset_eval_splits test_prefs \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1208_bsz64_13b_finetune_epoch_2_lr_5e-6_loss_type_sum__1__1733711678 \
    --chat_template_name tulu \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs 1 \
    --output_dir models/rm/rm_tulu_8b \
    --gradient_checkpointing \
    --push_to_hub \
    --with_tracking

# https://huggingface.co/allenai/open_instruct_dev/tree/reward_modeling__1__1733808608
# https://huggingface.co/allenai/open_instruct_dev/tree/reward_modeling__1__1733807985


# 7B RL
for beta in 0.03 0.05 0.07 0.1; do
for lr in 3e-7; do
for seed in 1; do
exp_name="1210_7b_gsm_math_if_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo23 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo2/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1206_dpo_7b_tune8e-7__allenai_open_instruct_dev__9__1733800022 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733808608 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done


for beta in 0.03 0.05 0.07 0.1; do
for lr in 3e-7; do
for seed in 1; do
exp_name="1210_7b_gsm_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo23 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo2/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer "{\"ai2-adapt-dev/gsm8k_ground_truth\": 1.0}" \
    --dataset_train_splits train \
    --dataset_eval_mixer "{\"ai2-adapt-dev/gsm8k_math_ground_truth\": 1.0}" \
    --dataset_eval_splits test \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1206_dpo_7b_tune8e-7__allenai_open_instruct_dev__9__1733800022 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733808608 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done


for beta in 0.03 0.05 0.07 0.1; do
for lr in 3e-7; do
for seed in 1; do
exp_name="1210_7b_lrm_gsm_math_if_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo23 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo2/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1206_dpo_7b_tune8e-7__allenai_open_instruct_dev__9__1733800022 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733808608 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done


# 13B RL
for beta in 0.03 0.05 0.07 0.1; do
for lr in 3e-7 4e-7; do
for seed in 52; do
exp_name="1210_13b_gsm_math_if_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo23 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo2/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1208_dpo_13b_tune8e-7__allenai_open_instruct_dev__7__1733807564 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733807985 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done


for beta in 0.03 0.05 0.07 0.1; do
for lr in 3e-7; do
for seed in 52; do
exp_name="1210_2_13b_gsm_math_if_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo23 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo2/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision 1208_dpo_13b_tune8e-7__allenai_open_instruct_dev__8__1733807565 \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733807985 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done


"/weka/oe-adapt-default/costah/models/olmo2/1210_2_13b_gsm_math_if_beta_0.1_lr_3e-7_12983"


for beta in 0.03 0.05 0.07 0.1; do
for lr in 3e-7; do
for seed in 52; do
exp_name="1210_13b_gsm_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo23 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo2/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer "{\"ai2-adapt-dev/gsm8k_ground_truth\": 1.0}" \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"ai2-adapt-dev/gsm8k_ground_truth": 128}' \
    --dataset_eval_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path /weka/oe-adapt-default/costah/models/olmo2/1210_2_13b_gsm_math_if_beta_0.1_lr_3e-7_12983_checkpoints/step_480/ \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733807985 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done



for beta in 0.03 0.05 0.07 0.1; do
for lr in 3e-7; do
for seed in 52; do
exp_name="1210_13b_gsm_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo23 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo2/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer "{\"ai2-adapt-dev/gsm8k_ground_truth\": 1.0}" \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"ai2-adapt-dev/gsm8k_ground_truth": 128}' \
    --dataset_eval_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path /weka/oe-adapt-default/costah/models/olmo2/1210_2_13b_gsm_math_if_beta_0.1_lr_3e-7_12983_checkpoints/step_480/ \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733807985 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done



for beta in 0.03 0.05 0.07 0.1; do
for lr in 3e-7; do
for seed in 52; do
exp_name="1210_13b_math_beta_${beta}_lr_${lr}_${RANDOM}"
echo $exp_name
python mason.py \
    --cluster ai2/jupiter --image nathanl/open_instruct_auto --pure_docker_mode \
    --workspace ai2/tulu-3-dev \
    --priority urgent \
    --preemptible \
    --num_nodes 4 \
    --image costah/open_instruct_ppo_olmo23 \
    --budget ai2/jupiter \
    --gpus 8 -- pip install git+https://github.com/vwxyzjn/transformers.git@olmo2-classification \&\& source configs/beaker_configs/ray_node_setup.sh \&\& python open_instruct/ppo_vllm_thread_ray_gtrl_olmo.py \
    --beta $beta \
    --learning_rate $lr \
    --exp_name $exp_name \
    --seed $seed \
    --output_dir "/weka/oe-adapt-default/costah/models/olmo2/${exp_name}" \
    --save_freq 60 \
    --try_launch_beaker_eval_jobs_on_weka \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --dataset_mixer "{\"ai2-adapt-dev/math_ground_truth\": 1.0}" \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"ai2-adapt-dev/math_ground_truth": 128}' \
    --dataset_eval_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path /weka/oe-adapt-default/costah/models/olmo2/1210_13b_gsm_beta_0.03_lr_3e-7_7470_checkpoints/step_180/ \
    --reward_model_path allenai/open_instruct_dev \
    --reward_model_revision reward_modeling__1__1733807985 \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --total_episodes 200000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 8 \
    --local_rollout_batch_size 8 \
    --actor_num_gpus_per_node 7 8 8 8 \
    --vllm_tensor_parallel_size 1 \
    --apply_verifiable_reward true \
    --num_evals 3 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
done
done
done
