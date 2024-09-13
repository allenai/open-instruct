beaker session create \
    --gpus 1 \
    --budget ai2/allennlp  \
    --workdir $PWD \
    --image beaker://costah/open_instruct_onlinedpo1 \
    --priority normal \
    --workspace ai2/costah
beaker session create \
    --gpus 1 \
    --budget ai2/allennlp  \
    --workdir $PWD \
    --image beaker://costah/open_instruct_dev_uv \
    --priority normal \
    --workspace ai2/costah


beaker session create \
    --gpus 3 \
    --budget ai2/allennlp  \
    --workdir $PWD \
    --image beaker://ai2/cuda11.8-cudnn8-dev-ubuntu20.04 \
    --priority normal \
    --workspace ai2/costah

beaker session create \
    --gpus 1 \
    --budget ai2/allennlp  \
    --bare \
    --image beaker://costah/open_instruct_onlinedpo \
    --priority normal \
    --workspace ai2/costah


accelerate launch --num_processes 2 open_instruct/online_dpo_vllm.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tldr \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 4 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 53 \
    --vllm_device cuda:2 --sanity_check


accelerate launch --num_processes 2 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/online_dpo_vllm.py \
    --dataset_name allenai/ultrafeedback_binarized_cleaned \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --sft_messages_key chosen \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tulu3 \
    --chat_template tulu \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 2 \
    --vllm_device cuda:2 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 100000 \
    --model_name_or_path allenai/llama-3-tulu-2-8b  \
    --reward_model_path allenai/reward_modeling__allenai_llama-3-tulu-2-8b_ultrafeedback \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 512 \
    --with_tracking \
    --push_to_hub \



accelerate launch --num_processes 1 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/online_dpo_vllm.py \
    --dataset_name allenai/ultrafeedback_binarized_cleaned \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --sft_messages_key chosen \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tulu3 \
    --chat_template tulu \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 2 \
    --vllm_device cuda:1 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 100000 \
    --model_name_or_path allenai/llama-3-tulu-2-8b  \
    --reward_model_path allenai/reward_modeling__allenai_llama-3-tulu-2-8b_ultrafeedback \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 512 \
    --with_tracking \
    --push_to_hub \


accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/online_dpo_vllm.py \
    --dataset_name allenai/ultrafeedback_binarized_cleaned \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --sft_messages_key chosen \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tulu3 \
    --chat_template tulu \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 2 \
    --vllm_device cuda:7 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 200000 \
    --model_name_or_path allenai/llama-3-tulu-2-8b  \
    --reward_model_path allenai/reward_modeling__allenai_llama-3-tulu-2-8b_ultrafeedback \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 512 \
    --with_tracking \
    --push_to_hub \






python open_instruct/online_dpo_vllm.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tldr \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 8 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 53


accelerate launch --num_processes 2 open_instruct/online_dpo_vllm_thread.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tldr \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 8 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 53 --vllm_device cuda:2 --sanity_check


python open_instruct/online_dpo_vllm_thread.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tldr \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 4 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-2.8b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 53

python open_instruct/online_dpo_vllm.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tldr \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 4 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-2.8b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 53




pip install git+https://github.com/vwxyzjn/vllm.git@costa-single-gpu-fix

docker build --build-arg CUDA=12.1.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 --build-arg REQUIRE=requirements.txt . -t open_instruct_onlinedpo2
beaker image delete $(whoami)/open_instruct_onlinedpo2 
beaker image create open_instruct_onlinedpo2 -n open_instruct_onlinedpo2 -w ai2/$(whoami)


accelerate launch --num_processes 2 open_instruct/online_dpo_vllm.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_tldr \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 4 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 10000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 53 \
    --vllm_device cuda:2 --sanity_check  --with_tracking

accelerate launch --num_processes 2 open_instruct/ppo_vllm.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo_tldr \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --local_rollout_forward_batch_size 16 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 10000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 53 \
    --vllm_device cuda:2 --sanity_check


accelerate launch --num_processes 2 open_instruct/online_dpo_vllm_thread.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo_tldr \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --local_rollout_forward_batch_size 16 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 10000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --num_evals 10 \
    --response_length 53 \
    --vllm_device cuda:2 --sanity_check

accelerate launch --num_processes 2 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/online_dpo_vllm.py \
    --dataset_name allenai/ultrafeedback_binarized_cleaned \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --sft_messages_key chosen \
    --learning_rate 5e-7 \
    --output_dir models/minimal/online_dpo_tulu2_llama333 \
    --chat_template tulu \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 4 \
    --vllm_device cuda:2 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000 \
    --model_name_or_path allenai/llama-3-tulu-2-8b  \
    --reward_model_path allenai/reward_modeling__allenai_llama-3-tulu-2-8b_ultrafeedback \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.03 \
    --num_evals 10 \
    --response_length 1024 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub --sanity_check


accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/online_dpo_vllm_thread.py \
    --dataset_name allenai/ultrafeedback_binarized_cleaned \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --sft_messages_key chosen \
    --learning_rate 5e-7 \
    --output_dir models/minimal/online_dpo_tulu2_llama333 \
    --chat_template tulu \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 2 \
    --vllm_device cuda:7 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 2000 \
    --model_name_or_path allenai/llama-3-tulu-2-8b  \
    --reward_model_path allenai/reward_modeling__allenai_llama-3-tulu-2-8b_ultrafeedback \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.03 \
    --num_evals 10 \
    --response_length 1024 \
    --gradient_checkpointing --sanity_check \

g = AutoModelForSequenceClassification.from_pretrained("allenai/llama-3-tulu-2-8b", num_labels=1)

accelerate launch --num_processes 2 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/online_dpo_vllm_thread.py \
    --dataset_name allenai/ultrafeedback_binarized_cleaned \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --sft_messages_key chosen \
    --learning_rate 5e-7 \
    --output_dir models/minimal/online_dpo_tulu2_llama333 \
    --chat_template tulu \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 2 \
    --vllm_device cuda:2 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000 \
    --model_name_or_path vwxyzjn/btulu  \
    --reward_model_path allenai/llama-3.1-tulu-2-8b-uf-mean-rm \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.03 \
    --num_evals 10 \
    --response_length 1024 \
    --gradient_checkpointing  --with_tracking


accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/online_dpo_vllm_thread.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --learning_rate 5e-7 \
    --output_dir models/minimal/online_dpo_tulu2_llama333 \
    --chat_template simple_concat_with_space \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --local_rollout_forward_batch_size 64 \
    --vllm_device cuda:7 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 10000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.03 \
    --num_evals 10 \
    --response_length 53 \
    --gradient_checkpointing  --with_tracking


accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/ppo_vllm_thread.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --learning_rate 5e-7 \
    --output_dir models/minimal/online_dpo_tulu2_llama333 \
    --chat_template simple_concat_with_space \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 2 \
    --vllm_device cuda:7 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.03 \
    --num_evals 10 \
    --response_length 53 \
    --gradient_checkpointing


accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml \
    open_instruct/online_dpo_vllm_thread.py \
    --dataset_name allenai/ultrafeedback_binarized_cleaned \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --sft_messages_key chosen \
    --learning_rate 5e-7 \
    --output_dir /output/ \
    --chat_template tulu \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 2 \
    --vllm_device cuda:7 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 300000 \
    --model_name_or_path vwxyzjn/btulu  \
    --reward_model_path allenai/llama-3.1-tulu-2-8b-uf-mean-rm \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.04 \
    --num_evals 1 \
    --response_length 1024 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub


accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml ds3.py




accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml \
    open_instruct/online_dpo_vllm_thread.py \
    --dataset_name allenai/ultrafeedback_binarized_cleaned \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --sft_messages_key chosen \
    --learning_rate 5e-7 \
    --output_dir /output/ \
    --chat_template tulu \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 4 \
    --vllm_device cuda:7 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 200000 \
    --model_name_or_path OLMoE/OLMoE-1B-7B-0824-SFT  \
    --reward_model_path allenai/llama-3.1-tulu-2-8b-uf-mean-rm \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.05 \
    --num_evals 1 \
    --response_length 1024 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub


accelerate launch --num_processes 8 --config_file configs/ds_configs/deepspeed_zero3.yaml \
    open_instruct/online_dpo.py \
    --dataset_name allenai/ultrafeedback_binarized_cleaned \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --max_token_length 512 \
    --max_prompt_token_lenth 256 \
    --sft_messages_key chosen \
    --learning_rate 5e-7 \
    --output_dir /output/ \
    --chat_template tulu \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 4 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 200000 \
    --model_name_or_path OLMoE/OLMoE-1B-7B-0824-SFT  \
    --reward_model_path allenai/llama-3.1-tulu-2-8b-uf-mean-rm \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.05 \
    --num_evals 1 \
    --response_length 512 \
    --gradient_checkpointing \
    --with_tracking \
    --push_to_hub


accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml \
        open_instruct/online_dpo_vllm_thread.py \
        --exp_name "online_dpo_vllm_thread_beta_${beta}" \
        --dataset_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
        --dataset_train_splits train_prefs \
        --dataset_eval_mixer '{"allenai/ultrafeedback_binarized_cleaned": 1.0}' \
        --dataset_eval_splits test_prefs \
        --max_token_length 1024 \
        --max_prompt_token_lenth 512 \
        --sft_messages_key chosen \
        --learning_rate 5e-7 \
        --output_dir /output/ \
        --chat_template tulu \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 32 \
        --local_rollout_forward_batch_size 1 \
        --vllm_device cuda:7 \
        --num_epochs 1 \
        --num_mini_batches 1 \
        --total_episodes 300000 \
        --model_name_or_path allenai/llama-3-tulu-2-8b  \
        --reward_model_path allenai/reward_modeling__allenai_llama-3-tulu-2-8b_ultrafeedback \
        --non_stop_penalty \
        --stop_token eos \
        --penalty_reward_value -10.0 \
        --beta $beta \
        --num_evals 1 \
        --response_length 1024 \
        --gradient_checkpointing \
        --with_tracking \
        --push_to_hub


python open_instruct/online_dpo_vllm_thread.py \
    --exp_name "online_dpo_vllm_thread_beta" \
    --dataset_mixer '{"HuggingFaceH4/no_robots": 1.0}' \
    --dataset_train_splits train \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --learning_rate 5e-7 \
    --output_dir /output/ \
    --chat_template tulu \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --no_async_mode \
    --gradient_accumulation_steps 32 \
    --local_rollout_forward_batch_size 1 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 300000 \
    --model_name_or_path allenai/open_instruct_dev  \
    --model_revision costa_finetune_tulu3_8b_norobot__meta-llama_Meta-Llama-3.1-8B__42__1725559869 \
    --reward_model_path vwxyzjn/reward_modeling__allenai_llama-3-tulu-2-8b \
    --reward_model_revision reward_modeling__1__1725631368 \
    --non_stop_penalty \
    --stop_token eos \
    --penalty_reward_value -10.0 \
    --beta 0.05 \
    --num_evals 1 \
    --response_length 1024 \
    --gradient_checkpointing \
    --vllm_device cuda:1 \
    --with_tracking \


python mason.py \
    --cluster ai2/pluto-cirrascale ai2/prior-cirrascale ai2/s2-cirrascale ai2/general-cirrascale \
    --priority normal \
    --resumable \
    --budget ai2/allennlp \
    --gpus 8 -- accelerate launch --num_processes 7 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/online_dpo_vllm_thread.py \
    --dataset_mixer '{"trl-internal-testing/tldr-preference-sft-trl-style": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"trl-internal-testing/tldr-preference-sft-trl-style": 1.0}' \
    --dataset_eval_splits validation \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_vllm_thread_tldr \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --response_length 53 \
    --with_tracking \
    --push_to_hub \
    --vllm_device cuda:7 \


accelerate launch --num_processes 3 --config_file configs/ds_configs/deepspeed_zero3.yaml \
     open_instruct/online_dpo_vllm_thread.py \
    --dataset_mixer '{"trl-internal-testing/tldr-preference-sft-trl-style": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"trl-internal-testing/tldr-preference-sft-trl-style": 1.0}' \
    --dataset_eval_splits validation \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_dpo_vllm_thread_tldr \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --non_stop_penalty \
    --stop_token eos \
    --beta 0.1 \
    --response_length 53 \
    --with_tracking \
    --push_to_hub \
    --vllm_device cuda:3 \