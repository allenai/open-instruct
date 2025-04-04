python open_instruct/reward_modeling.py \
    --dataset_mixer_list trl-internal-testing/sentiment-trl-style 400 \
    --model_name_or_path EleutherAI/pythia-14m \
    --tokenizer_name EleutherAI/pythia-14m \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_token_length 1024 \
    --max_prompt_token_length 1024 \
    --num_train_epochs 1 \
    --push_to_hub