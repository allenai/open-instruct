SPLITS=(
    # "human_datamodel_counts_7000_ID__03242bb1814b48978f32cfa8090308e2__SWAPS_2169"
    "human_datamodel_counts_7000_ID__12210931507643a7b7eed73e878eae1f__SWAPS_6507"
    # "human_datamodel_counts_7000_ID__3e2014f53dba499cbdac39fda22e24a5__SWAPS_4338"
    # "human_datamodel_counts_7000_ID__5d3de9a4893b4a0ab79234af754954e1__SWAPS_2169"
    # "human_datamodel_counts_7000_ID__5dc9108715934d989e0080d9111506bf__SWAPS_6507"
    # "human_datamodel_counts_7000_ID__9be2ce559efd4a59ba46478f3f9ed502__SWAPS_4338"
    # "human_datamodel_counts_7000_ID__a00a49d8a9404752bba39a60196dc650__SWAPS_2169"
    # "human_datamodel_counts_7000_ID__a2c10e356fa84d449b9849c911b50a72__SWAPS_6507"
    # "human_datamodel_counts_7000_ID__c4ee928ab76b483d9d9b2c8b3c1ef9c3__SWAPS_2169"
    # "human_datamodel_counts_7000_ID__c8d950699ce94fbc86b71eef9e1f5b52__SWAPS_4338"
    # "human_datamodel_counts_7000_ID__ca3ec66db56a41d48b94cfc1f2242657__SWAPS_2169"
    # "human_datamodel_counts_7000_ID__e6316c853a4d4974ab0eaa5c27571d5c__SWAPS_2169"
    # "hs2p_human_SWAPS_6766_SEED_42"
    # "hs2p_human_75_SWAPS_4938_SEED_42"
    # "hs2p_human_50_SWAPS_3060_SEED_42"
    # "hs2p_human_25_SWAPS_1433_SEED_42"
    # "hs2p_gpt4_SWAPS_0_SEED_42"
    # "hs2p_random_SWAPS_3031_SEED_42"
    # "hs2p_human_SWAPS_6766_SEED_10010"
    # "hs2p_human_75_SWAPS_4899_SEED_10010"
    # "hs2p_human_50_SWAPS_3062_SEED_10010"
    # "hs2p_human_25_SWAPS_1467_SEED_10010"
    # "hs2p_gpt4_SWAPS_0_SEED_10010"
    # "hs2p_random_SWAPS_3050_SEED_10010"
)

for split in "${SPLITS[@]}"; do
    python3 mason.py \
        --cluster ai2/jupiter-cirrascale-2 \
        --image nathanl/open_instruct_auto \
        --pure_docker_mode \
        --priority high \
        --workspace ai2/tulu-3-dev \
        --budget ai2/oe-adapt \
        --num_nodes 4 \
        --gpus 8 -- accelerate launch \
        --deepspeed_multinode_launcher standard \
        --num_processes 8 \
        --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/reward_modeling.py \
        --dataset_mixer '{"ai2-adapt-dev/helpsteer2-pref-subsamples": 0.95}' \
        --dataset_train_splits $split \
        --dataset_eval_mixer '{"ai2-adapt-dev/helpsteer2-pref-subsamples": 0.05}' \
        --dataset_eval_splits $split \
        --model_name_or_path meta-llama/Meta-Llama-3-70B \
        --chat_template tulu \
        --learning_rate 3e-6 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --max_token_length 2048 \
        --max_prompt_token_length 2048 \
        --num_train_epochs 2 \
        --output_dir models/rm/rm_llama3 \
        --gradient_checkpointing \
        --push_to_hub \
        --hf_entity "ai2-adapt-dev" \
        --hf_repo_id "hybrid-pref-dev" \
        --hf_repo_revision llama3_8b_hs2p_$split \
        --wandb_project_name "hybrid-pref" \
        --wandb_entity "ai2-llm" \
        --with_tracking

    # echo "Sleeping....($split is done)"
    # sleep 1500  # Sleep for 25 minutes (1500 seconds)
done

