SPLITS=(
    "human_datamodel_counts_7000_ID__03242bb1814b48978f32cfa8090308e2__SWAPS_2169"
    # "human_datamodel_counts_7000_ID__12210931507643a7b7eed73e878eae1f__SWAPS_6507"
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
)

for split in "${SPLITS[@]}"; do
    python mason.py \
        --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
        --image nathanl/open_instruct_auto --pure_docker_mode \
        --priority normal \
        --budget ai2/ljm-oe-adapt \
        --gpus 2 -- accelerate launch --config_file configs/ds_configs/deepspeed_zero3.yaml \
        open_instruct/reward_modeling.py \
        --dataset_mixer '{"ljvmiranda921/helpsteer2-pref-samples": 1.0}' \
        --dataset_train_splits $split \
        --model_name_or_path Qwen/Qwen2.5-7B \
        --chat_template tulu \
        --learning_rate 3e-6 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 32 \
        --max_token_length 1024 \
        --max_prompt_token_length 1024 \
        --num_train_epochs 1 \
        --output_dir models/rm/rm_qwen2_7b \
        --gradient_checkpointing \
        --push_to_hub \
        --with_tracking
done

