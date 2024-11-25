for lr in 7e-7 8e-7; do
python scripts/submit_dpo_job.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --datasets 01JD6E3WACB6K4ZS901MQ96M3H:/model  \
    --default_beaker_config configs/beaker_configs/default_dpo_multinode_olmo1124_augusta.yaml \
    --config configs/train_configs/dpo/olmo1124_7b_dpo.yaml \
    --exp_name "olmo1124_7b_dpo_${lr}" \
    --experiment_name "olmo1124_7b_dpo_${lr}" \
    --learning_rate $lr \
    --hf_metadata_dataset allenai/olmo-instruct-evals
done