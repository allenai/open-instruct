python3 -m scripts.submit_finetune_job \
    --cluster ai2/jupiter-cirrascale-2 \
    --priority urgent \
    --workspace ai2/tulu-3-dev \
    --num_nodes 4 \
    --default_beaker_config toolu_training/beaker_configs/default_finetune_multinode.yaml \
    --config toolu_training/train_configs/toolu_8b_sft.yaml \
    --exp_name "Llama3.1_toolu" \
    --reduce_loss sum \
    --hf_metadata_dataset allenai/olmo-instruct-evals