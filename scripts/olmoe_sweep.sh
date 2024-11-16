#!/bin/bash

# Directory containing the configs
CONFIG_DIR="/net/nfs.cirrascale/allennlp/nathanl/open-instruct/configs/train_configs/dpo/olmoe_sweep"

# Loop through X and Y values
for X in {1..3}; do
  for Y in {1..3}; do
    # Construct the config file name
    CONFIG="${CONFIG_DIR}/olmoe_tulu3_v${X}_${Y}.yaml"
    
    # Ensure the file exists before running the script
    if [ -f "$CONFIG" ]; then
      echo "Running with config: $CONFIG"
      
      # Execute the Python script
      python scripts/submit_dpo_job.py \
        --config "$CONFIG" \
        --default_beaker_config configs/beaker_configs/default_dpo_multinode.yaml \
        --cluster ai2/jupiter-cirrascale-2 \
        --num_nodes 4 \
        --num_gpus 8 \
        --workspace ai2/tulu-3-dev \
        --priority urgent \
        --datasets 01JCADMWGGGSE961PAWM2XTM3D:/model \
        --experiment_name "OLMoE_v3.9_dpo_sweep_${X}_${Y}" \
        --image=nathanl/open_instruct_auto
    else
      echo "Config file not found: $CONFIG"
    fi
  done
done