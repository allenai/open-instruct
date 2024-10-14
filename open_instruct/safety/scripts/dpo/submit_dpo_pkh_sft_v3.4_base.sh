python scripts/submit_dpo_job.py \
  --default_beaker_config configs/beaker_configs/default_dpo.yaml \
  --config configs/train_configs/dpo/llama-3.1-8b-dpo-pku.yaml\
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --experiment_name nd-DPO-tulu3-sft-wildgaurdmixtrain-mixv3.4-valmix-pkh \
  --num_gpus 8 \
  --datasets 01J9VR2PAE6V3Z4P7NPM3CSDFA:/model

