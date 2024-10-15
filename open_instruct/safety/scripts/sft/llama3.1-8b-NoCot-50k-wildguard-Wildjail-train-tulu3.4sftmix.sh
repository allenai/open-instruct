python scripts/submit_finetune_job.py \
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --num_nodes 8 \
  --image nathanl/open_instruct_auto-f0b3def-11239828100 \
  --reduce_loss sum \
  --default_beaker_config configs/beaker_configs/default_finetune_multinode.yaml \
  --config configs/train_configs/sft/safety_mix/llama3-8b-50k-wildgaurdmix-WildJail-mixv3.4.yaml \
  --exp_name nd-SFT-llama3-8b-50k-wildgaurdmix-Wildjail-mixv3.4





