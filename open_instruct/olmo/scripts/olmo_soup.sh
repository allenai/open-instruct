python scripts/submit_finetune_job.py \
  --default_beaker_config configs/beaker_configs/default_finetune_multinode.yaml \
  --config configs/train_configs/sft/olmo/olmo_7b_soup.yaml \
  --cluster  ai2/jupiter-cirrascale-2\
  --priority urgent \
  --workspace ai2/tulu-3-dev \
  --mount_on_weka=oe-training-default \
  --num_nodes 4  \
  --image nathanl/open_instruct_olmo_13 \
  --exp_name nd-SFT-olmo-soup \
  --num_gpus 8

