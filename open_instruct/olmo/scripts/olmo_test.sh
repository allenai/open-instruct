python scripts/submit_finetune_job.py \
  --default_beaker_config configs/beaker_configs/default_olmo_finetune.yaml \
  --config configs/train_configs/sft/olmo/olmo_7b_0924_v3.9_safety.yaml \
  --cluster  ai2/jupiter-cirrascale-2\
  --priority high \
  --num_nodes 4  \
  --image nathanl/open_instruct_olmo_13 \
  --exp_name nd-SFT-olmo_7b_0924_v3.9_safety \
  --num_gpus 8
