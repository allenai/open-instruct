python scripts/submit_finetune_job.py \
  --default_beaker_config configs/beaker_configs/default_finetune_multinode.yaml \
  --config configs/train_configs/sft/olmo/olmo_7b_0924_v3.9_safety.yaml \
  --cluster  ai2/jupiter-cirrascale-2\
  --priority urgent \
  --workspace ai2/tulu-3-dev \
  --mount_on_weka=oe-training-default \
  --num_nodes 4  \
  --image nathanl/open_instruct_olmo_13 \
  --exp_name nd-SFT-olmo_7b_0924_v3.9_safety \
  --num_gpus 8

max_seq_length: 4096
preprocessing_num_workers: 128
per_device_train_batch_size: 1
gradient_accumulation_steps: 4 # designed for 4 nodes
# gradient_accumulation_steps: 8 # designed for 2 nodes
# gradient_accumulation_steps: 16 # designed for 1 nodes
gradient_checkpointing: true
learning_rate: 2.0e-06
lr_scheduler_type: linear
warmup_ratio: 0.03
weight_decay: 0.0
num_train_epochs: 3
output_dir: /output/
with_tracking: true
# reduce_loss: mean
report_to:
  - wandb
logging_steps: 1
checkpointing_steps: epoch
add_bos: true