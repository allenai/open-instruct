python scripts/submit_finetune_job.py \
  --default_beaker_config configs/beaker_configs/default_finetune.yaml \
  --config configs/train_configs/sft/mmlu_mix/llama3-8b-mmlu36k-v3.4-100k.yaml\
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --exp_name nd-SFT-llama3-8b-mmlu36k-v3.4-100k \
  --num_gpus 8

