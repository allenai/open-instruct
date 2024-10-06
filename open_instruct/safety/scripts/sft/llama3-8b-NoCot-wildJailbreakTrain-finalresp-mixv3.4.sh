
python scripts/submit_finetune_job.py \
  --default_beaker_config configs/beaker_configs/default_finetune.yaml \
  --config configs/train_configs/sft/safety_mix/llama3-8b-finalresp-wildJailbreakTrain-mixv3.4.yaml\
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --exp_name nd-SFT-llama3-8b-finalresp-wildJailbreakTrain-mixv3.4.yaml \
  --num_gpus 8