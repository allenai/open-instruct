python scripts/submit_finetune_job.py \
  --default_beaker_config configs/beaker_configs/default_olmo_finetune.yaml \
  --config configs/train_configs/sft/olmo/olmo_7b_0924_v3.9_safety.yaml \
  --cluster ai2/jupiter-cirrascale-2 ai2/allennlp-cirrascale ai2/mosaic-cirrascale\
  --priority high \
  --image nouhad/open_instruct_olmo
  --exp_name nd-SFT-olmo_7b_0924_v3.9_safety \
  --num_gpus 6
