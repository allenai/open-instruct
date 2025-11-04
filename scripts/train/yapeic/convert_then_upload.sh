# python scripts/convert_and_upload_checkpoints.py \
#   --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
#   --checkpoint_path /weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_checkpoint_states/global_step100 \
#   --revision step100 \
#   --hf_repo_id yapeichang/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop \
#   --upload_dtype bf16 \
#   --max_shard_size 5GB \
#   --save_format safetensors \
#   --trust_remote_code

cd /weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template_checkpoint_states
python zero_to_fp32.py . global_step140/pytorch_model_fp32 --tag global_step140

python scripts/convert_and_upload_checkpoints.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --checkpoint_path /weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template_checkpoint_states/global_step140 \
  --revision step140 \
  --hf_repo_id yapeichang/grpo_qwen25-7b-inst_v3_ratio_10k_with_stop_fixed_template \
  --upload_dtype bf16 \
  --max_shard_size 5GB \
  --save_format safetensors \
  --trust_remote_code