# need to fix
python get-responses-from-model.py \
  --dataset_name "your/dataset" \
  --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
  --output_path "results.jsonl" \
  --num_gpus 2 \
  --num_responses 3 \
  --apply_chat_template