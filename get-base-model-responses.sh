# need to fix
python get-responses-from-model.py \
  --dataset_name "jacobmorrison/social-rl-eval-prompts-100" \
  --model_name "allenai/Olmo-3-1025-7B" \
  --tokenizer_name "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_1000/" \
  --output_path "responses-from-olmo-3-base-model.jsonl" \
  --num_gpus 1 \
  --num_responses 1 \
  --apply_chat_template \
  --temperature 1.0 \
  --top_p 1.0 \
  --max_tokens 4096 \
  --max_model_len 32768 \
  --apply_chat_template \
  --messages_column "messages"