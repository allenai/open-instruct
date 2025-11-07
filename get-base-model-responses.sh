# need to fix
python get-responses-from-model.py \
  --dataset_name "jacobmorrison/social-rl-eval-prompts" \
  --model_name "/weka/oe-training-default/ai2-llm/checkpoints/tylerr/long-context/olmo25_7b_lc_64k_6T_M100B_round5-sparkle_6634-pre_s2pdf_gzip2080_cweN-yake-all-olmo_packing_yarn-fullonly_50B-fb13a737/step11921-hf" \
  --tokenizer_name "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_1000/" \
  --output_path "responses-from-olmo-3-base-model.jsonl" \
  --num_gpus 1 \
  --num_responses 5 \
  --apply_chat_template \
  --temperature 1.0 \
  --top_p 1.0 \
  --max_tokens 16384 \
  --max_model_len 32768 \
  --apply_chat_template \
  --messages_column "messages"