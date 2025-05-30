for eval in minerva_math::hamish_zs_reasoning bbh:cot::hamish_zs_reasoning gsm8k::hamish_zs_reasoning minerva_math_500::hamish_zs_reasoning zebralogic::hamish_zs_reasoning aime::hamish_zs_reasoning agi_eval_english:0shot_cot::hamish_zs_reasoning gpqa:0shot_cot::hamish_zs_reasoning codex_humanevalplus:0-shot-chat::tulu-thinker ifeval::tulu popqa::tulu mmlu:mc::tulu bbh:cot-v1::tulu; do
for step in 50; do
python oe_eval/launch.py \
  --model test_grpo_rm_with_valpy_code__1__1748573647_step_${step} \
  --beaker-workspace ai2/tulu-3-results \
  --beaker-budget ai2/oe-adapt \
  --task $eval \
  --model-type vllm \
  --batch-size 10000 \
  --model-args "{\"model_path\":\"gs://ai2-llm/post-training//jacobm/output/test_grpo_rm_with_valpy_code__1__1748573647_checkpoints/step_${step}\", \"max_length\": 32768, \"process_output\": \"r1_style\"}" \
  --task-args '{ "generation_kwargs": { "max_gen_toks": 32768, "truncate_context": false, "stop_sequences": ["</answer>"] } }' \
  --gpus 1 \
  --gantry-args '{"env-secret": "OPENAI_API_KEY=openai_api_key", "env":"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1", "env-secret#2":"HF_TOKEN=HF_TOKEN", "mount": "/mnt/filestore_1:/filestore", "env#111": "HF_HOME=/filestore/.cache/huggingface", "env#112": "HF_DATASETS_CACHE=/filestore/.cache/huggingface", "env#113": "HF_HUB_CACHE=/filestore/.cache/hub"}' \
  --cluster ai2/augusta-google-1 \
  --beaker-retries 2 \
  --beaker-image oe-eval-beaker/oe_eval_auto \
  --beaker-priority high \
  --push-datalake \
  --datalake-tags run_id=https://wandb.ai/ai2-llm/open_instruct_internal/runs/mrq6k9hg,step=${step}
done
done