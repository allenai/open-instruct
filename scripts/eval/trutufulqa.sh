# python -m eval.truthfulqa.run_eval \
#     --data_dir data/eval/truthfulqa \
#     --save_dir results/trutufulqa/llama2-7B \
#     --model_name_or_path ../hf_llama2_models/7B \
#     --tokenizer_name_or_path ../hf_llama2_models/7B \
#     --metrics judge info mc \
#     --preset qa \
#     --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
#     --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
#     --eval_batch_size 32 \
#     --load_in_8bit


python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/trutufulqa/llama_7B_large_mix \
    --model_name_or_path ../checkpoints/llama_7B_large_mix/ \
    --tokenizer_name_or_path ../checkpoints/llama_7B_large_mix/ \
    --metrics judge info mc \
    --preset qa \
    --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --eval_batch_size 32 \
    --load_in_8bit \
    --use_chat_format


# python -m eval.truthfulqa.run_eval \
#     --data_dir data/eval/truthfulqa \
#     --save_dir results/trutufulqa/davinci \
#     --openai_engine davinci \
#     --metrics judge info \
#     --preset qa \
#     --gpt_judge_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
#     --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
#     --eval_batch_size 32 \
#     --load_in_8bit