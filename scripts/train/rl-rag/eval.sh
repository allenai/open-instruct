# gpqa diamond evaluation.
export S2_API_KEY='xxx'
python -m open_instruct.search_utils.gpqa_eval \
    --model_path /weka/oe-adapt-default/hamishi/model_checkpoints/testing_rl_rag

export MASSIVE_DS_URL='http://ceres-cs-aus-445.reviz.ai2.in:45489/search'
python -m open_instruct.search_utils.simpleqa_eval \
    --model_path /weka/oe-adapt-default/hamishi/model_checkpoints/rl_rag/testing_init_rl_rag \
    --output_dir simpleqa_eval_tmp