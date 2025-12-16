RAW_MODEL_PATH=$1
model_size=$2
og_name=$3

python scripts/weight_diff.py make_diff --path_raw ${RAW_MODEL_PATH}/${model_size} --path_tuned /model --path_diff /results/${og_name}-diff
python scripts/weight_diff.py recover --path_raw ${RAW_MODEL_PATH}/${model_size} --path_tuned test_recover --path_diff  /results/${og_name}-diff --original_model /model
