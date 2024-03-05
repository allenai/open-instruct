# Run model merges.
# Run from the the directory holding the script.

CONFIG_DIR=/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/experiments/davidw/merge_configs

# python merge_models.py \
#     --model_1_name science_adapt_continued_4k_per_task_200 \
#     --model_1_dataset 01HN9PPJGZR13M0PY7ZEQRX26D \
#     --model_2_name tuluv2_no_science \
#     --model_2_dataset 01HKG46RNVAP3NSHNDH019R5KB \
#     --config_file $CONFIG_DIR/50_50.yaml


# python merge_models.py \
#     --model_1_name science_adapt_continued_4k_per_task_200_tulu_ratio_1 \
#     --model_1_dataset 01HNK9MEQJD9V00NMWW2XM2RFV \
#     --model_2_name tuluv2_no_science \
#     --model_2_dataset 01HKG46RNVAP3NSHNDH019R5KB \
#     --config_file $CONFIG_DIR/50_50.yaml
