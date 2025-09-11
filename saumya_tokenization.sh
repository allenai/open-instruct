special_tokenizer_v1=/weka/oe-training-default/saumyam/olmo3/dolma2-tokenizer-special-tokens-v1-hf
default_tokenizer=/weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf
gantry run \
   --cluster ai2/saturn-cirrascale \
   --allow-dirty  -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
   --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
   --weka=oe-training-default:/weka/oe-training-default \
   --env-secret HF_TOKEN=HF_TOKEN \
   --gpus 4 \
   --priority urgent \
   --beaker-image nathanl/open_instruct_auto \
   -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
       --dataset_mixer_list \
           allenai/all_reasoning_sft_datasets_no_chains 1.0 \
       --tokenizer_name_or_path $special_tokenizer_v1 \
       --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/sft/usable-tulu-16k-rerun/valpy_sft_reasoning_prompts_no_chains \
       --visualize True \
       --chat_template_name olmo \
       --max_seq_length 16384