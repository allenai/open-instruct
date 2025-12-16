# check if there is $HF_TOKEN in the environment variables, or if the huggingface-cli is installed and logged in
if huggingface-cli whoami &>/dev/null; then
    echo "Logged in to HuggingFace."
elif [ -z "$HF_TOKEN" ]
then
    echo "Warning: HuggingFace dataset LIMA requires permissive access."
    echo "Warning: Please request the access at https://huggingface.co/datasets/GAIR/lima and set the HF_TOKEN environment variable before running this script."
    exit 1
fi

echo "Downloading Super-NaturalInstructions dataset..."
wget -P data/raw_train/super_ni/ https://github.com/allenai/natural-instructions/archive/refs/heads/master.zip
unzip data/raw_train/super_ni/master.zip -d data/raw_train/super_ni/ && rm data/raw_train/super_ni/master.zip
mv data/raw_train/super_ni/natural-instructions-master/* data/raw_train/super_ni/ && rm -r data/raw_train/super_ni/natural-instructions-master


echo "Downloading the flan_v2 chain-of-thought submix..."
wget -P data/raw_train/cot/ https://huggingface.co/datasets/hamishivi/tulu_mix_store/resolve/main/cot_zsopt.jsonl
wget -P data/raw_train/cot/ https://huggingface.co/datasets/hamishivi/tulu_mix_store/resolve/main/cot_fsopt.jsonl


echo "Downloading the flan_v2 collection, here we use two subsampled versions: for tulu v1 we subsampled 100K, for tulu v2 we subsampled 50K..."
mkdir -p data/raw_train/flan_v2/
wget -O data/raw_train/flan_v2/tulu_v1_resampled_flan_100k.jsonl https://huggingface.co/datasets/hamishivi/tulu_mix_store/resolve/main/flan_v2_resampled_100k.jsonl
wget -O data/raw_train/flan_v2/tulu_v2_resampled_flan_50k.jsonl https://huggingface.co/datasets/hamishivi/tulu_mix_store/resolve/main/flan_v2_resampled_50k.jsonl


echo "Downloading self-instruct data..."
wget -P data/raw_train/self_instruct/ https://raw.githubusercontent.com/yizhongw/self-instruct/main/data/gpt3_generations/batch_221203/all_instances_82K.jsonl


echo "Downloading unnatural-instructions data..."
wget -P data/raw_train/unnatural_instructions/ https://github.com/orhonovich/unnatural-instructions/raw/main/data/core_data.zip
unzip data/raw_train/unnatural_instructions/core_data.zip -d data/raw_train/unnatural_instructions/


echo "Downloading Stanford alpaca data..."
wget -P data/raw_train/stanford_alpaca/ https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json


echo "Downloading the dolly dataset..."
wget -P data/raw_train/dolly/ https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl


echo "Downloading the OpenAssistant data (oasst1)..."
wget -P data/raw_train/oasst1/ https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz
gzip -d data/raw_train/oasst1/2023-04-12_oasst_ready.trees.jsonl.gz


echo "Downloading the code alpaca dataset..."
wget -P data/raw_train/code_alpaca/ https://github.com/sahil280114/codealpaca/raw/master/data/code_alpaca_20k.json


echo "Downloading the gpt4-llm dataset..."
wget -P data/raw_train/gpt4_alpaca/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json
wget -P data/raw_train/gpt4_alpaca/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data_zh.json


echo "Downloading the baize dataset..."
wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/alpaca_chat_data.json
wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/medical_chat_data.json
wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/quora_chat_data.json
wget -P data/raw_train/baize/ https://github.com/project-baize/baize-chatbot/raw/main/data/stackoverflow_chat_data.json


echo "Downloading ShareGPT dataset..."
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json
echo "Splitting the ShareGPT dataset with 2048 max tokens per conversation..."
python scripts/data/sft/split_sharegpt_conversations.py \
    --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
    --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_2048.json \
    --model-name-or-path oobabooga/llama-tokenizer \
    --max-length 2048
echo "Splitting the ShareGPT dataset with 4096 max tokens per conversation..."
python scripts/data/sft/split_sharegpt_conversations.py \
    --in-files data/raw_train/sharegpt/sg_90k_part1_html_cleaned.json data/raw_train/sharegpt/sg_90k_part2_html_cleaned.json \
    --out-file data/raw_train/sharegpt/sharegpt_html_cleaned_and_split_4096.json \
    --model-name-or-path oobabooga/llama-tokenizer \
    --max-length 4096

echo "Downloading LIMA dataset..."
if huggingface-cli whoami &>/dev/null; then
    huggingface-cli download GAIR/lima --repo-type dataset --local-dir data/raw_train/lima/
else
    wget --header="Authorization: Bearer $HF_TOKEN" -P data/raw_train/lima/ https://huggingface.co/datasets/GAIR/lima/raw/main/train.jsonl
fi

echo "Downloading WizardLM dataset..."
# original data removed wget -P data/raw_train/wizardlm/ https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k/resolve/main/WizardLM_evol_instruct_V2_143k.json
wget -P data/raw_train/wizardlm/ https://huggingface.co/datasets/ai2-adapt-dev/wizardlm-backup/resolve/main/data/train-00000-of-00001.parquet


echo "Downloading the OpenOrca dataset..."
wget -P data/raw_train/open_orca/ https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/1M-GPT4-Augmented.parquet
wget -P data/raw_train/open_orca/ https://huggingface.co/datasets/Open-Orca/OpenOrca/resolve/main/3_5M-GPT3_5-Augmented.parquet


echo "Downloading the Science Instructions dataset..."
wget -P data/raw_train/science https://huggingface.co/datasets/hamishivi/tulu_mix_store/resolve/main/science_train.jsonl


echo "Downloading the HardCoded dataset..."
wget -P data/raw_train/hard_coded/ https://huggingface.co/datasets/hamishivi/tulu_mix_store/resolve/main/hard_coded_examples.xlsx


echo "Processing datasets..."
python scripts/data/sft/reformat_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/
