#!/bin/bash

echo "Converting Aya dataset..."
python -m scripts.data.sft.aya \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/aya

echo "Converting Coconot dataset..."
python -m scripts.data.sft.coconot \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --local_save_dir ./data/sft/coconot

echo "Converting CodeFeedback Filtered Instructions dataset..."
python -m scripts.data.sft.codefeedback_mix \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/codefeedback_filtered_instructions

echo "Converting Daring Anteater dataset..."
python -m scripts.data.sft.daring_anteater \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --remove_subsets open_platypus_commercial \
    --local_save_dir ./data/sft/daring_anteater

echo "Converting Evol CodeAlpaca dataset..."
python -m scripts.data.sft.evol_codealpaca \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/evol_codealpaca

echo "Converting Flan V2 dataset..."
python -m scripts.data.sft.flan \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/flan

echo "Converting LIMA dataset..."
python -m scripts.data.sft.lima \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/lima

echo "Converting LMSYS Chat 1M dataset..."
python -m scripts.data.sft.lmsys_chat \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --model_name_regex="gpt-4" \
    --local_save_dir ./data/sft/lmsys_chat

echo "Converting Metamath dataset..."
python -m scripts.data.sft.metamath \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/metamath

echo "Converting No Robots dataset..."
python -m scripts.data.sft.no_robots \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --local_save_dir ./data/sft/no_robots

echo "Converting NuminaMath dataset..."
python -m scripts.data.sft.numinamath \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/numinamath

echo "Converting Open Assistant v1 and v2 dataset..."
python -m scripts.data.sft.open_assistant \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --top_k=1 \
    --local_save_dir ./data/sft/open_assistant

echo "Converting OpenMathInstruct-2 dataset..."
python -m scripts.data.sft.open_math_instruct \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/open_math_instruct

echo "Converting SciRiff dataset..."
python -m scripts.data.sft.sciriff \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/sciriff

echo "Converting ShareGPT dataset..."
python -m scripts.data.sft.sharegpt \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/sharegpt

echo "Converting Slim Orca dataset..."
python -m scripts.data.sft.slim_orca \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/slim_orca

echo "Converting TableGPT dataset..."
python -m scripts.data.sft.table_gpt \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/table_gpt

echo "Converting WebInstruct dataset..."
python -m scripts.data.sft.web_instruct \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/web_instruct

echo "Converting WildChat dataset..."
python -m scripts.data.sft.wildchat \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/wildchat

echo "Converting WizardLM dataset..."
python -m scripts.data.sft.wizardlm \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --apply_keyword_filters \
    --apply_empty_message_filters \
    --local_save_dir ./data/sft/wizardlm

echo "Converting Tulu Hard Coded dataset..."
python -m scripts.data.sft.tulu_hard_coded \
    --push_to_hub \
    --hf_entity ai2-adapt-dev \
    --repeat_n=10 \
    --local_save_dir ./data/sft/tulu_hard_coded
