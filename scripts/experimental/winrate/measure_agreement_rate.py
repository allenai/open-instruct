import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm
from gpt_eval_judge import LLMJudgeConfig, llm_judge
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader

from open_instruct.dataset_processor import INPUT_IDS_CHOSEN_KEY, INPUT_IDS_REJECTED_KEY, SimplePreferenceCollator
from open_instruct.model_utils import get_reward


"""
# synthetic online RM
python -i measure_agreement_rate.py \
    --reward_model_path vwxyzjn/reward_modeling__allenai_open_instruct_dev \
    --reward_model_revision reward_modeling__1__1725760619 \
    --input_path csvs/allenai/open_instruct_dev_costa_offline_dpo_norobot_3pair_3peoch__allenai_open_instruct_dev__42__1726354080_judged.csv \
    --n -1
#   
python -i measure_agreement_rate.py \
    --reward_model_path vwxyzjn/reward_modeling__allenai_open_instruct_dev \
    --reward_model_revision reward_modeling__1__1725760619 \
    --input_path csvs/allenai/open_instruct_dev_costa_offline_dpo_norobot_3pair_3peoch__allenai_open_instruct_dev__42__1726354080_judged.csv \
    --n -1
"""


@dataclass
class Args:
    input_path: str
    reward_model_path: str
    reward_model_revision: Optional[str] = None
    n: int = 1000


def main(args: Args):
    df = pd.read_csv(args.input_path)
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, revision=args.reward_model_revision)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    prompts = df["prompt"].tolist()
    responses0 = df["response0"].tolist()
    responses1 = df["response1"].tolist()
    prompt_responses0 = [f"{prompt}{response}" for prompt, response in zip(prompts, responses0)]
    prompt_responses1 = [f"{prompt}{response}" for prompt, response in zip(prompts, responses1)]
    chosen = []
    rejected = []
    for i, (prompt_response0, prompt_response1) in enumerate(zip(prompt_responses0, prompt_responses1)):
        if df.iloc[i]["preferred"] == "response0":
            chosen.append(prompt_response0)
            rejected.append(prompt_response1)
        else:
            chosen.append(prompt_response1)
            rejected.append(prompt_response0)
    prompt_token = [tokenizer.encode(prompt) for prompt in prompts]
    chosen_token = [tokenizer.encode(chosen[i]) for i in range(len(chosen))]
    rejected_token = [tokenizer.encode(rejected[i]) for i in range(len(rejected))]
    batch_size = 4

    def pad_from_right(list_of_tokens: List[List[int]], pad_token_id):
        max_len = max(len(tokens) for tokens in list_of_tokens)
        return [tokens + [pad_token_id] * (max_len - len(tokens)) for tokens in list_of_tokens]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path, revision=args.reward_model_revision, torch_dtype=torch.bfloat16)
    model = model.to(device)
    agreement_rates = 0
    count = 0
    with torch.no_grad():
        for i in range(0, len(chosen_token), batch_size):
            prompt_batch = prompt_token[i : i + batch_size]
            chosen_batch = chosen_token[i : i + batch_size]
            rejected_batch = rejected_token[i : i + batch_size]
            cat_batch = chosen_batch + rejected_batch
            query_responses = torch.tensor(pad_from_right(cat_batch, tokenizer.pad_token_id)).to(device)
            _, predicted_reward, seq_lens = get_reward(model, query_responses, tokenizer.pad_token_id, 0)
            chosen_reward = predicted_reward[: len(chosen_batch)]
            rejected_reward = predicted_reward[len(chosen_batch) :]
            agreement_rates += (chosen_reward > rejected_reward).sum().item()
            count += len(chosen_batch)
            print(f"count: {count}, agreement_rate: {agreement_rates / count:.2%}")

if __name__ == "__main__":
    main(*HfArgumentParser((Args)).parse_args_into_dataclasses())
