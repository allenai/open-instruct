import json
import os
import sys
from datetime import datetime
import random

import numpy as np
import torch
from datasets import load_dataset
from fire import Fire
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import set_seed as hf_set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

datasets = [
            # 'gov_report',
            # 'summ_screen_fd',
            # 'qmsum',
            # 'qasper',
            # 'narrative_qa',
            'quality',
            # 'musique',
            # 'squality',
            # 'space_digest',
            # 'book_sum_sort'
]

model_to_max_input_tokens = {
    "google/flan-t5-xxl": 8192,
    "google/flan-t5-xl": 8192,
    "google/flan-t5-large": 8192,
    "google/flan-t5-base": 8192,
    "google/flan-t5-small": 8192,
    "google/flan-ul2": 8192,
    "bigscience/T0pp": 8192,
    "allenai/tulu-2-dpo-7b": 8192
}


def trim_doc_keeping_suffix(tokenizer, tokenized_input_full, example, suffix_index, max_tokens, device):
    seperator_and_suffix = f"{example['truncation_seperator'].strip()}\n\n{example['input'][suffix_index:].strip()}\n"
    tokenized_seperator_and_suffix = tokenizer(seperator_and_suffix, return_tensors="pt").input_ids.to(device)
    tokenized_input_trimmed = tokenized_input_full[:, :max_tokens - tokenized_seperator_and_suffix.shape[1]]
    tokenized_input = torch.cat([tokenized_input_trimmed, tokenized_seperator_and_suffix], dim=1)
    return tokenized_input


def process_model_input(tokenizer, example, max_tokens, device):
    tokenized_input_full = tokenizer(example["input"], return_tensors="pt").input_ids.to(device)
    if tokenized_input_full.shape[1] <= max_tokens:
        return tokenized_input_full

    seperator_and_query_text = example['truncation_seperator'] + example["input"][example['query_start_index']:]
    tokenized_seperator_and_query = tokenizer(seperator_and_query_text, return_tensors="pt").input_ids.to(device)
    input_without_query = example['input'][:example['query_start_index']]
    tokenized_input_without_query = tokenizer(input_without_query, return_tensors="pt").input_ids.to(device)
    tokenized_input_without_query = tokenized_input_without_query[:,
                                    :max_tokens - tokenized_seperator_and_query.shape[1]]

    tokenized_input = torch.cat([tokenized_input_without_query, tokenized_seperator_and_query], dim=1)
    return tokenized_input

from transformers import AutoTokenizer, AutoModel
def main(model_name="allenai/tulu-2-dpo-7b", generations_dir="generations", max_examples_per_task=-1):
    seed = 43
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)
    print("Params:")
    print(f"model: {model_name}")
    generations_dir = os.path.join(generations_dir, model_name.replace("/", "_").replace("-", "_"))
    print(f"generations_dir: {generations_dir}")
    print(f"max_examples_per_task: {max_examples_per_task}")
    print("=" * 50)
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f"time as start: {time}")

    print("Loading tokenizer")
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_input_length = model_to_max_input_tokens[model_name]

    # Load the model
    model = AutoModel.from_pretrained(model_name)

    model = model.eval()

    print(f"{model} model loaded!, device:{model.device}")

    print("Will write to:", generations_dir)
    os.makedirs(generations_dir, exist_ok=True)
    for dataset in datasets:
        generations = dict()
        print(f"Processing {dataset}")
        time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        print(f"time as start {dataset}: {time}")
        print(f"Loading {dataset}")
        data = load_dataset("tau/zero_scrolls", dataset)
        print(f"Loaded {dataset}")

        for i, example in enumerate(data["test"]):

            if 0 < max_examples_per_task == i:
                print(f"Reached {max_examples_per_task} for {dataset}. Breaking")
                break

            model_input = process_model_input(tokenizer, example, max_input_length, device)

            prediction_token_ids = model.generate(model_input,
                                                  max_new_tokens=1024,
                                                  do_sample=False,
                                                  top_p=0,
                                                  top_k=0,
                                                  temperature=1)

            predicted_text = tokenizer.decode(prediction_token_ids[0], skip_special_tokens=True)
            breakpoint()
            generations[example["id"]] = predicted_text

        out_file_path = os.path.join(generations_dir, f"preds_{dataset}.json")
        with open(out_file_path, 'w') as f_out:
            json.dump(generations, f_out, indent=4)

        print(f"Done generating {len(generations)} examples from {dataset}")
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f"time at end: {time}")
    print(f"Look for predictions in {generations_dir}")


if __name__ == '__main__':
    Fire(main)