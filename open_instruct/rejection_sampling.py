# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
)

from open_instruct.model_utils import get_reward

api = HfApi()


@dataclass
class Args:
    model_names_or_paths: List[str] = field(
        default_factory=lambda: ["cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"]
    )
    input_filename: str = "completions.jsonl"
    save_filename: str = "rejected_sampling_completions.jsonl"
    n: int = 1
    max_forward_batch_size: int = 8
    num_gpus: int = 1  # New argument for specifying the number of GPUs
    push_to_hub: bool = False
    hf_entity: Optional[str] = None
    hf_repo_id: str = "rejection_sampling"
    add_timestamp: bool = True


def process_shard(rank: int, model_name_or_path: str, args: Args, shard: List[str]):
    """
    This function processes a shard (subset) of data using a specified model. It tokenizes the data,
    runs it through the model to get reward scores, and handles out-of-memory errors by adjusting the batch size.

    Args:
        rank (int): The GPU rank (index) to use for processing.
        model_name_or_path (str): The path or name of the model to load.
        args (Args): The arguments passed to the script, containing various settings.
        shard (List[str]): A list of strings representing the shard of data to be processed.

    Returns:
        torch.Tensor: A tensor containing the reward scores for each item in the shard.
                      Shape: (num_items_in_shard,)
    """
    device = torch.device(f"cuda:{rank}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Convert the list of data items (shard) into a Hugging Face Dataset object
    ds = Dataset.from_list(shard)
    # Apply a tokenization function to each item in the dataset
    ds = ds.map(lambda x: {"input_ids": tokenizer.apply_chat_template(x["messages"])}, remove_columns=ds.column_names)

    # So this code handles only classification, I should also handle other models judges like Llama3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)

    # Initialize a data collator to handle dynamic padding of input sequences
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    current_batch_size = args.max_forward_batch_size
    # NOTE: two optimizations here:
    # 1. we sort by input_ids length to reduce padding at first
    # 2. we shrink the batch size if we run out of memory (so initially we can use a large batch size)
    input_ids_lengths = [len(x) for x in ds["input_ids"]]  # input_ids_lengths: (num_items_in_shard,)

    sorted_indices = np.argsort(input_ids_lengths)  # Get indices that would sort the input lengths

    # Initialize a list to store the scores for each item in the shard
    scores = []
    i = 0
    while i < len(ds):
        with torch.no_grad():
            data = ds[sorted_indices[i : i + current_batch_size]]
            try:
                input_ids = data_collator(data)["input_ids"].to(device)
                _, score, _ = get_reward(model, input_ids, tokenizer.pad_token_id, 0)
                # score = (batch_size, )
                scores.extend(score.cpu().tolist())  # convert the tensor score to a list
                i += current_batch_size
                print(f"processing: {i}:{i + current_batch_size}/{len(ds)}")
            except torch.cuda.OutOfMemoryError:
                if current_batch_size == 1:
                    raise ValueError("Out of memory even with batch size 1")
                current_batch_size //= 2
                print(f"Reducing batch size to {current_batch_size}")
                continue
    # restore the original order
    scores = np.array(scores)
    scores = scores[np.argsort(sorted_indices)]
    return torch.tensor(scores)


def majority_vote(offsets_per_model: dict[str, torch.tensor]) -> torch.tensor:
    """
    offsets_per_model: offsets returned by each model. each tensor is of shape (n_prompts,) indicating best/worst completion offset per prompt
    """
    # Determine the number of samples
    num_samples = offsets_per_model[next(iter(offsets_per_model))].size(0)
    # Initialize tensor to store the majority votes
    majority_votes = torch.zeros(num_samples, dtype=torch.long)

    # Tally the votes and determine the majority vote for each sample
    for i in range(num_samples):
        # Collect votes from all models for the current sample
        votes = [offsets_per_model[model][i].item() for model in offsets_per_model]
        # Determine the most common vote
        counter = Counter(votes)
        # Try to get ther majority vote, but if all models disagree, we randomly choose one
        if len(offsets_per_model) != len(counter):
            majority_vote = counter.most_common(1)[0][0]
        else:
            majority_vote = votes[np.random.randint(len(votes))]
        # Store the majority vote in the tensor
        majority_votes[i] = majority_vote

    return majority_votes


def main(args: Args):
    mp.set_start_method("spawn", force=True)

    # Load the completions from a file
    with open(args.input_filename, "r") as infile:
        completions = [json.loads(line) for line in infile]

    # Split the data into shards
    shard_size = len(completions) // args.num_gpus
    shards = [completions[i : i + shard_size] for i in range(0, len(completions), shard_size)]

    # Process shards in parallel
    best_offsets_per_model = {}
    worst_offsets_per_model = {}
    for model_name_or_path in args.model_names_or_paths:
        with mp.Pool(args.num_gpus) as pool:
            results = []
            for i in range(args.num_gpus):
                results.append(pool.apply_async(process_shard, (i, model_name_or_path, args, shards[i])))

            # Collect results
            scores = []
            for result in results:
                scores.append(result.get())

        # Combine scores from all GPUs
        scores = torch.cat(scores)

        # Rejection sampling
        scores_per_prompt = scores.reshape(-1, args.n)  # (n_prompts, n_completions)
        for i in range(len(completions)):
            if "score" not in completions[i]:
                completions[i]["score"] = {}
            completions[i]["score"][model_name_or_path] = scores[i].item()

        best_indices = torch.argmax(scores_per_prompt, dim=1)  # (n_prompts, 1) --> (n_prompts, )
        worst_indices = torch.argmin(scores_per_prompt, dim=1)  # (n_prompts, 1) --> (n_prompts, )
        best_indices_offset = torch.arange(0, len(best_indices) * args.n, args.n) + best_indices
        best_offsets_per_model[model_name_or_path] = best_indices_offset

        worst_indices_offset = torch.arange(0, len(worst_indices) * args.n, args.n) + worst_indices
        worst_offsets_per_model[model_name_or_path] = worst_indices_offset

    # Majority vote
    best_indices_offset = majority_vote(best_offsets_per_model)
    worst_indices_offset = majority_vote(worst_offsets_per_model)

    best_completions = [completions[i] for i in best_indices_offset]
    worst_completions = [completions[i] for i in worst_indices_offset]

    # Save results
    table = defaultdict(list)
    for i in range(len(best_completions)):
        table["chosen"].append(best_completions[i]["messages"])
        table["rejected"].append(worst_completions[i]["messages"])
        table["reference_completion"].append(worst_completions[i]["reference_completion"])
        assert worst_completions[i]["messages"][:-1] == best_completions[i]["messages"][:-1]
        table["chosen_score"].append(best_completions[i]["score"])
        table["rejected_score"].append(worst_completions[i]["score"])
    first_key = list(table.keys())[0]
    print(f"{len(table[first_key])=}")
    with open(args.save_filename, "w") as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")

    if args.push_to_hub:
        if args.hf_entity is None:
            args.hf_entity = api.whoami()["name"]
        full_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.add_timestamp:
            full_repo_id += f"_{int(time.time())}"
        api.create_repo(full_repo_id, repo_type="dataset", exist_ok=True)
        for f in [__file__, args.save_filename]:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f.split("/")[-1],
                repo_id=full_repo_id,
                repo_type="dataset",
            )
        print(f"Pushed to https://huggingface.co/datasets/{full_repo_id}/")


if __name__ == "__main__":
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
