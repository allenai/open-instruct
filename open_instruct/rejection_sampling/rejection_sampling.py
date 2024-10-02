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

import asyncio
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import Dataset
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    PreTrainedTokenizer,
)

from open_instruct.model_utils import get_reward
from open_instruct.rejection_sampling.generation import (
    GenerationArgs,
    format_conversation,
    generate_with_openai,
)

api = HfApi()
# we don't use `multiprocessing.cpu_count()` because typically we only have 12 CPUs
# and that the shards might be small
NUM_CPUS_FOR_DATASET_MAP = 4


@dataclass
class Args:
    model_names_or_paths: List[str] = field(default_factory=lambda: ["gpt-4"])
    input_filename: str = "completions.jsonl"
    save_filename: str = "rejected_sampling_completions.jsonl"
    save_filename_scores: str = "completion_scores.jsonl"
    num_completions: int = 1
    max_forward_batch_size: int = 64
    num_gpus: int = 1  # New argument for specifying the number of GPUs
    mode: str = "judgement"
    skill: str = "chat"
    include_reference_completion_for_rejection_sampling: bool = True

    # upload config
    hf_repo_id: str = os.path.basename(__file__)[: -len(".py")]
    hf_repo_id_scores: str = os.path.basename(__file__)[: -len(".py")] + "_scores"
    push_to_hub: bool = False
    hf_entity: Optional[str] = None
    add_timestamp: bool = True


def save_jsonl(save_filename: str, table: Dict[str, List]):
    first_key = list(table.keys())[0]
    dirname = os.path.dirname(save_filename)
    if dirname:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, "w") as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")


def process_shard(
    rank: int, model_name_or_path: str, args: Args, shard: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    # Convert the list of data items (shard) into a Hugging Face Dataset object
    raw_ds = Dataset.from_list(shard)

    device = torch.device(f"cuda:{rank}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Apply a tokenization function to each item in the dataset
    ds = raw_ds.map(
        lambda x: {"input_ids": tokenizer.apply_chat_template(x["messages"])},
        remove_columns=raw_ds.column_names,
        num_proc=NUM_CPUS_FOR_DATASET_MAP,
    )
    # So this code handles only classification, I should also handle other models judges like Llama3
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device)
    model.eval()

    # Initialize a data collator to handle dynamic padding of input sequences
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    scores = batch_processing_scores(args.max_forward_batch_size, device, tokenizer, ds, model, data_collator)

    return scores


def process_shard_api(model_name_or_path: str, args: Args, shard: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function processes a shard (subset) of data using api-based models.
    It feeds data through the model to get reward scores, and handles out-of-memory errors by adjusting the batch size.

    Args:
        model_name_or_path (str): The path or name of the model to load.
        args (Args): The arguments passed to the script, containing various settings.
        shard (List[str]): A list of strings representing the shard of data to be processed.

    Returns:
        torch.Tensor: A tensor containing the reward scores for each item in the shard.
                      Shape: (num_items_in_shard,)
        torch.Tensor: A tensor containing the reward scores for each reference completion in the shard.
    """

    # Convert the list of data items (shard) into a Hugging Face Dataset object
    raw_ds = Dataset.from_list(shard)

    # for judgement mode, we need to only generate `num_completions=1`
    gen_args = GenerationArgs(num_completions=1)

    ds = raw_ds.map(
        lambda x: {"prompt": format_conversation(x["messages"][:-1])},
        num_proc=NUM_CPUS_FOR_DATASET_MAP,
    )
    prompts = ds["prompt"]
    model_responses = ds["model_completion"]

    data_list_model_responses = [
        {"prompt": prompt, "response": response} for prompt, response in zip(prompts, model_responses)
    ]
    model_responses_scores = asyncio.run(
        generate_with_openai(model_name_or_path, data_list_model_responses, args, gen_args)
    )

    return torch.Tensor(model_responses_scores)


def batch_processing_scores(
    max_forward_batch_size: int,
    device: torch.device,
    tokenizer: PreTrainedTokenizer,
    ds: Dataset,
    model: torch.nn.Module,
    data_collator: DataCollatorWithPadding,
) -> torch.Tensor:
    # NOTE: two optimizations here:
    # 1. we sort by input_ids length to reduce padding at first
    # 1.1 note that this may cause slightly different results due to numerical issues.
    #   e.g., with sort: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_1723242217
    #   e.g., without sort: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_1723242476
    # 2. we shrink the batch size if we run out of memory (so initially we can use a large batch size)
    current_batch_size = max_forward_batch_size
    input_ids_lengths = [len(x) for x in ds["input_ids"]]  # input_ids_lengths: (num_items_in_shard,)

    # Get indices that would sort the input lengths
    sorted_indices = np.argsort(input_ids_lengths)
    # Initialize a list to store the scores for each item in the shard
    scores = []
    i = 0
    while i < len(ds):
        with torch.no_grad():
            data = ds[sorted_indices[i : i + current_batch_size]]
            try:
                print(f"processing: {i}:{i + current_batch_size}/{len(ds)}")
                input_ids = data_collator(data)["input_ids"].to(device)
                _, score, _ = get_reward(model, input_ids, tokenizer.pad_token_id, 0)
                # score = (batch_size, )
                scores.extend(score.cpu().tolist())  # convert the tensor score to a list
                i += current_batch_size
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

    # include the reference completion in the completions for efficient rejection sampling
    new_completions = []
    for i in range(len(completions)):
        if i % args.num_completions == 0:
            reference_completion = copy.deepcopy(completions[i])
            reference_completion["messages"][-1]["content"] = reference_completion["reference_completion"]
            reference_completion["model_completion"] = reference_completion["reference_completion"]
            new_completions.append(reference_completion)
        new_completions.append(completions[i])
    completions = new_completions
    actual_num_completions = args.num_completions + 1  # we have added the reference completion

    # Split the data into shards
    shard_size = len(completions) // args.num_gpus
    shards = [completions[i : i + shard_size] for i in range(0, len(completions), shard_size)]

    # Process shards in parallel
    best_offsets_per_model = {}
    worst_offsets_per_model = {}
    reference_completion_scores_per_model = {}
    for model_name_or_path in args.model_names_or_paths:
        results = []
        # if use openai
        if "gpt-3.5" in model_name_or_path or "gpt-4" in model_name_or_path:
            # when using LLM as a judge, num_gpus here refers to the number of shards as we query an API and we don't use GPUs
            for i in range(args.num_gpus):
                results.append(process_shard_api(model_name_or_path, args, shards[i]))
            scores = []
            for result in results:
                scores.append(result)
        else:
            with mp.Pool(args.num_gpus) as pool:  # NOTE: the `result.get()` need to live in this `mp.Pool` context
                for i in range(args.num_gpus):
                    results.append(pool.apply_async(process_shard, (i, model_name_or_path, args, shards[i])))
                # Collect results
                scores = []
                for result in results:
                    scores.append(result.get())

        # Combine scores from all GPUs
        scores = torch.cat(scores)
        scores_per_prompt = scores.reshape(-1, actual_num_completions)  # (n_prompts, n_completions)
        reference_completion_scores = scores_per_prompt[:, 0]
        reference_completion_scores_per_model[model_name_or_path] = reference_completion_scores.tolist()

        if not args.include_reference_completion_for_rejection_sampling:
            scores_per_prompt = scores_per_prompt[:, 1:]
            scores = scores_per_prompt.flatten()
            completions = [completions[i] for i in range(len(completions)) if i % actual_num_completions != 0]
            actual_num_completions -= 1

        assert len(completions) == len(scores)
        # Rejection sampling
        for i in range(len(scores)):
            if "score" not in completions[i]:
                completions[i]["score"] = {}
            completions[i]["score"][model_name_or_path] = scores[i].item()
            if "reference_completion_score" not in completions[i]:
                completions[i]["reference_completion_score"] = {}
            completions[i]["reference_completion_score"][model_name_or_path] = reference_completion_scores[
                i // actual_num_completions
            ].item()

        best_indices = torch.argmax(scores_per_prompt, dim=1)  # (n_prompts, 1) --> (n_prompts, )
        worst_indices = torch.argmin(scores_per_prompt, dim=1)  # (n_prompts, 1) --> (n_prompts, )
        best_indices_offset = (
            torch.arange(0, len(best_indices) * actual_num_completions, actual_num_completions) + best_indices
        )
        best_offsets_per_model[model_name_or_path] = best_indices_offset

        worst_indices_offset = (
            torch.arange(0, len(worst_indices) * actual_num_completions, actual_num_completions) + worst_indices
        )
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
        table["reference_completion_score"].append(
            {key: reference_completion_scores_per_model[key][i] for key in reference_completion_scores_per_model}
        )
        assert worst_completions[i]["messages"][:-1] == best_completions[i]["messages"][:-1]
        table["chosen_score"].append(best_completions[i]["score"])
        table["rejected_score"].append(worst_completions[i]["score"])
    save_jsonl(args.save_filename, table)

    table_scores = defaultdict(list)
    keys = list(completions[0].keys())
    for i in range(len(completions)):
        for key in keys:
            table_scores[key].append(completions[i][key])
    save_jsonl(args.save_filename_scores, table_scores)

    if args.push_to_hub:
        if args.hf_entity is None:
            args.hf_entity = api.whoami()["name"]
        full_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        timestamp = f"_{int(time.time())}"
        if args.add_timestamp:
            full_repo_id += timestamp
        api.create_repo(full_repo_id, repo_type="dataset", exist_ok=True)
        for f in [__file__, args.save_filename]:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f.split("/")[-1],
                repo_id=full_repo_id,
                repo_type="dataset",
            )
        repo_full_url = f"https://huggingface.co/datasets/{full_repo_id}"
        print(f"Pushed to {repo_full_url}")
        run_command = " ".join(["python"] + sys.argv)
        sft_card = RepoCard(
            content=f"""\
# allenai/open_instruct: Rejection Sampling Dataset

See https://github.com/allenai/open-instruct/blob/main/docs/algorithms/rejection_sampling.md for more detail

## Configs

```
args:
{pformat(vars(args))}
```

## Additional Information

1. Command used to run `{run_command}`
"""
        )
        sft_card.push_to_hub(
            full_repo_id,
            repo_type="dataset",
        )

        full_repo_id_scores = f"{args.hf_entity}/{args.hf_repo_id_scores}"
        if args.add_timestamp:
            full_repo_id_scores += timestamp
        api.create_repo(full_repo_id_scores, repo_type="dataset", exist_ok=True)
        for f in [__file__, args.save_filename_scores]:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f.split("/")[-1],
                repo_id=full_repo_id_scores,
                repo_type="dataset",
            )
        repo_full_url_scores = f"https://huggingface.co/datasets/{full_repo_id_scores}"
        print(f"Pushed to {repo_full_url_scores}")
        sft_card.push_to_hub(
            full_repo_id_scores,
            repo_type="dataset",
        )


if __name__ == "__main__":
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
