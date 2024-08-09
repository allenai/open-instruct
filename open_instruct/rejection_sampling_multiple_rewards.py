import os
import numpy as np

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import time
import torch
import torch.multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple, List
from transformers import (
    HfArgumentParser,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
)

from tqdm import tqdm
from datasets import Dataset
import json
from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from collections import Counter
from typing import Union

api = HfApi()


@dataclass
class Args:
    model_names_or_paths: List[str] = field(default_factory=lambda: ["allenai/llama-3-tulu-2-8b-uf-mean-rm", "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr", "weqweasdas/RM-Mistral-7B"])
    input_filename: str = "completions.jsonl"
    save_filename: str = "rejected_sampling_completions.jsonl"
    n: int = 1
    max_forward_batch_size: int = 8
    num_gpus: int = 1  # New argument for specifying the number of GPUs
    push_to_hub: bool = False
    hf_entity: Optional[str] = None
    hf_repo_id: str = "rejection_sampling"
    add_timestamp: bool = True


def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    """
    Finds the index of the first `True` value in each row of a boolean tensor. If no `True` value exists in a row,
    it returns the length of the row.

    Args:
        bools (torch.Tensor): A boolean tensor of shape (batch_size, sequence_length), where `True` values indicate
                              the positions of interest.
        dtype (torch.dtype): The data type to use for the output indices (default is torch.long).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the index of the first `True` value in each row.
                      If a row has no `True` value, the index will be the length of the row.
    """

    # Get the length of each row (i.e., the number of columns in the last dimension)
    # row_len is a scalar representing the length of each sequence (sequence_length)
    row_len = bools.size(-1)

    # Calculate the index positions for the first `True` in each row
    # ~bools: Invert the boolean values (True becomes False and vice versa)
    # ~bools.type(dtype): Convert the inverted boolean tensor to the specified dtype (0 for True, 1 for False)
    # row_len * (~bools).type(dtype): For `False` values, this will give `row_len`, for `True` values it gives 0.
    # torch.arange(row_len, dtype=dtype, device=bools.device): Generates a tensor with values [0, 1, 2, ..., row_len-1]
    # for each row. Shape: (sequence_length,)
    # zero_or_index: Shape (batch_size, sequence_length). This tensor contains the indices for `True` values and `row_len`
    # for `False` values.
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)

    # Return the minimum value in each row (i.e., the first `True` index or `row_len` if none exist)
    # torch.min(zero_or_index, dim=-1).values: This returns the minimum value in each row, which corresponds to the first
    # `True` value's index or `row_len` if there is no `True` in that row.
    # The returned tensor has shape (batch_size,)
    return torch.min(zero_or_index, dim=-1).values


def get_reward(
        model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function computes reward scores for a batch of query responses based on a pre-trained reward model.

    Args:
        model (torch.nn.Module): The pre-trained reward model.
        query_responses (torch.Tensor): Tensor containing the tokenized responses for which to compute rewards.
            Shape: (batch_size, sequence_length)
        pad_token_id (int): The ID used for padding tokens in the tokenized sequences.
        context_length (int): The length of the prompt or context preceding the completions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - reward_logits: The logits output from the model for all tokens in the sequences.
              Shape: (batch_size, sequence_length)
            - final_scores: The final reward scores, one for each sequence, after adjusting for sequence lengths.
              Shape: (batch_size,)
            - sequence_lengths: The lengths of each sequence (excluding padding).
              Shape: (batch_size,)
    """

    # Create an attention mask where tokens that are not padding have a value of 1, and padding tokens have a value of 0
    # Shape: (batch_size, sequence_length)
    attention_mask = query_responses != pad_token_id

    # Calculate position IDs for each token, considering the cumulative sum of the attention mask (to exclude padding)
    # Shape: (batch_size, sequence_length)
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum

    # Access the LM backbone from the reward model using its base model prefix
    lm_backbone = getattr(model, model.base_model_prefix)

    # Replace padding tokens with zeros in the input IDs (so padding tokens won't affect the model's processing)
    # Shape: (batch_size, sequence_length)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])  # (batch_size, sequence_length)

    # Calculate the length of each sequence by finding the first occurrence of a padding token after the context
    # sequence_lengths shape: (batch_size,)
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454

    # Return the reward logits for all tokens, the final reward scores for each sequence, and the sequence lengths
    return (
        # reward_logits shape: (batch_size, sequence_length)
        reward_logits,
        # final_scores shape: (batch_size,)
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1), # Shape: (batch_size,)
        sequence_lengths,
    )


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
    ds = ds.map(
        lambda x: {"input_ids": tokenizer.apply_chat_template(x["messages"])},
        remove_columns=ds.column_names
    )

    ### So this code handles only classification, I should also handle other models judges like Llama3
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
    input_ids_lengths = [len(x) for x in ds["input_ids"]] #  input_ids_lengths: (num_items_in_shard,)

    sorted_indices = np.argsort(input_ids_lengths) # Get indices that would sort the input lengths

    # Initialize a list to store the scores for each item in the shard
    scores = []
    i = 0
    while i < len(ds):
        with torch.no_grad():
            data = ds[sorted_indices[i:i + current_batch_size]]
            try:
                input_ids = data_collator(data)["input_ids"].to(device)
                _, score, _ = get_reward(model, input_ids, tokenizer.pad_token_id, 0)
                # score = (batch_size, )
                scores.extend(score.cpu().tolist()) # convert the tensor score to a list
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
    mp.set_start_method('spawn', force=True)

    # Load the completions from a file
    with open(args.input_filename, 'r') as infile:
        completions = [json.loads(line) for line in infile]

    # Split the data into shards
    shard_size = len(completions) // args.num_gpus
    shards = [completions[i:i + shard_size] for i in range(0, len(completions), shard_size)]

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
        scores_per_prompt = scores.reshape(-1, args.n) # (n_prompts, n_completions)
        for i in range(len(completions)):
            if "score" not in completions[i]:
                completions[i]["score"] = {}
            completions[i]["score"][model_name_or_path] = scores[i].item()

        best_indices = torch.argmax(scores_per_prompt, dim=1) # (n_prompts, 1) --> (n_prompts, )
        worst_indices = torch.argmin(scores_per_prompt, dim=1) # (n_prompts, 1) --> (n_prompts, )
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
    with open(args.save_filename, 'w') as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write('\n')

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