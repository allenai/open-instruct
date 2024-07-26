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
from datasets import Dataset
import json
from torch.utils.data import DataLoader
from huggingface_hub import HfApi
api = HfApi()


"""
python rejection_sampling.py \
    --input_filename completions.jsonl \
    --save_filename rejection_sampled_completions.jsonl \
    --n 3 \
    --num_gpus 2 \
    --push_to_hub \
"""

# 1. split up data manually
# 2. do map reduce style 

@dataclass
class Args:
    model_name_or_path: str = "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
    input_filename: str = "completions.jsonl"
    save_filename_prefix: str = "rejected_sampling_completions"
    n: int = 1
    forward_batch_size: int = 10
    num_gpus: int = 1  # New argument for specifying the number of GPUs
    push_to_hub: bool = False
    hf_entity: Optional[str] = None
    hf_repo_id: str = "rejection_sampling"


def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def get_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )

def process_shard(rank: int, args: Args, shard: List[str]):
    device = torch.device(f"cuda:{rank}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    ds = Dataset.from_list(shard)
    ds = ds.map(
        lambda x: {"input_ids": tokenizer.encode(x["prompt"] + x["completion"])},
        remove_columns=ds.column_names
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    model = model.to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(ds, batch_size=args.forward_batch_size, collate_fn=data_collator, pin_memory=True)
    scores = []
    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"].to(device)
            _, score, _ = get_reward(model, input_ids, tokenizer.pad_token_id, 0)
            scores.append(score.cpu())
    
    return torch.cat(scores)

def main(args: Args):
    mp.set_start_method('spawn', force=True)
    
    # Load the completions from a file
    with open(args.input_filename, 'r') as infile:
        completions = [json.loads(line) for line in infile]
    
    # Split the data into shards
    shard_size = len(completions) // args.num_gpus
    shards = [completions[i:i+shard_size] for i in range(0, len(completions), shard_size)]
    
    # Process shards in parallel
    with mp.Pool(args.num_gpus) as pool:
        results = []
        for i in range(args.num_gpus):
            results.append(pool.apply_async(process_shard, (i, args, shards[i])))
        
        # Collect results
        scores = []
        for result in results:
            scores.append(result.get())
    
    # Combine scores from all GPUs
    scores = torch.cat(scores)
    
    # Rejection sampling
    scores_per_prompt = scores.reshape(-1, args.n)
    for i in range(len(completions)):
        completions[i]["score"] = scores[i].item()
    best_indices = torch.argmax(scores_per_prompt, dim=1)
    worst_indices = torch.argmin(scores_per_prompt, dim=1)
    best_indices_offset = torch.arange(0, len(best_indices) * args.n, args.n) + best_indices
    worst_indices_offset = torch.arange(0, len(worst_indices) * args.n, args.n) + worst_indices
    best_completions = [completions[i] for i in best_indices_offset]
    worst_completions = [completions[i] for i in worst_indices_offset]
    
    # Save results
    with open(f"{args.save_filename_prefix}_best.jsonl", 'w') as outfile:
        for i in range(len(best_completions)):
            json.dump(best_completions[i], outfile)
            outfile.write('\n')
    with open(f"{args.save_filename_prefix}_worst.jsonl", 'w') as outfile:
        for i in range(len(worst_completions)):
            json.dump(worst_completions[i], outfile)
            outfile.write('\n')
    table = defaultdict(list)
    for i in range(len(best_completions)):
        table["chosen"].append([
            {"content": best_completions[i]["prompt"], "role": "user"},
            {"content": best_completions[i]["prompt"], "role": "assistant"},
        ])
        table["rejected"].append([
            {"content": worst_completions[i]["prompt"], "role": "user"},
            {"content": worst_completions[i]["prompt"], "role": "assistant"},
        ])
        assert worst_completions[i]["prompt"] == best_completions[i]["prompt"]
        table["chosen_score"].append(best_completions[i]["score"])
        table["rejected_score"].append(worst_completions[i]["score"])
    ds = Dataset.from_dict(table)
    if args.push_to_hub:
        if args.hf_entity is None:
            args.hf_entity = api.whoami()["name"]
        full_repo_id = f"{args.hf_entity}/{args.hf_repo_id}_{int(time.time())}"
        ds.push_to_hub(full_repo_id)
        api.upload_file(
            path_or_fileobj=__file__,
            path_in_repo=__file__.split("/")[-1],
            repo_id=full_repo_id,
            repo_type="dataset",
        )

if __name__ == "__main__":
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]
    main(args)