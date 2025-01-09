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

"""
python open_instruct/rejection_sampling/generation_check_ground_truth.py \
    --model_name_or_path allenai/open_instruct_dev \
    --revision L3.1-8B-v3.9-nc-fixed-2__meta-llama_Llama-3.1-8B__123__1730531285 \
    --num_completions 3 \
    --dataset_mixer_list ai2-adapt-dev/math_ground_truth 1.0 \
    --dataset_splits train \
    --dataset_end_idx 10


python open_instruct/rejection_sampling/generation_check_ground_truth.py \
    --model_name_or_path allenai/open_instruct_dev \
    --revision olmo_7b_soup_anneal_v3.9_4_DPO___model__42__1730863426 \
    --num_completions 5 \
    --dataset_mixer_list ai2-adapt-dev/gsm8k_ground_truth 1.0 \
    --dataset_splits train \
    --dataset_end_idx 20
"""
import asyncio
import copy
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Dict, List, Optional

from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from rich.pretty import pprint
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from open_instruct.dataset_processor import (
    INPUT_IDS_PROMPT_KEY,
    DatasetConfig,
    SFTDatasetProcessor,
)
from open_instruct.model_utils import apply_verifiable_reward
from open_instruct.utils import ArgumentParserPlus, check_hf_olmo_availability, combine_dataset

api = HfApi()
# we don't use `multiprocessing.cpu_count()` because typically we only have 12 CPUs
# and that the shards might be small
NUM_CPUS_FOR_DATASET_MAP = 4
if check_hf_olmo_availability():
    # allows AutoModel... to work with not in transformers olmo models
    import hf_olmo  # noqa
    from hf_olmo import OLMoTokenizerFast
    from open_instruct.olmo_adapter.olmo_new import OlmoNewForCausalLM
    from vllm.model_executor.models import ModelRegistry
    ModelRegistry.register_model("OLMoForCausalLM", OlmoNewForCausalLM)

@dataclass
class Args:
    dataset_mixer_list: List[str]
    dataset_splits: List[str] = None
    dataset_start_idx: int = 0
    dataset_end_idx: Optional[int] = None

    model_name_or_path: str = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
    revision: str = "main"
    save_filename: str = "completions.jsonl"

    # upload config
    hf_repo_id: str = os.path.basename(__file__)[: -len(".py")]
    push_to_hub: bool = False
    hf_entity: Optional[str] = None
    add_timestamp: bool = True


@dataclass
class GenerationArgs:
    num_completions: int = 3
    temperature: float = 0.8
    response_length: int = 2048
    top_p: float = 0.9
    tensor_parallel_size: int = 1


def save_jsonl(save_filename: str, table: Dict[str, List]):
    first_key = list(table.keys())[0]
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, "w") as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")


def generate_with_vllm(model_name_or_path: str, revision: str, prompt_token_ids: List[int], gen_args: GenerationArgs):
    llm = LLM(
        model=model_name_or_path,
        revision=revision,
        tokenizer_revision=revision,
        tensor_parallel_size=gen_args.tensor_parallel_size,
        max_model_len=gen_args.response_length,
    )

    # filter out prompts which are beyond the model's max token length
    max_model_len = llm.llm_engine.scheduler_config.max_model_len
    prompt_token_ids_len = len(prompt_token_ids)
    prompt_token_ids = [item for item in prompt_token_ids if len(item) < max_model_len]
    if len(prompt_token_ids) != prompt_token_ids_len:
        print(f"Filtered out {prompt_token_ids_len - len(prompt_token_ids)} prompts which exceeds max token length")

    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            n=gen_args.num_completions,
            temperature=gen_args.temperature,
            top_p=1.0,
            max_tokens=gen_args.response_length,
            include_stop_str_in_output=True,
        ),
    )

    response_ids = [list(out.token_ids) for output in outputs for out in output.outputs]
    return response_ids


def format_conversation(messages: list) -> str:
    formatted_conversation = []

    # Iterate through the messages
    for message in messages:  # Exclude the last assistant message
        role = "User A" if message["role"] == "user" else "User B"
        content = message["content"].strip()
        formatted_conversation.append(f"{role}: {content}")

    # Join the conversation with a single newline
    return "\n".join(formatted_conversation)


def main(args: Args, dataset_config: DatasetConfig, gen_args: GenerationArgs):
    if len(args.dataset_splits) != len(args.dataset_mixer_list) // 2:
        args.dataset_splits = ["train"] * (len(args.dataset_mixer_list) // 2)
        print(f"Using default dataset_splits: {args.dataset_splits} for {(len(args.dataset_mixer_list) // 2)} datasets")
    
    dataset = combine_dataset(
        args.dataset_mixer_list,
        splits=args.dataset_splits,
        columns_to_keep=[dataset_config.sft_messages_key, dataset_config.ground_truths_key, dataset_config.dataset_source_key],
    )
    if args.dataset_end_idx is None:
        args.dataset_end_idx = len(dataset)
    dataset = dataset.select(range(args.dataset_start_idx, args.dataset_end_idx))
    pprint([dataset_config, args, gen_args])


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.revision)
    dataset_processor = SFTDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    dataset = dataset_processor.tokenize(dataset)
    dataset = dataset_processor.filter(dataset)
    prompt_token_ids = dataset[INPUT_IDS_PROMPT_KEY]
    ground_truth = dataset[dataset_config.ground_truths_key]
    dataset_source = dataset[dataset_config.dataset_source_key]
    response_ids = generate_with_vllm(args.model_name_or_path, args.revision, prompt_token_ids, gen_args)


    # repeat prompt_token_ids, ground truth, dataset source args.num_completions times
    prompt_token_ids = [prompt_token_ids[i] for i in range(len(prompt_token_ids)) for _ in range(gen_args.num_completions)]
    ground_truth = [ground_truth[i] for i in range(len(ground_truth)) for _ in range(gen_args.num_completions)]
    dataset_source = [dataset_source[i] for i in range(len(dataset_source)) for _ in range(gen_args.num_completions)]

    # left pad prompt token ids with 0
    max_seq_len = max(len(item) for item in prompt_token_ids)
    padded_prompt_token_ids = [[0] * (max_seq_len - len(item)) + item for item in prompt_token_ids]
    # right pad response token ids with 0
    max_seq_len = max(len(item) for item in response_ids)
    padded_response_ids = [item + [0] * (max_seq_len - len(item)) for item in response_ids]
    padded_prompt_token_ids = torch.tensor(padded_prompt_token_ids)
    padded_response_ids = torch.tensor(padded_response_ids)
    query_response = torch.concat([padded_prompt_token_ids, padded_response_ids], dim=1)
    verifiable_reward, _ = apply_verifiable_reward(
        query_response,
        tokenizer,
        ground_truth,
        dataset_source,
        verify_reward=10,
    )
    import math
    verifiable_reward = verifiable_reward.reshape(-1, gen_args.num_completions)
    pass_at_k = (verifiable_reward.sum(dim=1) > 1).float().mean()
    maj_at_k = (verifiable_reward.sum(dim=1) > math.ceil(gen_args.num_completions / 2)).float().mean()
    printa = lambda i: print(tokenizer.decode(query_response[i]), ground_truth[i])
    print(f"{verifiable_reward=}")
    print(f"{pass_at_k=}")
    print(f"{maj_at_k=}")
    breakpoint()

    # save_jsonl(args.save_filename, table)

#     if args.push_to_hub:
#         if args.hf_entity is None:
#             args.hf_entity = api.whoami()["name"]
#         full_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
#         timestamp = f"_{int(time.time())}"
#         if args.add_timestamp:
#             full_repo_id += timestamp
#         api.create_repo(full_repo_id, repo_type="dataset", exist_ok=True)
#         for f in [__file__, args.save_filename]:
#             api.upload_file(
#                 path_or_fileobj=f,
#                 path_in_repo=f.split("/")[-1],
#                 repo_id=full_repo_id,
#                 repo_type="dataset",
#             )
#         repo_full_url = f"https://huggingface.co/datasets/{full_repo_id}"
#         print(f"Pushed to {repo_full_url}")
#         run_command = " ".join(["python"] + sys.argv)
#         sft_card = RepoCard(
#             content=f"""\
# # allenai/open_instruct: Generation Dataset

# See https://github.com/allenai/open-instruct/blob/main/docs/algorithms/rejection_sampling.md for more detail

# ## Configs

# ```
# args:
# {pformat(vars(args))}

# dataset_config:
# {pformat(vars(dataset_config))}

# gen_args:
# {pformat(vars(gen_args))}
# ```

# ## Reproduce this dataset

# 1. Download the `{[f.split("/")[-1] for f in [__file__, args.save_filename]]}` from the {repo_full_url}.
# 2. Run `{run_command}`
# """
#         )
#         sft_card.push_to_hub(
#             full_repo_id,
#             repo_type="dataset",
#         )


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, DatasetConfig, GenerationArgs))
    main(*parser.parse())
