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
import itertools
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pprint import pformat

from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from litellm import acompletion
from tqdm.asyncio import tqdm_asyncio
from transformers import HfArgumentParser

api = HfApi()
# we don't use `multiprocessing.cpu_count()` because typically we only have 12 CPUs
# and that the shards might be small
NUM_CPUS_FOR_DATASET_MAP = 4


@dataclass
class Args:
    model_names_or_paths: list[str] = field(default_factory=lambda: ["gpt-4"])
    input_filename: str = "completions.jsonl"
    save_filename: str = "rejected_sampling_completions.jsonl"
    num_completions: int = 1

    # upload config
    hf_repo_id: str = os.path.basename(__file__)[: -len(".py")]
    hf_repo_id_scores: str = os.path.basename(__file__)[: -len(".py")] + "_scores"
    push_to_hub: bool = False
    hf_entity: str | None = None
    add_timestamp: bool = True

    # judgement config
    model: str = "gpt-4o-2024-08-06"
    max_parallel_requests: int | None = None

    def __post_init__(self):
        # these are just experience numbers to avoid rate limits, but feel free to adjust
        if "gpt-3.5" in self.model or "gpt-4o" in self.model:
            self.max_parallel_requests = 100
        elif "gpt-4" in self.model:
            self.max_parallel_requests = 13


def save_jsonl(save_filename: str, table: dict[str, list]):
    first_key = list(table.keys())[0]
    dirname = os.path.dirname(save_filename)
    if dirname:
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    with open(save_filename, "w") as outfile:
        for i in range(len(table[first_key])):
            json.dump({key: table[key][i] for key in table}, outfile)
            outfile.write("\n")


# The prompt comes from https://arxiv.org/pdf/2203.02155, p. 37
TEMPLATE = r"""
You are given a text-based description of a task, submitted by a user. This task description may be in the form of an explicit instruction (e.g. "Write a story about a wise frog."). The task may also be specified indirectly, for example by using several examples of the desired behavior (e.g. given a sequence of movie reviews followed by their sentiment, followed by one movie review without a sentiment, you can assume the task is to predict the sentiment of the final review), or by generating the start of a desired output (e.g. given "There once was a wise frog named Julius," you can assume the task is to continue the story). You will also be given several text outputs, intended to help the user with their task. Your job is to evaluate these outputs to ensure that they are helpful, truthful, and harmless. For most tasks, being truthful and harmless is more important than being helpful.
By helpful, we mean that the output should follow the user's intention, and help the user solve their task. Some examples of helpful behavior:
- Writing in clear language.
- Answering the question they meant to ask, even if they mis-asked it.
- Being sensitive to internationality (e.g. "football" shouldn't mean American football, and "the president" doesn't necessarily mean the US president).
- If an instruction is too confusing, asking for clarification and explaining why the instruction is confusing.
- Not giving overly long or rambling answers, or repeating information from the question.
- Not assuming extraneous extra context outside of what's given (besides things like facts about the world), unless that's an implied part of the task. For example, if asked to "write a polite email response to this email: email body" the output shouldn't assume "I can't make it this time, but am free next weekend." But if asked to "Write an email to Socrates", then you can safely assume the output can make up a context.
By truthful, we mean that the output contains accurate information, and doesn't mislead the user. Some examples of truthful behavior:
- On tasks like summarization, where the output should only use information from input, not making up details that are not part of the input description.
- Not producing clearly false information about the world (e.g. making up facts or promoting conspiracies). For example, the output should not state that Hillary Clinton has served time in prison.
- Avoid generating misleading information, or information with questionable authenticity. For example, if the input asks "Why did Hillary Clinton go to jail?", the output should not say "It's not totally clear", but rather should refute the premise of the question.
By harmless, we mean that the output should not cause physical, psychological, or social harm to people; damage to or loss of equipment or property; damage to the environment; or harm to institutions or resources necessary to human wellbeing. Some examples of harmless behavior:
- Treating other humans with kindness, respect and consideration; not denigrating members of certain groups, or using biased language against a particular group.
- Not generating abusive, threatening, or offensive language, or promoting violence.
- Not writing sexual or violent content if it's not asked for.
- Not giving bad real-world advice, or promoting illegal activity.
Evaluating model outputs may involve making trade-offs between these criteria. These trade-offs will depend on the task. Use the following guidelines to help select between outputs when making these trade-offs:
For most tasks, being harmless and truthful is more important than being helpful. So in most cases, rate an output that's more truthful and harmless higher than an output that's more helpful. However, if: (a) one output is much more helpful than the other; (b) that output is only slightly less truthful / harmless; and (c) the task does not seem to be in a "high stakes domain" (e.g. loan applications, therapy, medical or legal advice, etc.); then rate the more helpful output higher. When choosing between outputs that are similarly helpful but are untruthful or harmful in different ways, ask: which output is more likely to cause harm to an end user (the people who will be most impacted by the task in the real world)? This output should be ranked lower. If this isn't clear from the task, then mark these outputs as tied.
A guiding principle for deciding on borderline cases: which output would you rather receive from a customer assistant who is trying to help you with this task?
Ultimately, making these tradeoffs can be challenging and you should use your best judgment.
### Task:
{{task}}
### Response 0:
{{response0}}
### Response 1:
{{response1}}
### Instructions:
FIRST, provide a one-sentence comparison of the two responses, explaining which you prefer and why. SECOND, on a new line, state only "0" or "1" to indicate your choice. Your response should use the format:
Comparison: <one-sentence comparison and explanation>
Preferred: <"0" or "1">
"""


def main(args: Args):
    # Load the completions from a file
    with open(args.input_filename) as infile:
        completions = [json.loads(line) for line in infile]

    all_comparison_pairs = []

    ADD_COMPARISON_WITH_REFERENCE = True
    for i in range(0, len(completions), args.num_completions):
        prompt_completions = completions[i : i + args.num_completions]
        reference_completion = completions[i]["reference_completion"]
        if ADD_COMPARISON_WITH_REFERENCE:
            prompt_referecen_completion = copy.deepcopy(completions[i])
            prompt_referecen_completion["messages"][-1]["content"] = reference_completion
            prompt_completions.append(prompt_referecen_completion)
        comparison_pairs = itertools.combinations(prompt_completions, 2)
        all_comparison_pairs.extend(comparison_pairs)

    async def get_judgement(model: str, comparison_pair: list[dict[str, str]], limiter: asyncio.Semaphore):
        task = comparison_pair[0]["messages"][:-1]
        # shuffle the order of the responses
        shuffled_index = random.randint(0, 1)
        response0 = comparison_pair[shuffled_index]["messages"][-1]
        response1 = comparison_pair[1 - shuffled_index]["messages"][-1]
        template = (
            TEMPLATE.replace("{{task}}", str(task))
            .replace("{{response0}}", str(response0))
            .replace("{{response1}}", str(response1))
        )
        messages = [{"content": template, "role": "user"}]
        MAX_TRIES = 3
        async with limiter:
            for i in range(MAX_TRIES):
                try:
                    response = await acompletion(model=model, messages=messages)
                    r = response.choices[0].message.content
                    comparison = r.split("Comparison:")[1].split("Preferred:")[0].strip()
                    preferred = r.split("Preferred:")[1].strip()

                    chosen = comparison_pair[0]["messages"]
                    rejected = comparison_pair[1]["messages"]
                    # reverse the preferred choice if the responses were shuffled
                    if preferred == "0" and shuffled_index == 1 and preferred == "1" and shuffled_index == 0:
                        chosen, rejected = rejected, chosen
                    messages.append({"content": r, "role": "assistant"})
                    return chosen, rejected, comparison, messages
                except Exception as e:
                    print(f"Error: {e}")
                    if i == MAX_TRIES - 1:
                        return None

    print(f"{len(all_comparison_pairs)=}")

    async def get_judgement_all(model: str, all_comparison_pairs: list[list[dict[str, str]]]):
        limiter = asyncio.Semaphore(args.max_parallel_requests)
        tasks = []
        for comparison_pair in all_comparison_pairs:
            task = asyncio.create_task(get_judgement(model, comparison_pair, limiter))
            tasks.append(task)

        return await tqdm_asyncio.gather(*tasks)

    items = asyncio.run(get_judgement_all(args.model, all_comparison_pairs))

    # Save results
    table = defaultdict(list)
    for i in range(len(items)):
        table["chosen"].append(items[i][0])
        table["rejected"].append(items[i][1])
        table["comparison"].append(items[i][2])
        table["whole_conversation"].append(items[i][3])
    save_jsonl(args.save_filename, table)

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
                path_or_fileobj=f, path_in_repo=f.split("/")[-1], repo_id=full_repo_id, repo_type="dataset"
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
        sft_card.push_to_hub(full_repo_id, repo_type="dataset")


if __name__ == "__main__":
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
