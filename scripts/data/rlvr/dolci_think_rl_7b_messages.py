"""Create a lean, retokenizable version of allenai/Dolci-Think-RL-7B.

The upstream dataset was pre-tokenized for the Olmo-3-7B tokenizer (input_ids,
input_ids_prompt, attention_mask, labels) and has no "messages" column, so it
can't be re-tokenized for a different model via the standard RLVR pipeline
(rlvr_tokenize_v3 requires a "messages" list). This script drops the
pre-tokenized fields and all extra metadata, keeping only messages,
ground_truth, dataset, custom_id, and original_dataset.

The "prompt" column is a flattened string starting with "user: ". Most rows
are single-turn, but ~1.4% (mostly from the WildChat source) are genuine
multi-turn conversations flattened with "\\nuser: " / "\\nassistant: "
separators; those are split back into individual turns. A handful of
IF-eval rows contain adversarial "user: ...user: ..." text (a fake embedded
conversation as part of a single jailbreak-style prompt) that does not
alternate user/assistant — those are intentionally kept as a single user
turn so their content isn't corrupted by a naive split.

Usage:

    uv run python scripts/data/rlvr/dolci_think_rl_7b_messages.py
    uv run python scripts/data/rlvr/dolci_think_rl_7b_messages.py --push_to_hub --hf_entity mnoukhov
"""

import re
from dataclasses import dataclass

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from transformers import HfArgumentParser

SOURCE_DATASET = "allenai/Dolci-Think-RL-7B"
DEFAULT_REPO_NAME = "dolci_think_rl_7b_messages"
TURN_RE = re.compile(r"(?:^|\n)(user|assistant): ")


@dataclass
class Args:
    push_to_hub: bool = False
    hf_entity: str | None = None
    repo_name: str = DEFAULT_REPO_NAME


def prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    matches = list(TURN_RE.finditer(prompt))
    assert matches and matches[0].group(1) == "user", f"prompt does not start with a user turn: {prompt[:100]!r}"

    roles = [m.group(1) for m in matches]
    expected = ["user" if i % 2 == 0 else "assistant" for i in range(len(roles))]
    if roles != expected:
        # Not a real alternating conversation (e.g. adversarial text embedding fake
        # "user:"/"assistant:" markers) -- keep the whole thing as a single user turn.
        return [{"role": "user", "content": prompt[matches[0].end() :]}]

    messages = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        messages.append({"role": roles[i], "content": prompt[start:end]})
    return messages


def process(example: dict) -> dict:
    example["messages"] = prompt_to_messages(example["prompt"])
    return example


def main(args: Args):
    dataset = load_dataset(SOURCE_DATASET, split="train")
    dataset = dataset.map(process, num_proc=8)
    dataset = dataset.select_columns(["messages", "ground_truth", "dataset", "custom_id", "original_dataset"])
    print(dataset)

    if args.push_to_hub:
        api = HfApi()
        entity = args.hf_entity or api.whoami()["name"]
        repo_id = f"{entity}/{args.repo_name}"
        print(f"Pushing dataset to Hub: {repo_id}")
        dataset.push_to_hub(repo_id)
        api.upload_file(
            path_or_fileobj=__file__, path_in_repo="create_dataset.py", repo_type="dataset", repo_id=repo_id
        )


if __name__ == "__main__":
    parser = HfArgumentParser(Args)
    main(*parser.parse_args_into_dataclasses())
