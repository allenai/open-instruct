from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.columns import Columns
import numpy as np
from transformers import HfArgumentParser


def maybe_markdown(content: str, markdown: bool) -> str:
    return Markdown(content) if markdown else content

def print_hf_chosen_rejected(chosen: List[Dict[str, str]], rejected: List[Dict[str, str]], markdown: bool = False, chosen_key: str = "chosen", rejected_key: str = "rejected"):
    """here we are assuming the chosen[:-1] == rejected[:-1]"""
    assert len(chosen) == len(rejected)
    assert chosen[:-1] == rejected[:-1]
    console = Console()
    colors = ["red", "green"]
    color_idx = 0
    console.rule(f"[bold yellow]The number of turns is {len(chosen)}")
    for i in range(len(chosen) - 1):
        message = chosen[i]
        role = message["role"]
        content = maybe_markdown(message["content"], markdown)
        console.print(Panel(content, title_align="left", title=role, border_style=colors[color_idx]))
        color_idx = (color_idx + 1) % 2
    half_width = int(0.48 * console.width)
    columns = Columns(
        [
            Panel(maybe_markdown(chosen[-1]["content"], markdown), width=half_width, title=chosen_key, border_style="green"),
            Panel(maybe_markdown(rejected[-1]["content"], markdown), width=half_width, title=rejected_key, border_style="red"),
        ],
    )

    console.print(Panel(columns, title=chosen[-1]["role"], border_style=colors[color_idx]))



@dataclass
class Args:
    rejection_sampled_dataset: str = "vwxyzjn/rejection_sampling_31313"
    shuffle: bool = False

def main(args: Args):
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    ds = load_dataset("vwxyzjn/rejection_sampling_31313", split="train")
    if args.shuffle:
        ds = ds.shuffle()

    print("🚀 Dataset loaded, starting to analyze...")
    chosen_scores = defaultdict(list)
    rejected_scores = defaultdict(list)
    reference_scores = defaultdict(list)
    chosen_lengths = []
    rejected_lengths = []
    reference_length = []
    for example in ds:
        chosen_lengths.append(len(example["chosen"][-1]["content"]))
        rejected_lengths.append(len(example["rejected"][-1]["content"]))
        reference_length.append(len(example["reference_completion"]))
        for key in example["chosen_score"]:
            chosen_scores[key].append(example["chosen_score"][key])
            rejected_scores[key].append(example["rejected_score"][key])
            reference_scores[key].append(example["reference_completion_score"][key])

    print(f"chosen:    mean length = {np.mean(chosen_lengths)}")
    print(f"rejected:  mean length = {np.mean(rejected_lengths)}")
    print(f"reference: mean length = {np.mean(reference_length)}")
    for key in example["chosen_score"]:
        print(f"{key=}")
        print(f"chosen:    mean score = {np.mean(chosen_scores[key])}")
        print(f"           std score = {np.std(chosen_scores[key])}")
        print(f"rejected:  mean score = {np.mean(rejected_scores[key])}")
        print(f"           std score = {np.std(rejected_scores[key])}")
        print(f"reference: mean score = {np.mean(reference_scores[key])}")
        print(f"           std score = {np.std(reference_scores[key])}")

    for i in range(len(chosen_scores[key])):
        if reference_scores[key][i] > chosen_scores[key][i]:
            print("reference is better than chosen")
            print_hf_chosen_rejected(
                ds[i]["chosen"][:-1] + [{"role": "assistant", "content": ds[i]["reference_completion"]}],
                ds[i]["chosen"],
                chosen_key="reference",
                rejected_key="chosen",
            )
            input("Press Enter to continue...")

if __name__ == "__main__":
    main(Args())