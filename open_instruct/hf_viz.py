from dataclasses import dataclass
from typing import Dict, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.columns import Columns
from datasets import load_dataset

def maybe_markdown(content: str, markdown: bool) -> str:
    return Markdown(content) if markdown else content


def print_hf_messages(messages: List[Dict[str, str]], markdown: bool = False):
    console = Console()
    colors = ["red", "green"]
    color_idx = 0
    console.rule(f"[bold yellow]The number of turns is {len(messages)}")
    for message in messages:
        role = message["role"]
        content = message["content"]
        console.print(Panel(maybe_markdown(content, markdown), title_align="left", title=role, border_style=colors[color_idx]))
        color_idx = (color_idx + 1) % 2


def print_hf_chosen_rejected(chosen: List[Dict[str, str]], rejected: List[Dict[str, str]], markdown: bool = False, title: Optional[str] = None, explanation: Optional[str] = None):
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
            Panel(maybe_markdown(chosen[-1]["content"], markdown), width=half_width, title="chosen", border_style="green"),
            Panel(maybe_markdown(rejected[-1]["content"], markdown), width=half_width, title="rejected", border_style="red"),
        ],
    )
    console.print(Panel(columns, title=chosen[-1]["role"], border_style=colors[color_idx]))
    if explanation is not None:
        console.print(Panel(maybe_markdown(explanation, markdown), title=title, border_style="blue"))


@dataclass
class Args:
    split: str = "train"
    """The split of the dataset to visualize."""
    sft: Optional[str] = None
    """The name of the Hugging Face SFT dataset."""
    sft_messages_column_name: str = "messages"
    """The name of the column containing the messages in the SFT dataset."""
    preference: Optional[str] = None
    """The name of the Hugging Face preference dataset."""
    preference_chosen_column_name: str = "chosen"
    """The name of the column containing the chosen messages in the preference dataset."""
    preference_rejected_column_name: str = "rejected"
    """The name of the column containing the rejected messages in the preference dataset."""
    markdown: bool = False
    """Whether to print the output in markdown."""


def print_all_sft_dataset(dataset_name: str, split: str, sft_messages_column_name: str, markdown: bool) -> None:
    """visualize the SFT dataset."""
    ds = load_dataset(dataset_name)
    i = 0
    while True:
        print_hf_messages(ds[split][i][sft_messages_column_name], markdown)
        i += 1
        input("Press Enter to continue...")


def print_all_preference_dataset(dataset_name: str, split: str, preference_chosen_column_name: str, preference_rejected_column_name: str, markdown: bool) -> None:
    """visualize the preference dataset."""
    ds = load_dataset(dataset_name)
    i = 0
    while True:
        print_hf_chosen_rejected(ds[split][i][preference_chosen_column_name], ds[split][i][preference_rejected_column_name], markdown)
        i += 1
        input("Press Enter to continue...")


if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    if args.sft is not None:
        print_all_sft_dataset(args.sft, args.split, args.sft_messages_column_name, args.markdown)
    elif args.preference is not None:
        print_all_preference_dataset(args.preference, args.split, args.preference_chosen_column_name, args.preference_rejected_column_name, args.markdown)
    else:
        print("Please provide the name of the dataset.")