"""
Add a 'tools' column to hamishivi/tmax-sft-full-20260403 containing the
SWERL bash tool definition, then push the updated dataset back to the Hub.

Usage:
    python scripts/data/add_tools_column_tmax_sft.py
"""
from datasets import load_dataset

DATASETS = [
    "hamishivi/tmax-sft-full-20260317",
    "hamishivi/tmax-sft-full-20260403",
]

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command. Each command runs in a new subshell.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
        },
    },
}

TOOLS = [BASH_TOOL]


def add_tools_column(dataset_id: str) -> None:
    print(f"Loading {dataset_id} ...")
    dataset = load_dataset(dataset_id)
    for split in dataset:
        n = len(dataset[split])
        print(f"  {split}: {n} rows")
        dataset[split] = dataset[split].add_column("tools", [TOOLS] * n)
    print(f"Pushing to {dataset_id} ...")
    dataset.push_to_hub(dataset_id)
    print(f"Done: {dataset_id}")


def main():
    for dataset_id in DATASETS:
        add_tools_column(dataset_id)


if __name__ == "__main__":
    main()
