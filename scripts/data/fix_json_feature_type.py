"""
Fix datasets saved with schema inconsistencies (mixed types, legacy 'Json'
feature type) by loading the raw JSONL and re-uploading to HuggingFace Hub.

Issues fixed:
  - tool_calls[].function.arguments fields with mixed number/string types
  - Legacy 'Json' feature type (datasets < 2.x)

Usage:
    python scripts/data/fix_json_feature_type.py \
        --source rl-rag/browsecomp-gptoss-clean-qwen35-sft \
        --dest hamishivi/browsecomp-gptoss-clean-qwen35-sft
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download


def normalize_value(v):
    """Recursively normalize all values to JSON-safe types with consistent schemas."""
    if isinstance(v, dict):
        return {k: normalize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [normalize_value(item) for item in v]
    # Normalize numbers that should be strings (e.g. tool call argument ids)
    return v


def normalize_tool_calls(tool_calls):
    if not tool_calls:
        return tool_calls
    result = []
    for tc in tool_calls:
        tc = dict(tc)
        if "function" in tc and isinstance(tc["function"], dict):
            fn = dict(tc["function"])
            if "arguments" in fn:
                args = fn["arguments"]
                # Normalize arguments: convert entire structure to string if mixed
                if isinstance(args, dict):
                    fn["arguments"] = {k: str(v) if not isinstance(v, str) else v for k, v in args.items()}
                elif not isinstance(args, str):
                    fn["arguments"] = json.dumps(args)
            tc["function"] = fn
        result.append(tc)
    return result


def normalize_messages(messages):
    if not messages:
        return messages
    result = []
    for msg in messages:
        msg = dict(msg)
        if "tool_calls" in msg and msg["tool_calls"]:
            msg["tool_calls"] = normalize_tool_calls(msg["tool_calls"])
        result.append(msg)
    return result


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" in row:
                row["messages"] = normalize_messages(row["messages"])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    print(f"Downloading snapshot of {args.source} ...")
    local_dir = snapshot_download(repo_id=args.source, repo_type="dataset")
    print(f"  Downloaded to {local_dir}")

    splits = {}
    for jsonl_file in sorted(Path(local_dir).glob("*.jsonl")):
        split_name = jsonl_file.stem  # e.g. "train"
        print(f"  Loading {jsonl_file.name} ...")
        rows = load_jsonl(jsonl_file)
        print(f"  {split_name}: {len(rows)} rows")
        splits[split_name] = Dataset.from_list(rows)

    if not splits:
        raise ValueError(f"No .jsonl files found in {local_dir}")

    dataset = DatasetDict(splits)
    for split in dataset:
        print(f"  {split} features: {dataset[split].features}")

    print(f"Pushing to {args.dest} ...")
    dataset.push_to_hub(args.dest, private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
