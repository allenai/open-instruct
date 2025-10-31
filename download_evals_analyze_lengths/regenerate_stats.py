"""Regenerate statistics.json from already-downloaded eval results."""

import argparse
import json
from pathlib import Path
from typing import List

from length_analysis import (
    calculate_statistics,
    extract_model_responses,
    load_jsonl_file,
    print_statistics,
    tokenize_responses,
)


def normalize_stats(stats):
    """Convert numpy scalar values to native Python scalars for serialization."""
    return {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in stats.items()
    }


def load_tokenizer():
    from transformers import AutoTokenizer

    print("Loading dolma2 tokenizer...")
    return AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")


def find_matching_files(root: Path, pattern: str) -> List[Path]:
    """Return all files under root matching pattern."""
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate statistics.json from already-downloaded eval results."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing the downloaded eval results",
    )
    parser.add_argument(
        "--pattern",
        default="*predictions.jsonl",
        help="Glob pattern for files used in length analysis (default: *predictions.jsonl).",
    )
    args = parser.parse_args()

    target_dir = Path(args.directory).expanduser().resolve()
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist.")
        return

    print(f"Analyzing files in: {target_dir}")
    matched_files = find_matching_files(target_dir, args.pattern)

    if not matched_files:
        print(f"No files matching '{args.pattern}' found in {target_dir}")
        return

    print(f"Found {len(matched_files)} files matching pattern '{args.pattern}'")

    tokenizer = load_tokenizer()
    all_token_counts: List[int] = []

    for file_path in matched_files:
        print(f"\nProcessing file: {file_path}")
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} entries")
        responses = extract_model_responses(data)
        print(f"  Found {len(responses)} model responses")

        if not responses:
            continue

        token_counts = tokenize_responses(responses, tokenizer)
        all_token_counts.extend(token_counts)
        print(f"  Accumulated {len(token_counts)} responses")

    if not all_token_counts:
        print("No model responses found in any files.")
        return

    stats = normalize_stats(calculate_statistics(all_token_counts))
    print_statistics(f"Analysis of {target_dir.name}", stats)

    # Since we're analyzing a single experiment directory, we'll create a simpler structure
    summary = {
        "experiment_name": target_dir.name,
        "statistics": stats,
        "total_responses": len(all_token_counts),
        "files_analyzed": len(matched_files),
    }

    print("\nStatistics summary:")
    print(json.dumps(summary, indent=2))

    stats_path = target_dir / "statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nStatistics saved to {stats_path}")


if __name__ == "__main__":
    main()

