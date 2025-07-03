import argparse
import re
from typing import Dict

from datasets import Sequence, Value, load_dataset

"""
Script to remove mentions of a date cutoff from post-training datasets.
Motivated by: realizing the SFT mix has lots of "as my last update in April 2023" snippets. for next version, we should really template these, so we can add exact cutoff date.

Run with:
python scripts/data/sft/filter_cutoff_date.py --dataset allenai/tulu-3-sft-mixture --column messages
"""

# More permissive pattern to catch variations
CUT_OFF_PATTERN = re.compile(
    r"""(?ix)                   # Case insensitive, verbose mode
    (?:                         # Non-capturing group for prefix variations
        as\s+(?:of\s+)?        # "as of" or "as"
        (?:my\s+)?             # Optional "my"
        (?:last|latest)\s+     # "last" or "latest"
        (?:update|knowledge)    # Followed by "update" or "knowledge"
        (?:\s+was)?\s+in       # Optional "was" and mandatory "in"
        |
        knowledge\s+cutoff     # Or "knowledge cutoff"
        |
        last\s+updated?\s+in   # Or "last update(d) in"
    )
    \s+
    (?:                        # Month-year pattern
        (?:January|February|March|April|May|June|July|
           August|September|October|November|December)
        \s+
    )?
    (\d{4})                    # Year
    """
)

def process_example(example: Dict, column: str) -> Dict:
    """
    Extract cutoff date mentions from the specified column and add them to the example.
    """
    messages = example.get(column, [])
    matches = []

    id = example.get("id", "")
    source = example.get("source", "")

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            found_matches = list(CUT_OFF_PATTERN.finditer(content))
            matches.extend(str(match.group(0)) for match in found_matches)

    new_features = {k: v for k, v in example.items()}
    new_features["cutoff_matches"] = matches
    new_features["id"] = id
    new_features["source"] = source
    return new_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="allenai/tulu-3-sft-mixture")
    parser.add_argument("--column", type=str, default="messages")
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--push_to_hf", action="store_true", default=False)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    split_name = args.split
    split_data = dataset[split_name]
    print(f"\nProcessing split: {split_name}")
    print(f"Number of examples in split: {len(split_data)}")

    # Add new features including cutoff_matches
    new_features = split_data.features.copy()
    new_features["cutoff_matches"] = Sequence(Value("string"))
    new_features["id"] = Value("string")
    new_features["source"] = Value("string")

    # Process examples with the new features
    processed = split_data.map(
        lambda ex: process_example(ex, column=args.column),
        num_proc=args.num_proc,
        desc="Extracting cutoff matches",
        features=new_features
    )

    # Filter examples with matches
    filtered = processed.filter(
        lambda ex: bool(ex["cutoff_matches"]),
        num_proc=args.num_proc,
        desc="Filtering examples with cutoff mentions"
    )

    # Collect all matches
    all_matches = [match for ex in filtered for match in ex["cutoff_matches"]]

    # print ids and counter of sources
    sources = {}
    for ex in filtered:
        source = ex["source"]
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    print("\nSources and counts:")
    for source, count in sources.items():
        print(f"- {source}: {count}")

    print(f"\nFound {len(filtered)} examples with cutoff mentions")
    print(f"Total mentions: {len(all_matches)}")
    if all_matches:
        print("\nExample matches:")
        for match in all_matches[:10]:
            print(f"- {match}")

    # remove entries in processed if id in ids of filtered
    remove_ids = [ex["id"] for ex in filtered]
    print(f"Removing {len(remove_ids)} examples from processed")
    processed = processed.filter(
        lambda ex: ex["id"] not in remove_ids,
        num_proc=args.num_proc,
        desc="Filtering processed examples"
    )
    print(f"Number of examples in processed after filtering: {len(processed)}")
    print(f"Number of examples in orig dataset: {len(split_data)}")

    # Save the filtered dataset if specified
    if args.push_to_hf:
        new_name = args.dataset + "-filter-datecutoff"
        # remove column "cutoff_matches"
        processed = processed.remove_columns(["cutoff_matches"])
        processed.push_to_hub(new_name)



if __name__ == "__main__":
    main()
