import argparse
import re

from datasets import Sequence, Value, load_dataset

import open_instruct.utils as open_instruct_utils

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


def process_example(example: dict, column: str, index: int = None) -> dict:
    """
    Extract cutoff date mentions from the specified column and add them to the example.
    """
    messages = example.get(column, [])
    matches = []

    # Use existing id if available, otherwise generate one
    id = example.get("id", "")
    if not id and index is not None:
        id = str(index)
    elif not id:
        # Fallback: create hash from first message content if available
        if messages and len(messages) > 0:
            first_content = str(messages[0].get("content", ""))[:50]
            id = str(hash(first_content))
        else:
            id = str(hash(str(example)))

    source = example.get("source", "unknown")

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
    parser.add_argument(
        "--output-entity",
        type=str,
        help="Output entity (org/user) for the filtered dataset. If not provided, uses the same entity as the input dataset.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, num_proc=open_instruct_utils.max_num_processes())

    split_name = args.split
    split_data = dataset[split_name]
    print(f"\nProcessing split: {split_name}")
    print(f"Number of examples in split: {len(split_data)}")
    print(f"Original dataset columns: {split_data.column_names}")
    print(f"Sample of first example: {split_data[0] if len(split_data) > 0 else 'No examples'}")

    # Add new features including cutoff_matches
    new_features = split_data.features.copy()
    new_features["cutoff_matches"] = Sequence(Value("string"))
    new_features["id"] = Value("string")
    new_features["source"] = Value("string")

    # Process examples with the new features
    processed = split_data.map(
        lambda ex, idx: process_example(ex, column=args.column, index=idx),
        num_proc=args.num_proc,
        desc="Extracting cutoff matches",
        features=new_features,
        with_indices=True,
    )

    # Filter examples with matches
    filtered = processed.filter(
        lambda ex: bool(ex["cutoff_matches"]), num_proc=args.num_proc, desc="Filtering examples with cutoff mentions"
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
    remove_ids = set(ex["id"] for ex in filtered)  # Use set for faster lookup
    print(f"Removing {len(remove_ids)} examples from processed")
    print(f"Sample remove_ids: {list(remove_ids)[:5]}")

    # Debug: Check if IDs exist in processed
    processed_ids = set(ex["id"] for ex in processed.select(range(min(100, len(processed)))))
    print(f"Sample processed IDs: {list(processed_ids)[:5]}")
    print(
        f"ID overlap check - first 5 remove_ids in processed: {[rid in processed_ids for rid in list(remove_ids)[:5]]}"
    )

    processed = processed.filter(
        lambda ex: ex["id"] not in remove_ids, num_proc=args.num_proc, desc="Filtering processed examples"
    )
    print(f"Number of examples in processed after filtering: {len(processed)}")
    print(f"Number of examples in orig dataset: {len(split_data)}")

    # Save the filtered dataset if specified
    if args.push_to_hf:
        # Generate output dataset name
        if args.output_entity:
            # Use custom output entity
            if "/" in args.dataset:
                _, dataset_name = args.dataset.split("/", 1)
            else:
                dataset_name = args.dataset
            new_name = f"{args.output_entity}/{dataset_name}-filter-datecutoff"
        else:
            # Use same entity as input dataset
            new_name = args.dataset + "-filter-datecutoff"

        print(f"Uploading filtered dataset to: {new_name}")
        print(f"Dataset size before upload: {len(processed)}")

        # Remove temporary columns if they exist
        columns_to_remove = []
        if "cutoff_matches" in processed.column_names:
            columns_to_remove.append("cutoff_matches")

        if columns_to_remove:
            print(f"Removing columns: {columns_to_remove}")
            processed = processed.remove_columns(columns_to_remove)

        print(f"Final dataset size: {len(processed)}")
        print(f"Final dataset columns: {processed.column_names}")

        processed.push_to_hub(new_name, private=True)
        print(f"Successfully uploaded dataset with {len(processed)} examples")


if __name__ == "__main__":
    main()
