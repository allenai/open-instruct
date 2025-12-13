import argparse

from datasets import load_dataset

import open_instruct.utils as open_instruct_utils
from open_instruct import logger_utils

"""
Used for stripping special tokens from text fields in a dataset.

Supports both text format and messages/conversation format.

For text format:
Prompt
<think>reasoning</think><answer>answer</answer>

For messages format, converts to:
User: Question.

Assistant: Answer.

User: Question.

Assistant: Answer.
...

e.g.
uv run python scripts/data/filtering_and_updates/filter_special_tokens.py --dataset_name allenai/big-reasoning-traces-reformatted-keyword-filter-datecutoff-chinese-ngram --push_to_hub
"""

# Set up logging
logger = logger_utils.setup_logger(__name__)

# Dictionary mapping special tokens to their replacements
SPECIAL_TOKEN_MAP = {"<think>": "\n", "</think>": "", "<answer>": "\n\n", "</answer>": ""}


def filter_special_tokens(text):
    """Remove special tokens from text using the mapping dictionary."""
    if not isinstance(text, str):
        return text

    filtered_text = text
    for token, replacement in SPECIAL_TOKEN_MAP.items():
        filtered_text = filtered_text.replace(token, replacement)

    return filtered_text


def convert_messages_to_text(messages):
    """Convert messages/conversation format to text format."""
    if not isinstance(messages, list):
        return ""

    text_parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "").lower()
        content = message.get("content", "")

        if not content:
            continue

        # Filter special tokens from content
        filtered_content = filter_special_tokens(content)

        if role in ["user", "human"]:
            text_parts.append(f"User: {filtered_content}")
        elif role in ["assistant", "ai"]:
            text_parts.append(f"Assistant: {filtered_content}")
        else:
            # Handle other roles generically
            text_parts.append(f"{role.capitalize()}: {filtered_content}")

    return "\n\n".join(text_parts)


def process_dataset_row(row, column_name="text"):
    """Process a single row of the dataset to remove special tokens from specified column."""
    processed_row = row.copy()

    if column_name in row:
        value = row[column_name]
        if isinstance(value, str):
            processed_row[column_name] = filter_special_tokens(value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            # Handle conversation-like structures (messages/conversation format)
            if column_name in ["messages", "conversation"]:
                # Convert to text format and overwrite the text column
                converted_text = convert_messages_to_text(value)
                processed_row["text"] = converted_text
                # Keep the original messages column as well
                processed_row[column_name] = value
            else:
                # Handle other list of dict structures
                processed_row[column_name] = []
                for item in value:
                    processed_item = {}
                    for sub_key, sub_value in item.items():
                        if isinstance(sub_value, str):
                            processed_item[sub_key] = filter_special_tokens(sub_value)
                        else:
                            processed_item[sub_key] = sub_value
                    processed_row[column_name].append(processed_item)

    return processed_row


def main():
    parser = argparse.ArgumentParser(description="Filter special tokens from dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--column", type=str, default="text", help="Column to filter (default: text)")
    parser.add_argument("--debug", action="store_true", help="Debug mode - process only first 100 samples")
    parser.add_argument("--push_to_hub", action="store_true", help="Push filtered dataset to HuggingFace Hub")

    args = parser.parse_args()

    logger.info(f"Loading dataset: {args.dataset_name}")
    logger.info(f"Filtering column: {args.column}")

    # Load dataset
    if args.debug:
        logger.info("Debug mode: Loading first 100 samples")
        dataset = load_dataset(
            args.dataset_name, split="train[:100]", num_proc=open_instruct_utils.max_num_processes()
        )
    else:
        dataset = load_dataset(args.dataset_name, split="train", num_proc=open_instruct_utils.max_num_processes())

    logger.info(f"Original dataset size: {len(dataset)}")

    # Check if column exists
    if args.column not in dataset.column_names:
        logger.error(f"Column '{args.column}' not found in dataset. Available columns: {dataset.column_names}")
        return

    # Auto-detect messages/conversation format
    if args.column in ["messages", "conversation"]:
        logger.info(f"Detected conversation format in column '{args.column}'. Will convert to text format.")

    # Process dataset
    logger.info("Filtering special tokens...")
    filtered_dataset = dataset.map(
        lambda row: process_dataset_row(row, args.column), desc="Filtering special tokens", num_proc=64
    )

    logger.info(f"Processed dataset size: {len(filtered_dataset)}")

    if args.push_to_hub:
        # Create new dataset name with suffix
        new_dataset_name = f"{args.dataset_name}-no-special-tokens"
        logger.info(f"Pushing filtered dataset to: {new_dataset_name}")

        if args.debug:
            new_dataset_name += "-debug"

        filtered_dataset.push_to_hub(new_dataset_name, private=True)
        logger.info(f"Successfully pushed dataset to {new_dataset_name}")
    else:
        logger.info("Skipping push to hub (use --push_to_hub to enable)")

    # Print some statistics
    logger.info("Processing complete!")

    # Show sample of filtered content for verification
    logger.info("Sample filtered content:")
    for i, row in enumerate(filtered_dataset):
        if i >= 3:  # Show first 3 examples
            break

        # Show the text column if it exists (especially for converted messages)
        if "text" in row:
            content = str(row["text"])[:300]
            logger.info(f"Row {i+1} - text: {content}...")
        elif args.column in row:
            content = str(row[args.column])[:300]
            logger.info(f"Row {i+1} - {args.column}: {content}...")


if __name__ == "__main__":
    main()
