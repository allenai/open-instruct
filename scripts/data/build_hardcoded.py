import argparse
from functools import partial

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi

import open_instruct.utils as open_instruct_utils
from open_instruct import logger_utils

"""
Script to build hardcoded data for Ai2's models, so
  the models know who they are and why they're trained.
Example use, for Tülu 3, run the following:

python scripts/data/build_hardcoded.py \
    --filter_tags olmo \
    --model_name "Tülu 3" \
    --base_model "Llama 3.1" \
    --posttrain_recipe "Tülu 3" \
    --context_length 4096 \
    --license "Llama 3.1 Community License Agreement" \
    --target_namespace "allenai" \\
    --weights_link "https://huggingface.co/allenai"

For OLMo 2, run:

python scripts/data/build_hardcoded.py \
    --filter_tags olmo \
    --model_name "OLMo 2" \
    --base_model "OLMo 2" \
    --posttrain_recipe "Tülu 3" \
    --context_length 4096 \
    --date_cutoff "November 2024" \
    --license "Apache 2.0" \
    --target_namespace "allenai" \
    --weights_link "https://huggingface.co/allenai"
"""

# --- Configuration ---
SOURCE_DATASET_REPO = "allenai/hardcoded-seed"
DEFAULT_TARGET_NAMESPACE = "allenai"  # Or your HF username

# --- Default Placeholder Values ---
DEFAULT_PLACEHOLDERS = {
    "<|MODEL_NAME|>": "OLMo 2",
    "<|POSTTRAIN_RECIPE|>": "Tülu 3.1",
    "<|CONTEXT_LENGTH|>": 4096,
    "<|BASE_MODEL|>": "OLMo 2",
    "<|DATE_CUTOFF|>": "November 2024",
    "<|LICENSE|>": "Apache 2.0",
    "<|WEIGHTS_LINK|>": "",  # New placeholder for model weights link
}

# Map argparse argument names to placeholder keys
ARG_TO_PLACEHOLDER_MAP = {
    "model_name": "<|MODEL_NAME|>",
    "posttrain_recipe": "<|POSTTRAIN_RECIPE|>",
    "context_length": "<|CONTEXT_LENGTH|>",
    "base_model": "<|BASE_MODEL|>",
    "date_cutoff": "<|DATE_CUTOFF|>",
    "license": "<|LICENSE|>",
    "weights_link": "<|WEIGHTS_LINK|>",  # New mapping for weights link
}

# Valid filter tags
VALID_FILTER_TAGS = ["olmo", "tulu", "date-cutoff", "no-tools", "english-only", "license", "availability"]

# --- Logging Setup ---
logger = logger_utils.setup_logger(__name__)


# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process the hardcoded-seed dataset using default placeholders, allowing overrides via arguments, and optionally push to Hugging Face Hub."
    )
    # Arguments now override defaults, so not required, default to None
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help=f"Value to override default for <|MODEL_NAME|> ('{DEFAULT_PLACEHOLDERS['<|MODEL_NAME|>']}').",
    )
    parser.add_argument(
        "--posttrain_recipe",
        type=str,
        default=None,
        help=f"Value to override default for <|POSTTRAIN_RECIPE|> ('{DEFAULT_PLACEHOLDERS['<|POSTTRAIN_RECIPE|>']}').",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
        help=f"Value to override default for <|CONTEXT_LENGTH|> ({DEFAULT_PLACEHOLDERS['<|CONTEXT_LENGTH|>']}).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help=f"Value to override default for <|BASE_MODEL|> ('{DEFAULT_PLACEHOLDERS['<|BASE_MODEL|>']}').",
    )
    parser.add_argument(
        "--date_cutoff",
        type=str,
        default=None,
        help=f"Value to override default for <|DATE_CUTOFF|> ('{DEFAULT_PLACEHOLDERS['<|DATE_CUTOFF|>']}').",
    )
    parser.add_argument(
        "--license",
        type=str,
        default=None,
        help=f"Value to override default for <|LICENSE|> ('{DEFAULT_PLACEHOLDERS['<|LICENSE|>']}').",
    )
    parser.add_argument(
        "--weights_link",
        type=str,
        default=None,
        help=f"Value to override default for <|WEIGHTS_LINK|> ('{DEFAULT_PLACEHOLDERS['<|WEIGHTS_LINK|>']}').",
    )
    # --- Other Arguments ---
    parser.add_argument(
        "--system_prompt_template",
        type=str,
        default="You are <|MODEL_NAME|>, a helpful assistant built by Ai2. Your date cutoff is <|DATE_CUTOFF|> and you do not have access to external tools such as search and running code, but you're very happy to help users find their way with it."
        + ("<|WEIGHTS_LINK|>" and " The model weights are available at <|WEIGHTS_LINK|>." or ""),
        help="Optional text for the system prompt. Can contain placeholders like <|MODEL_NAME|>.",
    )
    parser.add_argument(
        "--target_repo_name",
        type=str,
        default=None,
        help="Name for the target repository (e.g., 'hardcoded-my-model'). If not set, derived from the final model name.",
    )
    parser.add_argument(
        "--target_namespace",
        type=str,
        default=DEFAULT_TARGET_NAMESPACE,
        help=f"Namespace for the target repository (default: '{DEFAULT_TARGET_NAMESPACE}'). Use your username if pushing to your own space.",
    )
    parser.add_argument(
        "--source_repo",
        type=str,
        default=SOURCE_DATASET_REPO,
        help=f"Source dataset repository ID (default: '{SOURCE_DATASET_REPO}').",
    )
    parser.add_argument(
        "--filter_tags",
        type=str,
        nargs="+",
        choices=VALID_FILTER_TAGS,
        help=f"Filter out examples with specific tags. Valid tags: {', '.join(VALID_FILTER_TAGS)}.",
    )

    args = parser.parse_args()

    # Note: Target repo name determination moved to main() after replacements are finalized.

    return args


# --- Processing Functions (Unchanged) ---
def format_content(text, replacements):
    """Replaces all known placeholders in a given text string."""
    if not isinstance(text, str):  # Handle potential non-string data
        return text
    for placeholder, value in replacements.items():
        # Ensure the value exists and convert to string for replacement
        if value is not None:
            text = text.replace(placeholder, str(value))
    return text


def process_example(example, replacements, formatted_system_prompt=None):
    """
    Processes a single dataset example:
    1. Replaces placeholders in all messages.
    2. Prepends the formatted system prompt (if provided) ONLY if there isn't already a system prompt.
    """
    processed_messages = []

    # Check if the example already has a system prompt
    has_system_prompt = False
    if "messages" in example and example["messages"]:
        for message in example["messages"]:
            if isinstance(message, dict) and message.get("role") == "system":
                has_system_prompt = True
                break

    # 1. Add system prompt if provided AND example doesn't already have one
    if formatted_system_prompt and not has_system_prompt:
        processed_messages.append({"role": "system", "content": formatted_system_prompt})

    # 2. Process existing messages (including any existing system prompts)
    for message in example.get("messages", []):  # Use .get for safety
        if isinstance(message, dict) and "content" in message and "role" in message:
            processed_content = format_content(message.get("content", ""), replacements)
            processed_messages.append({"role": message["role"], "content": processed_content})
        else:
            logger.warning(f"Skipping malformed message in example ID {example.get('id', 'N/A')}: {message}")

    # Update the example with the new messages list
    example["messages"] = processed_messages
    return example


def should_keep_example(example, filter_tags):
    """
    Determine if an example should be kept based on its tags and the filter tags.
    Return True if the example should be kept, False if it should be filtered out.
    """
    if not filter_tags:
        return True

    example_tags = example.get("tags", [])
    if not example_tags:
        return True

    # Check if any of the filter tags match the example tags
    for tag in filter_tags:
        if (
            tag == "olmo"
            and any(t in ["olmo", "OLMo"] for t in example_tags)
            or tag == "tulu"
            and any(t in ["tulu", "Tülu", "Tulu"] for t in example_tags)
            or tag == "date-cutoff"
            and "date-cutoff" in example_tags
            or tag == "no-tools"
            and "no-tools" in example_tags
            or tag == "english-only"
            and "english-only" in example_tags
        ):
            return False

    return True


# --- Main Execution ---
def main():
    args = parse_arguments()

    # --- Prepare Replacements: Start with defaults, then override ---
    replacements = DEFAULT_PLACEHOLDERS.copy()
    logger.info(f"Loaded default placeholders: {replacements}")

    overrides_applied = {}
    for arg_name, placeholder_key in ARG_TO_PLACEHOLDER_MAP.items():
        arg_value = getattr(args, arg_name, None)  # Get argument value (e.g., args.model_name)
        if arg_value is not None:  # Check if user provided this argument
            replacements[placeholder_key] = arg_value
            overrides_applied[placeholder_key] = arg_value

    if overrides_applied:
        logger.info(f"Applied overrides from arguments: {overrides_applied}")
    else:
        logger.info("No placeholder overrides provided via arguments, using defaults.")

    logger.info(f"Final replacements to be used: {replacements}")

    # --- Determine Full Target Repo ID ---
    target_repo_name = args.target_repo_name
    if not target_repo_name:
        # Derive from the FINAL model name after considering defaults/overrides
        final_model_name = replacements["<|MODEL_NAME|>"]
        sanitized_model_name = (
            final_model_name.lower().replace(" ", "-").replace("/", "_").replace("<|", "").replace("|>", "")
        )
        target_repo_name = f"hardcoded-{sanitized_model_name}"
        logger.info(f"Target repository name automatically set to: {target_repo_name}")

    target_repo_full_id = f"{args.target_namespace}/{target_repo_name}"

    # --- Check HF Hub Access ---
    try:
        api = HfApi()
        try:
            repo_info = api.repo_info(repo_id=target_repo_full_id, repo_type="dataset")
            logger.info(f"Target repository '{target_repo_full_id}' exists.")
        except Exception:
            logger.info(
                f"Target repository '{target_repo_full_id}' not found or inaccessible. It will be created during push."
            )
    except Exception as e:
        logger.error(f"Error accessing Hugging Face Hub: {e}")
        logger.error("Ensure you are logged in with 'huggingface-cli login' before running this script.")
        return

    # --- Load Source Dataset ---
    try:
        logger.info(f"Loading source dataset '{args.source_repo}'...")
        original_dataset = load_dataset(args.source_repo, num_proc=open_instruct_utils.max_num_processes())
        logger.info(f"Dataset loaded successfully. Splits: {list(original_dataset.keys())}")
    except Exception as e:
        logger.error(f"Failed to load source dataset '{args.source_repo}': {e}")
        return

    # --- Apply Tag Filtering if Specified ---
    if args.filter_tags:
        logger.info(f"Filtering out examples with tags: {args.filter_tags}")
        filtered_dataset = DatasetDict()

        for split_name, split_dataset in original_dataset.items():
            filtered_split = split_dataset.filter(
                lambda example: should_keep_example(example, args.filter_tags), desc=f"Filtering {split_name} by tags"
            )
            filtered_dataset[split_name] = filtered_split

            logger.info(
                f"Split '{split_name}': Kept {len(filtered_split)}/{len(split_dataset)} examples after filtering"
            )

        original_dataset = filtered_dataset

    # --- Format System Prompt ---
    formatted_system_prompt = None
    if args.system_prompt_template:
        # Use the final 'replacements' dict here too
        formatted_system_prompt = format_content(args.system_prompt_template, replacements)
        logger.info(f"Formatted system prompt: '{formatted_system_prompt}'")
    else:
        logger.info("No system prompt template provided.")

    # --- Process Dataset Splits ---
    logger.info("Processing dataset splits...")
    processing_func = partial(
        process_example,
        replacements=replacements,  # Pass the final replacements dict
        formatted_system_prompt=formatted_system_prompt,
    )

    processed_dataset = original_dataset.map(
        processing_func, batched=False, desc="Applying replacements and system prompt"
    )
    logger.info("Dataset processing complete.")

    # --- Display Sample ---
    try:
        first_split_name = next(iter(processed_dataset.keys()))
        logger.info(f"\n--- Sample Processed Example (from split '{first_split_name}') ---")
        # Pretty print the dictionary for better readability
        import json

        print(json.dumps(processed_dataset[first_split_name][0], indent=2))
        logger.info("--- End Sample ---")
    except Exception as e:
        logger.warning(f"Could not display sample processed data: {e}")

    # --- Push to Hub ---
    try:
        logger.info(f"Pushing processed dataset to '{target_repo_full_id}'...")
        processed_dataset.push_to_hub(
            repo_id=target_repo_full_id.replace(" ", "-").replace("ü", "u"),  # Replace spaces with hyphens
            private=True,
        )
        logger.info(f"Dataset successfully pushed to: https://huggingface.co/datasets/{target_repo_full_id}")
    except Exception as e:
        logger.error(f"Failed to push dataset to '{target_repo_full_id}': {e}")
        logger.error("Check that you're logged in with 'huggingface-cli login' and have the necessary permissions.")
    else:
        logger.info("Processing complete. Dataset pushed to Hub successfully.")


if __name__ == "__main__":
    main()
