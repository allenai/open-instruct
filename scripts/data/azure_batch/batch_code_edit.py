"""This script is used to create Azure OpenAI batch API requests for code editing across multiple datasets.

Usage:

Cd into the directory of this file and run:
```
python batch_code_edit.py --datasets "user/dataset1,user/dataset2" --num-errors 3
```

"""

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List

import datasets
import openai
import tiktoken

# Set up logging with file name and line number
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model costs are in USD per million tokens.
MODEL_COSTS_PER_1M_TOKENS = {
    "o3-batch": {
        "input": 10 * 0.5,
        "output": 40 * 0.5,
    },
    "o3": {
        "input": 10 * 0.5,
        "output": 40 * 0.5,
    },
}

# Maximum number of prompts per batch file
MAX_PROMPTS_PER_BATCH = 95_000

# Dataset-specific column mappings
# TODO: Fill out these mappings for each dataset
DATASET_COLUMN_MAPPINGS = {
    "saurabh5/llama-nemotron-rlvr": {
        "code_column": "rewritten_solution",
        "test_column": "ground_truth",
        "description_index_fn": lambda x: x,
        "description_column": "rewritten_input",
        "id_column": "id",
        "language": "python",
    },
    "saurabh5/the-algorithm-python": {
        "code_column": "reference_solution",
        "test_column": "ground_truth",
        "description_column": "messages",
        "description_index_fn": lambda x: x[0]["content"],
        "id_column": "python_index",
        "language": "python",
    },
    "saurabh5/open-code-reasoning-rlvr": {
        "code_column": "rewritten_solution",
        "test_column": "ground_truth",
        "description_column": "input",
        "id_column": "id",
        "language": "python",
    },
    "saurabh5/tulu-3-personas-code-rlvr": {
        "code_column": "rewritten_solution",
        "test_column": "ground_truth",
        "description_column": "rewritten_input",
        "id_column": "id",
        "language": "python",
    },
}


@dataclasses.dataclass
class CodeError:
    tag: str
    prompt: str


SUPPORTED_LANGUAGES = ["python", "javascript", "java", "C++", "Rust", "Bash"]


# Language-specific coding errors
LANGUAGE_ERRORS: dict[str, list[CodeError]] = {
    "python": [
        CodeError(
            tag="eq-asgn",
            prompt="Replace a '==' equality test in a conditional expression with a single '=' assignment.",
        ),
        CodeError(
            tag="offby1",
            prompt="Adjust a loop boundary (e.g., change range(n) to range(n + 1) or the reverse) so the loop runs one iteration off.",
        ),
        CodeError(
            tag="miss-colon",
            prompt="Remove the trailing colon from a control-flow or function header (if/for/while/def/class).",
        ),
        CodeError(
            tag="unclose-par",
            prompt="Delete a closing parenthesis from a multi-token expression, leaving it unmatched.",
        ),
        CodeError(
            tag="bad-indent",
            prompt="Dedent an inner code block (such as a loop or conditional body) by two spaces, breaking its alignment.",
        ),
        CodeError(
            tag="str-quotes",
            prompt="Change a closing double quote \") to a single quote ' in a string literal, leaving the opening quote unchanged.",
        ),
        CodeError(
            tag="illegal-kw",
            prompt="Rename an identifier to a reserved keyword (for example, change a variable name to 'class').",
        ),
        CodeError(
            tag="ill-comment",
            prompt="Insert a stray '#' mid-line so the remainder becomes an unintended comment.",
        ),
        CodeError(
            tag="var-uninit",
            prompt="Delete the assignment of exactly one variable that is later used, leaving it uninitialised.",
        ),
        CodeError(
            tag="none-deref",
            prompt="Assign None to an object immediately before calling one of its methods.",
        ),
        CodeError(
            tag="idx-oob",
            prompt="Shift an index expression by +1 (e.g., arr[i] → arr[i+1]), risking an out-of-bounds access.",
        ),
        CodeError(
            tag="div-zero",
            prompt="Introduce a division by zero (e.g., replace x / y with x / 0).",
        ),
        CodeError(
            tag="inf-loop",
            prompt="Modify an existing loop condition so it never becomes false, creating an infinite loop.",
        ),
        CodeError(
            tag="long-loop",
            prompt="Double the iteration count of an existing loop (such as multiplying its upper bound by 2).",
        ),
        CodeError(
            tag="float-prec",
            prompt="Insert an equality comparison between two floats that differ only by floating-point precision error.",
        ),
        CodeError(
            tag="mut-default",
            prompt="Give a function a mutable default argument like {} or [] instead of None.",
        ),
        CodeError(
            tag="slow-loop",
            prompt="Invoke an expensive computation twice within the same iteration of a nested loop.",
        ),
        CodeError(
            tag="magic-num",
            prompt="Replace a variable or parameter with a hard-coded constant (a 'magic number') reused in multiple places.",
        ),
    ],
    "javascript": [
        CodeError(tag="missing-semi", prompt="Insert a missing semicolon."),
        CodeError(
            tag="undefined-var", prompt="Replace a variable with an undefined one."
        ),
        CodeError(
            tag="ill-decl",
            prompt="Change a function declaration to a variable assignment.",
        ),
        CodeError(
            tag="unclose-par",
            prompt="Remove the closing parenthesis of the longest expression.",
        ),
        CodeError(tag="bad-obj", prompt="Change an object literal to a function call."),
        CodeError(tag="miss-ret", prompt="Remove the return statement."),
        CodeError(tag="ill-arr", prompt="Change an array literal to a function call."),
        CodeError(
            tag="miss-decl", prompt="Insert a missing var/let/const declaration."
        ),
        CodeError(
            tag="str-concat",
            prompt="Swap one string concatenation for a template literal.",
        ),
        CodeError(
            tag="miss-try", prompt="Insert a try/catch block around the function."
        ),
    ],
    "java": [
        CodeError(tag="missing-import", prompt="Add a missing import statement."),
        CodeError(
            tag="undefined-var", prompt="Replace a variable with an undefined one."
        ),
        CodeError(
            tag="ill-decl",
            prompt="Change a method declaration to a variable assignment.",
        ),
        CodeError(tag="miss-semi", prompt="Insert a missing semicolon."),
        CodeError(
            tag="bad-class",
            prompt="Change a class declaration to a variable assignment.",
        ),
        CodeError(tag="miss-ret", prompt="Remove the return statement."),
        CodeError(tag="ill-arr", prompt="Change an array literal to a function call."),
        CodeError(tag="miss-access", prompt="Insert a missing access modifier."),
        CodeError(
            tag="str-concat",
            prompt="Swap one string concatenation for a template literal.",
        ),
        CodeError(
            tag="miss-try", prompt="Insert a try/catch block around the function."
        ),
    ],
    "C++": [
        CodeError(tag="missing-semi", prompt="Insert a missing semicolon."),
        CodeError(
            tag="undefined-var", prompt="Replace a variable with an undefined one."
        ),
        CodeError(
            tag="ill-decl",
            prompt="Change a function declaration to a variable assignment.",
        ),
        CodeError(
            tag="unclose-par",
            prompt="Remove the closing parenthesis of the longest expression.",
        ),
        CodeError(tag="bad-obj", prompt="Change an object literal to a function call."),
    ],
    "Rust": [
        CodeError(tag="missing-semi", prompt="Insert a missing semicolon."),
        CodeError(
            tag="undefined-var", prompt="Replace a variable with an undefined one."
        ),
        CodeError(
            tag="ill-decl",
            prompt="Change a function declaration to a variable assignment.",
        ),
        CodeError(
            tag="unclose-par",
            prompt="Remove the closing parenthesis of the longest expression.",
        ),
        CodeError(tag="bad-obj", prompt="Change an object literal to a function call."),
    ],
    "Bash": [
        CodeError(tag="missing-semi", prompt="Insert a missing semicolon."),
        CodeError(
            tag="undefined-var", prompt="Replace a variable with an undefined one."
        ),
        CodeError(
            tag="ill-decl",
            prompt="Change a function declaration to a variable assignment.",
        ),
        CodeError(
            tag="unclose-par",
            prompt="Remove the closing parenthesis of the longest expression.",
        ),
        CodeError(tag="bad-obj", prompt="Change an object literal to a function call."),
    ],
}


@dataclasses.dataclass
class PromptData:
    id: str
    prompt: str
    dataset_name: str
    original_row_id: str
    sampled_errors: List[str]


def get_dataset_columns(dataset_name: str) -> Dict[str, str]:
    """Get the column mappings for a specific dataset."""
    if dataset_name not in DATASET_COLUMN_MAPPINGS:
        raise ValueError(f"No column mapping found for dataset: {dataset_name}")
    return DATASET_COLUMN_MAPPINGS[dataset_name]


def sample_errors(language: str, num_errors: int) -> List[str]:
    """Sample random errors for the given language."""
    if language not in LANGUAGE_ERRORS:
        raise ValueError(f"No errors defined for language: {language}")

    available_errors = LANGUAGE_ERRORS[language]
    if num_errors > len(available_errors):
        logger.warning(
            f"Requested {num_errors} errors but only {len(available_errors)} available for {language}"
        )
        num_errors = len(available_errors)

    return random.sample(available_errors, num_errors)


def create_prompt(code: str, errors: List[CodeError], language: str) -> str:
    """Create the prompt for code editing."""
    errors_str = "\n".join(
        [f"{i + 1}. {error.prompt}" for i, error in enumerate(errors)]
    )
    return (
        f"Please edit this code:\n\n{code}\nto insert the following "
        f"{len(errors)} errors:\n\n{errors_str}\n\nWhen you write the code "
        f"out at the end, make sure to surround it with ```{language} "
        f"and ```, like this: \n```{language}\n{{code}}\n```\n\n"
        "If some of the errors are not applicable to the code, ignore them. "
        "If you don't have any valid errors to apply, just print 'ERROR'."
    )


def create_batch_files(
    prompts: List[PromptData],
    base_batch_file_name: str,
    model: str,
    timestamp: int,
    max_completion_tokens: int,
) -> List[str]:
    """Create multiple batch files in the format required by Azure OpenAI Batch API.
    Returns a list of created batch file paths."""
    batch_file_paths = []

    # Split prompts into chunks of MAX_PROMPTS_PER_BATCH
    for i in range(0, len(prompts), MAX_PROMPTS_PER_BATCH):
        chunk = prompts[i : i + MAX_PROMPTS_PER_BATCH]
        batch_file_name = f"{base_batch_file_name}_{i // MAX_PROMPTS_PER_BATCH}.jsonl"

        with open(batch_file_name, "w") as f:
            for prompt in chunk:
                # Format each request according to batch API requirements
                batch_request = {
                    "custom_id": f"{timestamp}_{prompt.id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt.prompt}],
                        "max_completion_tokens": max_completion_tokens,
                    },
                }
                f.write(json.dumps(batch_request) + "\n")

        batch_file_paths.append(batch_file_name)

    return batch_file_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Azure OpenAI batch API requests for code editing across multiple datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset configuration group
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--datasets",
        type=str,
        required=True,
        help='Comma-separated list of dataset names (e.g., "user/dataset1,user/dataset2")',
    )
    dataset_group.add_argument(
        "--split",
        type=str,
        default="train",
        help='Dataset split to use (default: "train")',
    )
    dataset_group.add_argument(
        "--sample-limit",
        type=int,
        help="Limit the number of samples to process per dataset. If not specified, processes all samples.",
    )

    # Language and error configuration
    language_group = parser.add_argument_group("Language and Error Configuration")
    language_group.add_argument(
        "--num-errors",
        type=int,
        default=3,
        help="Number of errors to sample per code snippet (default: 3)",
    )

    # Model configuration group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="o3-batch",
        choices=list(MODEL_COSTS_PER_1M_TOKENS.keys()),
        help=f"Model to use for completions. Available models: {', '.join(MODEL_COSTS_PER_1M_TOKENS.keys())}",
    )
    model_group.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8192,
        help="Maximum number of completion tokens to use (default: 8192)",
    )

    # Runtime options group
    runtime_group = parser.add_argument_group("Runtime Options")
    runtime_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without making any API calls",
    )
    runtime_group.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help='Directory to save batch files and metadata (default: "./output")',
    )

    args = parser.parse_args()

    # Validate arguments
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be a positive integer")

    if args.num_errors <= 0:
        parser.error("--num-errors must be a positive integer")

    return args


def make_requests_for_dataset(
    dataset_name: str,
    num_errors: int,
    split: str,
    sample_limit: int | None,
    model: str,
    max_completion_tokens: int,
    dry_run: bool,
    tokenizer: tiktoken.Encoding,
    timestamp: int,
) -> tuple[List[PromptData], List[Dict[str, Any]], int, int]:
    """Make requests for a dataset."""
    logger.info(f"Processing dataset: {dataset_name}")

    # Load dataset
    dataset = datasets.load_dataset(dataset_name, split=split)
    logger.info(f"Loaded {len(dataset)} rows from {dataset_name}")

    # Get column mappings
    column_mapping = get_dataset_columns(dataset_name)

    # Determine number of samples to process
    num_samples = (
        len(dataset) if sample_limit is None else min(sample_limit, len(dataset))
    )
    sampled_indices = random.sample(range(len(dataset)), num_samples)

    logger.info(f"Processing {num_samples} samples from {dataset_name}")

    prompts: List[PromptData] = []
    metadata: List[Dict[str, Any]] = []
    total_input_tokens, total_output_tokens = 0, 0

    for idx in sampled_indices:
        row = dataset[idx]

        # Extract data using column mappings
        code = row[column_mapping["code_column"]]
        description = row[column_mapping["description_column"]]
        if column_mapping["id_column"] == "python_index":
            row_id = idx
        else:
            row_id = row[column_mapping["id_column"]]

        # Sample errors
        sampled_errors = sample_errors(column_mapping["language"], num_errors)

        # Create prompt
        prompt_text = create_prompt(code, sampled_errors, column_mapping["language"])

        # Create unique ID for this prompt
        prompt_id = f"{dataset_name.replace('/', '_')}_{row_id}_{timestamp}"

        # Create PromptData object
        prompt_data = PromptData(
            id=prompt_id,
            prompt=prompt_text,
            dataset_name=dataset_name,
            original_row_id=row_id,
            sampled_errors=sampled_errors,
        )
        prompts.append(prompt_data)

        # Create metadata entry
        metadata_entry = {
            "prompt_id": prompt_id,
            "dataset_name": dataset_name,
            "original_row_id": row_id,
            "language": column_mapping["language"],
            "sampled_errors": [error.tag for error in sampled_errors],
            "description": description,
            "timestamp": timestamp,
        }
        metadata.append(metadata_entry)

        # Count tokens
        prompt_tokens = len(tokenizer.encode(prompt_text))
        total_input_tokens += prompt_tokens

        # Estimate output tokens (rough estimate)
        estimated_output_tokens = (
            len(tokenizer.encode(code)) * 2
        )  # Assume output is roughly 2x input
        total_output_tokens += estimated_output_tokens

    return prompts, metadata, total_input_tokens, total_output_tokens


def main(
    datasets: str,
    num_errors: int = 3,
    split: str = "train",
    sample_limit: int | None = None,
    model: str = "o3-batch",
    max_completion_tokens: int = 8192,
    dry_run: bool = False,
    output_dir: str = "./output",
) -> None:
    # Parse dataset list
    dataset_names = [name.strip() for name in datasets.split(",")]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/batch_files", exist_ok=True)

    timestamp = int(time.time())
    base_batch_file_name = f"{output_dir}/batch_files/{timestamp}"

    # Set random seed for reproducibility
    random.seed(42)

    all_prompts: List[PromptData] = []
    all_metadata: List[Dict[str, Any]] = []
    tokenizer = tiktoken.encoding_for_model(model)
    total_input_tokens, total_output_tokens = 0, 0

    # Track batch IDs per dataset
    dataset_batch_ids: Dict[str, List[str]] = {}

    for dataset_name in dataset_names:
        prompts, metadata, input_tokens, output_tokens = make_requests_for_dataset(
            dataset_name,
            num_errors,
            split,
            sample_limit,
            model,
            max_completion_tokens,
            dry_run,
            tokenizer,
            timestamp,
        )
        all_prompts.extend(prompts)
        all_metadata.extend(metadata)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

    # Calculate estimated cost
    estimated_cost = (
        total_input_tokens * MODEL_COSTS_PER_1M_TOKENS[model]["input"]
        + total_output_tokens * MODEL_COSTS_PER_1M_TOKENS[model]["output"]
    ) / 1_000_000

    logger.info("\nSummary:")
    logger.info(f"Total prompts: {len(all_prompts)}")
    logger.info(
        f"Input tokens: {total_input_tokens}, estimated output tokens: {total_output_tokens}"
    )
    logger.info(f"Estimated cost: ${estimated_cost:.2f}")

    metadata_path = f"{output_dir}/metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}.")

    if dry_run:
        logger.info("Dry run mode - exiting without making API calls. Prompts:")
        for prompt in all_prompts:
            logger.info(prompt.prompt)
        sys.exit(0)

    if estimated_cost > 100:
        logger.info(
            "\nWaiting 10 seconds to allow you to cancel the script if you don't want to proceed..."
        )
        time.sleep(10)

    # Initialize the client with your API key
    client = openai.AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-04-01-preview",
    )

    # Process each dataset separately to track batch IDs
    for dataset_name in dataset_names:
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Filter prompts for this dataset
        dataset_prompts = [p for p in all_prompts if p.dataset_name == dataset_name]
        
        if not dataset_prompts:
            logger.warning(f"No prompts found for dataset: {dataset_name}")
            continue
            
        # Create batch files for this dataset
        dataset_base_name = f"{base_batch_file_name}_{dataset_name.replace('/', '_')}"
        batch_file_paths = create_batch_files(
            dataset_prompts, dataset_base_name, model, timestamp, max_completion_tokens
        )
        
        # Submit batch jobs for this dataset
        dataset_batch_ids[dataset_name] = []
        logger.info(f"Submitting {len(batch_file_paths)} batch jobs for {dataset_name}...")
        
        for batch_file_path in batch_file_paths:
            batch_file = client.files.create(
                file=open(batch_file_path, "rb"), purpose="batch"
            )

            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            batch_id = batch_job.id
            dataset_batch_ids[dataset_name].append(batch_id)
            all_batch_ids.append(batch_id)
            logger.info(f"Submitted batch job with ID: {batch_id} for {dataset_name}")

    # Print batch IDs in the requested format
    logger.info("\nBatch job IDs by dataset:")
    for dataset_name, batch_ids in dataset_batch_ids.items():
        logger.info(f"{dataset_name}: {', '.join(batch_ids)}")
    all_batch_ids = [batch_id for batch_ids in dataset_batch_ids.values() for batch_id in batch_ids]
    logger.info(f"\nAll batches: {', '.join(all_batch_ids)}")
    logger.info("\nYou can check the status of your batch jobs using the IDs above.")


if __name__ == "__main__":
    args = parse_args()
    main(
        datasets=args.datasets,
        num_errors=args.num_errors,
        split=args.split,
        sample_limit=args.sample_limit,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )
