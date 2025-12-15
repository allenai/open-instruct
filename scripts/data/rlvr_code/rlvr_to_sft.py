import datasets
from tqdm import tqdm

from open_instruct import logger_utils

"""
Converts a RLVR dataset into a SFT dataset.

This script processes a specific format of RLVR dataset, assumed to have been processed
to include standard input/output (`stdio`) style problems and solutions. It filters this
dataset to create a high-quality SFT dataset for model training.

The script performs the following steps:
1.  Loads an input RLVR dataset from Hugging Face Hub (`INPUT_HF_DATASET`).
2.  Iterates through each row, validating that required fields are present.
3.  Filters out rows where the `good_program` flag is `False`, keeping only correct and high-quality examples.
4.  For each valid row, it constructs a `messages` list in the standard SFT format, containing a 'user' role with the problem statement and an 'assistant' role with the solution.
5.  Creates a new Hugging Face Dataset from the processed rows.
6.  Pushes the final SFT dataset to a specified repository on the Hugging Face Hub (`OUTPUT_HF_DATASET`).

The output dataset has the following columns:
- `messages`: A list of dictionaries for the conversation turn (e.g., `[{"role": "user", ...}, {"role": "assistant", ...}]`).
- `dataset`: A string identifier for the dataset source (e.g., "code").
- `good_program`: A boolean flag indicating the quality of the program.

"""

# Set up logging
logger = logger_utils.setup_logger(__name__)

INPUT_HF_DATASET = "saurabh5/open-code-reasoning-rlvr-stdio"
OUTPUT_HF_DATASET = "saurabh5/open-code-reasoning-rlvr-sft-stdio"

STDIN_INPUT_COLUMN_NAME = "rewritten_input"
STDIN_SOLUTION_COLUMN_NAME = "rewritten_solution"


def validate_row(row):
    """Validate that a row has all required fields."""
    required_fields = ["good_program", STDIN_INPUT_COLUMN_NAME, STDIN_SOLUTION_COLUMN_NAME]
    for field in required_fields:
        if field not in row:
            return False, f"Missing field: {field}"
        if row[field] is None:
            return False, f"Field {field} is None"
    return True, None


def get_original_input(row):
    return row["messages"][0]["content"]


def main():
    try:
        logger.info(f"Loading dataset: {INPUT_HF_DATASET}")
        input_ds = datasets.load_dataset(INPUT_HF_DATASET, split="train", num_proc=max_num_processes())
        logger.info(f"Loaded {len(input_ds)} rows")

        stdin_rows = []
        skipped_count = 0
        error_count = 0

        for i, row in enumerate(tqdm(input_ds, desc="Processing rows")):
            # Validate row
            is_valid, error_msg = validate_row(row)
            if not is_valid:
                logger.warning(f"Row {i}: {error_msg}")
                error_count += 1
                continue

            if not row["good_program"]:
                skipped_count += 1
                continue

            try:
                # fn_input = row[FN_INPUT_COLUMN_NAME]
                stdin_input = get_original_input(row)
                # fn_solution = row[FN_SOLUTION_COLUMN_NAME]
                stdin_solution = row[STDIN_SOLUTION_COLUMN_NAME]

                # construct messages
                # fn_messages = [
                #    {"role": "user", "content": fn_input},
                #    {"role": "assistant", "content": fn_solution}
                # ]

                stdin_messages = [
                    {"role": "user", "content": stdin_input},
                    {"role": "assistant", "content": stdin_solution},
                ]

                # fn_rows.append({
                #    "messages": fn_messages,
                #    "dataset": "code",
                #    "good_program": row["good_program"],
                # })

                stdin_rows.append({"messages": stdin_messages, "dataset": "code", "good_program": row["good_program"]})

            except Exception as e:
                logger.error(f"Error processing row {i}: {e}")
                error_count += 1
                continue

        logger.info(f"Processed {len(stdin_rows)} valid rows")
        logger.info(f"Skipped {skipped_count} rows (good_program=False)")
        logger.info(f"Errors: {error_count}")

        if len(stdin_rows) == 0:
            logger.error("No valid rows to process!")
            return

        logger.info("Creating datasets...")
        stdin_ds = datasets.Dataset.from_list(stdin_rows)

        logger.info(f"Pushing stdin dataset to: {OUTPUT_HF_DATASET}")
        stdin_ds.push_to_hub(OUTPUT_HF_DATASET)

        logger.info("Processing complete!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    main()
