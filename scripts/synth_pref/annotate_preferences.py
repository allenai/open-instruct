import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from scripts.synth_pref.utils.openai_api import format_for_openai_batch
from scripts.synth_pref.utils.ultrafeedback_template import system_prompt

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

load_dotenv()


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser("Perform batch inference using OpenAI", formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    # Define shared arguments
    shared_args = argparse.ArgumentParser(add_help=False)
    shared_args.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="Model to use.")

    # Convert dataframe into something parse-able for OpenAI
    parser_convert = subparsers.add_parser("convert", help="Convert dataframe from collate_results for OpenAI", parents=[shared_args])
    parser_convert.add_argument("--input_path", type=Path, help="Path to the JSONL file.")
    parser_convert.add_argument("--output_dir", type=Path, help="Directory to save the output JSONL files (10k each).")
    parser_convert.add_argument("--estimate_tokens", action="store_true", help="Estimate number of tokens.")
    parser_convert.add_argument("--rows_per_shard", type=int, default=250, help="Number of rows per shard.")
    parser_convert.add_argument("--prefix", type=str, help="Name prefix for the shards.")
    parser_convert.add_argument("--id_col", type=str, default="prompt_hash", help="ID column to use.")
    parser_convert.add_argument("--no_suffix", action="store_true", default=False, help="If set, no suffix will be created.")
    parser_convert.add_argument("--no_system_prompt", action="store_true", default=False, help="If set, no system prompt will be passed.")

    # Upload and then run the batch job
    parser_upload = subparsers.add_parser("upload", help="Upload file to OpenAI", parents=[shared_args])
    parser_upload.add_argument("--input_dir", type=Path, help="Directory to the input files.")
    parser_upload.add_argument("--output_dir", type=Path, help="Directory to store the reports.")
    parser_upload.add_argument("--delay", type=int, default=20, help="Number of seconds to delay between uploads.")
    parser_upload.add_argument("--description", type=str, default="Batch OpenAI job", help="Metadata of the file.")

    # Check status given the batch_id
    parser_download =subparsers.add_parser(name="download", help="Get information about the batch and download results if done.", parents=[shared_args])
    parser_download.add_argument("--batch_report", type=Path, help="OpenAI batch report CSV file.")
    parser_download.add_argument("--output_dir", required=False, default=None, help="If set, will download batches that were already done in the specified directory.")
    return parser.parse_args()


def main():
    args = get_args()

    api_key: str = os.getenv("OPENAI_API_KEY", default=None)
    if not api_key:
        msg = f"API key not found! Please set it in {env_name}."
        logging.error(msg)
        raise ValueError(msg)
    client = OpenAI(api_key=api_key)

    if args.command == "convert":
        df = pd.read_json(args.input_path, lines=True)

        # Get aspect from filename
        hdfs = format_for_openai_batch(
            df,
            system_prompt=None if args.no_system_prompt else system_prompt,
            url="/v1/chat/completions",
            model=args.model,
            id_col=args.id_col,
            rows_per_shard=args.rows_per_shard,
            custom_id_suffix=(
                "" if args.no_suffix else get_aspect_suffix(args.input_path.stem)
            ),
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, hdf in enumerate(hdfs):
            output_path = output_dir / f"{args.prefix}-shard-{str(idx).zfill(6)}.jsonl"
            hdf.to_json(output_path, lines=True, orient="records")
        logging.info(f"Saved shards to {output_dir}")

    if args.command == "upload":
        input_dir = Path(args.input_dir)
        files = list(input_dir.glob("*.jsonl"))

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / Path("batch_infer_openai_results.csv")
        headers = ["id", "input_file_id", "local_filepath", "shard"]
        pd.DataFrame(columns=headers).to_csv(file_path, index=False)

        # Create cache for retrying files if batching fails
        retry_file_path = output_dir / Path("retry_for_batch_infer.csv")
        headers = ["local_filepath", "shard"]
        pd.DataFrame(columns=headers).to_csv(retry_file_path, index=False)

        for file in tqdm(files):
            batch_input_file = client.files.create(
                file=open(file, "rb"), purpose="batch"
            )

            batch_input_file_id = batch_input_file.id
            logging.info(f"File: {file} ID: {batch_input_file_id}")

            # Keep trying to create batch until it succeeds
            batch = None
            while True:
                try:
                    batch = client.batches.create(
                        input_file_id=batch_input_file_id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={
                            "description": f"{args.description} from file: {file.name}",
                            "aspect": file.stem.removesuffix("-full"),
                        },
                    )
                    # If we get here, batch creation succeeded
                    break

                except Exception as e:
                    if "pending" in str(e).lower():
                        logging.info("Batch creation pending, retrying in 10s")
                        time.sleep(10)
                        continue
                    if "running" in str(e).lower():
                        logging.info("Batch creation running, retrying in 10s")
                        time.sleep(10)
                        continue
                    else:
                        # For non-pending errors, log warning and add to retry file
                        logging.warning(f"Please retry: {file} | ERROR: {str(e)}")
                        row = {
                            "local_filepath": str(file),
                            "shard": file.stem.removesuffix("-full"),
                        }
                        pd.DataFrame([row]).to_csv(
                            retry_file_path, mode="a", header=False, index=False
                        )
                        break

            if batch is not None:
                batch_dict = dict(batch)
                row = {
                    "id": batch_dict["id"],
                    "input_file_id": batch_dict["input_file_id"],
                    "local_filepath": str(file),
                    "shard": file.stem.removesuffix("-full"),
                }
                pd.DataFrame([row]).to_csv(
                    file_path, mode="a", header=False, index=False
                )
        logging.info(
            "Created a batch report as a CSV file. Please keep this as you'll use this to retrieve the results later on!"
        )

    if args.command == "status":
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        else:
            output_dir = None

        df = pd.read_csv(args.batch_report)
        batch_ids = df["id"].to_list()
        logging.info(f"Checking status of {len(batch_ids)} batch IDs")
        for batch_id in batch_ids:
            status = "validating"
            while status not in ("completed", "failed", "canceled"):
                batch_response = client.batches.retrieve(batch_id)
                status = batch_response.status
                print(
                    f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}"
                )

                if status == "completed" and output_dir:
                    file_id = batch_response.output_file_id
                    output_path = output_dir / f"{batch_id}.jsonl"
                    logging.info(f"Retrieving responses for batch id {batch_id}")
                    file_response = client.files.content(file_id)
                    raw_responses = file_response.text.strip().split("\n")
                    json_responses = [
                        json.loads(response) for response in raw_responses
                    ]
                    logging.info(f"Saving file {file_id} to {output_path}")
                    resp_df = pd.DataFrame(json_responses)
                    resp_df.to_json(output_path, lines=True, orient="records")

            if batch_response.status == "failed":
                for error in batch_response.errors.data:
                    print(f"Error code {error.code} Message {error.message}")


def get_aspect_suffix(filename: str) -> str:
    if "truthfulness" in filename:
        return "tru"
    elif "helpfulness" in filename:
        return "hel"
    elif "instruction_following" in filename:
        return "ins"
    elif "honesty" in filename:
        return "hon"
    else:
        raise ValueError(f"Unidentified filename to get aspect from {filename}")


if __name__ == "__main__":
    main()
