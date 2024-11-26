import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.synth_pref.utils.ultrafeedback_template import parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

aspects = ["helpfulness", "honesty", "instruction_following", "truthfulness"]


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser("Parse preferences to get the final preference dataset.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory that contains all the JSONL files of preferences.")
    parser.add_argument("--reference_file", type=Path, required=True, help="Path of the reference file containing the prompts and responses as reference.")
    parser.add_argument("--output_path", type=Path, required=True, help="Output to save the preferences in JSONL format.")
    parser.add_argument("--id_col", type=str, default="prompt_hash", help="ID column to use.")
    parser.add_argument("--text_col", type=str, default="raw_text", help="Column name of the instruction field.")
    parser.add_argument("--override_errors", action="store_true", help="Try best-case parse if warnings arise.")
    # OpenAI Parser
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    if args.command == "openai":
        # Read all JSONL files in a single dataframe
        input_dir = Path(args.input_dir)
        logging.info(f"Reading all files from {input_dir}")
        _dfs = []
        for f in tqdm(list(input_dir.glob("*jsonl"))):
            _df = pd.read_json(f, lines=True)
            _df["file_id"] = f.stem
            _dfs.append(_df)
        df = pd.concat(_dfs).reset_index(drop=True)
        # Run parser function
        pref_df = parse_openai(
            df,
            ref_df=pd.read_json(args.reference_file, lines=True),
            id_col=args.id_col,
            text_col=args.text_col,
        )

    pref_df.to_json(args.output_path, lines=True, orient="records")
    logging.info(f"Saved file ({len(pref_df)} instances) to {args.output_path}")


def parse_openai(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    id_col: str,
    text_col: str,
) -> pd.DataFrame:
    assert "custom_id" in df.columns, "Missing 'custom_id' in input files"
    df = df.rename(columns={"custom_id": "_custom_id"})
    aspects_map = {
        "hon": "honesty",
        "hel": "helpfulness",
        "ins": "instruction_following",
        "tru": "truthfulness",
    }
    aspects = list(aspects_map.values())

    def find_key(d: dict[str, list[str]], value: str) -> Optional[str]:
        return next((k for k, v in d.items() if value in v), None)

    def get_resp(resp: dict[str, Any]) -> str:
        message = resp["body"]["choices"][0]["message"]
        return message.get("content", "")

    # Preprocess the files and compute the ratings
    logging.info("openai: Preprocessing files...")
    df["aspect"] = df["_custom_id"].apply(lambda x: aspects_map.get(x.split("_")[1]))
    df["custom_id"] = df["_custom_id"].apply(lambda x: x.split("_")[0])
    if df.aspect.value_counts().nunique() != 1:
        logging.info("Possible missing files")
        print(df.aspect.value_counts())
    if df.custom_id.value_counts().nunique() != 1:
        logging.info("Possible duplicate files")
        print(df.custom_id.value_counts())

    df["output"] = df["response"].apply(lambda x: get_resp(x))
    df = df[["custom_id", "aspect", "output"]]
    df = (
        df.pivot(index="custom_id", columns="aspect", values="output")
        .reset_index()
        .fillna("")
        .rename(columns={"custom_id": "id"})
    )

    # Parse the responses
    logging.info("openai: Parsing responses...")
    for aspect in aspects:
        df[f"{aspect}_responses"] = df[aspect].apply(lambda x: parser(x, aspect=aspect))
        df[f"{aspect}_ratings"] = df[f"{aspect}_responses"].apply(
            lambda x: get_rating(x)
        )

    # Compute the mean ratings and get the chosen and rejected response
    # For Ultrafeedback, we get the chosen as the highest score, and rejected as the remaining three.
    logging.info("openai: Computing ratings for binarization...")
    df["mean_ratings"] = df.apply(lambda row: compute_mean_rating(row), axis=1)

    ref_df = ref_df.reset_index()
    columns_to_keep = [id_col, text_col, "completions", "index"]
    if "dataset" in ref_df.columns:
        columns_to_keep.append("dataset")
    if "models" in ref_df.columns:
        columns_to_keep.append("models")
    ref_df = ref_df[columns_to_keep].rename(columns={text_col: "prompt"})
    if "prompt_hash" in ref_df.columns:
        ref_df = ref_df.rename(columns={"prompt_hash": "id"})

    combined = df[["id", "mean_ratings"]].merge(ref_df, on="id", how="left")

    logging.info("openai: Binarizing preferences...")
    binarized = combined.apply(binarize_pref, axis=1)
    binarized = binarized.dropna().reset_index(drop=True)
    pref_df = pd.concat([combined, binarized], axis=1)
    pref_df["chosen"] = pref_df.apply(
        lambda x: [
            {"content": x["prompt"], "role": "user"},
            {"content": x["chosen_text"], "role": "assistant"},
        ],
        axis=1,
    )
    pref_df["rejected"] = pref_df.apply(
        lambda x: [
            {"content": x["prompt"], "role": "user"},
            {"content": x["rejected_text"], "role": "assistant"},
        ],
        axis=1,
    )

    columns_to_keep = [
        "prompt",
        "chosen",
        "rejected",
        "chosen_rating",
        "rejected_rating",
    ]

    if "dataset" in pref_df.columns:
        columns_to_keep.append("dataset")
    if "chosen_model" in pref_df.columns:
        columns_to_keep.append("chosen_model")
    if "rejected_model" in pref_df.columns:
        columns_to_keep.append("rejected_model")

    pref_df = pref_df[columns_to_keep]
    pref_df = pref_df.dropna().reset_index(drop=True)
    return pref_df


if __name__ == "__main__":
    main()
