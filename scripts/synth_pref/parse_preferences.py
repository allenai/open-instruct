import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from open_instruct import logger_utils
from scripts.synth_pref.utils.ultrafeedback_template import parser

logger = logger_utils.setup_logger(__name__)

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
        logger.info(f"Reading all files from {input_dir}")
        _dfs = []
        for f in tqdm(list(input_dir.glob("*jsonl"))):
            _df = pd.read_json(f, lines=True)
            _df["file_id"] = f.stem
            _dfs.append(_df)
        df = pd.concat(_dfs).reset_index(drop=True)
        # Run parser function
        pref_df = parse_openai(
            df, ref_df=pd.read_json(args.reference_file, lines=True), id_col=args.id_col, text_col=args.text_col
        )

    pref_df.to_json(args.output_path, lines=True, orient="records")
    logger.info(f"Saved file ({len(pref_df)} instances) to {args.output_path}")


def parse_openai(df: pd.DataFrame, ref_df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    assert "custom_id" in df.columns, "Missing 'custom_id' in input files"
    df = df.rename(columns={"custom_id": "_custom_id"})
    aspects_map = {"hon": "honesty", "hel": "helpfulness", "ins": "instruction_following", "tru": "truthfulness"}
    aspects = list(aspects_map.values())

    def find_key(d: dict[str, list[str]], value: str) -> str | None:
        return next((k for k, v in d.items() if value in v), None)

    def get_resp(resp: dict[str, Any]) -> str:
        message = resp["body"]["choices"][0]["message"]
        return message.get("content", "")

    # Preprocess the files and compute the ratings
    logger.info("openai: Preprocessing files...")
    df["aspect"] = df["_custom_id"].apply(lambda x: aspects_map.get(x.split("_")[1]))
    df["custom_id"] = df["_custom_id"].apply(lambda x: x.split("_")[0])
    if df.aspect.value_counts().nunique() != 1:
        logger.info("Possible missing files")
        print(df.aspect.value_counts())
    if df.custom_id.value_counts().nunique() != 1:
        logger.info("Possible duplicate files")
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
    logger.info("openai: Parsing responses...")
    for aspect in aspects:
        df[f"{aspect}_responses"] = df[aspect].apply(lambda x: parser(x, aspect=aspect))
        df[f"{aspect}_ratings"] = df[f"{aspect}_responses"].apply(lambda x: get_rating(x))

    # Compute the mean ratings and get the chosen and rejected response
    # For Ultrafeedback, we get the chosen as the highest score, and rejected as the remaining three.
    logger.info("openai: Computing ratings for binarization...")
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

    logger.info("openai: Binarizing preferences...")
    binarized = combined.apply(binarize_pref, axis=1)
    binarized = binarized.dropna().reset_index(drop=True)
    pref_df = pd.concat([combined, binarized], axis=1)
    pref_df["chosen"] = pref_df.apply(
        lambda x: [{"content": x["prompt"], "role": "user"}, {"content": x["chosen_text"], "role": "assistant"}],
        axis=1,
    )
    pref_df["rejected"] = pref_df.apply(
        lambda x: [{"content": x["prompt"], "role": "user"}, {"content": x["rejected_text"], "role": "assistant"}],
        axis=1,
    )

    columns_to_keep = ["prompt", "chosen", "rejected", "chosen_rating", "rejected_rating"]

    if "dataset" in pref_df.columns:
        columns_to_keep.append("dataset")
    if "chosen_model" in pref_df.columns:
        columns_to_keep.append("chosen_model")
    if "rejected_model" in pref_df.columns:
        columns_to_keep.append("rejected_model")

    pref_df = pref_df[columns_to_keep]
    pref_df = pref_df.dropna().reset_index(drop=True)
    return pref_df


def get_rating(resp: dict[str, Any]) -> str:
    def _parse_number(s: str) -> int:
        try:
            int(s)
        except ValueError:
            return -1
        else:
            return int(s)

    num_ratings = []
    for r in resp:
        str_rating = r["Rating"]
        num_ratings.append(_parse_number(str_rating))
    return num_ratings


def compute_mean_rating(row: dict[str, Any]) -> list[str]:
    def _vmeans(data: list[list[int]]) -> list[float] | None:
        try:
            array = np.array(data, dtype=float)
            return list(np.nanmean(array, axis=0))
        except ValueError:
            # Handle jagged lists by padding with NaN
            max_len = max(len(row) for row in data)
            padded = [row + [np.nan] * (max_len - len(row)) for row in data]
            array = np.array(padded, dtype=float)
            return list(np.nanmean(array, axis=0))

    rating_matrix = []
    for aspect in aspects:
        rating_matrix.append(row[f"{aspect}_ratings"])
    return _vmeans(rating_matrix)


def binarize_pref(row):
    ratings = row["mean_ratings"][:4]
    chosen_idx = int(np.argmax(ratings))
    if len(ratings) == 1:
        logger.warning(f"Potential parse error for instance id: {row['id']}")
        rejected_idx = chosen_idx
    else:
        rejected_idx = int(np.random.choice([i for i in range(len(ratings)) if i != chosen_idx], 1))

    try:
        data = {
            "chosen_text": row["completions"][chosen_idx],
            "rejected_text": row["completions"][rejected_idx],
            "chosen_rating": row["mean_ratings"][chosen_idx],
            "rejected_rating": row["mean_ratings"][rejected_idx],
        }
        if "models" in row:
            data["chosen_model"] = row["models"][chosen_idx]
            data["rejected_model"] = row["models"][rejected_idx]

        return pd.Series(data)

    except Exception:
        return None


if __name__ == "__main__":
    main()
