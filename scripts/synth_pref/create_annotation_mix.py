import argparse
import hashlib
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from jinja2 import BaseLoader, Environment
from tqdm import tqdm

from scripts.synth_pref.utils.ultrafeedback_template import user_prompts

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser("Collate all model responses.")
    parser.add_argument("-n", "--name", type=str, required=True, help="Name of the experiment or the preference dataset.")
    parser.add_argument("--input_dir", type=Path, help="Directory where all the model completions are saved.")
    parser.add_argument("--output_dir", type=str, required=False, default="annotation_mix", help="Local file directory to save full annotation mix formatted for preference annotation.")
    parser.add_argument("--prompt_template", type=str, default="ultrafeedback", choices=["ultrafeedback"], help="Prompt template to apply.")
    parser.add_argument("--id_col", default="prompt_hash", type=str, help="ID to use for combining prompts.")
    parser.add_argument("--one_side_model", default=None, help="If set, will keep one column to be the same model. Useful for on-policy set-up.")
    parser.add_argument("-x", "--ignore_model", nargs="+", help="List of models to exclude from the model pool.")
    parser.add_argument("-y", "--include_model", nargs="+", help="List of models to include in the model pool.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    input_dir = Path(args.input_dir)
    # Filter the models and get the final paths
    model_paths = filter_models(
        {folder.name: folder for folder in input_dir.iterdir() if folder.is_dir()},
        include_models=args.include_model,
        ignore_models=args.ignore_model,
    )
    logging.info(f"Using the {len(model_paths)} models")
    df = sample_responses(
        model_paths,
        source_dir=input_dir,
        id_col=args.id_col,
        one_side_model=args.one_side_model,
    )

    logging.info(f"*** Applying the prompt template '{args.prompt_template}' ***")
    aspect_dfs = {}
    for aspect, prompt in user_prompts.items():
        id_col = "prompt_hash" if "prompt_hash" in df.columns else "id"
        cols = [id_col, "text", "models", "completions"]
        aspect_df = df[cols].copy(deep=True).rename(columns={"text": "raw_text"})
        tqdm.pandas(desc=aspect)
        aspect_df["text"] = df.progress_apply(
            lambda row: render_template(
                instruction=row["text"],
                completions=row["completions"],
                prompt_template=prompt,
            ),
            axis=1,
        )
        aspect_df["aspect"] = aspect
        aspect_dfs[aspect] = aspect_df

    parent_output_dir = Path(args.output_dir) / str(args.name)
    full_output_dir = Path(args.output_dir) / "full"
    full_output_dir.mkdir(parents=True, exist_ok=True)
    rows_per_shard = 250
    logging.info(f"*** Saving shards for each aspect in {parent_output_dir} ***")
    for aspect, aspect_df in aspect_dfs.items():
        output_dir = parent_output_dir / aspect
        output_dir.mkdir(parents=True, exist_ok=True)
        full_aspect_df = aspect_df.reset_index()
        full_aspect_df.to_json(
            full_output_dir / f"{aspect}-full.jsonl",
            lines=True,
            orient="records",
        )

        logging.debug(f"Saving shards in {output_dir}")
        for idx, shard in enumerate(range(0, len(df), rows_per_shard)):
            shard_df = aspect_df.iloc[shard : shard + rows_per_shard]
            shard_df = shard_df.reset_index()
            shard_df.to_json(
                output_dir / f"{aspect}___shard-{str(idx).zfill(6)}.jsonl",
                lines=True,
                orient="records",
            )


def sample_responses(
    model_paths: dict[str, pd.DataFrame],
    source_dir: Path,
    id_col: str,
    one_side_model: Optional[str] = None,
) -> pd.DataFrame:
    """Sample responses and combine all dataframes"""

    def _filter_common_rows(
        dfs: dict[str, pd.DataFrame], unique_key: str = "prompt_hash"
    ) -> dict[str, pd.DataFrame]:
        common_ids = pd.concat(dfs.values())[unique_key].value_counts()
        common_ids = common_ids[common_ids == len(dfs)].index
        return {key: df[df[unique_key].isin(common_ids)] for key, df in dfs.items()}

    logging.debug(f"Reading JSONL files from cache {source_dir}")
    model_dfs: dict[str, pd.DataFrame] = {}
    for model, path in tqdm(model_paths.items()):
        model_df = pd.concat(
            pd.read_json(file, lines=True) for file in path.glob("*.jsonl")
        )
        model_df = model_df.reset_index(drop=True)
        # Create hash so that it's easier to reference them later on
        if "prompt_hash" not in model_df.columns:
            model_df["reference_str"] = model_df["text"] + model_df[id_col]
            model_df["prompt_hash"] = model_df["reference_str"].apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()
            )
        model_df = (
            model_df.drop_duplicates(subset=id_col)
            .dropna(subset="outputs")
            .reset_index(drop=True)
        )
        model_df["response"] = model_df["outputs"].apply(lambda x: x[0]["text"])
        model_df["model_name"] = model.replace("___", "/")

        model_dfs[model] = model_df
        n_unique_instances = model_df[id_col].nunique()
        tqdm.write(f"Loaded {model} (len={n_unique_instances})")

    # Compile responses from each model and then sample responses because of the Ultrafeedback pipeline
    model_dfs = _filter_common_rows(model_dfs, id_col)
    if one_side_model:
        assert one_side_model in model_dfs, f"Unknown model: {one_side_model}"
        logging.info(f"Value passed to --one_side_model: '{one_side_model}'")
        cols = ["prompt_hash", "response", "model_name"]
        one_side_df = model_dfs.pop(one_side_model)[cols].rename(columns={id_col: "id"})
        combined_df = pd.concat(model_dfs.values()).reset_index(drop=True)
        others_df = combined_df.groupby("prompt_hash").agg(
            id=("prompt_hash", "first"),
            text=("text", "first"),
            all_completions=("response", list),
            all_models=("model_name", list),
            # dataset=("dataset", "first"),
        )
        # fmt: off
        others_df["vs_choice_idx"] = [random.choice(range(len(model_dfs.keys()))) for _ in range(len(others_df))]
        others_df["vs_completion"] = others_df.apply(lambda row: row["all_completions"][row["vs_choice_idx"]], axis=1)
        others_df["vs_model"] = others_df.apply(lambda row: row["all_models"][row["vs_choice_idx"]], axis=1)
        # fmt: on
        vs_df = one_side_df.merge(others_df, on="prompt_hash").drop(
            columns=["vs_choice_idx"]
        )

        def _sample_models(row):
            if random.random() >= 0.5:
                row["models"] = [row["vs_model"], row["model_name"]]
                row["completions"] = [row["vs_completion"], row["response"]]
            else:
                row["models"] = [row["model_name"], row["vs_model"]]
                row["completions"] = [row["response"], row["vs_completion"]]
            return row

        vs_df = vs_df.apply(_sample_models, axis=1)
        result_df = vs_df[["prompt_hash", "text", "models", "completions"]]
    else:
        combined_df = pd.concat(model_dfs.values()).reset_index(drop=True)
        result_df = combined_df.groupby("prompt_hash").agg(
            id=(id_col, "first"),
            text=("text", "first"),
            all_completions=("response", list),
            all_models=("model_name", list),
            # dataset=("dataset", "first"),
        )
        # fmt: off
        # Sample four responses because of ultrafeedback format
        result_df["sampled_idxs"] = [random.sample(range(len(model_dfs.keys())), 4) for _ in range(len(result_df))]
        result_df["completions"] = result_df.apply(lambda row: [row["all_completions"][idx] for idx in row["sampled_idxs"]], axis=1)
        result_df["models"] = result_df.apply(lambda row: [row["all_models"][idx] for idx in row["sampled_idxs"]], axis=1)
        # fmt: on

    logging.debug(f"Compiled dataframe has {len(result_df)} instances")
    return result_df


def filter_models(
    model_paths: dict[str, Path],
    include_models: Optional[list[str]] = None,
    ignore_models: Optional[list[str]] = None,
) -> dict[str, Path]:
    model_names = model_paths.keys()
    if include_models:
        model_names = [model for model in model_names if model in include_models]
    if ignore_models:
        model_names = [model for model in model_names if model not in ignore_models]
    # Filter model_paths to only include models in model_names
    model_paths = {
        model: path for model, path in model_paths.items() if model in model_names
    }
    return model_paths


def render_template(
    instruction: str,
    completions: list[str],
    prompt_template: str,
) -> str:
    rtemplate = Environment(loader=BaseLoader()).from_string(prompt_template)
    user_prompt = rtemplate.render(
        instruction=instruction,
        completions=completions,
    )
    return user_prompt


if __name__ == "__main__":
    main()
