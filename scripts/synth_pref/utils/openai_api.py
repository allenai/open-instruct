"""
Source: https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a
"""

import logging
import sys
from typing import Optional

import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)


def create_openai_chat_fmt(
    user_prompt: str,
    system_prompt: Optional[str] = None,
) -> list[dict[str, str]]:
    """Format the text into OpenAI instances"""
    message = []
    if system_prompt:
        message.append({"role": "system", "content": system_prompt})
    message.append({"role": "user", "content": user_prompt})

    return message


def format_for_openai_batch(
    df: pd.DataFrame,
    model: str,
    system_prompt: Optional[str] = None,
    url: str = "/v1/chat/completions",
    id_col: str = "prompt_hash",
    rows_per_shard=10_000,
    custom_id_suffix: Optional[str] = "",
) -> list[pd.DataFrame]:
    df = df.reset_index()
    df = df[[id_col, "text"]].rename(columns={id_col: "custom_id"})
    df["method"] = "POST"
    df["url"] = url
    df["body"] = df["text"].apply(
        lambda x: {
            "model": model,
            "messages": create_openai_chat_fmt(
                user_prompt=x, system_prompt=system_prompt
            ),
            "temperature": 0.1,
        }
    )
    df["custom_id"] = df["custom_id"].apply(lambda x: str(x))
    if custom_id_suffix:
        df["custom_id"] = df["custom_id"].apply(
            lambda x: str(x) + "_" + custom_id_suffix
        )
    df = df.drop(columns=["text"])

    shards = []
    for _, shard in enumerate(range(0, len(df), rows_per_shard)):
        shard_df = df.iloc[shard : shard + rows_per_shard]
        shards.append(shard_df)
    return shards
