import asyncio
import random
import time
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, HfArgumentParser

import open_instruct.utils as open_instruct_utils
from open_instruct.generation import print_rich_table


@dataclass
class LLMJudgeConfig:
    n: int = 64
    model: str = "gpt-3.5-turbo-0125"
    max_parallel_requests: int | None = None

    def __post_init__(self):
        if "gpt-3.5" in self.model:
            # gpt-3.5 generates so fast that it will exceeds the
            # token limit per minute
            self.max_parallel_requests = 50
        elif "gpt-4" in self.model:
            self.max_parallel_requests = 13


@dataclass
class Args:
    csv: str = "gpt.csv"
    output_path: str | None = None
    num_trails: int = 1


TEMPLATE = r"""\
### Instructions:
You are a machine learning data cleanning engineer and your job is to figure out if the data quality is good. You are tasked to clean an instruction following dataset. You will be given a prompt and your job is to determine if the prompt is bad or does not make sense.

FIRST provide a one-sentence explanation on the quality of the prompt. SECOND, on a new line, state only "not bad" or "bad" to indicate your choice"
Your response should use the format:"
Explanation: <one-sentence reasoning and explanation>
Decision: <"not bad" or "bad">

### Prompt:
{{prompt}}

### Your answer:
"""


def llm_judge(ljc: LLMJudgeConfig, df: pd.DataFrame):
    async_client = AsyncOpenAI()

    async def process_text(prompt: str, i: int, limiter: asyncio.Semaphore):
        text = TEMPLATE.replace("{{prompt}}", prompt)
        async with limiter:
            response = None
            while response is None:
                try:
                    response = await async_client.chat.completions.create(
                        model=ljc.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": text},
                        ],
                    )
                    r = response.choices[0].message.content
                except Exception as e:
                    print(f"error in {i}: {e}")
                    time.sleep(30)  # deal with rate limit
                    continue

            try:
                explanation = r.split("Explanation:")[1].split("Decision:")[0].strip()
                decision = r.split("Decision:")[1].strip()
                return explanation, decision, i, text + r
            except Exception as e:
                print(f"error in {i} {e}")
                return "", random.choice(["A", "B"]), i, text + r

    async def main(ljc: LLMJudgeConfig, df: pd.DataFrame):
        """`df` should have columns: `prompt`, `response0`, `response1`"""
        limiter = asyncio.Semaphore(ljc.max_parallel_requests)
        tasks = []
        df["explanation"] = [None for _ in range(len(df))]
        df["decision"] = [None for _ in range(len(df))]
        df["entire_conversation"] = [None for _ in range(len(df))]
        r = range(min(ljc.n, len(df)))
        if ljc.n == -1:
            r = range(len(df))
        for i in r:
            prompt = df["prompt"].iloc[i].strip()
            task = asyncio.create_task(process_text(prompt, i, limiter))
            tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks)

        for _, (explanation, decision, i, entire_conversation) in enumerate(results):
            df.at[i, "explanation"] = explanation
            df.at[i, "entire_conversation"] = entire_conversation
            df.at[i, "decision"] = decision.lower()
        return df

    return asyncio.run(main(ljc, df))


if __name__ == "__main__":
    args, ljc = HfArgumentParser((Args, LLMJudgeConfig)).parse_args_into_dataclasses()
    raw_dataset = load_dataset(
        "allenai/tulu-v2-sft-mixture", split="train", num_proc=open_instruct_utils.max_num_processes()
    )
    raw_dataset = raw_dataset.select(range(64))
    tokenizer = AutoTokenizer.from_pretrained("allenai/llama-3-tulu-2-8b")
    ds = raw_dataset.map(
        lambda x: {"prompt": tokenizer.apply_chat_template(x["messages"][:-1], tokenize=False)},
        remove_columns=raw_dataset.column_names,
    )

    # Extract the prompts and reference completions
    df = ds.to_pandas()
    judge_df = llm_judge(ljc, df)
    print(judge_df["decision"].value_counts())
    print_rich_table(judge_df[["prompt", "decision", "explanation"]].iloc[2:5])
