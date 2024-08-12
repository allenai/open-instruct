# main.py

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import asyncio
import time
import random
from openai import AsyncOpenAI
from prompt_templates import get_generation_template, get_judgment_template


@dataclass
class LLMGenerationConfig:
    n: int = 64
    model: str = "gpt-3.5-turbo-0125"
    max_parallel_requests: Optional[int] = None

    def __post_init__(self):
        if "gpt-3.5" in self.model:
            self.max_parallel_requests = 11
        elif "gpt-4" in self.model:
            self.max_parallel_requests = 13


@dataclass
class Args:
    output_path: Optional[str] = None
    num_trials: int = 1
    skill: str = "summarization"
    mode: str = "generation"  # Can be "generation" or "judgment"


class LLMProcessor:
    def __init__(self, config: LLMGenerationConfig):
        self.config = config
        self.async_client = AsyncOpenAI()

    async def process_text(self, data: dict, i: int, limiter: asyncio.Semaphore, args: Args):
        if args.mode == "generation":
            template = get_generation_template(args.skill)
            text = template.format(**data)
        else:  # judgment mode
            template = get_judgment_template(args.skill)
            text = template.format(**data)

        async with limiter:
            while True:
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": text},
                        ],
                    )
                    r = response.choices[0].message.content
                    break
                except Exception as e:
                    print(f"Error in {i}: {e}")
                    await asyncio.sleep(30)

        return r, i, text + r

    async def process_batch(self, data_list: List[dict], args: Args):
        limiter = asyncio.Semaphore(self.config.max_parallel_requests)
        tasks = [self.process_text(data, i, limiter, args) for i, data in enumerate(data_list)]
        return await asyncio.gather(*tasks)


def main(args: Args):
    df = pd.read_csv(args.csv)
    config = LLMGenerationConfig()
    processor = LLMProcessor(config)

    data_list = df.to_dict('records')
    results = asyncio.run(processor.process_batch(data_list, args))

    # Process and save results as needed
    # ...


if __name__ == "__main__":
    args = Args(mode="generation", skill="summarization")  # Example usage
    main(args)