# main.py

import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional

from openai import AsyncOpenAI
from prompt_templates import get_generation_template, get_judgment_template
from tqdm.asyncio import tqdm


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


class LLMProcessor:
    def __init__(self, config: LLMGenerationConfig):
        self.config = config
        self.async_client = AsyncOpenAI()

    async def process_text(self, data: dict, i: int, limiter: asyncio.Semaphore, args: Args):
        if args.mode == "generation":
            template = get_generation_template(args.skill)
            text = template.format(prompt=data)
        else:  # judgment mode
            template = get_judgment_template(args.skill)
            text = template.format(prompt=data["prompt"], response=data["response"])

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
                    response = response.choices[0].message.content
                    if args.mode == "generation":
                        response = response
                    else:
                        match = re.search(r"Total score:\s*(\d+)", response)
                        if match:
                            total_score = int(match.group(1))
                        else:
                            total_score = -1
                        response = total_score
                    break
                except Exception as e:
                    print(f"Error in {i}: {e}")
                    await asyncio.sleep(30)

        return response

    async def process_batch(self, data_list: List[dict], args: Args):
        limiter = asyncio.Semaphore(self.config.max_parallel_requests)
        tasks = [self.process_text(data, i, limiter, args) for i, data in enumerate(data_list)]
        # Use tqdm to track progress
        return await tqdm.gather(*tasks, total=len(tasks), desc="Processing Batch")
