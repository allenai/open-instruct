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
    nb_completions: int = 64
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

    async def process_text(self, data: dict, i: int, limiter: asyncio.Semaphore, args: Args, gen_args: Args):
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
                        n=3,  # Request multiple completions
                        temperature=1,  # Sampling temperature
                        max_tokens=50,  # Maximum tokens in the response
                        top_p=0.9,  # Top-P (nucleus) sampling
                        stop=None,  # Add stopping criteria if needed
                    )
                    # Collect all completions if `n > 1`
                    completions = [choice.message.content for choice in response.choices]
                    breakpoint()
                    if args.mode == "generation":
                        response = completions
                    else:
                        # If in judgment mode, process the completions (for example, extracting scores)
                        responses = []
                        for completion in completions:
                            match = re.search(r"Total score:\s*(\d+)", completion)
                            if match:
                                total_score = int(match.group(1))
                            else:
                                total_score = -1
                            responses.append(total_score)
                        response = responses
                    break
                except Exception as e:
                    print(f"Error in {i}: {e}")
                    await asyncio.sleep(30)

        return response

    async def process_batch(self, data_list: List[dict], args: Args, gen_args: Args):
        limiter = asyncio.Semaphore(self.config.max_parallel_requests)
        tasks = [self.process_text(data, i, limiter, args, gen_args) for i, data in enumerate(data_list)]
        # Use tqdm to track progress
        return await tqdm.gather(*tasks, total=len(tasks), desc="Processing Batch")
