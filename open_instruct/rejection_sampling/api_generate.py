# main.py

import asyncio
import re
from dataclasses import dataclass

import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from open_instruct.rejection_sampling.prompt_templates import get_generation_template, get_judgment_template


@dataclass
class LLMGenerationConfig:
    num_completions: int = 64
    model: str = "gpt-3.5-turbo-0125"
    max_parallel_requests: int | None = None

    def __post_init__(self):
        if "gpt-3.5" in self.model:
            self.max_parallel_requests = 11
        elif "gpt-4" in self.model:
            self.max_parallel_requests = 13


@dataclass
class Args:
    output_path: str | None = None
    num_trials: int = 1


class LLMProcessor:
    def __init__(self, config: LLMGenerationConfig):
        self.config = config
        self.async_client = AsyncOpenAI()

    async def process_text(self, data: dict, i: int, limiter: asyncio.Semaphore, args: Args, gen_args: Args):
        if args.mode == "generation":
            template = get_generation_template(args.skill)
            text = template.format(prompt=data)
        elif args.mode == "judgement":  # judgment mode
            template = get_judgment_template(args.skill)
            text = template.format(prompt=data["prompt"], response=data["response"])
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

        async with limiter:
            while True:
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": text},
                        ],
                        n=gen_args.num_completions,  # Request multiple completions
                        temperature=gen_args.temperature,  # Sampling temperature
                        max_tokens=gen_args.response_length,  # Maximum tokens in the response
                        top_p=gen_args.top_p,  # Top-P (nucleus) sampling
                        stop=None,  # Add stopping criteria if needed
                    )
                    # Collect all completions if `n > 1`
                    completions = [choice.message.content for choice in response.choices]
                    if args.mode == "generation":
                        response = completions
                    elif args.mode == "judgement":
                        # If in judgment mode, process the completions (for example, extracting scores)
                        scores = []
                        for completion in completions:
                            match = re.search(r"Total score:\s*(\d+)", completion)
                            score = int(match.group(1)) if match else -1
                            scores.append(score)
                        response = np.mean(scores)
                    else:
                        raise ValueError(f"Invalid mode: {args.mode}")
                    break
                except Exception as e:
                    print(f"Error in {i}: {e}")
                    await asyncio.sleep(30)

        return response

    async def process_batch(self, data_list: list[dict], args: Args, gen_args: Args):
        limiter = asyncio.Semaphore(self.config.max_parallel_requests)
        tasks = [self.process_text(data, i, limiter, args, gen_args) for i, data in enumerate(data_list)]
        # Use tqdm to track progress
        return await tqdm.gather(*tasks, total=len(tasks), desc="Processing Batch")
