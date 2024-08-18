# main.py

import asyncio
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from open_instruct.rejection_sampling.prompt_templates import (
    get_generation_template,
    get_judgment_template,
)

import anthropic
import genai
import os
import time

# Constants for retries and error handling
API_MAX_RETRY = 3
API_ERROR_OUTPUT = "error"
API_RETRY_SLEEP = 5  # seconds


ANTHROPIC_MODEL_LIST = (
    "claude-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-instant-1",
    "claude-instant-1.2",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
)

OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "gpt-4o-mini-2024-07-18",
)

GEMINI_MODEL_LIST = ("gemini-1.5-flash-001", "gemini-1.5-pro-001")

def chat_completion_anthropic(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None and "api_key" in api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if conv["messages"][0]["role"] == "system":
        sys_msg = conv["messages"][0]["content"]
        conv["messages"] = conv["messages"][1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=conv["messages"],
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg,
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output.strip()


def chat_completion_gemini(model, conv, temperature, max_tokens, api_dict=None):
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    api_model = genai.GenerativeModel(model)

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = api_model.generate_content(
                conv,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
                request_options={"timeout": 1000},
                safety_settings={
                    genai.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.HarmBlockThreshold.BLOCK_NONE,
                    genai.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.HarmBlockThreshold.BLOCK_NONE,
                    genai.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.HarmBlockThreshold.BLOCK_NONE,
                    genai.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.HarmBlockThreshold.BLOCK_NONE,
                },
            )

            if response.prompt_feedback == "block_reason: OTHER":
                print("Weird safety block, continuing!")
                output = "error"
                break
            try:
                output = response.text
            except ValueError:
                print("Erroneous response, not API error")
                print(f"Prompt feedback {response.prompt_feedback}")
                print(f"Finish reason {response.candidates[0].finish_reason}")
                print(f"Safety ratings {response.candidates[0].safety_ratings}")
            else:
                break
        except Exception as e:
            print(f"Failed to connect to Gemini API: {e}")
            time.sleep(API_RETRY_SLEEP)

    return output


@dataclass
class LLMGenerationConfig:
    num_completions: int = 64
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


class LLMProcessorOpenAI:
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
                            if match:
                                score = int(match.group(1))
                            else:
                                score = -1
                            scores.append(score)
                        response = np.mean(scores)
                    else:
                        raise ValueError(f"Invalid mode: {args.mode}")
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


class LLMProcessorAnthropic:
    def __init__(self, config: LLMGenerationConfig):
        self.config = config
        self.api_dict = {"api_key": args.api_key}  # Ensure you pass the API key if necessary

    async def process_text(self, data: dict, i: int, limiter: asyncio.Semaphore, args: Args, gen_args: Args):
        prompt = data['prompt']  # Adjust according to your data format

        async with limiter:
            while True:
                try:
                    # Use the chat_completion_anthropic function to get responses
                    response = chat_completion_anthropic(
                        model=self.config.model,
                        conv={"messages": [{"role": "user", "content": prompt}]},  # Assuming messages are in this format
                        temperature=gen_args.temperature,
                        max_tokens=gen_args.response_length,
                        api_dict=self.api_dict,
                    )
                    return response
                except Exception as e:
                    print(f"Error in {i}: {e}")
                    await asyncio.sleep(30)

    async def process_batch(self, data_list: List[dict], args: Args, gen_args: Args):
        limiter = asyncio.Semaphore(self.config.max_parallel_requests)
        tasks = [self.process_text(data, i, limiter, args, gen_args) for i, data in enumerate(data_list)]
        # Use tqdm to track progress
        return await tqdm.gather(*tasks, total=len(tasks), desc="Processing Batch")


class LLMProcessorGemini:
    def __init__(self, config: LLMGenerationConfig, api_key: str):
        self.config = config
        self.api_key = api_key

    async def process_text(self, data: dict, i: int, limiter: asyncio.Semaphore, args: Args, gen_args: GenerationArgs):
        prompt_token_ids = data['prompt_token_ids']  # Adjust according to your data format
        conv = {"messages": [{"role": "user", "content": prompt_token_ids}]}

        async with limiter:
            while True:
                try:
                    response = chat_completion_gemini(
                        model=self.config.model,
                        conv=conv,
                        temperature=gen_args.temperature,
                        max_tokens=gen_args.response_length,
                        api_dict={"api_key": self.api_key},
                    )
                    return response
                except Exception as e:
                    print(f"Error in {i}: {e}")
                    await asyncio.sleep(30)

    async def process_batch(self, data_list: List[dict], args: Args, gen_args: Args):
        limiter = asyncio.Semaphore(self.config.max_parallel_requests)
        tasks = [self.process_text(data, i, limiter, args, gen_args) for i, data in enumerate(data_list)]
        # Use tqdm to track progress
        return await tqdm.gather(*tasks, total=len(tasks), desc="Processing Batch")
