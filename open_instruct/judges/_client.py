import logging
import asyncio
import instructor
import openai
import litellm
from openai import AzureOpenAI, AsyncAzureOpenAI

import time
import os
import re
import random
from typing import Optional, Union, Any

from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt, AsyncRetrying

import easyapi


openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Or DEBUG, depending on verbosity


PRICING_PER_1M_TOKENS = {
    "gpt-4": {"input": 0.00003, "output": 0.00006}, 
    "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002}, 
    "gpt-4-1106-preview": {"input": 0.00001, "output": 0.00003},
    "gpt-4o": {"input": 0.0000025, "output": 0.000001},
    "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
    "claude-sonnet": {"input": 0.000003, "output": 0.000015}, 
    "deepseek-chat": {"input": 0.00000007, "output": 0.000001}, 
    "deepseek-reasoner": {"input": 0.00000014, "output": 0.000002}, 
    "claude-3-7-sonnet-20250219": {"input": 0.000003, "output": 0.000015}
}


# Response wrapper to safely hold metadata
class CompletionWithMetadata:
    def __init__(self, response):
        self.response = response
        self.cost = 0
        self.response_time = 0.0
        self.usage = getattr(response, "usage", None)
        self.choices = getattr(response, "choices", None)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        response_obj = self.__dict__.get('response')
        if response_obj is not None:
            try:
                return getattr(response_obj, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __str__(self):
        return str(self.response)

    def __repr__(self):
        return repr(self.response)


def build_fallback_response(reason: str, elapsed_time: float = 0.0):
    """Return a structured fallback response with a reasoning and zero score."""
    logger.warning(f"Returning fallback response: {reason}")
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f'{{"REASONING": "{reason}", "SCORE": 0}}'
                }
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        },
        "cost": 0.0,
        "response_time": elapsed_time,
    }

async def call_azure_sync(client, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: client.chat.completions.create(**kwargs))


def llm_client(model_type: str = "openai"):
    if model_type == "huggingface":
        import litellm
        litellm.set_verbose=True
        client = litellm
    else:
        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    return client 

### V2 of async completion with retry
def track_cost_callback(kwargs, completion_response, start_time, end_time):
    try:
        model_name = kwargs.get("model", "")
        usage = getattr(completion_response, "usage", None)
        if usage is None and hasattr(completion_response, 'response'):
             usage = getattr(completion_response.response, 'usage', None)

        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        pricing = PRICING_PER_1M_TOKENS.get(model_name, {"input": 0.0, "output": 0.0})
        cost = (
            pricing.get("input", 0.0) * prompt_tokens
            + pricing.get("output", 0.0) * completion_tokens
        )

        if isinstance(completion_response, CompletionWithMetadata):
             completion_response.cost = cost
             completion_response.response_time = round(end_time - start_time, 4)
        else:
            try:
                 completion_response.cost = cost
                 completion_response.response_time = round(end_time - start_time, 4)
            except AttributeError:
                 print("Warning: Could not attach cost/time metadata directly to response object.")

    except Exception as e:
        print(f"Callback error: {e}")
        if isinstance(completion_response, CompletionWithMetadata):
             completion_response.cost = 0.0001
             completion_response.response_time = 0.0

def get_sglang_endpoint(api: easyapi.Api, model: str) -> str:
    return random.choice(api.get_sglang_endpoints(model))

async def async_get_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    response_format: dict = None,
    response_model: BaseModel = None,
    easyapi: easyapi.Api = None,
    client: Optional[Union[openai.AsyncOpenAI, Any]] = None,
):
    # Determine client type and create one *only* if not provided
    client_needs_creation = client is None
    if client_needs_creation:
        client_or_module = llm_client(model_type="huggingface" if '/' in model else "openai")
    else:
        # Use the provided client directly
        client_or_module = client

    async for attempt in AsyncRetrying(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    ):
        with attempt:
            start_time = time.time()

            try: 
                if response_format and response_model:
                    raise Exception("response_format and response_model cannot both be provided. please provide only one.")

                # Check the type of the client_or_module *after* potential creation or reception
                is_openai_client = isinstance(client_or_module, openai.AsyncOpenAI) or isinstance(client_or_module, AsyncAzureOpenAI)
                is_litellm_module = hasattr(client_or_module, "__name__") and client_or_module.__name__ == "litellm"

                if response_model and response_format is None:
                    if is_openai_client:
                        # Patch the potentially provided or newly created client
                        instructor_client = instructor.patch(client_or_module)
                        response = await instructor_client.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=messages,
                            seed=seed,
                            response_model=response_model,
                        )
                    elif is_litellm_module:
                        instructor_client = instructor.from_litellm(litellm.acompletion)
                        response = await instructor_client.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=messages,
                            seed=seed,
                            response_model=response_model,
                        )
                    else:
                        raise Exception("unknown client type for response_model handling.")

                # --- Case 2: Not using instructor (no response_model) ---
                else:
                    if is_openai_client:
                        # Use the potentially provided or newly created client
                        response = await client_or_module.chat.completions.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=messages,
                            seed=seed,
                            response_format=response_format,
                        )

                    elif is_litellm_module:
                        # Litellm path remains mostly the same, using the module directly
                        if easyapi is not None: # Use sglang endpoint only if easyapi provided
                            sglang_endpoint = get_sglang_endpoint(easyapi, model)
                            # Create a *temporary* client for this specific call if using sglang via litellm path
                            # This might still need careful handling regarding loops if used extensively
                            temp_sglang_client = openai.AsyncOpenAI(api_key="dummy-key", base_url=f"{sglang_endpoint}")
                            response = await temp_sglang_client.chat.completions.create(
                                model=model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                messages=messages,
                                seed=seed,
                            )
                            await temp_sglang_client.close() # Close the temporary client
                        else:
                            litellm_kwargs = {
                                # Ensure model name is correct for litellm
                                "model": model if '/' in model else f"huggingface/{model}",
                                "max_tokens": max_tokens,
                                "temperature": temperature,
                                "messages": messages,
                                "seed": seed,
                            }
                            # Use the litellm module directly
                            response = await client_or_module.acompletion(**litellm_kwargs)
                    else:
                        raise Exception("unknown client type for standard completion handling.")

                    end_time = time.time()
                    track_cost_callback({"model": model}, response, start_time, end_time)

                if response_model is None:
                    response = CompletionWithMetadata(response)
                    # return build_fallback_response("No response received.", elapsed_time=end_time - start_time)
                    
                return response

            # except content filter error for openai and litellm
            except Exception as e:
                error_str = str(e).lower()
                status_code = getattr(e, "status_code", None)

                # Handle content filter errors from either client
                if status_code == 400 and "content_filter" in error_str:
                    return build_fallback_response("Content filter triggered by model.", elapsed_time=0.0)

                # Optional: handle rate limit or other known error types here if needed
                # elif status_code == 429:
                #     ...

                # Re-raise unknown errors
                raise


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
)
def get_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    response_format: dict = None,
    response_model: BaseModel = None,
    easyapi: easyapi.Api = None,
):
    return asyncio.run(async_get_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        response_format=response_format,
        response_model=response_model,
        easyapi=easyapi
    ))
