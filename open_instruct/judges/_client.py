import logging
import asyncio
import instructor
import openai
import litellm
from openai import AzureOpenAI

import time
import os

from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt, AsyncRetrying


openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)

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

    def __getattr__(self, name):
        # Delegate attribute access to the underlying response
        return getattr(self.response, name, None)

import asyncio

async def call_azure_sync(client, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: client.chat.completions.create(**kwargs))


def llm_client():
    
    try:
        # # Attempt to import litellm
        import litellm
        # fall back to azure openai if litellm is not available
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview", #"2024-07-18", # 2024-11-20 for gpt-4o
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    except ImportError:
        # fallback to async openai if litellm is not available
        print("litellm not found, falling back to openai.AsyncOpenAI")
        client = openai.AsyncOpenAI()
        return client

    else:
        # Return client/litellm module object if import succeeds
        return litellm # client #litellm

### V2 of async completion with retry
def track_cost_callback(kwargs, completion_response, start_time, end_time):

    try:
        model_name = kwargs.get("model", "")
        prompt_tokens = getattr(completion_response.usage, "prompt_tokens", 0)
        completion_tokens = getattr(completion_response.usage, "completion_tokens", 0)

        # pricing
        pricing = PRICING_PER_1M_TOKENS.get(model_name, {"input": 0.0, "output": 0.0})
        cost = (
            pricing.get("input", 0.0) * prompt_tokens
            + pricing.get("output", 0.0) * completion_tokens
        )

        completion_response.cost = cost
        completion_response.response_time = round(end_time - start_time, 4)

    except Exception as e:
        print(f"Callback error: {e}")
        completion_response.cost = 0.0001
        completion_response.response_time = 0.0

    # try:
    #     model_name = kwargs.get("model", "gpt-4o-mini")
    #     usage = getattr(completion_response, "usage", {})

    #     prompt_tokens = usage.get("prompt_tokens", 0)
    #     completion_tokens = usage.get("completion_tokens", 0)

    #     input_cost = (prompt_tokens) * PRICING_PER_1M_TOKENS.get(model_name, {}).get("input", 0)
    #     output_cost = (completion_tokens) * PRICING_PER_1M_TOKENS.get(model_name, {}).get("output", 0)
    #     total_cost = input_cost + output_cost

    #     response_time = end_time - start_time

    #     # Attach attributes directly
    #     completion_response.cost = total_cost
    #     completion_response.response_time = response_time

    # except Exception as e:
    #     print(f"Callback error: {e}")



async def async_get_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    response_format: dict = None,
    response_model: BaseModel = None,
):
    client_or_module = llm_client() # Returns openai.AsyncOpenAI or litellm module
    
    async for attempt in AsyncRetrying(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    ):
        with attempt:
            start_time = time.time()

            if response_format and response_model:
                raise Exception("response_format and response_model cannot both be provided. please provide only one.")

            # --- Case 1: Using instructor with a response_model ---
            if response_model and response_format is None:
                if isinstance(client_or_module, openai.AsyncOpenAI):
                    # Use instructor with the already async OpenAI client
                    instructor_client = instructor.patch(client_or_module) # Use instructor.patch for async client
                    response = await instructor_client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                        seed=seed,
                        response_model=response_model,
                    )
                elif hasattr(client_or_module, "__name__") and client_or_module.__name__ == "litellm":
                    # Use instructor with litellm's async completion
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
                # breakpoint()
                if isinstance(client_or_module, openai.AsyncOpenAI):
                    # Use AsyncOpenAI client directly
                    response = await client_or_module.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                        seed=seed,
                        response_format=response_format,
                    )

                # elif isinstance(client_or_module, openai.lib.azure.AzureOpenAI):
                #     # Use openai.lib.azure.AzureOpenAI client directly
                #     response = await client_or_module.chat.completions.create(
                #         model=f"{model}-standard",
                #         max_tokens=max_tokens,
                #         temperature=temperature,
                #         messages=messages,
                #         seed=seed,
                #         response_format=response_format,
                #     )
                elif isinstance(client_or_module, openai.lib.azure.AzureOpenAI):
                    response = await call_azure_sync(
                        client_or_module,
                        model=f"{model}-standard",
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                        seed=seed,
                        response_format=response_format,
                    )
                    # breakpoint()

                elif hasattr(client_or_module, "__name__") and client_or_module.__name__ == "litellm":
                    # litellm.acompletion is already async
                    response = await client_or_module.acompletion(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=messages,
                        seed=seed,
                        response_format=response_format,
                    )
                else:
                    raise Exception("unknown client type for standard completion handling.")

            end_time = time.time()
            track_cost_callback({"model": model}, response, start_time, end_time)

            return response

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
):
    client = llm_client()

    if response_format and response_model:
        raise Exception("response_format and response_model cannot both be provided. please provide only one.")
    
    def track_cost_callback(kwargs, completion_response, start_time, end_time):
        try:
            model_name = kwargs.get("model", "gpt-4o-mini")
            usage = getattr(completion_response, "usage", {})

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            input_cost = (prompt_tokens) * PRICING_PER_1M_TOKENS.get(model_name, {}).get("input", 0)
            output_cost = (completion_tokens) * PRICING_PER_1M_TOKENS.get(model_name, {}).get("output", 0)
            total_cost = input_cost + output_cost

            response_time = end_time - start_time

            # Attach attributes directly
            completion_response.cost = total_cost
            completion_response.response_time = response_time

        except Exception as e:
            print(f"Callback error: {e}")

    # Register the callback
    litellm.success_callback = [track_cost_callback]
    
    # Capture start time
    start_time = time.time()
    
    if response_model and response_format is None:
        if client.__class__.__name__ == "OpenAI":
            client = instructor.from_openai(client)
        elif hasattr(client, "__name__") and client.__name__ == "litellm":
            client = instructor.from_litellm(client.completion)
        else:
            raise Exception("unknown client. please create an issue on GitHub if you see this message.")
        
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            seed=seed,
            response_model=response_model,
        )
    else:
        if client.__class__.__name__ == "OpenAI":
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                seed=seed,
                response_format=response_format,
            )
        elif hasattr(client, "__name__") and client.__name__ == "litellm":
            response = client.completion(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                seed=seed,
                response_format=response_format,
            )
    end_time = time.time()

    track_cost_callback({"model": model}, response, start_time, end_time)

    return response


# @retry(
#     wait=wait_random_exponential(min=1, max=60),
#     stop=stop_after_attempt(5),
# )
# def get_completion(
#     messages: list[dict[str, str]],
#     model: str,
#     temperature: float,
#     max_tokens: int,
#     seed: int,
#     response_format: dict = None,
#     response_model: BaseModel = None,
# ):
#     client = llm_client()

#     if response_format and response_model:
#         raise Exception("response_format and response_model cannot both be provided. please provide only one.")
    
#     # track_cost_callback - Automatically stores results
#     def track_cost_callback(kwargs, wrapped_response, start_time, end_time):
#         try:
#             response_cost = kwargs.get("response_cost", 0) # completion_response.get("usage", {}).get("response_cost", 0)
#             response_time = (end_time - start_time).total_seconds()  # Convert timedelta to float

#             # Attach cost & response time as attributes to response object
#             # setattr(completion_response, "cost", response_cost)
#             # setattr(completion_response, "response_time", response_time)
#             wrapped_response.cost = response_cost
#             wrapped_response.response_time = response_time


#         except Exception as e:
#             print(f"Error in callback: {e}")

#     # Set callback
#     litellm.success_callback = [track_cost_callback]
    
#     # Capture start time
#     start_time = time.time()
    
#     if response_model and response_format is None:
#         if client.__class__.__name__ == "OpenAI":
#             client = instructor.from_openai(client)
#         elif hasattr(client, "__name__") and client.__name__ == "litellm":
#             client = instructor.from_litellm(client.completion)
#         else:
#             raise Exception("unknown client. please create an issue on GitHub if you see this message.")
        
#         response = client.chat.completions.create(
#             model=model,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             messages=messages,
#             seed=seed,
#             response_model=response_model,
#         )
#     else:
#         if client.__class__.__name__ == "OpenAI":
#             response = client.chat.completions.create(
#                 model=model,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 messages=messages,
#                 seed=seed,
#                 response_format=response_format,
#             )
#         elif hasattr(client, "__name__") and client.__name__ == "litellm":
#             response = client.completion(
#                 model=model,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 messages=messages,
#                 seed=seed,
#                 response_format=response_format,
#             )
#     end_time = time.time()

#     # Wrap and manually trigger callback
#     wrap_response = CompletionWithMetadata(response)
#     track_cost_callback({}, wrap_response, start_time, end_time)
#     # breakpoint()
#     return wrap_response
