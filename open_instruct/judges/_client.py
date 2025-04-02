import logging

import instructor
import openai
import litellm

import time

from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt

openai._utils._logs.logger.setLevel(logging.WARNING)
openai._utils._logs.httpx_logger.setLevel(logging.WARNING)



def llm_client():
    try:
        import litellm
    except ImportError:
        # fallback to openai
        client = openai.OpenAI()
        return client
    else:
        return litellm


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
    
    # track_cost_callback - Automatically attaches results to response
    def track_cost_callback(kwargs, completion_response, start_time, end_time):
        try:
            response_cost = getattr(completion_response, "usage", {}).get("total_cost", 0)
            response_time = end_time - start_time  # Convert to seconds

            # Attach cost & response time as attributes to response object
            setattr(completion_response, "cost", response_cost)
            setattr(completion_response, "response_time", response_time)

        except Exception as e:
            print(f"Error in callback: {e}")

    # Set callback
    litellm.success_callback = [track_cost_callback]
    
    breakpoint()
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

    return response
