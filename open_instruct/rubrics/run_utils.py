"""Utility functions for running LLM judge calls via LiteLLM."""

import asyncio
import json
import os
import weakref
from typing import Any

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# Per-event-loop concurrency control for LiteLLM async calls to avoid event loop binding issues
_LITELLM_SEMAPHORES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Lazy-loaded litellm module reference
_litellm = None


def _get_litellm():
    """Lazily import and configure litellm."""
    global _litellm
    if _litellm is None:
        import litellm  # noqa: PLC0415

        # Configure LiteLLM to drop unsupported parameters instead of raising errors
        litellm.drop_params = True
        _litellm = litellm
    return _litellm


def _get_litellm_semaphore() -> asyncio.Semaphore:
    """Return a per-event-loop semaphore limiting concurrent LiteLLM async requests.

    Limit can be configured with env var `LITELLM_MAX_CONCURRENT_CALLS` (default 256).
    """
    loop = asyncio.get_running_loop()
    sem = _LITELLM_SEMAPHORES.get(loop)
    if sem is None:
        max_concurrent = int(os.environ.get("LITELLM_MAX_CONCURRENT_CALLS", "256"))
        sem = asyncio.Semaphore(max_concurrent)
        _LITELLM_SEMAPHORES[loop] = sem
    return sem


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract JSON object from a response string, handling various edge cases."""
    json_end = response.rfind("}") + 1
    if json_end == 0:
        return None

    # Try to find valid JSON by testing different starting positions
    json_start = response.find("{")
    while json_start != -1 and json_start < json_end:
        json_str = response[json_start:json_end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Clean the JSON string of potential invisible characters and extra whitespace
                cleaned_json = json_str.strip().encode("utf-8").decode("utf-8-sig")
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                try:
                    # Fix doubled braces (e.g., '{{' -> '{', '}}' -> '}')
                    fixed_braces = json_str.replace("{{", "{").replace("}}", "}")
                    return json.loads(fixed_braces)
                except json.JSONDecodeError:
                    # Try next { position
                    json_start = response.find("{", json_start + 1)
                    continue
        break

    logger.warning(f"Could not decode JSON from response: {repr(response)}")
    return None


def run_litellm(model_name: str, user_prompt: str, system_prompt: str | None = None, **chat_kwargs) -> str:
    """
    Run litellm for the given model.
    We assume that the right env vars are set for the model.
    e.g. for vLLM, need HOSTED_VLLM_API_BASE
    e.g., for azure, need AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        user_prompt: User prompt
        system_prompt: Optional system prompt (defaults to None)
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """
    litellm = _get_litellm()

    # Set default parameters
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    chat_kwargs["max_tokens"] = chat_kwargs.get("max_tokens", 800)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)
    chat_kwargs["num_retries"] = chat_kwargs.get("num_retries", 5)
    chat_kwargs["fallbacks"] = chat_kwargs.get("fallbacks", ["gpt-4.1-mini"])

    # Prepare messages
    msgs = (
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )

    # Create chat completion
    try:
        response = litellm.completion(messages=msgs, model=model_name, **chat_kwargs)
    except Exception:
        # if we get an error, return an empty string
        return ""

    return response.choices[0].message.content


async def run_litellm_async(
    model_name: str,
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    **chat_kwargs,
) -> str:
    """
    Async version of run_litellm for the given model.
    We assume that the right env vars are set for the model.

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        user_prompt: User prompt
        system_prompt: Optional system prompt (defaults to None)
        messages: Optional list of messages to use instead of system_prompt/user_prompt
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """
    litellm = _get_litellm()

    # Set default parameters
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    chat_kwargs["max_tokens"] = chat_kwargs.get("max_tokens", 16384)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)
    chat_kwargs["num_retries"] = chat_kwargs.get("num_retries", 5)
    chat_kwargs["fallbacks"] = chat_kwargs.get("fallbacks", [])

    # Prepare messages
    if messages is not None:
        msgs = messages
    else:
        msgs = (
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            if system_prompt is not None
            else [{"role": "user", "content": user_prompt}]
        )

    # Apply default timeout if not provided
    chat_kwargs["timeout"] = chat_kwargs.get("timeout", float(os.environ.get("LITELLM_DEFAULT_TIMEOUT", "600")))

    # Guard concurrent calls with a global semaphore
    try:
        semaphore = _get_litellm_semaphore()
        async with semaphore:
            # Create chat completion
            response = await litellm.acompletion(messages=msgs, model=model_name, **chat_kwargs)
    except Exception as e:
        # if we get an error, return an empty string
        logger.warning(f"Error in run_litellm_async: {e}")
        return ""

    return response.choices[0].message.content
