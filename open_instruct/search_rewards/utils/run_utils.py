import json
import logging
import os
from typing import Any, Dict, Optional, List

import jsonlines
import litellm
from openai import AzureOpenAI

# Configure LiteLLM to drop unsupported parameters instead of raising errors
litellm.drop_params = True

LOGGER = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        return None

    json_str = response[json_start:json_end]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # Clean the JSON string of potential invisible characters and extra whitespace
            cleaned_json = json_str.strip().encode('utf-8').decode('utf-8-sig')
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            try:
                # Fix doubled braces (e.g., '{{' -> '{', '}}' -> '}')
                fixed_braces = json_str.replace('{{', '{').replace('}}', '}')
                return json.loads(fixed_braces)
            except json.JSONDecodeError:
                try:
                    # Last resort: try adding closing brackets (for incomplete arrays/objects)
                    return json.loads(json_str + "]}")
                except json.JSONDecodeError:
                    LOGGER.warning(f"Could not decode JSON from response: {repr(json_str)}")
                    return None


def run_chatopenai(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    json_mode: bool = False,
    **chat_kwargs,
) -> str:
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    if json_mode:
        chat_kwargs["response_format"] = {"type": "json_object"}
    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )
    resp = litellm.completion(
        model=model_name,
        messages=msgs,
        **chat_kwargs,
    )

    return resp.choices[0].message.content


def load_jsonlines(file):
    with jsonlines.open(file, "r") as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode="w") as writer:
        writer.write_all(data)


def run_azure_openai(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
    **chat_kwargs,
) -> str:
    """
    Run Azure OpenAI model with the given prompts.

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        system_prompt: Optional system prompt
        user_prompt: User prompt
        endpoint: Azure OpenAI endpoint URL
        deployment: Azure OpenAI deployment name (defaults to model_name if not specified)
        api_version: Azure OpenAI API version
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """
    # Get Azure credentials from environment
    subscription_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not subscription_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

    # Use provided endpoint or get from environment
    if endpoint is None:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint must be provided or set in AZURE_OPENAI_ENDPOINT environment variable"
            )

    # Use provided deployment or default to model_name
    if deployment is None:
        deployment = model_name

    # Set default parameters
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    chat_kwargs["max_completion_tokens"] = chat_kwargs.get("max_completion_tokens", 800)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)

    # Create Azure OpenAI client
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    # Prepare messages
    msgs = (
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )

    # Create chat completion
    response = client.chat.completions.create(
        messages=msgs,
        model=deployment,
        **chat_kwargs,
    )

    return response.choices[0].message.content


def run_litellm(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    **chat_kwargs,
) -> str:
    """
    Run litellm for the given model.
    matches api for the run_azure_openai function.
    We assume that the right env vars are set for the model.
    e.g. for vLLM, need HOSTED_VLLM_API_BASE
    e.g., for azure, need AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        system_prompt: Optional system prompt (defaults to None)
        user_prompt: User prompt
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """

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
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt is not None
        else [{"role": "user", "content": user_prompt}]
    )

    # Create chat completion
    response = litellm.completion(
        messages=msgs,
        model=model_name,
        **chat_kwargs,
    )

    return response.choices[0].message.content


async def run_litellm_async(
    model_name: str,
    user_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    **chat_kwargs,
) -> str:
    """
    Async version of run_litellm for the given model.
    matches api for the run_azure_openai function.
    We assume that the right env vars are set for the model.
    e.g. for vLLM, need HOSTED_VLLM_API_BASE
    e.g., for azure, need AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION

    Args:
        model_name: The model name (used as deployment if deployment not specified)
        system_prompt: Optional system prompt (defaults to None)
        user_prompt: User prompt
        **chat_kwargs: Additional arguments to pass to the chat completion

    Returns:
        The response content from the model
    """

    # Set default parameters
    chat_kwargs["temperature"] = chat_kwargs.get("temperature", 0)
    chat_kwargs["max_tokens"] = chat_kwargs.get("max_tokens", 800)
    chat_kwargs["top_p"] = chat_kwargs.get("top_p", 1.0)
    chat_kwargs["frequency_penalty"] = chat_kwargs.get("frequency_penalty", 0.0)
    chat_kwargs["presence_penalty"] = chat_kwargs.get("presence_penalty", 0.0)
    chat_kwargs["num_retries"] = chat_kwargs.get("num_retries", 5)
    chat_kwargs["fallbacks"] = chat_kwargs.get("fallbacks", ["gpt-4.1-mini"])

    # Prepare messages
    if messages is not None:
        msgs = messages
    else:
        msgs = (
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if system_prompt is not None
            else [{"role": "user", "content": user_prompt}]
        )

    # Create chat completion
    response = await litellm.acompletion(
        messages=msgs,
        model=model_name,
        **chat_kwargs,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Simple test case for run_chatopenai function
    def test_run_chatopenai():
        """Test the run_chatopenai function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_chatopenai function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # Note: This will require a valid model name and API credentials
            # Uncomment the line below to actually test (requires OPENAI_API_KEY)
            response = run_chatopenai("gpt-3.5-turbo", system_prompt, user_prompt)
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    def test_run_azure():
        """Test the run_azure_openai function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_azure_openai function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # Note: This will require valid Azure OpenAI credentials
            # Use the correct deployment name that matches test_azure_api.py
            response = run_azure_openai(
                "gpt-4.5-preview", system_prompt, user_prompt, deployment="gpt-4.5-preview-standard"
            )
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    def test_run_litellm():
        """Test the run_litellm function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_litellm function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # assert env vars are set
            # for now, azure
            assert os.environ.get("HOSTED_VLLM_API_BASE") is not None

            response = run_litellm("hosted_vllm/Qwen/QwQ-32B-Preview", system_prompt, user_prompt)
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    async def test_run_litellm_async():
        """Test the run_litellm_async function with a simple prompt"""
        try:
            # Test with a simple prompt
            system_prompt = "You are a helpful assistant."
            user_prompt = "What is 2 + 2?"

            print("Testing run_litellm_async function...")
            print(f"System prompt: {system_prompt}")
            print(f"User prompt: {user_prompt}")

            # assert env vars are set
            # for now, azure
            assert os.environ.get("HOSTED_VLLM_API_BASE") is not None

            response = await run_litellm_async("hosted_vllm/Qwen/QwQ-32B-Preview", system_prompt, user_prompt)
            print(f"Response: {response}")

            print("Test completed successfully!")

        except Exception as e:
            print(f"Test failed with error: {e}")

    # test_run_chatopenai()
    # test_run_azure()
    # test_run_litellm()
    
    # Run async test
    import asyncio
    asyncio.run(test_run_litellm_async())
