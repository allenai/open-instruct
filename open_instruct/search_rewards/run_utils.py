import json
import jsonlines
import logging
from typing import Any, Dict, Optional

import litellm
LOGGER = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    json_start = response.find("{")
    json_end = response.rfind("}") + 1
    if json_start == -1 or json_end == -1:
        return None

    try:
        return json.loads(response[json_start:json_end])
    except json.JSONDecodeError:
        try:
            return json.loads(response[json_start:json_end]+"]}")
        except json.JSONDecodeError:
            LOGGER.warning(
                f"Could not decode JSON from response: {response[json_start:json_end]}"
            )
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
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)



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
    
    test_run_chatopenai()
    