"""
python open_instruct/tool_utils/tool_vllm.py
"""

import re
import time
import traceback
from dataclasses import dataclass
from typing import Union

import requests
from vllm import LLM


@dataclass
class ToolOutput:
    output: str
    called: bool
    error: str
    timeout: bool
    runtime: float
    start_str: str = "<output>\n"
    end_str: str = "\n</output>"


class Tool:
    def __init__(self, start_str: str, end_str: str):
        self.start_str = start_str
        self.end_str = end_str

    def __call__(self, prompt: str) -> ToolOutput:
        raise NotImplementedError("Subclasses must implement this method")


class MaxCallsExceededTool(Tool):
    def __call__(self, prompt: str) -> ToolOutput:
        return ToolOutput(output="Max tool calls exceeded.", called=False, error="", timeout=False, runtime=0)


class PythonCodeTool(Tool):
    """@vwxyzjn: I recommend using something like a FastAPI for this kind of stuff; 1) you
    won't accidentally block the main vLLM process and 2) way easier to parallelize via load balancing."""

    def __init__(self, api_endpoint: str, *args, **kwargs):
        self.api_endpoint = api_endpoint
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        r"""
        NOTE: We avoid using `r'<tool>\s*(.*?)\s*</tool>'` because it will fail in this case  # noqa: W605
        Let's implement this in Python using the `<code>` tag to execute the code and get the result.
        </think>

        <code>
        def find_sum_of_a():
            total_sum = 0
            for n in range(100):  # Arbitrary large range for n
                for m in range(100):  # Arbitrary large range for m
                    a = 2**n * 3**m
                    if 6*n > a or 6*m > a:
                        continue
                    total_sum += a
            return total_sum

        result = find_sum_of_a()
        print(result)
        </code>

        Instead, Use negative look-behind approach to find the code block.
        """
        re_str = r"(?s)(?<!`)<tool>\s*(.*?)\s*</tool>"
        re_str = re_str.replace("<tool>", "<code>").replace("</tool>", "</code>")

        code_blocks = re.findall(re_str, prompt, re.DOTALL)
        all_outputs = []
        timeout = False
        error = ""
        if len(code_blocks) == 0:
            return ToolOutput(output="", called=False, error="", timeout=False, runtime=0)

        # Only execute the last code block
        code = code_blocks[-1]

        # Define timeout in seconds
        timeout_seconds = 3
        start_time = time.time()
        try:
            # Call the FastAPI endpoint to execute the code with client-side timeout
            response = requests.post(
                self.api_endpoint,
                json={"code": code, "timeout": timeout_seconds},  # Server-side timeout (keeping this)
                timeout=timeout_seconds,  # Client-side timeout
            )

            # Parse the response
            result = response.json()

            # Process the API response
            output = result["output"]
            error = result.get("error") or ""

            all_outputs.append(output)
            if len(error) > 0:
                all_outputs.append("\n" + error)

        except requests.Timeout:
            # Handle client-side timeout specifically
            all_outputs.append(f"Timeout after {timeout_seconds} seconds")
            timeout = True

        except Exception as e:
            # Capture any other exceptions that occur during the API call
            error_message = f"Error calling API: {str(e)}\n"
            error_traceback = traceback.format_exc()
            all_outputs.append(error_message + error_traceback)

        # Return all captured outputs as a single string
        return ToolOutput(
            output="\n".join(all_outputs), called=True, error=error, timeout=timeout, runtime=time.time() - start_time
        )


# Compatibility wrapper for ToolUseLLM - now uses standard vLLM with manual tool processing
class ToolUseLLM:
    """
    Compatibility wrapper that provides the ToolUseLLM interface.
    This is a simplified version that doesn't have the full async tool execution
    of the integrated LLMRayActor version, but provides basic tool functionality.

    For production use with Ray, use LLMRayActor from vllm_utils3.py instead.
    """

    def __init__(self, tools: dict[str, Tool] = None, max_tool_calls: Union[int, dict[str, int]] = 4, *args, **kwargs):
        self.tools = tools or {}
        # Convert max_tool_calls to a dict if it's an int
        if isinstance(max_tool_calls, int):
            self.max_tool_calls = {end_str: max_tool_calls for end_str in tools.keys()} if tools else {}
        else:
            self.max_tool_calls = max_tool_calls

        # Create underlying vLLM instance
        self.llm = LLM(*args, **kwargs)
        self.llm_engine = self.llm.llm_engine

    def generate(self, prompts=None, sampling_params=None, prompt_token_ids=None, use_tqdm=True):
        """Simple generate method that processes tools synchronously after generation."""
        # Use the underlying vLLM generate
        outputs = self.llm.generate(
            prompts=prompts, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids, use_tqdm=use_tqdm
        )

        # Post-process outputs to add tool metadata (simplified version)
        for output in outputs:
            for completion in output.outputs:
                # Initialize tool metadata
                setattr(completion, "mask", [1] * len(completion.token_ids))
                setattr(completion, "num_calls", 0)
                setattr(completion, "timeout", False)
                setattr(completion, "tool_error", "")
                setattr(completion, "tool_output", "")
                setattr(completion, "tool_runtime", 0.0)
                setattr(completion, "tool_called", False)

                # Check if output ends with a tool stop string
                for stop_str in sampling_params.stop:
                    if stop_str in self.tools and completion.text.endswith(stop_str):
                        # Execute tool synchronously (simplified)
                        tool = self.tools[stop_str]
                        tool_result = tool(completion.text)

                        # Update metadata
                        completion.tool_called = True
                        completion.tool_output = tool_result.output
                        completion.tool_error = tool_result.error
                        completion.timeout = tool_result.timeout
                        completion.tool_runtime = tool_result.runtime
                        completion.num_calls = 1
                        break

        return outputs

    def get_tokenizer(self):
        """Pass through to underlying LLM."""
        return self.llm.get_tokenizer()

    def get_model_config(self):
        """Pass through to underlying LLM."""
        return self.llm.get_model_config()


# Note: For production use with Ray actors, use LLMRayActor from vllm_utils3.py
# which has full async tool execution support integrated into the generation loop
