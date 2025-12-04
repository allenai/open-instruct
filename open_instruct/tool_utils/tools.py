import re
import time
import traceback
from dataclasses import dataclass

import requests


def _sanitize_name(token: str) -> str:
    stripped = token.strip().strip("<>")
    return stripped.lstrip("/") or "tool"


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
    def __init__(self, start_str: str, end_str: str, name: str | None = None):
        self.start_str = start_str
        self.end_str = end_str
        self.name = name or _sanitize_name(start_str) or self.__class__.__name__.lower()

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

    def __call__(self, input_text: str) -> ToolOutput:
        re_str = r"(?s)(?<!`)<tool>\s*(.*?)\s*</tool>"
        re_str = re_str.replace("<tool>", "<code>").replace("</tool>", "</code>")

        code_blocks = re.findall(re_str, input_text, re.DOTALL)
        code = code_blocks[-1] if code_blocks else input_text.strip()

        if not code:
            return ToolOutput(output="", called=False, error="", timeout=False, runtime=0)

        all_outputs = []
        timeout = False
        error = ""

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
