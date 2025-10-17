# Tools System

This directory contains the tool-use system for open-instruct RL training and inference.
This is an async tool-calling system that allows models to call tools asynchronously and receive results in a streaming manner *during RL training*.
We also provide a simple vLLM wrapper for easy downstream inference.

For now, we use a basic xml-style tool calling format. Tool calls are wrapped in `<tool>...</tool>` tags, and the tool output is wrapped in `<tool_output>...</tool_output>` tags (where the tag names are specific to each tool). We are planning to support more varied/flexible formats in the future.

We provide basic code and search tools, but hopefully it's easy to add your own!

## Built-in Tools

### Python Code Tool
**Tool name**: `code`

Executes Python code in an isolated server environment.

**Usage**: Wrap code in `<code>...</code>` tags:
```python
<code>
result = sum([1, 2, 3, 4, 5])
print(result)
</code>
```

**Features**:
- Pre-imported packages: pandas, numpy, sympy, math, networkx
- Default timeout: 3 seconds
- Isolated execution via FastAPI server

### Search Tools
**Tool names**: `search_s2`, `search_you`, `search_massive_ds`

Search external data sources for relevant information.

**Usage**: Wrap queries in `<query>...</query>` tags:
```
<query>machine learning transformers</query>
```

**Available backends**:
- `search_s2`: Semantic Scholar academic papers (requires `S2_API_KEY` env var)
- `search_you`: You.com web search (requires `YOUCOM_API_KEY` env var)
- `search_massive_ds`: Alternative search backend (requires setting up a [massive-serve](https://github.com/RulinShao/massive-serve) server and setting the `MASSIVE_DS_URL` env var or passing the `search_api_endpoint` argument to the script)

## Adding Your Own Tool

### Step 1: Implement the Tool Class

Create a new Python file (e.g., `my_tool/tool.py`), and create a Tool class that inherits from `open_instruct.tools.utils.tool_classes.Tool`.
The Tool class should have a `__call__` method that takes a prompt and returns a `ToolOutput` object. Note that the prompt will have the full conversation history prepended to it, so you should extract the tool input by splitting on `self.start_str` and `self.end_str`.

```python
from open_instruct.tools.utils.tool_classes import Tool, ToolOutput
import time

class MyCustomTool(Tool):
    def __init__(self, start_str: str = "<my_tool>\n", end_str: str = "\n</my_tool>", **kwargs):
        super().__init__(start_str, end_str)
        # Initialize any resources here
        self.timeout = kwargs.get("timeout", 5)

    def __call__(self, prompt: str) -> ToolOutput:
        """Execute the tool with the given prompt."""
        start_time = time.time()

        # Check if tool was called
        if self.start_str not in prompt:
            return ToolOutput(
                output="",
                called=False,
                error="",
                timeout=False,
                runtime=0,
                start_str=self.start_str,
                end_str=self.end_str
            )

        try:
            # Extract input between start_str and end_str
            tool_input = prompt.split(self.start_str)[-1]

            # Execute your tool's logic
            result = self._process(tool_input)

            # Return result
            return ToolOutput(
                output=result,
                called=True,
                error="",
                timeout=False,
                runtime=time.time() - start_time,
                start_str=self.start_str,
                end_str=self.end_str
            )

        except Exception as e:
            return ToolOutput(
                output="",
                called=True,
                error=str(e),
                timeout=False,
                runtime=time.time() - start_time,
                start_str=self.start_str,
                end_str=self.end_str
            )

    def _process(self, tool_input: str) -> str:
        """Your tool's implementation goes here."""
        return f"Processed: {tool_input}"
```

### Step 2: Register the Tool

Add your tool to `tool_actor.py`:

```python
TOOL_CLASS_REGISTRY: Dict[str, str] = {
    # ... existing tools ...
    "my_custom": "open_instruct.tools.my_tool.tool:MyCustomTool",
}
```

### Step 3: Use the Tool in training

You can then use the tool in training scripts by passing the tool name (as specified in the `TOOL_CLASS_REGISTRY`) to the `--tools` flag in `grpo_fast.py`.
Note that you can pass multiple tools to the `--tools` flag, allowing the model to use multiple tools!

**Important**: The model will not implicitly know which tools are available to it. You should add prompts to your data (via system prompt or otherwise) that tell the model which tools are available to it and how to use them.

### Step 3 (Optional): Use the Tool in downstream inference

We provide a simple vLLM wrapper for easy downstream inference in `open_instruct.tools.utils.tool_vllm.ToolUseLLM`.
This wrapper allows you to use the tools in a vLLM-compatible way, and it will automatically handle the tool calls and results for you.

Check out the bottom of `open_instruct/tool_utils/tool_vllm.py` for an example on how to use!

