# Tool Training with GRPO

Tool training allows models to learn how to effectively use external tools (like code execution, web search, and browsing) during reinforcement learning. This guide covers how to set up and run tool-augmented GRPO training.

## Overview

Tool training in open-instruct works by:

1. **Generating responses**: The model generates text that may include tool calls
2. **Parsing tool calls**: A tool parser extracts tool calls from the model's output
3. **Executing tools**: The tools are executed and their outputs are returned
4. **Continuing generation**: The model continues generating based on tool outputs
5. **Computing rewards**: Verifiable rewards are computed on the final answer

This enables training models to use tools effectively for tasks like mathematical reasoning (using Python), information retrieval (using search), and reading web content (using browse).

Note that **right now, we only allow interleaved tool calling** - we do not allow e.g. sub agent calls, or agents that change their history midway through generation. If you require this, let someone know!

In the future, I (hamish) hope to add more sophisticated environment setups. Let me know if you have particular RL envs you care about!

## Quick Start

Here's a minimal example to run tool training with OLMo-3:

```bash
# Set required API keys
export SERPER_API_KEY="your-serper-key"
export JINA_API_KEY="your-jina-key"

# Run the test script
./scripts/train/build_image_and_launch.sh scripts/train/debug/tools/olmo_3_parser_multigpu.sh
```

## Available Tools

### Built-in Tools

| Tool Name | Call Name | Description | Required Environment Variables |
|-----------|-----------|-------------|-------------------------------|
| `python` | `code` | Executes Python code via a FastAPI server | None (uses API endpoint) |
| `serper_search` | `search` | Google search via Serper API | `SERPER_API_KEY` |
| `jina_browse` | `browse` | Fetches webpage content using Jina Reader | `JINA_API_KEY` |
| `s2_search` | `s2_search` | Semantic Scholar paper search | `S2_API_KEY` |
| `crawl4ai_browse` | `crawl4ai_browse` | Fetches webpage content using Crawl4A proxy server (ask Luca) | `CRAWL4AI_API_URL`, `CRAWL4AI_API_KEY`, `CRAWL4AI_BLOCKLIST_PATH` |

Feel free to add more built-in tools if you need! See below for how to add.

### MCP Tools

Open-instruct supports connecting to any MCP (Model Context Protocol) server via the `generic_mcp` tool:

```bash
--tools generic_mcp \
--tool_configs '{"server_url": "http://localhost:8000", "transport": "http"}'
```

MCP tools can automatically discover available tools from the server at startup. We make the following assumptions:
1. **Tools will not change over training**
2. **Tools only involve text out** (that is, we use the *unstructured* output from the mcp server by default).

## Configuration Options

### Basic Tool Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tools` | list[str] | `[]` | List of tool names to enable |
| `--tool_call_names` | list[str] | (same as tools) | Override names used in tool calls |
| `--tool_configs` | list[str] | `['{}']` | JSON configs for each tool |
| `--tool_parser_type` | str | `"legacy"` | Parser type for extracting tool calls |
| `--max_tool_calls` | int | `5` | Maximum tool calls per generation |
| `--pass_tools_to_chat_template` | bool | `True` | Pass tool definitions to chat template |

`tool_call_names` allows you to change what tools you use without changing how the model sees them. For example, you might want to train with a cheap search tool, and then swap training later to a more expensive one. For this, you could swap the `tools` args, and leave `tool_call_names` unchanged.

`pass_tools_to_chat_template` is mainly useful for cases where you want to pass in a custom system prompt or have the tool prompts hardcoded in your data.

### Tool Parser Types

Parsers handle parsing out tool calls, separate from tools. Note that if a tool parser exists in vLLM, it is pretty easy to wrap it into an open-instruct parser! We have already implemented the olmo3, hermes, and llama3 parsers for convenience.

| Parser Type | Format | Best For |
|-------------|--------|----------|
| `legacy` | `<tool_name>content</tool_name>` | Custom prompts, Search-R1 style|
| `vllm_olmo3` | Pythonic function calls | OLMo-3 models |
| `vllm_hermes` | Hermes-style JSON | Qwen2.5/3, Hermes models |
| `vllm_llama3_json` | Llama 3 JSON format | Llama 3.x models |
| `dr_tulu` | DR Tulu Parser | For DR Tulu-style training |


### Reward Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--only_reward_good_outputs` | bool | `False` | Only reward non-erroring tool outputs |

## Example Configurations

### OLMo-3 with vLLM Parser (Recommended)

```bash
python open_instruct/grpo_fast.py \
    --model_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
    --tools python serper_search jina_browse \
    --tool_call_names code search browse \
    --tool_configs '{"api_endpoint": "https://open-instruct-tool-server-10554368204.us-central1.run.app/execute", "timeout": 3}' '{}' '{}' \
    --tool_parser_type vllm_olmo3 \
    --max_tool_calls 5 \
    # ... other GRPO arguments
```

### Legacy Parser Style

For models without native tool calling support, use the legacy parser:

```bash
python open_instruct/grpo_fast.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --chat_template_name r1_simple_chat_postpend_think_tool_vllm \
    --tools python serper_search jina_browse \
    --tool_call_names code search browse \
    --tool_configs '{"api_endpoint": "...", "timeout": 3}' '{}' '{}' \
    --tool_parser_type legacy \
    --max_tool_calls 5 \
    # ... other GRPO arguments
```

### Per-Sample Tool Configuration

Datasets can specify which tools are allowed per sample using a `tools` column:

```json
{
    "messages": [...],
    "ground_truth": "42",
    "tools": ["code", "search"]  // Only code and search allowed for this sample
}
```

If a sample has an empty `tools` list, no tools will be used for that sample. If it is set to `None`, all tools become available.

## Tool Implementation Details

### Python Code Execution Tool

The Python tool uses a FastAPI server that:

- Pre-imports common packages (pandas, numpy, sympy, math, networkx)
- Executes code in isolated processes with timeouts
- Returns stdout and any errors

The public endpoint is:
```
https://open-instruct-tool-server-10554368204.us-central1.run.app/execute
```

You can also run your own server:

```bash
cd open_instruct/tools/servers/python_server
docker build -t tool-server .
docker run -p 1212:8080 tool-server
```

### Search Tools

**Serper Search** (`serper_search`):

- Uses Google search via the Serper API
- Returns formatted snippets with titles and source URLs
- Get an API key at https://serper.dev/

**Semantic Scholar** (`s2_search`):

- Searches academic papers
- Returns relevant paper snippets
- Get an API key at https://www.semanticscholar.org/product/api

### Browse Tools

**Jina Browse** (`jina_browse`):

- Converts webpages to clean markdown
- Free tier available
- Get an API key at https://jina.ai/reader/

**Crawl4AI Browse** (`crawl4ai_browse`):

- More advanced webpage crawling with AI2 configuration
- Supports caching, link filtering, and blocklists
- Requires Docker deployment

## Training Metrics

When tools are enabled, additional metrics are logged:

| Metric | Description |
|--------|-------------|
| `tools/{name}/avg_calls_per_rollout` | Average number of calls to this tool |
| `tools/{name}/failure_rate` | Rate of tool call failures |
| `tools/{name}/avg_runtime` | Average execution time |
| `tools/{name}/avg_excess_calls_per_rollout` | Calls that exceeded max limit |
| `tools/aggregate/*` | Aggregate metrics across all tools |

## Debugging Tool Training

### Local Single-GPU Testing

Use the debug scripts to test locally:

```bash
# Legacy parser (simpler, works with custom prompts)
bash scripts/train/debug/tools/legacy_parser_debug.sh

# vLLM Hermes parser (for Qwen models)
bash scripts/train/debug/tools/qwen3_vllm_hermes_parser_debug.sh

# MCP weather server example
bash scripts/train/debug/tools/mcp_weather_debug.sh
```

## Writing Custom Tools

To add a new tool:

1. **Create a Tool class** in `open_instruct/tools/tools.py`:

```python
class MyCustomTool(Tool):
    config_name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Input query"}
        },
        "required": ["query"],
    }

    def __init__(self, call_name: str, some_param: str) -> None:
        self.call_name = call_name
        self.some_param = some_param

    async def execute(self, query: str) -> ToolOutput:
        # Your tool logic here
        result = do_something(query)
        return ToolOutput(
            output=result,
            called=True,
            error="",
            timeout=False,
            runtime=0.1,
        )
```

2. **Create a config class**:

```python
@dataclass
class MyCustomToolConfig(BaseToolConfig):
    tool_class: ClassVar[type[Tool]] = MyCustomTool
    some_param: str = "default"
```

3. **Register the tool** in `TOOL_REGISTRY`:

```python
TOOL_REGISTRY: dict[str, type[BaseToolConfig]] = {
    # ... existing tools ...
    MyCustomToolConfig.tool_class.config_name: MyCustomToolConfig,
}
```

4. **Use your tool**:

```bash
--tools my_tool \
--tool_configs '{"some_param": "value"}'
```

