# Tools Module Refactor

## What Changed

### New `open_instruct/tools/` Module

```
tools/
├── __init__.py       # Package init
├── utils.py          # Tool, ToolCall, ToolOutput, BaseToolConfig base classes
├── new_tools.py      # Tool implementations (PythonCodeTool, SerperSearchTool, etc.) + TOOL_REGISTRY
├── parsers.py        # ToolParser ABC + implementations
├── servers/
│   └── python_server/  # Python execution server
└── tests/
    └── test_parsers.py
```

### The Tool Pattern

Every tool follows the same shape:

```python
@dataclass
class MyToolConfig(BaseToolConfig):
    tool_class: ClassVar[type[Tool]] = MyTool
    some_option: int = 10
    # override_name inherited from BaseToolConfig

class MyTool(Tool):
    _default_tool_function_name = "mytool"
    _default_tool_description = "Does something useful"
    _default_tool_parameters = {  # JSON Schema for your args
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["query"],
    }

    def __init__(self, some_option: int = 10, override_name: str | None = None):
        self._override_name = override_name
        # ...

    def __call__(self, query: str, limit: int = 10) -> ToolOutput:
        # Tools can accept any arguments - define what you need
        # The parser extracts args from model output and passes them as kwargs
        return ToolOutput(output=..., called=True, ...)
```

Then add to registry:
```python
TOOL_REGISTRY = {
    "mytool": MyToolConfig,  # Just the config - tool_class is a ClassVar on the config
}
```

CLI args are auto-generated. All tools are wrapped in Ray actors via `ToolProxy` for safe cross-process communication.

Access individual tool configs via `config.get_tool_config("tool_name")`.

### CLI Usage

Tools are specified with `--tools` and configured via `--tool_configs` (a 1:1 list of JSON objects):

```bash
--tools mcp python --tool_configs '{"tool_names": "snippet_search"}' '{"api_endpoint": "http://..."}'
```

Use `{}` for defaults:
```bash
--tools s2_search serper_search --tool_configs '{}' '{"num_results": 10}'
```

### Parser System

We support three parser families:

- **`legacy`**: Original `<tool_name>content</tool_name>` XML style
- **`dr_tulu`**: For MCP tools with their own routing (from DR Tulu project)
- **`vllm_*`**: Native vLLM tool parsers (`vllm_hermes`, `vllm_llama3_json`, `vllm_qwen3_coder`)

Parsers handle:
1. Extracting tool calls from model output
2. Formatting tool responses back
3. Providing stop sequences

vLLM parsers are wrappers around their vllm classes. In the future, we can hopefully remove more and more of the scaffolding around the vLLM parsers, and completely remove the legacy and dr_tulu parsers. For now, I would like to keep them around, since a tight integration isn't really possible (and, there are some higher-level concerns).

### Per-Tool Metrics

Now tracking per-tool call rates and counts:

```
val/tool_search_call_rate      # % samples that called search
val/tool_search_total_calls    # Total calls across all samples
val/tool_search_calls_per_sample
```

We do this for all tools in the training job.

### Override Name

We can override the names tools use (for simpler names or to match model training format):

```bash
--tools s2_search --tool_override_names search
```

Now `S2SearchTool` responds to `<search>` instead of `<s2_search>`, and its OpenAI function definition uses `search` as the function name.

Multiple tools:
```bash
--tools s2_search serper_search --tool_override_names papers web
```

## Available Tools

| Name | Description |
|------|-------------|
| `python` | Code execution via FastAPI server |
| `serper_search` | Google search via Serper API |
| `massive_ds_search` | Wikipedia/doc retrieval via massive_ds |
| `s2_search` | Semantic Scholar papers |
| `mcp` | DR Agent MCP tools wrapper (snippet_search, google_search, massive_serve, browse_webpage) |

## Adding a New Tool

1. Create `YourTool` class in `tools.py` extending `Tool`:
   - Set `_default_tool_function_name`, `_default_tool_description`, `_default_tool_parameters`
   - Implement `__call__(self, **kwargs) -> ToolOutput` with whatever arguments your tool needs

Note that the names, descriptions, and parameters are used to auto generate things like CLI args and system prompts, so make sure they are correct and reasonably well-written!

2. Create `YourToolConfig` dataclass in `tools.py`:
   - Inherit from `BaseToolConfig`
   - Set `tool_class: ClassVar[type[Tool]] = YourTool`
   - Add configuration fields (with defaults)

3. Register in `config.py`:
   ```python
   TOOL_REGISTRY = {
       ...,
       "yourtool": YourToolConfig,
   }
   ```
