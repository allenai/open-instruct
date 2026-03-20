# Tool Use and RL Environments

Open-Instruct supports multi-turn GRPO rollouts where the model can interact with external tools and stateful RL environments during generation.

In the current codebase, both are implemented through the same `RLEnvironment` interface:

- Stateless tools like Python execution, search, and browsing are exposed as environment-like workers.
- Stateful tasks like guessing games, sandboxes, and text environments run through the same rollout loop.
- Rewards can come from the environment itself, from verifiers on the final answer, or both.

This page replaces the older "tool training" view with the current setup in the repository.

## Overview

A tool- or environment-enabled rollout in `open_instruct/grpo_fast.py` works like this:

1. The dataloader builds a prompt and optionally attaches per-sample `tools` and `env_config`.
2. Open-Instruct creates Ray-backed pools for each configured tool or environment.
3. The model generates one turn of text.
4. A tool parser extracts structured calls from that text, or forwards the whole text to a text environment.
5. The selected tool or environment returns an observation, optional reward, and optional done signal.
6. The observation is appended back into the conversation and generation continues until `max_steps` is reached or the rollout ends.
7. Final rewards are aggregated with verifiers such as exact-match, code, rubric, or other ground-truth-based checks.

This makes it possible to train:

- Tool-using models that call Python, search, browse, or MCP tools.
- Stateful RL agents that solve tasks such as counter, guess-number, sandbox, or Wordle.
- Hybrid setups where the environment provides intermediate rewards and a verifier checks the final answer.

## What Is Supported

### Built-in tools

| Tool | Purpose | Notes |
| --- | --- | --- |
| `python` | Execute Python through a code API | Requires an `api_endpoint` in `tool_configs` |
| `jina_browse` | Fetch webpage content through Jina Reader | Requires `JINA_API_KEY` |
| `s2_search` | Retrieve Semantic Scholar snippets | Requires `S2_API_KEY` |
| `serper_search` | Google-style web search through Serper | Requires `SERPER_API_KEY` |
| `crawl4ai_browse` | Richer browsing via Crawl4AI | Uses AI2-specific deployment assumptions |
| `generic_mcp` | Connect to any MCP server | Can auto-discover tools at startup |
| `dr_agent_mcp` | DR-Tulu style MCP tool wrapper | Requires `uv sync --extra dr-tulu` |

### Built-in RL environments

| Environment | Type | Notes |
| --- | --- | --- |
| `counter` | Tool-style env | Increment/decrement/submit toy task |
| `guess_number` | Tool-style env | Guess a hidden integer with feedback |
| `generic_sandbox` | Tool-style env | Sandbox for command/editor style tasks |
| `wordle` | Text env | Model emits `<guess>...</guess>` text, env responds as `user` |

### Tool parser types

| Parser | Best for | Notes |
| --- | --- | --- |
| `legacy` | Custom XML-tag prompts | Expects `<tool_name>...</tool_name>`; effectively single-string-arg tools |
| `vllm_hermes` | Qwen / Hermes-style tool calling | Good default for many chat models |
| `vllm_llama3_json` | Llama 3 JSON tool calling | Uses vLLM native parser |
| `vllm_olmo3` | OLMo 3 pythonic tool calling | Uses vLLM native parser |
| `vllm_qwen3xml` | Qwen3 XML tool calling | Uses vLLM native parser |
| `dr_tulu` | DR-Tulu-style `<call_tool ...>` prompting | Requires `dr_agent_mcp` and stop sequences |

## Key Arguments

### Environment and tool selection

| Argument | Meaning |
| --- | --- |
| `--tools` | Names from the tool/env registry to enable |
| `--tool_call_names` | Names the model should emit for those tools |
| `--tool_configs` | JSON config for each tool/env |
| `--tool_parser_type` | Parser used to detect tool calls |
| `--max_steps` | Maximum rollout turns for tools/envs |
| `--per_turn_max_tokens` | Token cap for each turn inside the loop |
| `--pool_size` | Number of Ray workers per tool/env pool |
| `--pass_tools_to_chat_template` | Pass tool definitions into the chat template instead of relying on a custom system prompt |

### Reward and masking behavior

| Argument | Meaning |
| --- | --- |
| `--only_reward_good_outputs` | Ignore errored tool outputs when computing rewards |
| `--mask_tool_use` | Mask tool/environment tokens from the policy loss |
| `--reward_aggregator` | Aggregate per-turn rewards using `last` or `sum` |

`mask_tool_use` is especially important in the current `grpo_fast.py` path. Some rollout and logprob settings assert that tool-use tokens stay masked.

## Dataset Format

The standard chat dataset can be extended with two optional columns:

- `tools`: per-sample list of allowed tool names
- `env_config`: per-sample environment configuration override

Tool-gated example:

```json
{
  "messages": [
    {"role": "system", "content": "Use tools when needed."},
    {"role": "user", "content": "Use code and search to answer the question."}
  ],
  "ground_truth": "42",
  "tools": ["code", "search"]
}
```

Environment-configured example:

```json
{
  "messages": [
    {"role": "system", "content": "You are playing a number guessing game."},
    {"role": "user", "content": "Guess the hidden number efficiently."}
  ],
  "ground_truth": "7",
  "env_config": {
    "max_steps": 10,
    "env_configs": [
      {
        "env_name": "guess_number",
        "number": "7"
      }
    ]
  }
}
```

Notes:

- If `tools` is `null`, all configured tools remain available for that sample.
- If `tools` is an empty list, no tools are available for that sample.
- `env_config` is normalized into a canonical `{"env_configs": [...]}` structure during dataset preprocessing.
- Per-sample `env_config` overrides the base config supplied on the CLI for matching environment names.
- Values in `tools` should match the tool-call names exposed to the model.
- Values in `env_config.env_name` should match the configured pool target, which is usually the tool or environment name unless you override it with `--tool_call_names`.

## Quick Starts

### Tool use with a vLLM parser

For a local Qwen-based tool run:

```bash
bash scripts/train/debug/tools/qwen3_vllm_hermes_parser_debug.sh
```

That script demonstrates:

- `--tools python serper_search jina_browse`
- `--tool_call_names code search browse`
- `--tool_parser_type vllm_hermes`
- per-sample tool activation through a dataset `tools` column

### Generic MCP

For a local MCP example with the weather demo server:

```bash
bash scripts/train/debug/tools/mcp_weather_debug.sh
```

This uses:

```bash
--tools generic_mcp \
--tool_configs '{"server_url": "http://localhost:8765/mcp", "transport": "http", "timeout": 30}' \
--tool_parser_type vllm_hermes
```

If `tool_name` is omitted from the MCP config, Open-Instruct discovers the available MCP tools once at startup and expands them into separate tool pools.

### DR-Tulu style tools

For a DR-Tulu style run:

```bash
bash scripts/train/debug/tools/dr_tulu_parser_debug.sh
```

This path uses:

- `--tools dr_agent_mcp`
- `--tool_parser_type dr_tulu`
- a custom system prompt from `scripts/train/debug/tools/dr_tulu_system_prompt.txt`
- `--pass_tools_to_chat_template false`

### RL environments

For a simple built-in environment:

```bash
bash scripts/train/debug/envs/guess_number_1gpu.sh
```

For a larger text-environment run:

```bash
bash scripts/train/debug/envs/wordle_8gpu.sh
```

These runs do not need external search APIs. The environment itself provides the interaction loop and can emit rewards before the final verifier step.

## RL Environment Notes

Open-Instruct now supports both tool-style and text-style environments.

### Tool-style environments

These expose OpenAI-style tool definitions and are invoked through parsed calls. `counter` and `guess_number` fall into this bucket.

### Text environments

These consume the model's full text output instead of structured tool calls. `wordle` is the main example. Internally, Open-Instruct wraps the text into a synthetic environment call so it can reuse the same rollout loop.

Current constraint:

- Only one text environment can be active in a rollout at a time.

## MCP Notes

`generic_mcp` supports HTTP, SSE, and stdio transports. A few practical details matter:

- Discovery happens at startup, not continuously during training.
- The default integration currently assumes text outputs.
- If your server's tool list changes during training, Open-Instruct will not automatically refresh it.

## Common Caveats

- `legacy` parsing is best for tools whose first required argument can absorb the full XML body as a single string.
- `dr_tulu` requires exactly the `dr_agent_mcp` tool and depends on parser stop sequences to detect calls.
- Dataset tool names must match configured tool call names seen by the model.
- Some older comments in debug scripts are stale; prefer the actual command flags over nearby comments if they disagree.

## Adding Your Own Tool or Environment

To add a new integration:

1. Subclass `Tool`, `RLEnvironment`, or `TextRLEnvironment`.
2. Create a matching config dataclass that subclasses `BaseEnvConfig`.
3. Register that config in `TOOL_REGISTRY`.
4. Pass it through `--tools` and `--tool_configs`.

Use a `Tool` when the component is effectively stateless and request/response shaped. Use an `RLEnvironment` when you need episode state, intermediate rewards, or custom termination rules.
