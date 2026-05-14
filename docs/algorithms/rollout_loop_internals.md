# Rollout Loop Internals: Prompt Construction and Token Management

This document explains exactly how the multi-turn interaction loop works at the implementation level — how the initial prompt is built, how the model is queried, how tool results are injected, and how tokens are accumulated for training. It complements the higher-level overview in [rl_with_environments.md](rl_with_environments.md).

---

## Key design principle: everything is token-level after dataset preprocessing

There is **no messages dict** after the initial tokenization step. The entire rollout loop works on a flat `list[int]` called `current_prompt` that grows as new turns are appended. This is important to understand before reading any of the code.

`apply_chat_template` (which encodes all the model-specific special tokens and turn delimiters) is called exactly **once per sample**, at dataset preprocessing time. After that, new content is appended by manually constructing the correct token string using hardcoded role templates.

---

## Phase 1: Dataset Preprocessing

**File:** `open_instruct/dataset_transformation.py` → `rlvr_tokenize_v1`

The dataset contains a `messages` list (standard OpenAI chat format). At preprocessing time:

1. Tool definitions (OpenAI function schemas) are collected from all registered environments.
2. `tokenizer.apply_chat_template(prompt, tools=tool_definitions, add_generation_prompt=True)` is called, which:
   - Injects tool schemas into the system prompt in the model's native format
   - Adds all correct special tokens (`<|im_start|>`, `<|start_header_id|>`, etc.)
   - Returns a flat `list[int]`
3. The result is stored as `INPUT_IDS_PROMPT_KEY` in the dataset row.

This is the **only** place `apply_chat_template` is called. The model sees tool schemas in the system prompt from the very start of each episode.

Per-sample tool filtering: if the dataset row has an `active_tools` column, only the listed tools are passed to `apply_chat_template`, allowing different samples to activate different subsets of tools.

---

## Phase 2: Rollout Request Setup

**File:** `open_instruct/grpo_fast.py` → `LLMRayActor.generate`

When the trainer submits a batch for rollout:

- `INPUT_IDS_PROMPT_KEY` (flat token list) is stored in `actor.request_metadata[request_id]["prompt_token_ids"]`
- `active_tools` (which tool names are enabled for this sample) is stored alongside
- `env_config` (per-episode config: `max_steps`, per-env kwargs) is stored alongside
- One async coroutine (`process_request`) is launched per sample × rollout replica

---

## Phase 3: The Multi-Turn Rollout Loop

**File:** `open_instruct/vllm_utils.py` → `process_request`

This is the core loop. One coroutine runs per sample. All state is local — no shared mutable state across samples.

```
current_prompt: list[int] = list(original_prompt)   # grows each turn
response_tokens: list[int] = []                      # model + tool tokens, for training
response_masks:  list[int] = []                      # 1=train on, 0=masked out
response_logprobs: list[float] = []

while step_count < max_steps and token_budget_remaining:
```

### Step A: Generate

```python
api_response = await vllm_client.completions.create(
    model=actor.model_name,
    prompt=current_prompt,          # raw token IDs, not messages
    max_tokens=current_max_tokens,
    ...
)
model_tokens = list(output.token_ids)
current_prompt.extend(model_tokens)     # append assistant turn in-place
response_tokens.extend(model_tokens)
response_masks.extend([1] * len(model_tokens))
```

vLLM is called via its OpenAI-compatible `/completions` endpoint with `prompt=` as token IDs (not `messages=`). This avoids any re-tokenization.

`per_turn_max_tokens` optionally caps how many tokens the model can generate per turn (before a tool call is expected), independent of the total `response_length` budget.

### Step B: Parse Tool Calls

```python
tool_calls = actor.tool_parser.get_tool_calls(output.text)
```

`get_tool_calls` delegates to a vLLM native parser (e.g., `Hermes2ProToolParser`, `Llama3JsonToolParser`) which understands the model's tool call syntax. Returns a list of `EnvCall(id, name, args)`.

**Text environment special case:** For `TextRLEnvironment` subclasses (e.g., Wordle), a **shadow `EnvCall`** is always injected regardless of whether the model called a tool:

```python
for text_env_name in text_env_names:
    tool_calls.append(EnvCall(id="", name=text_env_name, args={"text": output.text}))
```

The environment's `step()` method extracts `args["text"]` and passes the full model output to `text_step()`. This means text environments fire on every generation step unconditionally.

If there are no tool calls (and no text env), the loop `break`s — the trajectory ends here.

### Step C: Step Environment

```python
step_result: StepResult = await asyncio.wait_for(
    target_actor.step.remote(EnvCall(id=..., name=tc.name, args=tc.args)),
    timeout=actor.tool_call_timeout,
)
observations.append((step_result.result, tool_response_roles.get(tc.name, "tool")))
rollout.rewards.append(step_result.reward)
```

`target_actor` is a Ray remote actor acquired from the environment's `EnvironmentPool`. Multiple tool calls in the same turn are dispatched sequentially in the current implementation (one `await` per call).

`tool_response_roles` maps tool names to the chat role their responses should be injected with (e.g., `"tool"` for `swerl_sandbox`, `"user"` for `WordleTextEnv`). This is set per-environment in `_acquire_and_reset_pools`.

Timeout and exception handling both produce an error string observation and `reward=0.0`, then set `rollout.timeout=True`.

### Step D: Format and Inject Tool Response

```python
formatted_str = tool_parser.format_tool_outputs([observation], role=role)
tokens = tokenizer.encode(formatted_str, add_special_tokens=False)
current_prompt.extend(tokens)
response_tokens.extend(tokens)
response_masks.extend([0 if mask_tool_use else 1] * len(tokens))
```

`format_tool_outputs` uses a **hardcoded role template** from the `VLLM_PARSERS` registry (see below). It does **not** call `apply_chat_template`. The formatted string is encoded directly and appended to `current_prompt`.

If `--mask_tool_use` is set, tool response tokens get `mask=0` — they are present in the sequence for context but excluded from the training loss.

The `output_postfix` (e.g., `<|im_start|>assistant\n`) is appended after the tool response to prime the model to continue as the assistant on the next turn.

### Loop continues

Back to Step A with the now-longer `current_prompt`, which includes all previous turns.

---

## Phase 4: Rollout Completion

**File:** `open_instruct/vllm_utils.py` → `process_completed_request`

When the loop exits, the rollout is packaged:

```
CompletionOutput:
  token_ids      = response_tokens    (all model + tool tokens)
  logprobs       = response_logprobs
  mask           = response_masks     (1=in loss, 0=excluded)
  rollout_state  = {rewards, step_count, tool_output, tool_error, timeout, ...}
```

This is placed on `actor.completion_queue` for the trainer to consume.

---

## Phase 5: Training

The trainer receives the flat token lists and:

1. Concatenates prompt + response tokens to form the full sequence
2. Computes advantages from `rollout_state.rewards` using the configured `reward_aggregator` (`last` or `sum`)
3. Applies optional verifier-based rewards on top
4. Runs the GRPO loss, masking out positions where `mask=0`
5. Updates model weights and syncs to vLLM engines

No message reconstruction happens here — training operates entirely on token sequences.

---

## Tool Parser Types and Role Templates

**File:** `open_instruct/environments/tools/parsers.py` → `VLLM_PARSERS`

The role templates are hardcoded per model family. They must match what `apply_chat_template` would produce for the same model. There is no automatic validation — if you use the wrong parser for your model, the injected tokens will be malformed and training will degrade silently.

| `--tool_parser_type` | Models | Tool role token | Postfix |
|---|---|---|---|
| `vllm_hermes` | Qwen 2.5, Qwen 3, ChatML | `<\|im_start\|>tool\n<tool_response>\n{output}\n</tool_response>\n<\|im_end\|>\n` | `<\|im_start\|>assistant\n` |
| `vllm_llama3_json` | Llama 3.x | `<\|start_header_id\|>ipython<\|end_header_id\|>\n\n{output}<\|eot_id\|>` | `<\|start_header_id\|>assistant<\|end_header_id\|>\n\n` |
| `vllm_olmo3` | Olmo 3 | `<\|im_start\|>environment\n{output}<\|im_end\|>\n` | `<\|im_start\|>assistant\n` |
| `vllm_qwen3_xml` | Qwen 3.5 XML | `<\|im_start\|>user\n<tool_response>\n{output}\n</tool_response>\n<\|im_end\|>\n` | `<\|im_start\|>assistant\n<think>\n` |

Note that `vllm_qwen3_xml` injects tool responses with the `user` role (not `tool`) and opens a `<think>` block in the postfix, matching Qwen 3.5's expected format.

For `TextRLEnvironment`, the role is `"user"` by default (set via `TextRLEnvironment.response_role`), so the `user` template is used regardless of parser type.

---

## Environment Types Summary

| Base class | Fires when | Model action | Response role |
|---|---|---|---|
| `RLEnvironment` | Model explicitly calls a tool | Parsed `EnvCall(name, args)` | `"tool"` (default) |
| `TextRLEnvironment` | Every generation step (shadow call) | Full model output text | `"user"` (default) |

### Registered environments

| Name (`--tools`) | Class | Type |
|---|---|---|
| `swerl_sandbox` | `SWERLSandboxEnv` | RLEnvironment (Podman container) |
| `generic_sandbox` | `GenericSandboxEnv` | RLEnvironment (Podman/Docker container) |
| `guess_number` | `GuessNumberEnv` | RLEnvironment (example) |
| `counter` | `CounterEnv` | RLEnvironment (example) |
| `wordle` | `WordleTextEnv` | TextRLEnvironment |
| `python` | `PythonCodeTool` | Tool (stateless) |
| `jina_browse` | `JinaBrowseTool` | Tool (stateless) |
| `crawl4ai_browse` | `Crawl4AIBrowseTool` | Tool (stateless) |
| `s2_search` | `S2SearchTool` | Tool (stateless) |
| `serper_search` | `SerperSearchTool` | Tool (stateless) |
| `dr_agent_mcp` | `DrAgentMCPTool` | Tool (MCP) |
| `generic_mcp` | `GenericMCPTool` | Tool (MCP) |

---

## Data Flow Summary

```
Dataset row (messages list)
        │
        ▼  apply_chat_template(tools=tool_defs)   [once, at preprocessing]
        │
flat token list (INPUT_IDS_PROMPT_KEY)
        │
        ▼  stored in request_metadata
        │
current_prompt = list(original_prompt)
        │
        ┌─────────────────────────────────┐
        │  ROLLOUT LOOP (per turn)        │
        │                                 │
        │  vLLM /completions(current_prompt)
        │         │                       │
        │         ▼                       │
        │  model_tokens ──extend──► current_prompt
        │                                 │
        │  tool_parser.get_tool_calls()   │
        │         │                       │
        │         ▼                       │
        │  env_actor.step.remote()        │
        │         │                       │
        │         ▼                       │
        │  format_tool_outputs(role_template)
        │         │                       │
        │         ▼                       │
        │  tool_tokens ───extend──► current_prompt
        │                                 │
        └─────────────────────────────────┘
        │
response_tokens + response_masks
        │
        ▼  GRPO loss (masked)
        │
weight update
```
