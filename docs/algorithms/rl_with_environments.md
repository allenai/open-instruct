# RL Training with Environments and Tools (open-instruct)

This document covers how [open-instruct](open-instruct/) supports reinforcement learning with external environments and tools, including how to add new environments, the agent loop, async support, and how to launch training.

The canonical upstream reference is [open-instruct/docs/algorithms/tool_training.md](open-instruct/docs/algorithms/tool_training.md).

---

## Table of Contents

1. [Adding a New Environment](#1-adding-a-new-environment)
2. [Containerization](#2-containerization)
3. [Multi-task RL (Multiple Environments)](#3-multi-task-rl-multiple-environments)
4. [Agent Loop](#4-agent-loop)
5. [Async RL Support](#5-async-rl-support)
6. [Launching RL Training](#6-launching-rl-training)

---

## 1. Adding a New Environment

**Directory:** `open-instruct/open_instruct/environments/`

Two base classes exist in `open_instruct/environments/base.py`:

### Option A: `RLEnvironment` (structured tool calls)

Use this when the model interacts with the environment via OpenAI-style function/tool calls.

```python
from open_instruct.environments.base import RLEnvironment, StepResult, EnvCall, State
from open_instruct.environments.base import BaseEnvConfig
from dataclasses import dataclass
from typing import ClassVar

class MyEnv(RLEnvironment):
    config_name = "my_env"  # key used in TOOL_REGISTRY

    async def reset(self, **kwargs) -> tuple[StepResult, list[dict]]:
        """Called once per episode with per-sample kwargs from the dataset.
        Returns (initial observation, list of OpenAI-format tool schemas)."""
        return StepResult(result="Episode started."), self.get_tool_definitions()

    async def step(self, call: EnvCall) -> StepResult:
        """Called per tool call. `call.name` is the function name, `call.args` is
        a parsed dict of arguments. Return observation, reward, and done flag."""
        if call.name == "my_action":
            value = call.args.get("value")
            reward = 1.0 if value == self.target else 0.0
            done = reward > 0
            return StepResult(result=f"Got {value}", reward=reward, done=done)
        return StepResult(result="Unknown action", reward=0.0, done=False)

    def state(self) -> State:
        """Return current episode state (used for logging/metrics)."""
        return State(step_count=self.step_count)

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        """Return OpenAI function schemas for all available tools."""
        return [{
            "type": "function",
            "function": {
                "name": "my_action",
                "description": "Perform an action.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer", "description": "The value to submit."}
                    },
                    "required": ["value"]
                }
            }
        }]
```

### Option B: `TextRLEnvironment` (raw text interaction)

Use this when the model's full text output (not parsed tool calls) is the action.

```python
from open_instruct.environments.base import TextRLEnvironment, StepResult

class MyTextEnv(TextRLEnvironment):
    response_role = "user"  # role used for observations in the conversation

    async def _reset(self, **kwargs) -> StepResult:
        return StepResult(result="Game started. Guess a word.")

    async def text_step(self, text: str) -> StepResult:
        """Process the model's full text output. Return feedback, reward, done."""
        guess = text.strip().lower()
        if guess == self.target:
            return StepResult(result="Correct!", reward=1.0, done=True)
        return StepResult(result=f"Wrong. Try again.", reward=0.0, done=False)
```

### Config Dataclass

Every environment needs a config class inheriting from `BaseEnvConfig`:

```python
@dataclass
class MyEnvConfig(BaseEnvConfig):
    tool_class: ClassVar[type] = MyEnv
    target: int = 42
    max_steps: int = 10
```

### Registry Entry

Register your environment in `open_instruct/environments/tools/tools.py`:

```python
from open_instruct.environments.tools.tools import TOOL_REGISTRY

TOOL_REGISTRY["my_env"] = MyEnvConfig
```

### Optional Hooks

- `setup()` — called once at training start; use for one-time resource initialization (e.g., loading data, connecting to a service).
- `get_metrics()` — return custom per-episode metrics for logging.

### Reference Examples

| File | Environment | Description |
|------|-------------|-------------|
| `environments/examples.py` | `CounterEnv`, `GuessNumberEnv` | Simple state machine / interactive game |
| `environments/examples.py` | `WordleTextEnv` | Text-based game (TextRLEnvironment) |
| `environments/generic_sandbox.py` | `GenericSandboxEnv` | Bash shell + file editor (Docker-backed) |
| `environments/tools/generic_mcp.py` | `GenericMCPTool` | Wraps any MCP server as an environment |
| `environments/tools/tools.py` | `PythonCodeTool`, `SerperSearchTool`, etc. | Stateless tool wrappers |

---

## 2. Containerization

**Only the `GenericSandboxEnv`** uses Docker containers, via a `SandboxBackend` in `environments/backends.py`. Configure with `backend: "docker"`.

**Lifecycle — on-demand per episode, not pre-launched:**

1. Each episode calls `reset()`, which spins up a fresh backend (fresh Docker container for sandbox envs).
2. The container persists state (bash env vars, cwd, files) across `step()` calls within one episode.
3. On the next episode, `reset()` tears down the old container and creates a new one — ensuring clean state isolation.

Most other environments (search tools, Python eval, etc.) run in-process or as subprocesses — no containers involved.

**MCP Servers** (`GenericMCPTool`) support HTTP, SSE, and stdio transport. They are launched as child processes or connected to externally running servers, and are managed separately from Docker containers.

---

## 3. Multi-task RL (Multiple Environments)

Multiple environments can be active simultaneously during a single training run. Configure via CLI:

```bash
--tools python serper_search guess_number wordle \
--tool_call_names code search guess_env text_game \
--tool_configs '{}' '{}' '{"target": 5}' '{"word": "crane"}'
```

**How it works:**

- A separate **Ray actor pool** is created per tool/environment.
- During rollout, the parser extracts all tool calls from the model output.
- Each tool call is dispatched to the matching pool by name.
- Multiple pools execute concurrently (async dispatch).

**Per-sample tool availability** can be restricted via a `tools` column in the dataset:

```json
{
  "prompt": "...",
  "tools": ["code", "search"],
  "env_config": {"max_steps": 10}
}
```

Only the listed tools will be active for that sample, even if the training run has more tools registered globally.

---

## 4. Agent Loop

The agent loop (implemented in `open_instruct/vllm_utils.py`, `_generate_with_tools`) is **ReAct-style** (interleaved reasoning and acting):

```
while step_count < max_steps and token_budget_remaining:
    1. Generate text (vLLM, up to per_turn_max_tokens)
    2. Parse tool calls from model output
    3. For each tool call:
         a. Acquire actor from the matching pool
         b. Call env.step(EnvCall) asynchronously
         c. Collect StepResult (observation, reward, done flag)
         d. Release actor back to pool
    4. Append observations to conversation (as tool/user messages)
    5. Update accumulated response tokens and log-probs
    6. If done=True or max_steps/token budget reached → stop
    7. Otherwise → next generation turn (back to step 1)
```

**Key data structures:**

| Type | Fields | Description |
|------|--------|-------------|
| `EnvCall` | `id, name, args` | Parsed tool call from model output |
| `StepResult` | `result, reward, done, metadata` | Environment response |
| `RolloutState` | `rewards[], step_count, tool_output, ...` | Accumulated rollout state |

**Reward aggregation** across steps is controlled by `--reward_aggregator`:
- `last` — use only the final step's reward (default)
- `sum` — sum all per-step rewards

Verifier-based rewards (ground truth checking via `--apply_verifiable_reward`) can be layered on top.

---

## 5. Async RL Support

Three layers of async/concurrency:

### Ray Actor Pools (`environments/pool.py`)

`EnvironmentPool` is an async Ray actor managing a pool of environment instances:
- `acquire()` — blocks until an actor is available (asyncio.Queue, no polling)
- `release(actor)` — returns the actor to the pool and wakes any waiters
- Default pool size = `num_unique_prompts_rollout × num_samples_per_prompt_rollout`
- Override with `--pool_size N`

Example: 16 unique prompts × 4 rollouts per prompt = 64 concurrent actors.

### Concurrent Tool Dispatch (`vllm_utils.py`)

Multiple tool calls within one rollout turn are dispatched **concurrently** to different actors:

```python
for tc in tool_calls:
    actor = await pool.acquire()
    result = await actor.step.remote(EnvCall(...))  # async, non-blocking
    await pool.release(actor)
```

### Data Pipeline Pipelining (`data_loader.py`)

`DataPreparationActor` prepares rollout batches `async_steps` steps ahead (default: 8). While the trainer is updating weights on batch N, rollouts for batch N+8 are being generated in the background — preventing generation from being the bottleneck.

---

## 6. Launching RL Training

**Main entry point:** `open-instruct/open_instruct/grpo_fast.py`

### Minimal example

```bash
python open_instruct/grpo_fast.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mixer_list my_rl_dataset 1.0 \
  --dataset_mixer_list_splits train \
  --tools guess_number \
  --tool_call_names guess_env \
  --tool_configs '{"target": 42}' \
  --tool_parser_type vllm_hermes \
  --max_steps 5 \
  --num_unique_prompts_rollout 16 \
  --num_samples_per_prompt_rollout 4 \
  --response_length 1024 \
  --learning_rate 3e-7 \
  --beta 0.01 \
  --total_episodes 200 \
  --output_dir output/my_run
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `--tools` | Space-separated list of tool/env names (from `TOOL_REGISTRY`) |
| `--tool_call_names` | Names the model uses to call each tool (order matches `--tools`) |
| `--tool_configs` | JSON config string per tool (order matches `--tools`) |
| `--tool_parser_type` | Parser for extracting tool calls from model output (see below) |
| `--max_steps` | Max tool calls per rollout episode |
| `--per_turn_max_tokens` | Optional per-generation-turn token limit |
| `--pool_size` | Number of concurrent environment actors |
| `--reward_aggregator` | `last` or `sum` — how to combine per-step rewards |
| `--apply_verifiable_reward` | Also apply ground-truth-based reward |
| `--num_unique_prompts_rollout` | Unique prompts per batch |
| `--num_samples_per_prompt_rollout` | Rollouts per prompt (for variance reduction) |
| `--async_steps` | Pipeline depth for data preparation (default: 8) |
| `--save_traces` | Save full rollout trajectories for inspection |
| `--pass_tools_to_chat_template` | Pass tool schemas into the model's chat template |
| `--beta` | KL penalty coefficient (0 = no KL penalty) |

### Tool parser types

| Parser | Models |
|--------|--------|
| `vllm_hermes` | Qwen 2.5, Qwen 3, most modern models (ChatML format) |
| `vllm_llama3_json` | Llama 3.x |
| `vllm_olmo3` | OLMo 3 |
| `vllm_qwen3xml` | Qwen 3 XML-style |
| `legacy` | Open Instruct original XML tag format |
| `dr_tulu` | DR-Tulu |

### Reference debug scripts

Located in `open-instruct/scripts/train/debug/envs/`:

```bash
# Single GPU debug run with GuessNumber environment
bash scripts/train/debug/envs/guess_number_1gpu.sh

# 8-GPU run with Wordle (TextRLEnvironment)
bash scripts/train/debug/envs/wordle_8gpu.sh

# Tool calling with Hermes parser
bash scripts/train/debug/tools/qwen3_vllm_hermes_parser_debug.sh
```

### Infrastructure options

```bash
--deepspeed_stage 2              # DeepSpeed ZeRO stage (2 or 3)
--vllm_tensor_parallel_size 1    # vLLM tensor parallelism
--vllm_gpu_memory_utilization 0.3
--single_gpu_mode                # for local debugging
```
