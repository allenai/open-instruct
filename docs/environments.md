# RL Environments

Open Instruct supports training language models with interactive RL environments. Instead of static reward functions, environments provide multi-turn feedback: the model generates tool calls, the environment executes them and returns observations, and the model continues until the task is complete.

## Quick Start

Run a single-GPU counter environment training:

```bash
bash scripts/train/debug/envs/counter_1gpu.sh
```

This trains a model to use `increment`, `decrement`, and `submit` tools to reach a target number. No external dependencies required.

## Architecture

### How It Works

1. **Dataset** — Each sample has an `env_config` column specifying its environment type (e.g., `"env_name": "counter"`)
2. **Discovery** — At startup, `discover_env_tool_definitions()` scans datasets for unique `env_name` values and fetches tool definitions for each
3. **Prompt Injection** — During tokenization, `rlvr_tokenize_v3` injects the correct tools into each sample's chat template based on its `env_name`
4. **Rollout** — During generation, the vLLM engine parses tool calls, routes them to environment actors, and appends observations back into the conversation
5. **Reward** — Per-turn rewards accumulate in `EnvironmentState` and feed into GRPO training

```
Dataset ──► discover_env_tool_definitions() ──► env_tool_map
                                                    │
Sample ──► rlvr_tokenize_v3(env_tool_map) ──► prompt with tools
                                                    │
vLLM engine ──► tool parser ──► env pool ──► observations + rewards
```

### Core Classes

| Class | File | Purpose |
|-------|------|---------|
| `RLEnvironment` | `environments/base.py` | Abstract base (extends `Tool`) |
| `StepResult` | `environments/base.py` | Return type from `reset()` and `step()` |
| `EnvironmentState` | `environments/base.py` | Accumulated per-episode rewards |
| `EnvironmentPool` | `environments/pool.py` | Ray actor pool for concurrent rollouts |
| `EnvConfig` | `tools/utils.py` | CLI configuration dataclass |

## Built-in Environments

### CounterEnv

Simple counter game. Increment/decrement to reach a target, then submit.

- **Registry name:** `counter`
- **Tools:** `increment`, `decrement`, `submit`
- **Reward:** +1.0 correct submit, -0.5 wrong submit, -0.1 per step
- **Dataset:** `hamishivi/rlenv-counter-nothink`

### GuessNumberEnv

Binary search game. Guess a secret number between 1 and 100.

- **Registry name:** `guess_number`
- **Tools:** `guess(number: int)`
- **Reward:** Closeness score `1.0 - distance / range`
- **Dataset:** `hamishivi/rlenv-guess-number-nothink`

### SandboxLMEnv

Code execution environment with bash and file editing. Mirrors the llm-in-sandbox tool interface.

- **Registry name:** `sandbox_lm`
- **Tools:** `execute_bash(command)`, `str_replace_editor(command, path, ...)`
- **Backends:** `docker`, `e2b`, `daytona`
- **Dataset:** `allenai/Dolci-RLZero-Math-7B` (math tasks solved via code)

### AgentTaskEnv

Per-sample coding tasks with test-based evaluation. Extends SandboxLMEnv.

- **Registry name:** `agent_task`
- **Tools:** `execute_bash`, `str_replace_editor`, `submit`
- **Backends:** `docker`, `daytona`
- **Dataset:** `hamishivi/agent-task-combined`

The `submit` tool runs a per-task test script and reads the reward from `/logs/verifier/reward.txt`.

Task data directory structure:

```
{task_data_dir}/{task_id}/
├── instruction.md           # Task prompt
├── environment/seeds/       # Files copied to /workspace/
├── tests/test.sh           # Test script (exit code = pass/fail)
├── image.txt               # Optional Docker image override
└── setup.sh                # Optional setup commands
```

### AppWorldEnv

Interactive environment with 9 apps (Spotify, Amazon, Venmo, etc.) and 457 APIs.

- **Registry name:** `appworld`
- **Tools:** `execute(code: str)` — runs Python with `apis.{app}.{api}()` calls
- **Dataset:** `hamishivi/rlenv-appworld-nothink`
- **Requires:** `appworld` package and data

### OpenEnv Environments

HTTP-based environments connecting to external OpenEnv servers.

- **Registry names:** `openenv_text`, `openenv_repl`
- **Config:** `--env_base_url http://localhost:8765`
- **Example:** Wordle via TextArena (`hamishivi/rlenv-wordle-nothink`)

## Creating a New Environment

### 1. Define the Environment Class

```python
# open_instruct/environments/my_env.py
from open_instruct.environments.base import RLEnvironment, StepResult, register_env
from open_instruct.tools.utils import ToolCall

@register_env("my_env")
class MyEnv(RLEnvironment):
    """My custom environment."""

    max_steps = 20

    _tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "my_action",
                "description": "Do something",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "The input"}
                    },
                    "required": ["input"],
                },
            },
        },
    ]

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return cls._tool_definitions

    async def reset(self, task_id: str | None = None, **kwargs) -> StepResult:
        """Initialize episode. Called once per sample."""
        self._state = {}  # reset internal state
        return StepResult(
            observation="Environment ready. Use my_action to proceed.",
            tools=self._tool_definitions,
        )

    async def step(self, tool_call: ToolCall) -> StepResult:
        """Process a tool call. Called after each model tool use."""
        if tool_call.name == "my_action":
            result = self._process(tool_call.args["input"])
            return StepResult(
                observation=result,
                reward=0.0,     # per-step reward
                done=False,     # True to end episode
            )
        return StepResult(observation="Unknown tool", reward=-0.1, done=False)
```

### 2. Register the Import

Add your module to `open_instruct/environments/__init__.py`:

```python
from open_instruct.environments import my_env  # noqa: F401
```

### 3. Create a Dataset

Each sample needs an `env_config` column with `env_name` matching your registry name:

```python
# scripts/data/create_my_env_dataset.py
samples = []
for task in tasks:
    samples.append({
        "messages": [
            {"role": "system", "content": "You are playing my game."},
            {"role": "user", "content": task["prompt"]},
        ],
        "ground_truth": task["answer"],
        "dataset": "passthrough",  # uses PassthroughVerifier (env provides reward)
        "env_config": {
            "task_id": task["id"],
            "env_name": "my_env",
        },
    })
```

The `"dataset": "passthrough"` verifier source tells the reward system to use the environment's own rewards (via `LastRewardAggregator` or `SumRewardAggregator`).

Push to HuggingFace Hub or save as a local `.jsonl` file.

### 4. Write a Training Script

```bash
#!/bin/bash
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list my_org/my_env_dataset 1.0 \
    --dataset_mixer_list_splits train \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --env_pool_size 16 \
    --env_max_steps 20 \
    --tool_parser_type vllm_hermes \
    --save_traces \
    --no_filter_zero_std_samples \
    --dataset_skip_cache \
    ... # other training args
```

No `--env_name` needed — it's auto-discovered from the dataset's `env_config` column.

## Configuration Reference

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--env_name` | None | Registry name (optional — auto-discovered from dataset) |
| `--env_class` | None | Custom class import path (e.g., `mymodule.MyEnv`) |
| `--env_backend` | None | Sandbox backend: `docker`, `e2b`, or `daytona` |
| `--env_pool_size` | 64 | Number of concurrent environment Ray actors |
| `--env_max_steps` | 50 | Max tool calls per episode before forced termination |
| `--env_timeout` | 60 | Timeout in seconds for environment operations |
| `--env_base_url` | None | URL for OpenEnv HTTP servers |
| `--env_task_data_dir` | None | Directory with per-task data (for agent_task) |
| `--env_image` | None | Docker image override |
| `--over_limit_penalty` | None | Penalty reward when max_steps exceeded |

### Per-Sample Override

Any CLI setting can be overridden per sample via the `env_config` column:

```json
{
    "env_config": {
        "env_name": "agent_task",
        "task_id": "task_042",
        "image": "python:3.12-slim"
    }
}
```

The per-sample `env_config` is merged with CLI defaults. Per-sample values take priority.

### Multi-Environment Training

You can mix environment and non-environment datasets in a single training run:

```bash
uv run python open_instruct/grpo_fast.py \
    --dataset_mixer_list \
        my_org/counter_dataset 0.5 \
        my_org/guess_number_dataset 0.5 \
    --dataset_mixer_list_splits train train \
    --env_pool_size 32 \
    --env_max_steps 20 \
    ...
```

Each sample gets tools injected based on its own `env_config.env_name`. Non-environment samples (no `env_config`) get no environment tools.

## Sandbox Backends

Sandbox-based environments (SandboxLMEnv, AgentTaskEnv) support pluggable backends:

| Backend | Flag | Requires | Notes |
|---------|------|----------|-------|
| Docker | `--env_backend docker` | Docker daemon + `docker` pip package | Local execution, Docker socket must be available |
| E2B | `--env_backend e2b` | `E2B_API_KEY` env var | Cloud microVMs, ~200ms startup |
| Daytona | `--env_backend daytona` | `DAYTONA_API_KEY` env var | Cloud sandboxes |

For Beaker jobs, the Docker socket is automatically mounted from the host node.

## Debug Scripts

All debug scripts are in `scripts/train/debug/envs/`:

| Script | Environment | GPUs | Notes |
|--------|------------|------|-------|
| `counter_1gpu.sh` | CounterEnv | 1 | No dependencies |
| `guess_number_1gpu.sh` | GuessNumberEnv | 1 | No dependencies |
| `sandbox_lm_1gpu.sh` | SandboxLMEnv | 1 | Needs Docker |
| `sandbox_lm_8gpu.sh` | SandboxLMEnv | 8 | Beaker launch |
| `agent_task_1gpu.sh` | AgentTaskEnv | 1 | Needs Docker + task data |
| `agent_task_8gpu.sh` | AgentTaskEnv | 8 | Beaker launch |
| `appworld_1gpu.sh` | AppWorldEnv | 1 | Needs appworld data |
| `appworld_8gpu.sh` | AppWorldEnv | 8 | Beaker launch |
| `wordle_1gpu.sh` | OpenEnvText | 1 | Needs openenv server |

Run 1-GPU scripts directly:

```bash
bash scripts/train/debug/envs/counter_1gpu.sh
```

Launch 8-GPU scripts on Beaker:

```bash
./scripts/train/build_image_and_launch.sh scripts/train/debug/envs/agent_task_8gpu.sh
```

## Rewards and Verifiers

### How Rewards Work

Environment rewards flow through three stages:

1. **Per-turn accumulation** — During rollout, each tool call returns a reward that's appended to `EnvironmentState.rewards`
2. **Verifier scoring** — After the episode, a verifier function evaluates the final output against ground truth
3. **Aggregation** — Per-turn rewards and verifier scores are combined into a single scalar for training

```
Rollout:  step1 → reward=0.0    step2 → reward=0.0    step3 → reward=1.0
                                                              ↓
Verifier: PassthroughVerifier → score=0.0  (env provides its own rewards)
                                                              ↓
Aggregation: LastRewardAggregator([0.0, 0.0, 1.0]) → 1.0  (final training reward)
```

### Verifier Sources

Each dataset sample has a `dataset` field (verifier source) that determines how it's evaluated. For environments, use `"passthrough"`:

| Verifier Source | Behavior |
|----------------|----------|
| `"passthrough"` | No-op verifier (score=0.0). Rewards come entirely from the environment. |
| `"gsm8k"` | Extracts last number from response, checks against ground truth |
| `"math"` | Checks boxed/Minerva/LaTeX math answers |
| `"ifeval"` | Instruction-following evaluation |
| `"llm_judge"` | Uses an LLM to judge response quality |
| `"code"` | Executes code against test cases |

For backward compatibility, `"env_last"` and `"env_sum"` also map to `PassthroughVerifier`.

**Environment datasets should use `"passthrough"`** (or the legacy `"env_last"` / `"env_sum"`):

```python
{
    "messages": [...],
    "ground_truth": "7",
    "dataset": "passthrough",
    "env_config": {"env_name": "counter", "task_id": "7"},
}
```

### Reward Aggregators

The `--reward_aggregator` flag controls how per-turn rewards are combined:

| Aggregator | Flag | Behavior | Use Case |
|-----------|------|----------|----------|
| Last | `--reward_aggregator last` (default) | Takes the final reward | Sparse reward envs (reward only at end) |
| Sum | `--reward_aggregator sum` | Sums all rewards | Dense reward envs (reward at each step) |

The aggregator applies **after** the verifier score is added to the last turn:

```
turn_rewards = [0.0, -0.1, 1.0]     # from environment
verifier_score = 0.0                  # from PassthroughVerifier
turn_rewards[-1] += verifier_score    # [0.0, -0.1, 1.0]

# With --reward_aggregator last:
final_reward = 1.0

# With --reward_aggregator sum:
final_reward = 0.9
```

### Combining Environment and Verifier Rewards

You can use environment rewards *and* a verifier together. For example, sandbox_lm uses `--apply_verifiable_reward true` with a math verifier — the model gets rewards both from the environment (per-step feedback) and from correctness checking (was the final answer right?):

```bash
uv run python open_instruct/grpo_fast.py \
    --apply_verifiable_reward true \
    --verification_reward 10 \
    ...
```

The `--verification_reward` is a **multiplier** on the verifier score. If the math verifier returns 0.8, the verifier contribution is `10 * 0.8 = 8.0`, added to the last turn's environment reward before aggregation.

### Multi-Verifier Datasets

A single sample can be evaluated by multiple verifiers by using lists:

```python
{
    "messages": [...],
    "ground_truth": ["42", "The answer is 42"],
    "dataset": ["math", "string_match"],
    "env_config": {...},
}
```

Each verifier's weighted score is summed. Verifier weights default to 1.0 but can be configured.

### Reward Flow Summary

```
Environment            Verifier                  Aggregator
─────────             ─────────                  ──────────
step() → reward 0.0 ─┐
step() → reward 0.0 ─┤
step() → reward 1.0 ─┤   PassthroughVerifier     LastRewardAggregator
                      ├── → score = 0.0      ──► ([0.0, 0.0, 1.0]) = 1.0
                      │   added to last turn       ──► training reward
                      │
                      │   OR with math verifier:
                      ├── MathVerifier
                      │   → score = 8.0      ──► ([0.0, 0.0, 9.0]) = 9.0
                      │   added to last turn       ──► training reward
```

### CLI Flags for Rewards

| Flag | Default | Description |
|------|---------|-------------|
| `--reward_aggregator` | `last` | `"last"` or `"sum"` — how to combine per-turn rewards |
| `--apply_verifiable_reward` | `true` | Whether to run verifier functions |
| `--verification_reward` | `10` | Multiplier for verifier scores |
| `--apply_r1_style_format_reward` | `false` | Bonus for correct `<think>` formatting |
| `--r1_style_format_reward` | `1.0` | Format reward value |
| `--additive_format_reward` | `false` | Add format reward to score (vs. gate on it) |
| `--non_stop_penalty` | `false` | Penalize responses that didn't stop naturally |
| `--non_stop_penalty_value` | `0.0` | Penalty value for non-stop responses |

## Inspecting Rollouts

When `--save_traces` is enabled, rollouts are saved as JSONL files. Inspect them to debug environment interactions:

```python
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

with open("path/to/rollouts.jsonl") as f:
    for line in f:
        r = json.loads(line)
        prompt = tokenizer.decode(r["prompt_tokens"])
        response = tokenizer.decode(r["response_tokens"])
        print(f"Reward: {r['reward']}")
        print(f"Tool calls: {r['request_info']['num_calls']}")
        print(f"Prompt: {prompt[:500]}")
        print(f"Response: {response[:500]}")
        print("---")
```

Key fields in each rollout:

- `reward` — Final episode reward
- `request_info.num_calls` — Number of tool calls made
- `request_info.tool_call_stats` — Per-call timing and results
- `prompt_tokens` / `response_tokens` — Token IDs for decoding
- `finish_reason` — `"stop"` (model stopped) or `"length"` (hit max tokens)
