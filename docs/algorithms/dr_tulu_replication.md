# DR Tulu Replication: Adaptive Rubrics and Tools

This page documents the closest current Open-Instruct equivalent of a DR-Tulu-style setup: tool-augmented GRPO with rubric-based reward signals.

One naming note up front: in this repository, the "adaptive rubric" work is implemented under the name **evolving rubrics**. The codebase contains strong support for:

- DR-Tulu-style tool parsing and MCP-backed tool execution
- static rubric scoring through `RubricVerifier`
- evolving-rubric generation and buffer utilities

But these pieces are not all wired together into one fully online training loop today. This page separates what is runnable now from what is still research plumbing.

## What You Can Run Today

### 1. DR-Tulu-style tool use

The runnable tool path is:

- install the optional dependency group with `uv sync --extra dr-tulu`
- run an MCP backend
- train with `--tools dr_agent_mcp --tool_parser_type dr_tulu`
- provide a DR-Tulu-style system prompt

The reference script is:

```bash
bash scripts/train/debug/tools/dr_tulu_parser_debug.sh
```

That script starts the MCP backend, loads search keys, and launches `open_instruct/grpo_fast.py` with:

- `--tools dr_agent_mcp`
- `--tool_parser_type dr_tulu`
- `--tool_configs` selecting MCP-exposed tools such as `snippet_search`, `google_search`, and `browse_webpage`
- `--system_prompt_override_file scripts/train/debug/tools/dr_tulu_system_prompt.txt`
- `--pass_tools_to_chat_template false`

### 2. Static rubric scoring

Open-Instruct already supports rubric-based verification through `RubricVerifier`.

The ground-truth payload should contain:

- `query` or `Question`
- `rubrics`, a list of rubric objects with `description` and `weight`

Example:

```json
{
  "query": "Explain why regularization helps in supervised learning.",
  "rubrics": [
    {
      "description": "Explains that regularization reduces overfitting by discouraging overly complex solutions.",
      "weight": 1.0
    },
    {
      "description": "Avoids claiming that regularization always improves training-set performance.",
      "weight": -1.0
    }
  ]
}
```

During reward computation, `RubricVerifier` uses an LLM judge to score each rubric from `0` to `2`, normalizes that to `0` to `1`, and returns a weighted aggregate score.

### 3. Tools plus rubric/verifier reward

A practical replication path today is:

1. Train with DR-Tulu-style tools.
2. Keep `--apply_verifiable_reward true`.
3. Use a dataset whose verifier or rubric payload matches the desired target behavior.

This gives you a stable tool-plus-judge setup without needing online rubric generation inside the GRPO loop.

## DR-Tulu Prompt Contract

The default prompt contract lives in:

```text
scripts/train/debug/tools/dr_tulu_system_prompt.txt
```

It expects the model to:

- reason in `<think>...</think>`
- call tools with `<call_tool name="...">...</call_tool>`
- support claims with citations from retrieved snippets
- finish with `<answer>...</answer>`

This format is important because the `dr_tulu` parser does not parse individual tool arguments itself. Instead, it detects that a DR-Tulu tool call happened and forwards the full text to `dr_agent_mcp`.

## Required Setup

### Dependencies

```bash
uv sync --extra dr-tulu
```

This optional group adds the `dr_agent` dependency used by `dr_agent_mcp`.

### MCP backend

The debug script starts the backend for you, but the direct shape is:

```bash
uv run --extra dr-tulu python -m dr_agent.mcp_backend.main \
  --host 0.0.0.0 \
  --port 8000 \
  --path /mcp
```

### Environment variables

Typical DR-Tulu search runs need at least:

- `SERPER_API_KEY`
- `S2_API_KEY`

Rubric scoring additionally needs:

- `OPENAI_API_KEY` or `AZURE_API_KEY`

Useful overrides:

- `RUBRIC_JUDGE_MODEL` for rubric scoring
- `RUBRIC_GENERATION_MODEL` for evolving-rubric generation utilities

## The Adaptive Rubric Story

### Static rubrics are fully usable

The static rubric path is end-to-end:

- dataset carries rubric ground truth
- `RubricVerifier` scores the final answer
- GRPO can optimize against that score through the standard verifiable reward path

If your dataset uses a different verifier label, you can remap it. For example:

```bash
--remap_verifier general_rubric=rubric
```

### Evolving rubrics exist as utilities

The repository also contains utilities for generating and maintaining evolving rubrics:

- rubric generation prompts
- LLM calls to propose positive and negative rubrics from rollout responses
- a per-query rubric buffer
- filtering helpers that deactivate low-signal rubrics
- cache helpers for saving rubric-generation artifacts

The closest experiment script is:

```bash
bash scripts/train/debug/evolving_rubric_mini_test.sh
```

### Important limitation

The evolving-rubric utilities are **not currently wired into** the main `RewardConfig` path in `open_instruct/grpo_fast.py`.

In practice, that means:

- the repo contains evolving-rubric generation code
- the config exposes flags like `--apply_evolving_rubric_reward`
- but the live GRPO reward path still constructs rewards from the existing verifier stack, not from online rubric generation inside training

So if you want a faithful "adaptive rubrics + tools" run today, there are two realistic options:

1. Use tools plus static rubric scoring during training.
2. Generate evolving rubrics offline or in a side pipeline, write them back into dataset ground truth, then train with the normal rubric verifier.

## Recommended Replication Paths

### Stable path: tools plus static rubrics

Use this if you want a run that matches current production wiring:

1. Start from `scripts/train/debug/tools/dr_tulu_parser_debug.sh`.
2. Keep the DR-Tulu prompt and parser setup.
3. Provide rubric-based ground truths in the dataset.
4. Use the standard verifiable reward path.

This is the best current choice for reproducible training.

### Research path: evolving rubrics plus tools

Use this if you want to extend the repo:

1. Start from the DR-Tulu tool setup.
2. Use the evolving-rubric utilities in `open_instruct/rubrics/`.
3. Thread generated rubrics and per-rubric metrics into the reward path used by `grpo_fast.py`.
4. Decide how to aggregate persistent and evolving rubric scores.

The code for generation, buffering, and filtering is already present, but the reward plumbing and metrics logging still need to be connected.

## Parser and Integration Caveats

- `dr_tulu` requires exactly one configured tool definition: `dr_agent_mcp`.
- The parser needs stop sequences; without them, calls are never detected.
- DR-Tulu tools and evolving rubrics are currently orthogonal features in the codebase.
- The evolving-rubric metrics helpers are available, but they are not yet part of the default training logs produced by `grpo_fast.py`.

## Suggested Starting Commands

Tool-focused DR-Tulu run:

```bash
bash scripts/train/debug/tools/dr_tulu_parser_debug.sh
```

Rubric-focused exploratory run:

```bash
bash scripts/train/debug/evolving_rubric_mini_test.sh
```

If your goal is a clean replication on current Open-Instruct, start with the first command and treat the second as a development scaffold rather than a fully integrated benchmark recipe.
