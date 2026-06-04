# Terminal-RL run analysis

Tools to inspect a Terminal / agentic RL run (GRPO with tools/sandbox) from its
wandb logs **and** its saved trajectories. Full write-up:
[`docs/algorithms/terminal_rl_trajectory_analysis.md`](../../../../../docs/algorithms/terminal_rl_trajectory_analysis.md)
and the metric glossary in
[`docs/algorithms/monitoring_and_debugging_runs.md`](../../../../../docs/algorithms/monitoring_and_debugging_runs.md).

## Quick start

```bash
cd scripts/general_agent/terminal/rl/analysis

# Everything, from just the wandb URL (reads exp_name / rollouts path / response_length from config):
uv run python analyze_terminal_rl.py https://wandb.ai/ai2-llm/oe-general-agents/runs/<run_id>

# Faster trajectory pass (sample 96 rollouts/step):
uv run python analyze_terminal_rl.py <url> --per-step-cap 96

# Also decode example trajectory tails to eyeball the failure buckets:
uv run python analyze_terminal_rl.py <url> --decode-examples 1

# wandb metrics only (skip the multi-GB trace scan):
uv run python analyze_terminal_rl.py <url> --skip-trajectories
```

`wandb` is a `uv` dependency, so use `uv run`. `WANDB_API_KEY` is normally
already set in this environment.

## Scripts

| script | what it does |
|---|---|
| `analyze_terminal_rl.py` | **Orchestrator.** Runs the wandb analysis, then drives the trajectory analysis using `exp_name` / `rollouts_save_path` / `response_length` read from the run config. Start here. |
| `analyze_wandb.py` | Pulls the metrics this repo logs, prints learning / stability / behavior / infra tables + wall-clock ETA, and emits automated **FLAGS** (dead reward key, degenerate advantages, KL runaway, high truncation / non-submitting, staleness/preemption, tail regression, …). Dependency-light (wandb + numpy). Returns the run config. |
| `analyze_trajectories.py` | Streams the `--save_traces` JSONL shards and classifies failures into **truncation** (budget) vs **stopped-but-wrong** (genuine), using the training code's own truncation rule. Reports the trend over training, per-prompt group consistency, the aggregate split, and (with `--decode-examples`) decoded example tails. Handles multiple restart instances. |

## Key flags (trajectory analysis)

- `--per-step-cap N` — sample at most N rollouts/step (0 = all). 64–128 is plenty
  for fractions and is much faster on the multi-GB shards. (Note: capping keeps
  full sample-per-prompt groups since records are written prompt-contiguous.)
- `--response-length N` — must match the run's `--response_length` (the
  orchestrator passes it automatically) so truncation is labeled correctly.
- `--decode-examples K` / `--step S` — decode K example tails per bucket at step S
  (needs `transformers`; default tokenizer `hamishivi/Qwen3.5-4B`).

## What "truncated" means

Same definition as training ([`open_instruct/data_loader.py`](../../../../../open_instruct/data_loader.py)):

```
truncated  ⇔  finish_reason != "stop"   OR   len(response_tokens) >= response_length
```

The second clause catches budget exhausted *inside* the multi-turn tool loop,
where the last stored `finish_reason` can still be `"stop"`.
