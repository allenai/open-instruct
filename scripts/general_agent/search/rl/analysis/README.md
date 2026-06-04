# Deep-Research (DR-Tulu) RL run analysis

Tools to inspect a Deep-Research RL run — GRPO with **search/browse tools** and a
**continuous evolving-rubric reward** — from its wandb logs **and** its saved
trajectories. Launched by [`scripts/general_agent/search/rl/rl_qwen35_4b_drtulu.sh`](../rl_qwen35_4b_drtulu.sh).
Full write-up:
[`docs/algorithms/dr_research_rl_trajectory_analysis.md`](../../../../../docs/algorithms/dr_research_rl_trajectory_analysis.md)
and the metric glossary in
[`docs/algorithms/monitoring_and_debugging_runs.md`](../../../../../docs/algorithms/monitoring_and_debugging_runs.md).

This mirrors the Terminal-RL analysis tooling (`scripts/general_agent/terminal/rl/analysis/`)
but is rebuilt around what makes DR different: reward is **continuous** (0..`max_possible_score`),
the rubric bar is **non-stationary** (rubrics evolve / get stricter over training), and the
tools are **search/browse** rather than a code sandbox.

## Quick start

```bash
cd scripts/general_agent/search/rl/analysis

# Everything, from just the wandb URL (reads exp_name / rollouts path / response_length / max_possible_score):
uv run python analyze_dr_research_rl.py https://wandb.ai/ai2-llm/oe-general-agents/runs/<run_id>

# Faster trajectory pass (sample 96 rollouts/step on the multi-GB shards):
uv run python analyze_dr_research_rl.py <url> --per-step-cap 96

# Also decode example trajectory tails to eyeball the reward buckets:
uv run python analyze_dr_research_rl.py <url> --decode-examples 1

# wandb metrics only (skip the multi-GB trace scan):
uv run python analyze_dr_research_rl.py <url> --skip-trajectories
```

`wandb` is a `uv` dependency, so use `uv run`. `WANDB_API_KEY` is normally already set.

## Scripts

| script | what it does |
|---|---|
| `analyze_dr_research_rl.py` | **Orchestrator.** Runs the wandb analysis, then drives the trajectory analysis using `exp_name` / `rollouts_save_path` / `response_length` / `max_possible_score` read from the run config. Start here. |
| `analyze_wandb.py` | Pulls the keys this repo logs and prints learning / **evolving-rubric** / stability / **tools** / infra tables + wall-clock ETA, and emits automated **FLAGS** tuned for DR (reward-key identity, non-stationary target, rubric valid_rate/skipped, budget-bound truncation, the `non_submitting==1.0` artifact, per-tool failure rates, KL/clip/grad, staleness, tail regression). Dependency-light (wandb + numpy). Returns the run config. |
| `analyze_trajectories.py` | Streams the `--save_traces` JSONL shards and reports the **continuous reward** distribution (zero / partial / full), the **truncated-vs-completed mean reward** gap (budget), **per-tool** call counts + failure rates (from `request_info.tool_call_stats`), turns/rollout, and **within-group reward spread** (the gradient magnitude) — all as a trend over training. Handles restart instances. |

## What's different from the Terminal-RL tooling

| | Terminal-RL | DR-Tulu (this) |
|---|---|---|
| Reward | binary pass/fail (1.0 / 0.0) | continuous rubric score `0..max_possible_score` |
| "correct_rate" | n/a | **fraction of rollouts with reward > 0** (rising = fewer total failures) |
| Failure analysis | truncation vs stopped-but-wrong | reward distribution + **truncated-vs-completed reward gap** |
| Truncation | ≈ guaranteed failure | the **norm**, only mildly penalized (rubric scores the partial answer) |
| Reward target | fixed | **non-stationary** — rubrics evolve, so raw score can fall as the model improves |
| Tools | bash sandbox | `google_search` / `browse_webpage` / `snippet_search` (per-tool failure rates matter) |
| `non_submitting` | real protocol signal | **constant 1.0 artifact** (`not rollout_state["done"]`; env never sets done) |

## Key flags (trajectory analysis)

- `--per-step-cap N` — sample at most N rollouts/step (0 = all). 64–128 is plenty for
  fractions and much faster on the multi-GB shards.
- `--response-length N` — must match the run's `--response_length` (orchestrator passes it).
- `--max-score N` — must match the run's `--max_possible_score` (reward scale; default 10).
- `--decode-examples K` / `--step S` — decode K example tails per reward bucket (zero / partial /
  full) at step S, with a per-tool + rubric summary (needs `transformers`; default tokenizer `Qwen/Qwen3.5-4B`).

## What "truncated" means

Same definition as training ([`open_instruct/data_loader.py`](../../../../../open_instruct/data_loader.py)):

```
truncated  ⇔  finish_reason != "stop"   OR   len(response_tokens) >= response_length
```

In DR the second clause dominates: long multi-turn search/browse transcripts run to the
`response_length` cap while the last stored `finish_reason` is still `"stop"`.
