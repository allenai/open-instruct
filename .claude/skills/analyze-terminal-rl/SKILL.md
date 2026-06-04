---
name: analyze-terminal-rl
description: Analyze a Terminal / agentic RL training run (GRPO with tools/sandbox) and report health + failure-mode insights. Use when the user gives a wandb run URL (or exp_name) for a Terminal RL / SWE-RL / tmax run and asks to check it, see if it's learning, diagnose it, or asks "does anything look suspect". Pulls wandb metrics AND classifies saved trajectories into truncation-vs-genuine failures.
---

# Analyze a Terminal-RL run

When the user points you at a Terminal-RL / agentic-RL run (a wandb URL like
`https://wandb.ai/<entity>/<project>/runs/<id>`, or an `exp_name`), run the
analysis tooling and report insights — don't make them ask for each piece.

## Instructions

1. **Run the one-shot analyzer** from the repo root:
   ```bash
   cd scripts/general_agent/terminal/rl/analysis
   uv run python analyze_terminal_rl.py <wandb-url> --per-step-cap 96
   ```
   - It prints (a) wandb metric tables + automated FLAGS, then (b) the trajectory
     error analysis (truncation vs genuine-wrong) with the over-training trend.
   - `--per-step-cap 96` keeps the multi-GB trajectory scan fast; drop it for an
     exact full pass, or add `--skip-trajectories` for a wandb-only quick look.
   - The trajectory scan can take a few minutes (10s of GB); run it in the
     background and keep working, or cap harder.
   - If wandb auth is needed, `WANDB_API_KEY` is usually already set; `wandb` is a
     `uv` dep (use `uv run`, not bare `python`).

2. **To eyeball example trajectories** (confirm the failure buckets are real),
   add `--decode-examples 1` (decodes the tails of one truncated / stopped /
   solved rollout; needs the tokenizer, default `hamishivi/Qwen3.5-4B`).

3. **Interpret and report** using the framing in
   `docs/algorithms/monitoring_and_debugging_runs.md` and
   `docs/algorithms/terminal_rl_trajectory_analysis.md`. Lead with the answer to
   "is it learning?" then surface the flags. Specifically:
   - **Learning?** Trust `val/avg_group_performance_pre_filter` (cleaner) over the
     noisy per-token `scores`. A binary-reward env means `scores` ≈ solve rate.
   - **The reward-key trap:** `objective/verifiable_reward` flat 0.0 is EXPECTED
     when reward comes from the environment path — not a bug. Real signal is
     `scores` / `avg_group_performance`.
   - **Why is it failing?** Report the truncation-vs-genuine split. Truncation =
     budget-bound (lever: `response_length` / conciseness / curriculum).
     Stopped-but-wrong = genuine difficulty (real learning headroom).
   - **beta=0** ⇒ KL is monitor-only; watch `objective/kl2_avg` yourself.
   - **Call out**: tail regression, degenerate advantages, KL runaway, high
     truncation/non-submitting, `stale_results_dropped` bursts (usually
     preemption), staleness gap, throughput/ETA.

4. **Offer the deeper drills** (don't run unprompted unless the user wants depth):
   per-step (uncapped) failure trend, decoding more examples from a specific step,
   or distinguishing a stale post-restart batch from a real regression.

## Notes

- Scripts: `analyze_terminal_rl.py` (orchestrator), `analyze_wandb.py` (metrics +
  flags), `analyze_trajectories.py` (trace classification). See the directory
  `README.md`.
- The orchestrator reads `exp_name`, `rollouts_save_path`, and `response_length`
  from the wandb run config, so a URL is enough. If only an `exp_name` is given,
  run `analyze_trajectories.py --exp-name <name>` directly.
- Truncation is defined exactly as the training code does it:
  `finish_reason != "stop" OR len(response_tokens) >= response_length`.
