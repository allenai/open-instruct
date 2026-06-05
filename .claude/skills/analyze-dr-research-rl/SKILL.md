---
name: analyze-dr-research-rl
description: Analyze a Deep-Research (DR-Tulu) RL training run — GRPO with search/browse tools and a continuous evolving-rubric reward — and report health + failure-mode insights. Use when the user gives a wandb run URL (or exp_name) for a DR-Tulu / Deep-Research / search-agent RL run (launched by rl_qwen35_4b_drtulu.sh) and asks to check it, see if it's learning, diagnose it, or asks "does anything look suspect". Pulls wandb metrics AND analyzes saved trajectories (reward distribution, budget, per-tool usage).
---

# Analyze a Deep-Research (DR-Tulu) RL run

When the user points you at a Deep-Research / DR-Tulu / search-agent RL run (a
wandb URL like `https://wandb.ai/<entity>/<project>/runs/<id>`, or an `exp_name`),
run the analysis tooling and report insights — don't make them ask for each piece.
These runs are launched by
[`scripts/general_agent/search/rl/rl_qwen35_4b_drtulu.sh`](../../../scripts/general_agent/search/rl/rl_qwen35_4b_drtulu.sh).

This is the **DR sibling** of `analyze-terminal-rl`. Use *this* skill when reward
is a **continuous evolving-rubric score** and tools are **search/browse**; use
`analyze-terminal-rl` for **binary** pass/fail sandbox/SWE runs. Almost every
interpretation differs — don't apply the binary-env playbook here.

## Instructions

1. **Run the one-shot analyzer** from the repo root:
   ```bash
   cd scripts/general_agent/search/rl/analysis
   uv run python analyze_dr_research_rl.py <wandb-url> --per-step-cap 96
   ```
   - It prints (a) wandb metric tables (learning / evolving-rubric / stability /
     tools / infra) + automated FLAGS, then (b) the trajectory reward/tool analysis.
   - `--per-step-cap 96` keeps the multi-GB trajectory scan fast; drop it for an
     exact full pass, or add `--skip-trajectories` for a wandb-only quick look.
   - The trajectory scan can take a few minutes (multi-GB shards); cap harder or run
     it in the background and keep working.
   - `WANDB_API_KEY` is usually already set; `wandb` is a `uv` dep (use `uv run`).

2. **To eyeball example trajectories**, add `--decode-examples 1` (decodes the tails
   of one zero / partial / full-reward rollout, with a per-tool + rubric summary;
   needs the tokenizer, default `Qwen/Qwen3.5-4B`).

3. **Interpret and report** using the framing in
   [`docs/algorithms/dr_research_rl_trajectory_analysis.md`](../../../docs/algorithms/dr_research_rl_trajectory_analysis.md)
   and the glossary in `docs/algorithms/monitoring_and_debugging_runs.md`. Lead with
   "is it learning?" then surface the flags. Specifically:
   - **Reward is continuous, not pass/fail.** Headline = **mean reward** (on the
     0..`max_possible_score` scale, default 10) and the zero/partial/full split — not a
     solve rate. `scores` == `objective/rubric_reward` == `objective/verifiable_reward`
     (the rubric judge IS the verifier via `--remap_verifier`), and
     `val/avg_group_performance_*` = that score / max. Treat them as ONE signal.
   - **The bar moves.** Rubrics evolve (`evolving_rubrics/num_active_rubrics`,
     `total_active` grow), so a **falling raw reward can be the stricter bar, not
     regression**. The cleanest learning signal is **`objective/rubric_correct_rate`**
     (= fraction of rollouts with reward > 0); cross-check it and, if available, a
     **frozen-rubric eval**.
   - **Truncation ≠ failure.** DR is heavily budget-bound (~80–95% truncated) but
     truncated rollouts still earn most of the reward (the rubric scores the partial
     answer). Report the **truncated-vs-completed mean reward gap** — small gap ⇒
     `response_length` buys quality, not pass/fail.
   - **`non_submitting_completion_fraction` == 1.0 is a constant artifact** (`not
     rollout_state["done"]`; the env never sets done). Ignore it — do NOT report it as
     a protocol failure.
   - **Per-tool failure rates** (`google_search`/Serper, `browse_webpage`/Jina,
     `snippet_search`/S2): a single flaky backend is an infra lever, not a model issue.
   - **Rubric quality:** `evolving_rubrics/valid_rate` < 0.9 or high `skipped` ⇒ noisy
     reward from the GPT-4.1 judge/generator.
   - **beta=0.001** ⇒ KL IS in the loss (small); also watch grad-norm, clipfrac (DAPO),
     advantage spread, `stale_results_dropped` bursts, staleness gap, throughput/ETA,
     and tail regression.

4. **Offer the deeper drills** (don't run unprompted unless the user wants depth):
   uncapped full trajectory pass, decoding more examples from a specific step,
   distinguishing a stale post-restart batch from a real regression, or — for
   *qualitative* depth — `inspect_trajectories.py` (see below) to read a full
   multi-turn transcript or break reward down by rubric composition.

## Notes

- Scripts: `analyze_dr_research_rl.py` (orchestrator), `analyze_wandb.py` (metrics +
  flags), `analyze_trajectories.py` (aggregate reward/tool/trajectory stats),
  `inspect_trajectories.py` (qualitative: `--view` renders one rollout's full
  multi-turn transcript — reasoning, tool calls, results, answer — under the rubric
  header; `--rubrics` gives a structural + correlational per-rubric reward analysis,
  since the judge only persists the weighted-average score, not per-rubric verdicts).
  See the directory `README.md`.
- The orchestrator reads `exp_name`, `rollouts_save_path`, `response_length`, and
  `max_possible_score` from the wandb run config, so a URL is enough. If only an
  `exp_name` is given, run `analyze_trajectories.py --exp-name <name>` directly (pass
  `--response-length` and `--max-score` to match the run).
- Truncation is defined exactly as the training code does it:
  `finish_reason != "stop" OR len(response_tokens) >= response_length`.
