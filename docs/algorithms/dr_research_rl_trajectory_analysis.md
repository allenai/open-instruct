# Deep-Research (DR-Tulu) RL run analysis: wandb metrics + trajectory analysis

This is the hands-on companion to [monitoring_and_debugging_runs.md](monitoring_and_debugging_runs.md) for **Deep-Research RL** runs — GRPO with **search/browse tools** and a **continuous evolving-rubric reward**, launched by [`scripts/general_agent/search/rl/rl_qwen35_4b_drtulu.sh`](../../scripts/general_agent/search/rl/rl_qwen35_4b_drtulu.sh). Where that doc explains *what every metric means*, this one is the concrete recipe for analyzing a DR run end-to-end: pull the wandb metrics, then crack open the saved rollouts to find out **what the reward is doing, whether the token budget is the bottleneck, and how the agent is using its tools**.

It is the DR sibling of [terminal_rl_trajectory_analysis.md](terminal_rl_trajectory_analysis.md). If you've read that, the structure is the same — but **almost every interpretation flips**, because DR reward is continuous and the bar moves. The scripts live in [`scripts/general_agent/search/rl/analysis/`](../../scripts/general_agent/search/rl/analysis/).

## TL;DR — run it

```bash
cd scripts/general_agent/search/rl/analysis

# One command: wandb metrics + flags, then trajectory / reward / tool analysis.
# Reads exp_name / rollouts_save_path / response_length / max_possible_score from the run config.
uv run python analyze_dr_research_rl.py https://wandb.ai/ai2-llm/oe-general-agents/runs/<run_id>

# Faster trajectory pass (sample 96 rollouts/step instead of all):
uv run python analyze_dr_research_rl.py <url> --per-step-cap 96

# Also decode a few example trajectory tails per reward bucket:
uv run python analyze_dr_research_rl.py <url> --decode-examples 1

# The two halves can also be run on their own:
uv run python analyze_wandb.py <url>
uv run python analyze_trajectories.py --exp-name <exp_name> --per-step-cap 96
```

## The one thing to internalize first: reward is continuous *and the bar moves*

Terminal-RL reward is binary — a rollout passed the tests or it didn't. DR-Tulu reward is a **rubric score**: a GPT-4.1 judge grades the agent's final answer against a set of per-query criteria ("did it state the program goals?", "did it cite a source?", each with a weight) and returns a score on a `0..max_possible_score` scale (default **10**). So a rollout can earn 0, 10, or anything in between — most earn something in between.

Two consequences drive everything below:

1. **There is no "solve rate."** The headline number is **mean reward** (and where it sits in the zero / partial / full distribution), not a pass fraction. The closest thing to a solve rate is `*_correct_rate` = *fraction of rollouts that earned __any__ reward (> 0)*.
2. **The rubric set evolves.** Every step the judge can mint new criteria for a query (`evolving_rubrics/*`), and the buffer of active rubrics grows. The model is therefore graded against a **stricter bar over time**. A *falling* raw reward can mean the bar rose faster than the model improved — **not** that the model regressed. This is the single biggest trap in reading a DR run, and the analogue of the active-sampling "thermostat" in Terminal-RL: a control loop keeps the headline number from being a clean thermometer.

## Part 1 — wandb metrics (`analyze_wandb.py`)

Pulls the keys this repo logs and prints five grouped tables — **learning signal**, **evolving rubrics**, **stability / vital signs**, **agent / tools / behavior**, **efficiency / infra** — each showing `first / early / late / min / max / Δ-over-run`. Then it estimates wall-clock and emits **automated FLAGS** tuned for DR.

### The reward keys, and why three of them are the same number

`--remap_verifier general_rubric=rubric` routes the evolving-rubric judge's output onto the *verifiable-reward* path, and it's the only reward in play. So:

```
scores  ==  objective/rubric_reward  ==  objective/verifiable_reward      (all on 0..max_possible_score)
val/avg_group_performance_pre_filter  ==  scores / max_possible_score     (the same thing, 0..1)
```

These are **one signal, not five**. (Contrast Terminal-RL, where `objective/verifiable_reward` sits flat at 0 because reward comes from the *environment* path — the opposite trap.) The script's first FLAG states the identity and the scale so you don't go hunting for a discrepancy that isn't there.

### What actually tells you it's learning

| Question | Look at | Why |
|---|---|---|
| Are fewer rollouts failing outright? | **`objective/rubric_correct_rate`** (= frac reward > 0) | The cleanest monotone-ish signal; insensitive to the exact rubric bar — it only asks "did it earn *anything*". Rising = good. |
| Is mean answer quality improving? | `scores` / `val/avg_group_performance_pre_filter`, **read against the rubric counts** | Direct quality, but measured against the *moving* bar — interpret with `evolving_rubrics/num_active_rubrics` & `total_active`. |
| Is the bar moving? | `evolving_rubrics/num_active_rubrics`, `evolving_rubrics/total_active` | Rising = stricter grading over time → a flat/falling raw score may still be progress. |
| Ground truth (best, when present) | a held-out **eval** with a *frozen* rubric set | The only bar that doesn't move. |

The FLAGS encode this: if `avg_group_performance` is falling **while** active rubrics are growing, that's flagged **INFO** ("non-stationary target, not necessarily regression"), not WARN. If `correct_rate` is *also* falling, that's the genuine-regression WARN.

### Evolving-rubric health (the EVOLVING RUBRICS table)

These are reward-*quality* vitals, not model vitals:

- **`evolving_rubrics/valid_rate`** — fraction of judge rubric-generations that parsed into valid rubrics. < 0.9 ⇒ the GPT-4.1 generator is erroring/malforming for a chunk of queries → noisy reward. (FLAG.)
- **`evolving_rubrics/skipped`** — queries whose rubric update was skipped this step (judge errors, empty responses). High ⇒ stale reward for those queries. (FLAG.)
- **`num_active_rubrics`** (per query, capped by `--max_active_rubrics`) and **`total_active`** (buffer-wide) — the bar. **`total_persistent`** is the seed/ground-truth rubric count (constant). `avg_gt_rubrics` is the persistent count per query.
- **`num_overrides`** — ground-truth overrides pushed back to vLLM each step (= number of unique prompts in the batch).

### Budget, behavior, and the `non_submitting` artifact

- **`val/truncated_completion_fraction`** is typically **very high** (0.8–0.95): DR transcripts are long. This is normal — see Part 2 for why it's not the disaster it would be in a binary env.
- **`val/non_submitting_completion_fraction` is pinned at 1.0 and means nothing here.** It is defined as `not rollout_state["done"]`, and the search env never sets `done=True`, so *every* rollout counts as non-submitting. It is a constant artifact, not a protocol failure, and with `mask_non_submitting_completions=False` it doesn't touch training. The script FLAGS this so you don't chase it. (Do **not** confuse it with Terminal-RL's `non_submitting`, which is a real signal there.)
- **`tools/<tool>/failure_rate`** — per-tool error/timeout rate for `google_search` (Serper), `browse_webpage` (Jina), `snippet_search` (S2). A high rate on one tool points at that backend (key/quota/flakiness), not the model. The script FLAGS aggregate failure > 15%.

It is dependency-light (wandb + numpy) so it runs without importing the training stack.

## Part 2 — trajectory / reward / tool analysis (`analyze_trajectories.py`)

This turns "mean reward is 4.4/10" into "…and 28% earn zero, 20% earn full marks, truncated rollouts score 4.27 vs 5.04 for completed ones, and `snippet_search` fails 17% of the time."

### Where the data comes from

With `--save_traces`, training writes one JSONL record per rollout (see `save_rollouts_to_disk` / `RolloutRecord` in [`open_instruct/rl_utils.py`](../../open_instruct/rl_utils.py)) to `--rollouts_save_path`. Each record carries exactly what we need:

| field | use |
|---|---|
| `step` | group by training step (survives restarts) |
| `reward` | the continuous rubric score (0..max) |
| `finish_reason` | `"stop"` vs cut off |
| `response_tokens` | length, to detect the budget wall |
| `prompt_idx` | group rollouts of the same prompt (for within-group spread) |
| `request_info.num_calls` | tool calls per rollout |
| `request_info.tool_call_stats` | **per-call** `{tool_name, success, runtime}` — the per-tool breakdown |
| `request_info.rollout_state.step_count` | turns taken; `done` is always False (see below) |
| `ground_truth` | the query + rubric set (`rubrics`, `rubrics_types` = persistent/evolving) |

### The truncation definition (must match training)

Same rule the training code uses ([`open_instruct/data_loader.py`](../../open_instruct/data_loader.py)):

```
truncated  ⇔  finish_reason != "stop"   OR   len(response_tokens) >= response_length
```

In DR the **second clause dominates**: the multi-turn search/browse loop runs the concatenated transcript to the `response_length` cap (16384) while the last stored `finish_reason` is still `"stop"` from the final model turn. Checking only `finish_reason` would report ~2% truncation; the real figure is ~85%. `--response-length` must match the run (the orchestrator reads it from config).

### What it computes

1. **Reward distribution over training (binned)** — mean reward, zero% / full%, truncation%, **truncated mean reward vs completed mean reward**, avg length, turns, tool calls, tool-error%.
2. **Tool usage (aggregate)** — calls/rollout, failure%, and call-share per tool (`google_search` / `browse_webpage` / `snippet_search`), parsed from `tool_call_stats`.
3. **Aggregate reward** — overall mean (% of max), zero/partial/full split, the truncated-vs-completed gap with an interpretation, and avg calls/turns/length.
4. **Within-group reward spread** — see below.

### Reading the output — what to look at

- **The reward distribution, not a solve rate.** zero% is the share earning *nothing* (mirrors `1 − correct_rate`); full% is the share scoring the max. A healthy run grows full% and/or shrinks zero%, *modulo the moving bar*.
- **Truncated vs completed mean reward — the budget question.** This is the DR replacement for Terminal-RL's "truncation ≈ guaranteed failure". Here truncation is the **norm** and only **mildly** penalized: the rubric scores whatever partial answer exists, so a truncated rollout still earns most of what a completed one would (e.g. 4.27 vs 5.04). The interpretation line tells you which regime you're in:
  - small gap (< ~15% of max) + high truncation ⇒ budget is *not* the main lever; raising `response_length` buys answer *quality*, not a pass/fail flip.
  - large gap ⇒ budget *is* costing reward; raise `response_length` / reward conciseness.
- **Per-tool failure rates.** One flaky backend (e.g. `snippet_search`/S2) shows up as a much higher failure% than the others — that's an *infra* fix (key/quota/retries), and it wastes budget and injects noise into observations.
- **Turns and call mix.** `avg turns` vs `--max_steps`: a spike at the cap means the agent is using its whole budget. The call-share tells you whether it leans on search vs browse vs snippet.
- **The mean-reward trend is reported with fitted-line endpoints** (so the printed numbers match the RISING/FALLING arrow even when the last bin is a noisy 1–2-step bounce), and always with the reminder to cross-check against the rubric-count growth.

### Within-group reward spread — what GRPO actually learns from

GRPO's advantage is `reward − group_mean`, so a group of K rollouts for the same prompt contributes gradient **in proportion to its reward standard deviation**. With binary reward that's the "all-solved / all-failed / mixed" story; with **continuous** reward it's a *spread*, so the script reports **mean within-group reward std** (the gradient magnitude, on the 0..max scale) and the **zero-std share** (groups where every rollout scored identically → ~no gradient).

The same active-sampling caveat as Terminal-RL applies: with `--active_sampling` + `filter_zero_std_samples`, the *saved* (trained-on) batch is pre-selected to be **informative**, so the zero-std share here is ~0% **by construction** — it confirms the selection is working, it is **not** the model's generation-time accuracy. For true task progress against the unfiltered population, read `val/avg_group_performance_pre_filter` on wandb (interpreted against the moving rubric bar). See ["Why the trace solve rate is flat"](terminal_rl_trajectory_analysis.md#why-the-trace-solve-rate-is-flat-and-why-that-is-not-its-not-learning) in the Terminal doc for the full walk-through of this thermostat dynamic — it transfers directly.

### Verifying the buckets (`--decode-examples K`)

Don't trust a classifier you haven't eyeballed. `--decode-examples` decodes the last ~300 tokens of K examples per reward bucket (zero / partial / full), with a per-tool + rubric summary line (needs `transformers` + the tokenizer). Sanity check:

- **zero** — the answer missed every rubric criterion (off-topic, no real answer, or cut off before answering).
- **partial** — hit some criteria, missed others; this is the bulk and the productive learning zone.
- **full** — addressed every active rubric for the query.

## Worked example (Qwen3.5-4B DR-Tulu, steps 0–26, run `m8xd8yr4`)

What this tooling surfaced on the early window of a real run (~5.4k rollouts, single instance, ~2h in, reward scale 0–10):

- **Reward is continuous:** mean **4.39/10**; **28.3% zero**, 52.1% partial, **19.7% full**. `*_correct_rate` ≈ 0.70 (≈ 1 − zero%).
- **The bar is rising fast:** `num_active_rubrics` 3.6 → 6.4 per query, buffer `total_active` **43 → 664**. So…
- **…raw reward falls but it's the moving target, not regression:** `avg_group_performance` 0.48 → 0.35 (`scores` 4.8 → 3.1) while `correct_rate` stays ~0.70. Flagged **INFO** (non-stationary target), corroborated by the trajectory mean-reward trend reported against the evolving bar. The cleaner "is total failure shrinking" signal (`correct_rate`) is roughly flat over this very early window — verdict: *too early to call learning; watch `correct_rate` and a frozen-rubric eval*.
- **Massively budget-bound but barely penalized:** **84% truncated**, yet truncated rollouts score **4.27** vs **5.04** completed — a <8%-of-max gap. So budget isn't a pass/fail wall here; raising `response_length` would buy quality, not flips.
- **One flaky tool:** `snippet_search` (S2) fails **17.4%** of calls vs 3.7% (`google_search`) and 1.8% (`browse_webpage`) — an infra lever (the aggregate tool failure rate also drifts up over the window).
- **Stable optimization:** KL tiny (beta=0.001, in the loss), clipfrac 0 (DAPO), grad-norm ~2–4, advantages a wide ±8 spread (continuous reward → healthy gradient).
- **`non_submitting` = 1.0 throughout** — the constant artifact, correctly ignored by a FLAG.

The takeaways: the run is healthy and stable but **too young to declare learning**; read `correct_rate` and a frozen-rubric eval rather than the raw (moving-bar) reward; the dominant "failure" mode (truncation) is mostly cosmetic for reward; and the most actionable concrete fix is the flaky `snippet_search` backend.

## See also

- [terminal_rl_trajectory_analysis.md](terminal_rl_trajectory_analysis.md) — the binary-reward sibling; read it for the active-sampling thermostat walk-through, which transfers directly.
- [monitoring_and_debugging_runs.md](monitoring_and_debugging_runs.md) — the full metric glossary and diagnostic playbook.
- [rollout_loop_internals.md](rollout_loop_internals.md) — how multi-turn tool rollouts are generated (why budget runs out *inside* the search/browse loop).
- [grpo_fast_internals.md](grpo_fast_internals.md) — the async pipeline behind staleness/dropped-rollout metrics.
