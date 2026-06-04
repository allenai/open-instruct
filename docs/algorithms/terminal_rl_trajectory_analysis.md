# Terminal-RL run analysis: wandb metrics + trajectory error analysis

This is the hands-on companion to [monitoring_and_debugging_runs.md](monitoring_and_debugging_runs.md). Where that doc explains *what every metric means*, this one is the concrete recipe for analyzing a **Terminal / agentic RL** run end-to-end: pull the wandb metrics, then crack open the saved rollouts to find out **why** the model is failing — is it running out of token budget (truncation) or genuinely getting the answer wrong?

The scripts live in [`scripts/general_agent/terminal/rl/analysis/`](../../scripts/general_agent/terminal/rl/analysis/).

## TL;DR — run it

```bash
cd scripts/general_agent/terminal/rl/analysis

# One command: wandb metrics + flags, then trajectory error analysis.
# Reads exp_name / rollouts_save_path / response_length straight from the run config.
uv run python analyze_terminal_rl.py https://wandb.ai/ai2-llm/oe-general-agents/runs/<run_id>

# Faster trajectory pass (sample 96 rollouts/step instead of all):
uv run python analyze_terminal_rl.py <url> --per-step-cap 96

# Also decode a few example trajectory tails so you can eyeball the buckets:
uv run python analyze_terminal_rl.py <url> --decode-examples 1

# The two halves can also be run on their own:
uv run python analyze_wandb.py <url>
uv run python analyze_trajectories.py --exp-name <exp_name> --per-step-cap 96
```

## Part 1 — wandb metrics (`analyze_wandb.py`)

Pulls the keys this repo logs and prints four grouped tables — **learning signal**, **stability / vital signs**, **agent / behavior**, **efficiency / infra** — each showing `first / early / late / min / max / Δ-over-run` per metric. Then it estimates wall-clock (planned steps from `total_episodes / num_unique_prompts_rollout` × observed step time) and emits **automated FLAGS** encoding the heuristics from the monitoring guide, e.g.:

- **Live reward key:** warns when `objective/verifiable_reward` is flat 0.0 but `scores` is non-zero — i.e. reward comes from the *environment* path, not the verifier path (a classic "reward looks broken but isn't" trap; see the monitoring guide §5.1).
- **Learning:** slope of `val/avg_group_performance_pre_filter` (the cleaner signal than the noisy per-token `scores`).
- **Degenerate advantages:** `advantages_min ≈ max ≈ 0` ⇒ no gradient signal.
- **KL:** trend of the stable `objective/kl2_avg`, with a reminder that `beta=0` makes KL monitor-only.
- **Truncation / non-submitting:** budget-bound and protocol failures.
- **Staleness / preemption:** `stale_results_dropped` bursts and the `training_step − model_step` gap vs `async_steps`.
- **Tail regression:** last 3 steps dropping >20% below the preceding window (catches a run that just started falling apart).

It is dependency-light (wandb + numpy) so it runs without importing the training stack.

## Part 2 — trajectory error analysis (`analyze_trajectories.py`)

This is the part that turns "45% of rollouts fail" into "…and of those failures, X% ran out of budget and Y% finished and were wrong" — which point to completely different fixes.

### Where the data comes from

With `--save_traces`, training writes one JSONL record per rollout (see `save_rollouts_to_disk` / `RolloutRecord` in [`open_instruct/rl_utils.py`](../../open_instruct/rl_utils.py)) to `--rollouts_save_path`, sharded as `{run_name}_rollouts_{shard:06d}.jsonl` (10k rollouts/shard). Each record has exactly what we need:

| field | use |
|---|---|
| `step` | group by training step (stored, so it survives restarts) |
| `reward` | solved (`>= 1.0`) vs failed |
| `finish_reason` | `"stop"` = model chose to end; else cut off |
| `response_tokens` | length, to detect the budget wall |
| `request_info.num_calls` | tool calls per rollout |
| `request_info`, `logprobs` | extra context / decoding |

### The truncation definition (must match training)

A rollout counts as **truncated** using the *same* rule the training code uses ([`open_instruct/data_loader.py`](../../open_instruct/data_loader.py)):

```
truncated  ⇔  finish_reason != "stop"   OR   len(response_tokens) >= response_length
```

The second clause matters for **multi-turn agents**: budget can be exhausted *inside* the tool loop (the loop exits without another vLLM call) while the last stored `finish_reason` is still `"stop"` from an earlier turn. Checking only `finish_reason` undercounts truncation. `--response-length` must match the run's `--response_length` (the orchestrator reads it from config automatically).

### What it computes

For every step: solved / zero-reward counts, and within the zero-reward failures the split between **truncated** (budget) and **stopped-but-wrong** (genuine). It also tracks unsolved-response length and tool calls. Then it reports:

1. **Run-instances** — restarts produce new timestamped files that continue the same step counter; it lists each instance's step range and warns about overlaps (a step re-done after a restart).
2. **Trend over training** (binned) — the headline being the **truncation-share-of-failures** slope: *is the model increasingly failing because it runs out of budget (rising) or learning to finish in time (falling)?*
3. **Aggregate** — overall solve rate, truncation rate, the failure decomposition, and whether reward is binary or has partial credit.

### Reading the output — what to look at

- **Truncation ≈ guaranteed failure?** Check `solved` count among truncated rollouts. If truncated rollouts essentially never get reward, every truncation is a wasted rollout.
- **Failure split.** If failures are mostly *truncation*, you're **budget-bound** → raise `response_length` (or reward conciseness / curriculum toward shorter tasks). If mostly *stopped-but-wrong*, that's genuine task difficulty → the model is getting fair shots and losing; that's real learning headroom, not a budget problem.
- **Unsolved length vs the cap.** Unsolved responses pressing the `response_length` cap corroborates budget-bound failures (cross-check with `val/sequence_lengths_unsolved` on wandb).
- **The trend.** A *rising* truncation share means the well-known "RL makes responses longer over time" dynamic is starting to cost you — the conciseness/budget levers get more valuable as training proceeds.

### Why the trace solve rate is flat — and why that is NOT "it's not learning"

This is the single most confusing thing about reading these traces, so here it is from the ground up. The short version: **the traces are a deliberately biased sample of prompts, and the bias is engineered to hold the solve rate near 50% no matter how good the model gets.** Walk through why.

**Step 1 — GRPO only learns from disagreement within a group.** For each prompt we sample K rollouts (here K=8) and the "advantage" of each rollout is its reward *minus the average reward of its group*. That subtraction is the whole trick — it's how GRPO decides "was this rollout better or worse than my other attempts at the same prompt." Now look at what happens to a prompt where **all 8 rollouts get the same reward**:

- All 8 solved (reward 1) → group average is 1 → every advantage is `1 − 1 = 0`.
- All 8 failed (reward 0) → group average is 0 → every advantage is `0 − 0 = 0`.

A zero advantage produces **zero gradient**. So a prompt the model *always* solves, or *never* solves, teaches it **nothing** this step — there's no "do more of this, less of that" signal because all attempts came out identical. The only prompts that carry a learning signal are the **mixed** ones: some of the 8 pass, some fail (a non-zero spread, i.e. non-zero standard deviation). We call all-same groups "zero-std".

**Step 2 — the repo throws the useless groups away.** Two settings act on this:

- `filter_zero_std_samples` literally drops the all-solved and all-failed groups before the gradient step (no point spending compute on zero-gradient data).
- `--active_sampling` goes further: it keeps *generating extra prompts* until it has refilled the batch with enough **mixed** groups. So instead of "generate 32 prompts, train on whatever's left after filtering," it's "keep going until you have 32 *useful* (mixed) prompts to train on."

**Step 3 — therefore the batch that gets trained on (and saved to traces) is a biased subset.** It is *not* a random sample of the dataset — it's specifically the prompts the model is currently getting **partially right** (roughly 1–7 out of 8). That's exactly why our group-consistency table comes out **~100% mixed, 0% all-solved, 0% all-failed**: the all-solved and all-failed prompts were filtered out before they ever reached the trainer. The 0% isn't a measurement of the model — it's a fingerprint of the filter.

**Step 4 — so the trace solve rate is a thermostat, not a thermometer.** Think of active sampling as an **adaptive tutor**: it only hands you problems you're getting right about half the time, because those are the ones you can still learn from. If you get good at a problem, the tutor stops giving it to you and substitutes a harder one you're still shaky on. Your score *on the problems the tutor is currently giving you* therefore stays pinned near ~50% **forever** — not because you're not improving, but because the tutor keeps adjusting the difficulty to keep you in the productive zone. Our ~54% trace solve rate is that pinned number. **A flat trace solve rate is the expected steady state, not evidence of failure to learn.**

**Step 5 — so where does improvement actually show up?** In the prompts that "graduate." Every time the model gets reliably good at a prompt, that prompt flips from *mixed* to *all-solved* and gets filtered out. The metric that counts those is **`val/avg_group_performance_pre_filter`** on wandb — "pre-filter" means it's measured over **all** the prompts the model generated on, *before* the all-solved ones are removed. As the model improves, more prompts land in the all-solved bucket, and this pre-filter number climbs (in our run, 0.30 → 0.45) **even though the post-filter / trace solve rate stays flat**. That gap — pre-filter rising while the trained-batch solve rate is flat — *is* the learning, made visible: the tutor's whole curriculum is getting easier for the student, so it keeps reaching for harder material.

**What to actually trust, then:**

| Question | Look at | Why |
|---|---|---|
| Is the model improving at the task? | `val/avg_group_performance_pre_filter` (wandb), and ideally a held-out **eval** | Measured before the difficulty-filter, so it can move; eval is unfiltered truth |
| Is the trace/`scores` solve rate flat? | (expected) | Held near the informative point *by* active sampling — it's a control signal, not accuracy |
| What is the gradient learning from right now? | the trajectory analysis here (truncation vs genuine) | These traces *are* the trained-on mixed batch — so this split is exactly the right lens for "what's in the gradient," even though the absolute solve rate isn't eval accuracy |

> One caveat to the caveat: all of the above assumes `--active_sampling` is on (it is, in these scripts). **Without** active sampling, the saved traces would include all-solved/all-failed groups, the group-consistency table would no longer be ~100% mixed, and a *growing* all-solved+all-failed share over training would itself be a readable "model is sharpening" signal. The script's group-consistency output detects which regime you're in and prints the matching interpretation.

### Verifying the buckets (`--decode-examples K`)

Don't trust a classifier you haven't eyeballed. `--decode-examples` decodes the last ~260 tokens of K example trajectories per bucket (needs `transformers` + the tokenizer). Sanity check:

- **truncated-zero** should end **mid-`<think>`/mid-tool-call** — cut off, never submitted.
- **stopped-zero** should end with the model having **run the tests and failed** / submitted something wrong / given up.
- **solved** should end with the submit sentinel and passing tests.

## Worked example (Qwen3.5-4B SWE-RL tmax-15k, steps 129–157)

What this tooling surfaced on a real run (full history, steps 0–157, ~40k rollouts across 2 restart instances):

- **Reward is binary** (pass/fail) — no partial credit.
- **Trace solve 54% / fail 46%.** Of failures: **~41.5% truncation, ~58.5% finished-but-wrong.**
- **Truncation ≈ a guaranteed loss** — only ~0.6% of truncated rollouts scored.
- **Unsolved rollouts run ~22k–24k tokens**, pressing the 32,768 cap.
- **Truncation-share-of-failures is slightly RISING** over training (~36% → ~40%) — responses lengthening.
- **Group consistency ~100% mixed** → active sampling is selecting informative groups, so the 54% trace
  solve rate is held flat by selection and is *not* eval accuracy (see ["Why the trace solve rate is
  flat"](#why-the-trace-solve-rate-is-flat-and-why-that-is-not-its-not-learning) above); wandb's
  `val/avg_group_performance_pre_filter` rose 0.30→0.45 over the same span, which *is* the learning signal.
- **Tail regression** at steps 155→157: solve rate ~37%→~20%, truncation up to 50%, lengths ballooning —
  flagged automatically by `analyze_wandb.py`; likely a post-preemption stale batch (step 153 had a
  `stale_results_dropped` burst).

The takeaways: the model is genuinely learning (wandb group-performance rising; most *trained-on* failures
are real attempts, not starvation), a large minority of failures are pure budget exhaustion (and growing),
and the very tail of the run shows a regression worth watching.

## See also

- [monitoring_and_debugging_runs.md](monitoring_and_debugging_runs.md) — the full metric glossary and diagnostic playbook.
- [rollout_loop_internals.md](rollout_loop_internals.md) — how multi-turn rollouts are generated (why budget runs out inside the tool loop).
- [grpo_fast_internals.md](grpo_fast_internals.md) — the async pipeline behind staleness/dropped-rollout metrics.
