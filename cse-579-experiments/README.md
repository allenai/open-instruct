# CSE-579 Experiment Log

One markdown file per real RL training run. Smoke tests and dev-time validation
runs do not get tracked here — only experiments whose results inform the writeup.

## Index

| File | Purpose | Beaker | State | Eval |
|------|---------|--------|-------|------|
| [qwen_4b_base_baseline.md](qwen_4b_base_baseline.md) | No-shaping baseline (Jacob's run) on Qwen3-4B-Base RL-Zero | (Jacob's training; checkpoint on weka) | training complete | retrieved (step_1000) |
| [qwen_4b_base_linear_alpha1.md](qwen_4b_base_linear_alpha1.md) | First length-shaping run on Qwen3-4B-Base RL-Zero, linear α=1.0 | [01KQTJDA…](https://beaker.org/ex/01KQTJDAE5J37VZ0VRXKEHGWTY) | training complete (reward-hacked) | retrieved (step_1000) |
| [qwen_4b_base_linear_alpha1_warmup_linear.md](qwen_4b_base_linear_alpha1_warmup_linear.md) | linear α=1.0 + step-based warmup (frac=0.5) — does ramping the penalty avoid collapse? | [01KSTQDC…](https://beaker.org/ex/01KSTQDCJ9RF60W1Z5885BTEYE) | running | not started |
| [qwen_4b_base_linear_alpha1_warmup_solve_rate.md](qwen_4b_base_linear_alpha1_warmup_solve_rate.md) | linear α=1.0 + latched solve-rate warmup (thr=0.55) — gate penalty on competence | [01KSTR2C…](https://beaker.org/ex/01KSTR2CJ3QF43NYGGKQJ1CPVY) | running | not started |

## Design follow-ups

See [design_followups.md](design_followups.md) for the running list of
shaping/advantage design ideas raised in discussion but not yet implemented
(reward-vs-advantage shaping options, denominator tweaks, the reporting-metric
bug, etc.).

## Results

Fetched eval metrics live under `results/<run_dir>/<task>/` (metrics.json +
length_stats.json per task). They're checked in so anyone reading the .md
docs can audit / re-render the numbers.

```bash
# Fetch results for a set of Beaker lmeval experiments
bash cse-579-experiments/fetch_eval_results.sh <run_dir> <exp_id> [<exp_id> ...]

# Render a markdown summary table from disk
uv run python cse-579-experiments/summarize.py [<run_dir> ...]
```

## Conventions

- **One file per Beaker experiment.** If a run is preempted and restarted, append
  a new section to the same file rather than creating a new one.
- **Filename**: `<model>_<base|sft|dpo>_<shaping_method>_<key_param>.md` —
  e.g. `qwen_4b_base_linear_alpha1.md`, `olmo_7b_sft_exponential_lambda1.md`.
- **Update status promptly** when the run terminates, when evals are retrieved,
  or when you spot something worth recording.
- **Pair every shaping run with a baseline.** When recording the shaping run,
  link the matching no-shaping baseline run (or note that one is needed).

## Template

Copy [TEMPLATE.md](TEMPLATE.md) when adding a new experiment.
