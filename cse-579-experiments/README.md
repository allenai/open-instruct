# CSE-579 Experiment Log

One markdown file per real RL training run. Smoke tests and dev-time validation
runs do not get tracked here — only experiments whose results inform the writeup.

## Index

| File | Purpose | Beaker | Status |
|------|---------|--------|--------|
| [qwen_4b_base_linear_alpha1.md](qwen_4b_base_linear_alpha1.md) | First length-shaping run on Qwen3-4B-Base RL-Zero, linear α=1.0 | [01KQTD4D…](https://beaker.org/ex/01KQTD4DJ57C1SY8A1MFNS3GFC) | running |

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
