---
name: track-terminal-evals
description: >
  Maintain the single terminal-bench eval-results tracking sheet (TB2.1 + TBlite in one CSV).
  Use when the user wants to add a model/checkpoint's eval results, refresh in-progress evals,
  or otherwise update the tracking spreadsheet. NOT for launching evals (run-terminal-eval) or
  analyzing a training run (analyze-terminal-rl).
---

# Track terminal-bench eval results

ONE self-contained sheet + ONE tool. Each row = a model/checkpoint with BOTH benchmarks side by
side; each row remembers its own eval-experiment URLs, so scores are (re)extracted directly from
Beaker — no intermediate per-benchmark sheets.

## Canonical files (in `/weka/nora-default/shashankg/code/tmax/scripts/beaker/`)
- **Sheet: `terminalbench_combined_evals.csv`** — one row per model/checkpoint. Columns:
  `model_name, step, max_model_len,` `tb21_pass@1/pass@5/pass1_adj/err_rate/beaker_url,`
  `tblite_pass@1/pass@5/pass1_adj/err_rate/beaker_url,` `train_grp_perf_w5, train_kl2, train_seq_len,
  wandb_url, train_beaker_url, workspace`.
- **Tool: `combined_evals.py`** (`refresh`, `add`). **Run with the open-instruct uv env**
  (per [[feedback_use_open_instruct_env_as_base]]):
  ```bash
  cd /weka/nora-default/shashankg/code/open-instruct && \
    uv run python /weka/nora-default/shashankg/code/tmax/scripts/beaker/combined_evals.py <cmd>
  ```
- **Row order (fixed, applied on every save):** raw Qwen (Qwen3 → Qwen3.5) · published Tmax SFT ·
  local SFT (small → big; Qwen3 → Qwen3.5) · published Tmax RL · RL checkpoints by step.

## The two commands
```bash
# fill scores for any row that has a beaker_url but no pass@1 yet (run after evals finish).
# --force re-extracts every row (idempotent). Only touches eval scores; train_* are carried as-is.
combined_evals.py refresh [--force]

# add (or update) one model/checkpoint row, then auto-refresh it:
combined_evals.py add --model <NAME> [--step N] --max-len <32768|65536> --workspace ai2/oe-agents \
    --tb21 <TB2.1_eval_exp_id> --tblite <TBlite_eval_exp_id> \
    [--wandb <url> --train-exp <train_beaker_id> --grp-w5 X --kl2 X --seq-len X]   # train metrics: RL/SFT ckpts only
```

## How scores are extracted (so you can trust/debug it)
- Reads the scoring job's log for each URL: `pass@1`, `pass@5`, the exception-count table, and the
  `X/Y trials` count. `AgentTimeoutError` = model failure; everything else (`RuntimeError` = vLLM/conn,
  `Verifier*`, `RewardFileNotFound`, …) = infra. `err_rate = total/n_trials`;
  `pass1_adj = pass@1·n_trials/(n_trials−infra)` (drops infra-failed trials; blank if no exception table).
- **n_trials**: read from the log; TB2.1 = 89 tasks, TBlite = 100 tasks (× k, usually 5).
- Edge case: a few early logs emit only a 2-line summary (no exception table) → tool falls back to the
  `errors=N` progress line for the total and leaves the agent/infra split + `pass1_adj` blank.

## Conventions / gotchas
- Track only the user's own runs. Eval **names are unreliable** (a TB2.1 run can be named
  `...terminal-bench-2-0`) — identity comes from the launch config, not the name.
- Standard configs: released/base + RL = 64k; SFT = 32k; k=5. TBlite eval launches use
  `--dataset openthoughts-tblite@2.0` (registry, 100 tasks); TB2.1 uses `--dataset-path .../terminal-bench-2-1`.
- The sheet is git-tracked in tmax; the per-benchmark source sheets + old builders
  (`track_evals.py`, `build_tblite_sheet.py`, `build_combined.py`) were retired — do not recreate them.
- Launch commands for the eval + training jobs live in `eval_runs_2026-07-08.md` (same dir).
