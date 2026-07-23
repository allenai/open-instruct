---
name: track-terminal-evals
description: >
  Compile / update the terminal-bench (harbor) eval-results tracking CSV. Use when the user
  wants to add eval experiments to the results file, refresh in-progress evals, pull train-time
  metrics for native checkpoints, dig up existing evals from a workspace, or otherwise maintain
  the growing eval spreadsheet. NOT for launching evals (run-terminal-eval) or analyzing a
  training run (analyze-terminal-rl).
---

# Track terminal-bench eval results

Maintains a single growing CSV of harbor eval results (pass@1/pass@5 + error disaggregation +
optional train-time metrics + clickable Beaker/W&B links), upserted idempotently by `beaker_url`.

## Canonical locations
- **Script:** `/weka/nora-default/shashankg/code/tmax/scripts/beaker/track_evals.py`
- **Default CSV:** `/weka/nora-default/shashankg/code/tmax/scripts/beaker/dppo9b_4n64k_tb21_evals.csv`
  (next to `eval_runs_2026-07-08.md`; pass `--csv` for a different file). Git-untracked.
- **ALWAYS run with the open-instruct uv env** (has `wandb`), per `feedback_use_open_instruct_env_as_base`:
  ```bash
  cd /weka/nora-default/shashankg/code/open-instruct && \
    uv run python /weka/nora-default/shashankg/code/tmax/scripts/beaker/track_evals.py <subcmd> ...
  ```

## Conventions (the judgment the script doesn't hardcode)
- **Only track the user's own runs**: `--author shashankg` (default in `discover`).
- **Default comparison config = TB2.1, k=5, 64k**: gate with `--require-tb21 --require-k 5 --require-maxlen 65536`
  on `add` so mismatches are skipped. The eval **name suffix is unreliable** (new launches get named
  `...terminal-bench-2-0` even when run on TB2.1 via `--dataset-path`) — the script keys off the actual
  `DATASET_PATH`/`N_ATTEMPTS`/`MAX_MODEL_LEN` spec envs, so trust those, not the name.
- **model_name**: for native RL checkpoints use a stable family name (e.g. `swerl-qwen35-9b-dppo-4n64k`)
  + `--step N`; for released/base models the HF path (`allenai/tmax-9b`, `Qwen/Qwen3.5-9B`), no step.
- **Error buckets**: `AgentTimeoutError` = the model's own 120s bash timeouts (model behavior). Everything
  else (`RuntimeError` = vLLM/connection crashes, `Verifier/RewardFileNotFound/BadRequest/...`) = `infra_err`.
  `pass1_infra_adj` = pass@1 recomputed excluding infra-failed trials (upper-ish estimate; assumes infra
  failures are difficulty-random). Released-model runs whose log lacks the exception table (e.g. tmax-2b)
  get blank buckets — flag, don't guess.
- **Train metrics** apply ONLY to native checkpoints (not released models — skip those). Solve-signal is
  noisy per-step; the `_w5` (±5-step smoothed) columns are the informative ones. Keep the set small
  (grp_perf/scores @step + w5, kl2, seq_len).

## Common tasks
```bash
# 1) Discover candidate evals in a workspace, then add the ones you want:
... track_evals.py discover --workspace ai2/oe-agents --filter tb21
... track_evals.py add --exp 01AAA 01BBB --model-name allenai/tmax-9b --require-tb21 --require-k 5 --require-maxlen 65536

# 2) Refresh every row currently status=running (flips running->done, fills scores):
... track_evals.py refresh

# 3) Fill train-time metrics for native-checkpoint rows across a RESUME CHAIN of W&B runs
#    (list earliest-first; earliest run wins on overlapping steps). Also sets wandb_url +
#    train_beaker_url (resolved by scanning workspaces for the wandb id in the exp description):
... track_evals.py train --wandb ut64ii7s gyw0z542 be3l7oy4
```

## Notes / gotchas
- Rows are keyed by `beaker_url` → re-running `add`/`refresh` updates in place, never duplicates,
  and only overwrites with non-empty values (so `train_*` set by `train` survive a later `refresh`).
- `n_trials` assumes 89 TB tasks × k; fine for TB2.1/TBlite-ish. If a dataset has a different task
  count, correct `n_trials`/`error_rate` afterward.
- The wandb resume-chain must be supplied by hand (`--wandb ...`) — each resume = a new W&B run;
  find them from the training beaker experiment descriptions. For THIS run the chain is
  `ut64ii7s`(steps 1-282) → `gyw0z542`(281-366) → `be3l7oy4`(361-440); later resumes append new runs.
- After updating, the CSV is the source of truth; paste into Google Sheets (URL columns auto-linkify).
