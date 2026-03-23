---
name: fetch-beaker-evals
description: Fetch AIME or other evaluation scores from Beaker for training runs. Use when user asks for eval scores, AIME results, or wants to compare training run performance.
allowed-tools: Bash(beaker *), Bash(uv run python *), Bash(cat *), Bash(ls *), Write, Read
argument-hint: "<run_name_1> <run_name_2> ... [--task aime] [--format csv]"
---

# Fetch Beaker Eval Scores

Fetch evaluation metrics from Beaker experiments in the `ai2/tulu-3-results` workspace for the specified training runs.

## How it works

Eval experiments are stored in the `ai2/tulu-3-results` Beaker workspace (ID: `01J4AD2DDXB4KHRX2ZYDK3S10P`). Experiment names follow the pattern `lmeval-None_step_{N}-on-{task}-{hash}`. The `model_config.model_path` field in each experiment's `metrics.json` contains the training run name, which is how we match experiments to runs.

## Steps

1. Parse the run names from `$ARGUMENTS`. Run names look like `vip_ppo_gt_base_2303_qwen3_4b_math__1__1774291333`.

2. Write and execute a Python script that:
   - Uses the Beaker API to paginate through experiments in the workspace: `beaker api "workspaces/01J4AD2DDXB4KHRX2ZYDK3S10P/experiments?limit=200&name_prefix=lmeval-None_step" --format json`
   - Pagination: use the `nextCursor` field from each response to get the next page. Paginate up to 10 pages (2000 experiments) to cover recent runs.
   - Filter experiments whose names contain the eval task (default: `aime`).
   - Download results in parallel using `concurrent.futures.ThreadPoolExecutor(max_workers=20)`:
     ```
     beaker experiment results <exp_id> --output /tmp/beaker_aime/<exp_id>
     ```
   - Read `metrics.json` from each result: `/tmp/beaker_aime/<exp_id>/main/metrics.json`
   - Match `model_config.model_path` against the requested run names.
   - Extract `all_primary_scores` field for the scores.
   - Extract the step number from the experiment name: `lmeval-None_step_{N}-on-...`
   - Cache results: skip downloading if `/tmp/beaker_aime/<exp_id>/main/metrics.json` already exists.

3. Print results grouped by run, sorted by step. Show both AIME 2024 and 2025 scores.

4. If `--format csv` is specified, also output a comma-separated table suitable for pasting into Google Sheets, with columns for each run and rows for each step. Use this layout:
   ```
   ,AIME 2024,,,AIME 2025,,
   Step,Run1,Run2,Run3,Run1,Run2,Run3
   1,score,score,score,score,score,score
   ```

## Example usage

```
/fetch-beaker-evals vip_ppo_gt_base_2303_qwen3_4b_math__1__1774291333 vip_grpo_base_2103_qwen3_4b_math__1__1774131604
```

## Important notes

- The workspace has 388k+ experiments. NEVER try to list all of them. Always use `name_prefix` filtering and limit pagination.
- Each eval experiment has 2 scores: AIME 2024 and AIME 2025 (identified by `2024` or `2025` in the score string).
- The `all_primary_scores` field is a list of strings like `"aime:zs_cot_r1::pass_at_32_2025_rlzero: 0.270833"`.
- Some experiments may have failed or have missing metrics. Skip those silently.
- Always present results as a clean table. Default to CSV format for easy Google Sheets pasting.
