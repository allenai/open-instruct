Get AIME 2024/2025 primary scores (pass@1) from Beaker for one or more model runs, broken down by training step.

## Arguments
$ARGUMENTS - One or more model path identifiers (the checkpoint directory names, e.g. `vip_grpo_base_p32_2403_qwen3_4b_math__1__1774385112`), separated by newlines or spaces.

## Process

1. Fetch all experiments from the `ai2/tulu-3-results` Beaker workspace matching `on-aime` in the name:
   ```
   beaker workspace experiments ai2/tulu-3-results --text "on-aime" --format json
   ```
   Cache the result to `/tmp/aime_all_exps.json` to avoid re-fetching.

2. For each model identifier provided, filter experiments whose spec JSON contains the model identifier string. Extract:
   - Step number from the experiment name (regex `step_(\d+)`)
   - Year (2024 or 2025) from the `"years"` field in the spec
   - Experiment ID
   - Status (succeeded or not)

3. For each succeeded experiment, download its metrics:
   ```
   beaker experiment results <exp_id> --output /tmp/aime_results/<exp_id> --prefix metrics
   ```
   Read `metrics.json` and extract `primary_score` from each task entry. Use the alias to determine year (look for `2024` or `2025` in the alias string).

4. Output results as comma-separated values (CSV), one table per model, suitable for pasting into Google Sheets:
   ```
   Step,AIME 2024 pass@1,AIME 2025 pass@1
   1,0.0750,0.0490
   100,0.2021,0.1677
   ...
   ```

5. Note any experiments that are still running or failed.
