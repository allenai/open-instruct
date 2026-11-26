# Paper Follow-Up TODOs (for Tech Report Section 4.1)

Actionable follow-up experiments and analyses identified while drafting the RL-zero section of the Olmicro technical report (`sections/post_training_results.tex` and `sections/appendicies/rl_zero.tex`). Ordered by impact per unit compute.

> **Paper figures:** `figures/make_rl_zero_figures.py` in the Overleaf repo generates `figures/rl_zero_scaling.pdf` and `figures/rl_zero_offline.pdf` from local CSV snapshots (`figures/rl_zero_in_loop_evals.csv`, `figures/rl_zero_minerva_aime_evals.csv`). After any new run below, re-run `tools/gather_results.py`, copy the updated CSVs into `figures/`, and regenerate both PDFs via `uv run --with matplotlib python3 figures/make_rl_zero_figures.py`.

---

## P1 — Highest value, low compute

### 1. Run 810M combined pool with online zero-variance filtering (`filter_zero_std_groups = true`)
- **Why it matters for the paper:**
  - Right now 810M is the only sub-1B model trained without `filter_zero_std_groups = true`, even though only 19–35% of groups on its combined pool carry reward signal (`tables/signal_fraction.csv`).
  - Enabling zero-variance filtering lifted 450M from 12.1 → 14.8 MATH-500 (2.9× base) and 275M from 4.8 → 6.3, whereas 810M without it went 10.5 → 22.2 (2.1× base).
  - Running this both checks whether 810M improves further and simplifies the paper's headline table (Table 1) from three recipes into a clean two-regime split: **≥1.4B (own pass band)** vs. **≤810M (combined pool + online zero-variance filter)**.
- **How to run:**
  - Copy `configs/olmicro-810m-lc-math-rl-refresh-32k-combined-20261119.toml` to a new config (e.g. `olmicro-810m-lc-math-rl-refresh-32k-combined-zstd-<date>.toml`).
  - Change only:
    - `name`, `[output] root`, and `[tracking] wandb_group`
    - `[training] filter_zero_std_groups = true`
  - Validate, preflight, and submit via `tools/submit_excluding.py --exclude holmes-cs-aus-511.reviz.ai2.in` on 1 × 8-B300 node (2 trainer + 6 inference GPUs, identical to the 450M/275M zstd runs).
- **Estimated cost:** ~3–4 h on 1 × 8-B300 node (the unfiltered 810M combined run took 1.73 h; ~20–35% keep rate means ~3–5× generation early, shrinking as signal rises).
- **Status (2026-11-24): done.** The run ran from 2:42 to 5:24 PM PST (2.7 h) as Beaker experiment `01M87VE3DZ6BV3EG17N4G37V9C`; the config is `configs/olmicro-810m-lc-math-rl-refresh-32k-combined-zstd-20261124.toml`.
  - **Outputs:** all 400 updates; HF export at `/weka/oe-adapt-default/jacobm/olmicro/runs/olmicro-810m-lc-math-rl-refresh-32k-combined-zstd-20261124/export-hf`. Core checkpoints were kept every 50 updates.
  - **Result: it beats 22.2 and becomes the 810M headline run.**
    - MATH-500 goes 11.3 → **23.8** (peak 24.0 at u150), against 10.5 → 22.2 (peak 24.0 at u350) without the filter.
    - GSM8K Platinum goes 32.5 → **50.6**, against 47.7 without the filter.
    - MATH-500 by update (0–400 every 50): 11.3, 14.0, 20.4, 24.0, 23.0, 23.2, 22.9, 22.5, 23.8. The run rises much faster, then plateaus from u150.
  - **Caveat:** the MATH-500 gain (+1.6) is close to eval noise. The same base scored 10.2 / 10.5 / 11.3 in the initial evals of the three 810M runs, and the ratio to base is the same 2.1×. The GSM8K gain (+2.8) and the faster rise are clearer.
  - **Signal and lengths:** 21% of groups carry signal at updates 0–49 and 37% at 350–399. Mean length goes 1.27k → 0.62k, the same shrinking as the other ≤810M runs.
  - **Cost:** 1.55× the unfiltered wall time (2.7 h vs 1.7 h). It generates 48 responses/s and trains on 17/s (`tables/throughput.csv`).
  - **Recipe split:** the two-regime split now holds: **≥1.4B** trains on its own pass band, and **≤810M** on the combined pool with the zero-std filter. The results-branch README and `tables/` are updated.
  - **Still to do on the paper side (Overleaf):** update `tab:rl-scaling`, `tab:rl-small-models` and `tab:rl-all-runs` and regenerate `figures/rl_zero_scaling.pdf`. Use 2.7 h for the GPU-hours.
- **Post-run:** Re-run `tools/gather_results.py` to refresh `tables/runs.csv`, `tables/in_loop_evals.csv`, and `tables/signal_fraction.csv`. If final MATH-500 beats 22.2, update Tables `tab:rl-scaling` and `tab:rl-small-models` in `sections/post_training_results.tex`, Table `tab:rl-all-runs` in `sections/appendicies/rl_zero.tex`, and regenerate `figures/rl_zero_scaling.pdf` (also update the GPU-hours range in Section 4.3 and Section 1 if wall time changes).

---

## P2 — Zero GPU cost (CPU-only analysis on Weka)

### 2. Extract training response-length trajectories for all five sizes
- **Why it matters for the paper:**
  - We currently know from `run-log.md` that 2.7B (~10k tokens) and 1.4B (~8.5k tokens) grow long chains of thought during RL, while 810M on its own band shrank from ~1.3k to ~500 tokens.
  - We do **not** have the training response-length curves in `tables/` for `olmicro-810m-lc-math-rl-refresh-32k-combined-20261119`, `olmicro-450m-lc-math-rl-refresh-32k-combined-zstd-20261120`, and `olmicro-275m-lc-math-rl-refresh-32k-combined-zstd-20261120` (nor for 810M zstd once run).
  - Pulling mean/median response length per 50-update block for every headline run lets us state cleanly whether long-CoT length growth emerges strictly at ≥1.4B or also appears at ≤810M under the combined + zstd recipe, and optionally add a length panel to the paper figures.
- **How to run:**
  - Run `tools/length_by_version.py <RUN_NAME> 0 50 100 150 200 250 300 350` on Weka (or extend `tools/gather_results.py` to emit a `tables/response_lengths.csv` with mean, median, p90, and truncation rate per 50-update bin for every run in `tables/runs.csv`).
- **Estimated cost:** CPU-only, <5 min.
- **Status (2026-11-24): done.** `tools/gather_results.py` now writes `tables/response_lengths.csv`, which has a row for each run and each 50-update policy-version bin.
  - **Columns:** response count, mean / median / p90 tokens, truncation %, correct %.
  - **Subsets:** `subset = all` covers every generated group. `subset = trained` exists only for zero-std runs and drops the groups the filter removed.
  - **Excluded:** groups finished under the post-update-399 weights. They were drained at shutdown, never trained on, and skew long.
  - Mean tokens below are for updates 0–49 → 350–399; medians are in parentheses.
  - **Long-CoT growth appears only at ≥1.4B, and the corrected numbers are larger than the ~10k / ~8.5k above:**
    - **2.7B 32k:** 3.4k → 14.6k (0.9k → 14.5k), truncation ≤0.8%.
    - **1.4B own band:** 1.3k → 8.9k (0.66k → 8.9k).
    - **1.4B combined:** 1.5k → 10.1k.
    - The jump starts at updates 50–99 for the 2.7B and around 150–199 for the 1.4B.
    - **2.7B 8k cap:** rises to 5.5k by updates 100–149 (42% truncated). The cap then pushes it back to 3.9k (2.5% truncated).
  - **≤810M, no growth under any recipe:** the median stays flat or shrinks, and the mean drops as the long tail of rambling, truncated responses disappears.
    - **810M own band:** 1.3k → 0.59k (0.61k → 0.43k).
    - **810M combined:** 1.2k → 0.54k (0.59k → 0.47k).
    - **450M combined + zstd:** 1.2k → 0.63k (0.43k → 0.48k).
    - **275M combined + zstd:** 1.7k → 0.57k (0.57k → 0.37k).
    - **Without zstd,** the 450M / 275M means stay at ~1.0–1.1k at the end and medians are flat (~0.55k). Zero-std filtering mainly removes the long tail at these sizes.

---

## P3 — Optional follow-ups (nice to have, not blockers)

### 3. Larger `pass@k` evaluation (base vs. RL checkpoints)
- **Why it could help:** Currently AIME 2026 is evaluated at $n=32$ samples per problem (showing 2.7B pass@32 rising from 16.7% to 36.7%, peaking at 50.0% at step 350). Testing larger $k$ (e.g., $n=128$ or $256$) on matched R1-template / 32k-limit prompts would let us directly check whether RL expands coverage at large $k$ or whether the base model catches up (the Yue et al. 2025 vs. Wen et al. 2025 question).
- **How to run:**
  - Cheapest check: `AIME 2026` at $n=256$ samples per problem ($30 \times 256 = 7,680$ rollouts, ~15–25 min on 1 node per checkpoint) for `2.7B base` and `2.7B 32k final` (and optionally `1.4B base` / `1.4B final`) via `tools/format_probe.py --eval-files` + `tools/score_extra_evals.py`. Note that with 30 problems, 1 problem = 3.3% granularity.
  - Lower-variance check: `MATH-500` at $n=128$ or $256$ under the same R1 template / 32k limit for `2.7B base` vs. `2.7B 32k final` (and optionally `1.4B`).
- **Estimated cost:** ~0.5–1 node-hour for AIME $n=256$; a few node-hours for MATH-500 $n=256$.

### 4. General-capability and long-context retention check on 2.7B RL-zero (32k)
- **Why it could help:** Evaluates whether math RL-zero preserves or degrades Olmicro 2.7B's general base-suite scores (MC STEM/non-STEM, Gen QA, Code) and 64k long-context scores (HELMET / RULER-10).
- **How to run:** Run the existing `olmo-eval` base and long-context evaluation suite on `/weka/oe-adapt-default/jacobm/olmicro/runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-4x12-20261118/export-hf`.
- **Estimated cost:** Standard single-model eval job.

### 5. Continue 2.7B 32k past 400 updates (or targeted add-back of newly solved prompts)
- **Why it could help:** Unlike the 275M and 450M zero-std runs (which largely plateaued by updates 250–300: 275M peaked 6.5 at u300 → 6.3 at u400; 450M went 14.1 → 14.4 → 13.9 → 14.8), the **2.7B 32k run was still climbing steadily at update 400** (MATH-500: 73.9 → 74.9 → 76.1 → 77.3).
- **Two options:**
  1. Resume from `rollout_0000399` for another 200–400 updates on the same 4,900-prompt band to see where 2.7B peaks.
  2. Sample the 2.7B u400 HF export on the 8,350 base-unsolved prompts (~1–2 node-hours), add newly solvable prompts to the pool, and continue training (testing an iterative curriculum at 2.7B).
- **Note:** Only worth running if we want a stronger final 2.7B checkpoint; keeping all five sizes at a matched 400-update budget is already clean for the tech report.

### 6. Online zero-variance filtering at 1.4B (combined pool)
- **Why it could help:** Optional check on whether `1.4B combined + zero-std` beats `1.4B own band` (49.9 MATH-500), which would allow using a single unified `combined + zero-std` recipe across all sizes ≤1.4B. Low priority since 1.4B own-band already has 51–77% signal and did not overfit.

### 7. Inference throughput (`rollouts/sec` / tokens/sec) summary for Section 4.3
- **Why it could help:** Yashas asked in Section 4.3 whether we can summarize `rollouts/sec` (and optionally compare against external small models) so readers have concrete throughput numbers alongside B300 GPU-hours.
- **How to run:** First extract steady-state `rollouts/sec` and generated `tokens/sec/GPU` per size from Weka/W&B logs of our existing runs (CPU-only). Optionally run a short throughput benchmark (e.g., 20 steps or standalone SGLang rollout sweep at 32k) comparing an Olmicro size against a parameter-matched pure transformer (e.g., Qwen3 1.7B/4B) to highlight the 4:1 hybrid throughput advantage.
- **Status (2026-11-24): CPU part done.** `tools/gather_results.py` now writes `tables/throughput.csv`, covering updates 100–399 of each run.
  - **Where the numbers come from:** the end-to-end rates are everything generated (filtered groups included) divided by wall time, so in-loop evals, checkpoint saves and weight syncs are included. The decode rates are SGLang's per-engine `gen throughput` log lines; engines are TP1, so per engine means per GPU.
  - **Headline runs:**

    | Run (trainer + inference GPUs) | Responses/s | Generated tok/s (per inference GPU) | s/update median (mean) |
    |---|---|---|---|
    | 2.7B 32k (4+12) | 2.97 | 36.0k (3.0k) | 74 (86) |
    | 2.7B 8k (4+4) | 6.28 | 27.8k (7.0k) | 32 (41) |
    | 1.4B own band (4+4) | 4.73 | 29.4k (7.3k) | 44 (54) |
    | 810M combined (2+6) | 22.2 | 16.6k (2.8k) | 4.9 (11.7) |
    | 810M combined + zstd (2+6) | 48.2 generated, 17.1 trained | 29.4k (4.9k) | 10.1 (15.0) |
    | 450M combined + zstd (2+6) | 30.7 generated, 8.2 trained | 25.4k (4.2k) | 19 (31) |
    | 275M combined + zstd (2+6) | 31.4 generated, 6.6 trained | 29.2k (4.9k) | 28 (39) |

  - **Zero-std filter cost:** the filter generates 2.8× more responses than it trains at 810M, and 3.7–4.7× more at 450M/275M.
  - **Caveat before quoting these numbers:** they are what our RL recipe achieved, not hardware limits.
    - **Inference is concurrency-bound.** `in_flight = 512` is split across the engines, and per-GPU decode rate tracks requests per engine. The 4-engine runs average 74–92 requests per engine and decode 6.8–10.7k tok/s per GPU. The 6- and 12-engine runs average 21–34 requests and decode 3.6–6.0k tok/s per GPU.
    - **Sub-1B runs without the filter are trainer- and eval-bound.** Their median is 4–11 s/update but their mean is 10–20 s; the gap is mostly the eval and checkpoint pauses every 50 updates.
    - **For an architecture claim** (e.g. Olmicro vs. Qwen3 throughput), run the optional standalone SGLang sweep at fixed high concurrency rather than quoting these numbers.

### 8. Base `pass@k` vs. post-RL predictability (cross-model / ladder check)
- **Why it could help:** Yashas asked in Section 4 whether base `pass@1` or `pass@k` better predicts post-RL accuracy. Across our 5 Olmicro sizes alone, base MATH `pass@1` and `pass@64` are collinear ($r = 0.98$), so separating them requires running matched RL-zero on external models (e.g., Qwen3 / Qwen3.5) or architecture-ladder checkpoints that differ in `pass@k` at similar `pass@1`. Only pursue if co-authors want a cross-model RL comparison.

---

### 9. ChatML prompt ablation at 2.7B, 8k cap (added 2026-11-25, Michael's suggestion) — done
- **What it tests:** whether RL-zero still works if the base model is prompted in the SFT-style ChatML format instead of R1.
  - **Prompt:** the format probe's `chatml` template. There is no system turn; the user turn is the question plus "Please reason step by step, and put your final answer within \boxed{}."
  - **Data:** the same 4,353-prompt 8k band and the same recipe as the R1 8k run.
  - **Run:** `olmicro-2.7b-lc-math-rl-v2-refresh-8k-chatml-20261125`, Beaker experiment `01M8A3689ZMXK3WEGT2K9KWHY8`. The successful attempt ran 11:59 AM – 5:17 PM PST.
  - **Config:** `configs/olmicro-2.7b-lc-math-rl-v2-refresh-8k-chatml-20261125.toml`.
- **Result: ChatML ends close to R1.** README finding 9 has the full comparison.
  - **MATH-500:** 38.1 → 72.8 (peak 73.9 at u350), against R1 37.3 → 73.6.
  - **GSM8K Platinum:** 54.0 → 89.7, against R1 67.3 → 90.9. ChatML trails by 1.2–2.6 points from u100 on.
  - **Truncation:** step-0 eval truncation at 7,680 tokens was 30.5% (R1 7.0%). Training-rollout truncation fell from 47% to 3%.
  - **End tokens:** the model ends with `<|endoftext|>`, never `<|im_end|>`.
- **For the paper:** this can support a sentence that the R1 template is a convenience, not a requirement. RL-zero from the long-context base learns to end ChatML turns and reach the same accuracy at 8k.
  - **Caveat:** it was tested only at 2.7B and 8k. The original template choice came from the midtrain checkpoint.
- **Optional follow-ups (not run):**
  - offline Minerva / AIME on the ChatML final (export at `runs/olmicro-2.7b-lc-math-rl-v2-refresh-8k-chatml-20261125/export-hf`);
  - cross-template evals (the ChatML final under R1, and vice versa).

## Explicitly deprioritized / skip for the paper
- **Offline AIME 2026 on 275M / 450M / 810M:** Base scores are 0.0 / 0.0 / 0.1 pass@1; not worth compute.
- **Offline Minerva MATH on 275M / 450M / 810M:** In-loop MATH-500 tracks full 5,000-problem Minerva MATH within <1 pt at both 1.4B (49.9 vs. 49.6) and 2.7B (77.3 vs. 77.1), and Table 2 in the paper is scoped cleanly to 1.4B and 2.7B without empty cells.
- **1.4B rest-of-pool sweep (~640 mostly 1/16 prompts):** Unlikely to move 1.4B results meaningfully unless moving 1.4B to a combined + zstd recipe.
