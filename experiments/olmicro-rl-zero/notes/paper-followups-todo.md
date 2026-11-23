# Paper Follow-Up TODOs (for Tech Report Section 4.2)

Actionable follow-up experiments and analyses identified while drafting the RL-zero section of the Olmicro technical report (`sections/post_training_results.tex` and `sections/appendicies/rl_zero.tex`). Ordered by impact per unit compute.

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
- **Post-run:** Re-run `tools/gather_results.py` to refresh `tables/runs.csv`, `tables/in_loop_evals.csv`, and `tables/signal_fraction.csv`. If final MATH-500 beats 22.2, update Tables 1 and 3 in `sections/post_training_results.tex`, Table `tab:rl-all-runs` in `sections/appendicies/rl_zero.tex`, and Figure 1.

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

---

## Explicitly deprioritized / skip for the paper
- **Offline AIME 2026 on 275M / 450M / 810M:** Base scores are 0.0 / 0.0 / 0.1 pass@1; not worth compute.
- **Offline Minerva MATH on 275M / 450M / 810M:** In-loop MATH-500 tracks full 5,000-problem Minerva MATH within <1 pt at both 1.4B (49.9 vs. 49.6) and 2.7B (77.3 vs. 77.1), and Table 2 in the paper is scoped cleanly to 1.4B and 2.7B without empty cells.
- **1.4B rest-of-pool sweep (~640 mostly 1/16 prompts):** Unlikely to move 1.4B results meaningfully unless moving 1.4B to a combined + zstd recipe.
