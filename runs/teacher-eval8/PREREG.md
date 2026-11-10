# Matched 8-answer evaluation of the original MOPD teachers (recorded 2026-11-10 PT, before launch)

Kevin, 2026-11-10: "can we get 8 per prompt to compare properly". This is a measurement follow-up. It adds no hypothesis, no ceiling claim and no decision rule, and it changes no earlier rule or claim.

## Why

The only saved scores for the original teachers are 1-answer in-training evals at update 99:
- math .719 on math;
- IF .438 on IF;
- code .680 on code.

The main-study table uses 8 answers per prompt. There is no matched 8-answer artifact for these teachers (searched: run configs, the decision log, the eval roots that seedstats3 reads).

## Cells (three only; no cross-task grid)

| Config | Checkpoint (frozen, read-only) | Eval set |
|---|---|---|
| teacher-eval8-qwen35-9b-math-v4-100u-hf99 | runs/teacher-qwen35-9b-math-v4-100u/hf-99 | eval_math |
| teacher-eval8-qwen35-9b-ifeval-v4-100u-hf99 | runs/teacher-qwen35-9b-ifeval-v4-100u/hf-99 | eval_ifeval |
| teacher-eval8-qwen35-9b-code-v4-100u-hf99 | runs/teacher-qwen35-9b-code-v4-100u/hf-99 | eval_code |

These are the hf-99 checkpoints the MOPD 100, MOPD 300 and hybrid configs name as teachers. They are not the Phase 1 specialists and not the equal- or smart-split teachers.

Each teacher is one training run, not three seeds.

## Contract (identical to the main-study capture reruns)

- **Config:** each file is a copy of `all3-eval8-qwen35-9b-mixrl-hf299-cap.toml`, changing only `name`, `model.source`, `output.root`/`assets`, and `eval_prompt_data` (reduced to the matching task). The exact diffs are `{math,ifeval,code}.diff`.
- **Eval data:** the frozen v3 256-prompt eval sets in the data-v4 directory, scored with the v4 verifiers (`verifiers.json`).
- **Sampling:** 8 answers per prompt, temperature 1, top-p 1, max response 16,384, context 18,432, the same chat template and the same eval-only GRPO path. Every answer is saved (`rollout_sample_rate = 1.0`).
- **Runtime:** image 01M5XNG1GRGN7GJ9X15TW9AFH4 with a code overlay of a commit whose wrapper code equals cb457390b, as for the capture reruns.
- **Metric:** mean reward over 256 × 8 answers (pass@1 estimate), computed from the saved answers the same way as seedstats3. Not best-of-8.

## Gates

- **G1:**
  - each run finishes with exit 0;
  - 256 distinct prompts × 8 scored answers = 2,048 per cell, 6,144 in total;
  - 0 failed samples.
- **Reported alongside the score:**
  - the truncation (cap-hit) count;
  - the run's in-training eval log line, if present.

## Rules

- A mechanical failure may be relaunched identically.
- No score-driven reruns.
- The 1-answer values stay in the historical record.
- In-training learning curves stay labelled 1-answer.

## Spend

- 3 jobs × 8 GPUs (Holmes, urgent, min_runtime 1h, timeout 3h), ≈1–1.5 h each, so ≈25–35 GPU-h.
- Peak 24 GPUs; total in use stays ≤ 64.
