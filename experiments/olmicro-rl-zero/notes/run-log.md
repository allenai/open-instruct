# Olmicro 2.7B RL-zero math — overnight log (2026-11-18)

## Morning summary

**v0 completed: 400 updates, exit 0. MATH-500 29.2% → 64.6%, GSM8K Platinum 51.0% → 83.4%.**
[Beaker](https://beaker.org/ex/01M7PV187ND4JH9QGPJ4KCABGF) · W&B `ai2-llm/open_instruct_internal`, group `olmicro-2.7b-math-rl-v0-r1-20261118`

| eval (temp 1.0, n=1) | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| MATH-500 (497) | 29.2 | 45.7 | 51.1 | 49.1 | 56.5 | 56.9 | 61.4 | 63.6 | **64.6** |
| GSM8K Platinum (1,209) | 51.0 | 72.6 | 78.2 | 78.7 | 80.4 | 81.5 | 83.2 | 83.3 | **83.4** |
| MATH-500 mean length | 544 | 667 | 645 | 640 | 679 | 854 | 959 | 1,056 | 1,348 |
| MATH-500 truncated % | 3.6 | 4.8 | 3.4 | 3.2 | 3.0 | 5.2 | 7.2 | 8.9 | 10.1 |

- **Artifacts (verified):** final native checkpoint `runs/olmicro-2.7b-math-rl-v0-r1-20261118/checkpoints/core/rollout_0000399` (400 steps, world size 2, cursor SHA matches; also 299/349); HF export `.../export-hf` (`.complete`, 5.38 GB); workflow `complete`. 400/400 grad norms finite and nonzero; no NaN; only errors were benign startup retries.
- **Wall clock:** 3 h 14 min on 8 B300s (2 trainer + 6 inference), including ~4.5 min startup. Speed fell from 17.6 to ~38 s/update as responses grew (trainer-bound; generation wait ~1 s).
- **What changed vs the plan:**
  1. Workspace → ai2/olmo3p5-training, all urgent.
  2. Smoke graph probe disabled after a benign near-tie greedy flip; all other gates passed.
  3. **Prompt template switched to R1 style** after a 5-template probe (training truncation 76% → 3.5%).
  4. No difficulty filtering (learning was clear).
- **Things to decide for the next run:**
  1. **Response length.** Training-rollout truncation rose 0% → 13–15% at the 4,096 cap (truncated = 0 reward), and MATH-500 length is still growing. Raise `max_response_length`; the model was trained at 8,192 context.
  2. **Difficulty filtering.** v0 saw 12,834 prompts × 8 samples: **68% never solved**, 0.5% always solved. Candidate set prepared (not launched): `data/dolci-math-base-r1-v1-passband-from-v0`, 4,062 prompts with 1/8 ≤ pass ≤ 7/8. Pass rates come from v0's improving policy, not the base model.
  3. **Continue v0 vs restart.** Resume from `rollout_0000399` (same 2 trainer GPUs) for more epochs/longer responses, or a fresh run with the filtered set.
  4. **Throughput.** Now trainer-bound: consider 4 trainer + 4 inference GPUs for longer responses. A new trainer world size cannot resume v0's checkpoint.

**Plan (approved by Jacob):** smoke → if clean, v0 on one 8-GPU B300 node → monitor → morning report.
Fix what I can. If the model gets ~0 correct, debug formatting with hand-run generations; if the
problems are simply too hard, pivot to difficulty filtering and rerun. One node at a time.

## Inputs (verified 2026-11-18)

- Model: `/weka/oe-adapt-default/jacobm/olmicro/models/olmicro-2.7b-base/stage2-step0011921` (HF `allenai/olmicro-2.7b-base@stage2-step0011921`, SHA256SUMS OK)
- Core config: `/weka/oe-adapt-default/jacobm/olmicro/models/olmicro-2.7b-base/olmo-core/stage2-step0011921/config.json` (HF bucket, SHA256SUMS OK)
- Data: `/weka/oe-adapt-default/jacobm/olmicro/data/dolci-math-base-boxed-v1` (13,260 train / 497 MATH-500 / 1,209 GSM8K Platinum)
- Image: `michaeln/olmicro-miles-84848b6` = `01M6HMYBFW8D9ZK4SKGJ1XD6HW`; Open-Instruct checkout `15413672d` (differs from image source only by one example TOML)
- Configs: `open-instruct/runs/olmicro-2.7b-math-smoke-20261118.toml`, `open-instruct/runs/olmicro-2.7b-math-rl-v0-20261118.toml`

## Timeline (UTC)

- 06:25 — Smoke submitted: https://beaker.org/ex/01M7PNAH2X9FPR1D1NJF47KTBK (job `01M7PNAH6QSRCRNP5RE7TABSHH`). Receipts in `/weka/oe-adapt-default/jacobm/olmicro/monitor/`.
- 06:29 — Smoke queued (normal FailedScheduling while waiting for slots). Prepared contingencies:
  - v0 now records every scored group to `/weka/oe-adapt-default/jacobm/olmicro/inference-records` (text for 5% of groups); observability only, re-planned/validated.
  - `data-prep/filter_by_pass_rate.py` turns those records into a pass-rate band-filtered train set (tested on synthetic records).
  - Core rejects `debug_rollout_only`; a base-model pass-rate sweep, if needed, would be an eval-over-train run with rollout capture.
- 06:33 — Per Jacob: everything runs at **urgent**. Raised the queued smoke job in place (`beaker job update-priority ... urgent`, confirmed); both TOMLs now say `priority = "urgent"` and revalidate.
- 06:39 — Per Jacob: scaling-ladders is clogged; moved to workspace **ai2/olmo3p5-training** (urgent allowed, default budget ai2/oe-other, jacobm_WANDB_API_KEY/jacobm_HF_TOKEN present). Cancelled the never-scheduled smoke `01M7PNAH6QSRCRNP5RE7TABSHH` (no output written); both TOMLs updated, validate + preflight pass.
- 06:40 — **Smoke resubmitted:** https://beaker.org/ex/01M7PP592S3WZBVXYJ9JRQ3JD9 (job `01M7PP59712F91NCS27MBY0MWY`), urgent; scheduled at 06:40:36, 10 s after creation.
- 06:46 — **Smoke attempt a exited 1 at the graph/eager probe, before training.** Diagnosis (artifacts in `runs/olmicro-2.7b-math-smoke-20261118/`, log `monitor/smoke-2-logs.txt`):
  - Teacher-forced selected-token parity exact (max logprob error 0.0 in all 7 cases, up to 3,077 tokens); the probe itself reports `passed: true`.
  - The smoke wrapper additionally requires all 7 greedy continuations to match; 1 of 7 diverged (case 2, output position 25 of 32).
  - It is a near-tie flip: after the model already answered "12 plus 30 is 42." it rambles; eager picks " identify" (p=0.079), graph " determine" (p=0.077), a gap of ~0.03 nats.
  - Graph-vs-eager logprob noise along greedy paths is 0.058 at 2.7B vs **0.053 in Michael's passing 450M run**; queued-request agreement 0.00008. The 450M simply hit no near-ties.
  - Conclusion: ordinary bf16 graph/eager kernel noise, not a correctness bug. CUDA graphs stay on.
- 06:49 — **Smoke-b submitted** with only `OI_NATIVE_GRAPH_PROBE` removed (fresh name/root; all other gates kept): https://beaker.org/ex/01M7PPNG808DDGFMCNTHVVCH00 (job `01M7PPNGBY65FED38ZQ4C7DWDT`), scheduled in 18 s.
- 07:31 — Smoke-b process 0 finished updates 1–5 (~5 min/update incl. per-update eval/checkpoint); resume process running. **Base-model behavior on Dolci with the qa_boxed template:**
  - Only ~10–15% of prompts get ≥1 correct in 8 samples (23–51 all-zero groups dropped per 4 kept; no all-correct groups).
  - **~46% of kept training samples hit the 4,096 cap** (74/160) — above the agreed ~30% hold threshold for v0.
  - Text previews show the cause is formatting: `Question:/Answer:` primes a bare short answer ("Answer: $15$.", "Answer: 0", "Answer: C.") with no reasoning/box, then either the `\nQuestion:` stop fires (it works) or the model rambles to the cap.
  - 8-prompt smoke eval reward 0.0–0.5.
- 07:35 — **Holding v0; running a prompt-template probe** (Jacob's "debug with test responses" step): 5 templates (qa_boxed, qa_boxed_think, problem_solution, chatml, r1) × 96 prompts (64 train + 32 MATH-500) × 8 samples at RL sampling settings, scored with MathVerifier, truncated = 0. https://beaker.org/ex/01M7PS9REH4TXAN32F45Q4SE64 (4 GPUs; with smoke-b = one node). Script `tools/format_probe.py`; results to `monitor/format-probe-20261118/`.
  - `data-prep/prepare_olmicro_math.py` now takes `--template` (default reproduces v1 byte-for-byte) and records the matching stop setting.
- 07:42 — **Smoke-b PASSED (exit 0).** Verified from artifacts, not just status:
  - `native-smoke-complete.json`: passed; nonzero-gradient updates 1–5 (process 0) and 6 (process 1); checkpoint 5 restored in a separate process; `NATIVE_SMOKE_RESUME_PASSED` in the job log.
  - Weight equality: 16/16 `[check_tensors]` reports list every tensor equal (652 SGLang tensor entries), none unequal; initial publication sent all 568 tensors / 5.377 GB.
  - Train/rollout logprob abs diff 0.0063–0.0073 (KL ~1e-4), about half of the 450M's 0.013–0.019.
  - So the 2.7B GQA shape, the HF-bucket Core config, and publication are all qualified on B300 with decode graphs.
- 07:50 — Format probe attempt a: generation worked, but scoring crashed in-process: the image's main env has ANTLR 4.9 (for OmegaConf) and `is_equiv` re-raises sympy's ANTLR-4.11 ImportError. **Training rewards are unaffected**: the training reward runs MathVerifier in a child process with `OPEN_INSTRUCT_MATH_VERIFIER_PYTHONPATH=/opt/core-rl/verifier-deps` (image ENV, `runtime/miles/Dockerfile.base:44`), consistent with the smoke's positive rewards. Fix: probe now only generates (raw outputs), and `tools/score_format_probe.py` scores on CPU with ANTLR 4.11 (tested on synthetic outputs).
- 07:55 — Format probe b submitted: https://beaker.org/ex/01M7PT7Q5M20QW7QYVB40ACENW (job `01M7PT7Q93YSPDDA9ESGQNRPWV`).
- 08:02 — **Format probe results** (96 prompts × 8 samples, temp 1.0, 4,096 cap, MathVerifier, truncated = 0; raw + scored files in `monitor/format-probe-20261118/`):

  | template | train reward | MATH-500 reward | train pass@8 | MATH-500 pass@8 | train truncated | boxed | mean tokens |
  |---|---|---|---|---|---|---|---|
  | qa_boxed (v1) | 0.012 | 0.188 | 0.078 | 0.594 | **0.760** | 0.32 | 3,087 |
  | qa_boxed_think | 0.029 | 0.215 | 0.172 | 0.562 | 0.371 | 0.43 | 2,228 |
  | problem_solution | 0.025 | 0.121 | 0.141 | 0.500 | 0.123 | 0.40 | 982 |
  | chatml | 0.025 | 0.094 | 0.125 | 0.531 | 0.344 | 0.43 | 1,989 |
  | **r1** | **0.039** | 0.211 | 0.141 | 0.531 | **0.035** | **0.62** | **613** |

  - **Chose r1**: best train reward, ~tied best on MATH-500, truncation 76% → 3.5%, responses ~5× shorter (cheaper generation, room to grow). Responses are genuine step-by-step reasoning; finishes are 65% EOS, 33% `\nUser:` stop, 3% length. 38% of responses still lack `\boxed{}` — learnable format signal for RL.
  - The training set is hard for this base model: only ~14% of Dolci prompts have ≥1 correct in 8 (MATH-500: ~53%). No difficulty filtering in v0, as agreed; watch the zero-gradient fraction.
- 08:05 — Data `data/dolci-math-base-r1-v1` (13,251 train / 497 MATH-500 / 1,209 GSM8K; 63 train prompts over 512 tokens dropped). Config `open-instruct/runs/olmicro-2.7b-math-rl-v0-r1-20261118.toml` = agreed v0 draft with only data paths, `rollout_stop = ["\nUser:"]` and name/root changed; plan/validate/preflight pass; stop matches dataset manifest.
- 08:05 — **v0 launched:** https://beaker.org/ex/01M7PV187ND4JH9QGPJ4KCABGF (job `01M7PV18B68QX42ERZ9T4J148A`), 8 B300s (2 trainer + 6 inference), urgent, ai2/olmo3p5-training. Inference records → `/weka/oe-adapt-default/jacobm/olmicro/inference-records`.
- 08:13 scheduled, 08:15 started (8 min queue for a full node).
- 08:37 — **Early check at update 30: healthy, no stop conditions.**
  - **Base model (step-0 eval, R1 template, temp 1.0, n=1): MATH-500 29.2%, GSM8K Platinum 51.0%** (truncated 3.6% / 0.5%).
  - Speed: warm cycle **17.6 s/update** (training median 10.1 s, generation wait 4.4 s, publication 0.8 s); trainer-bound now, not generation-bound. Startup ~4.5 min (serving 144 s). Projected finish ≈ 10:45 UTC.
  - Signal: **22–23% of groups have mixed rewards** (steady across the run so far); 77% all-zero; 1 all-correct of 1,156 groups. Batch reward 0.045 (updates 0–9) → 0.051 (20–29): too early to call.
  - Training-rollout truncation 0.0%; mean response 460–650 tokens.
  - Grad norm 32/32 finite and nonzero (0.06–0.18); no NaN/inf; 0 stale/expired/aborted groups; avg staleness ~0.8.
  - Only errors: 6 `/freeze_gc` connection-refused retries during SGLang startup (same as smoke, benign).
- 09:20 — **Update 170: learning clearly, no stop conditions.** Checkpoints at 49/99/149 saved.

  | eval (temp 1.0, n=1) | step 0 | 50 | 100 | 150 |
  |---|---|---|---|---|
  | MATH-500 (497) | 29.2% | 45.7% | **51.1%** | 49.1% |
  | GSM8K Platinum (1,209) | 51.0% | 72.6% | 78.3% | **78.7%** |

  - Training batch reward 0.045 (updates 0–24) → 0.089 (150–170); mixed-group fraction 22.5% → 30.5%; mean response length 550 → 836 tokens; truncation ≤0.2%; 0 dropped groups.
  - Grad norm 170/170 finite and nonzero (0.034–0.176). The one "nan" log match is the word inside a math problem's text preview. Tracebacks still only the 6 startup retries.
  - MATH-500 at 100 vs 150 (51.1 vs 49.1) is within noise (SE ≈ 2.2 pts). Not the "flat by 150" case, so **no filtering pivot**; v0 continues to 400.
- 10:03 — Update 271 (the job watcher hit its own 2 h deadline; job still running; watcher re-armed). Evals at 200/250: **MATH-500 56.5% / 56.9%**, GSM8K Platinum 80.4% / 81.5%; eval MATH-500 response length 679 → 854. Grad norm 271/271 nonzero; no new errors.
- 11:05 — Update 369, still healthy; **slower** than projected (~38 s/update vs 17.6) because responses grew. Finish now ≈ 11:25–11:30.
  - Evals at 300/350: **MATH-500 61.4% / 63.6%**; GSM8K Platinum 83.2% / 83.3%. MATH-500 eval truncation 3.6% → 8.9%.
  - Training: mean response length 550 → 1,861 tokens; tokens/update 141k → 476k; training 9.8 → 29.3 s/update (generation wait ~1 s, so trainer-bound); batch reward 0.045 → 0.130.
  - **Training-rollout truncation at the 4,096 cap rose 0% → 13–15%** (truncated = reward 0). Normal RL-zero length growth, not a stop condition, but the main knob for the next run (longer `max_response_length`).
  - Grad norm 369/369 nonzero (0.023–0.176); tracebacks still only the 6 startup retries. Checkpoints through 349.
- 11:29 — **v0 exited 0, workflow complete.** Final evals at 400: MATH-500 **64.6%**, GSM8K Platinum **83.4%**. Verified checkpoint `rollout_0000399` (completed_steps 400, world_size 2, cursor SHA OK) and HF export (`.complete`, `mainline_ladder` 30×1536, 5.38 GB). 400/400 grad norms nonzero.
- 11:35 — Prepared (not launched) a difficulty-filtered candidate from v0's inference records: `data/dolci-math-base-r1-v1-passband-from-v0`.
  - 4,062 prompts kept (1/8 ≤ pass ≤ 7/8, ≥8 samples).
  - Dropped: 8,709 never solved, 63 always solved, 417 unobserved.
  - Pass-rate histogram by decile: 0.0: 8,709 · 0.1: 1,775 · 0.2: 899 · 0.3: 531 · 0.5: 352 · 0.6: 235 · 0.7: 158 · 0.8: 110 · 0.9+: 63.
- All jobs from tonight have exited; no GPUs held. Pending wake-ups cancelled.

## v1 (approved by Jacob, 2026-11-18 daytime)

**Plan:** two arms in parallel, one node each, both from the base model, R1 template, 7,680-token responses / 8,192 context, 4 trainer + 4 inference GPUs, 4 eval samples per prompt, urgent in ai2/olmo3p5-training.
- Arm A: full dolci-math-base-r1-v1.
- Arm B: offline difficulty filter from **base-model** pass rates (sweep), band 1/16 ≤ p ≤ 14/16 over 16 samples.
- The 4+4 / 8k topology is qualified first by a 3-update test: weight equality at startup and every update, stop after 2, relaunch to resume for 3.
- Configs are generated from the v0 config by `tools/make_v1_configs.py` (asserted exact edits).

- 14:26 — **Pass-rate sweep** submitted: https://beaker.org/ex/01M7QGT8VC0030RBZ8CET6Q2DM (8 GPUs). Settings: all 13,251 training prompts, verbatim prepared inputs, 16 samples, temp 1.0, 7,680 new tokens, 8,192 context, stop `\nUser:`. Output `monitor/passrate-sweep-20261118/`, scored on CPU by `tools/score_format_probe.py` (truncated = 0). `format_probe.py` gained `--from-input`, sharding and IDs (CPU-checked: shards cover every prompt once; inputs equal the R1 render).
- 14:27 — **4+4 / 8k qualification test** submitted: https://beaker.org/ex/01M7QGX6EVXH6DME4PA6JFJE3G (config `runs/olmicro-2.7b-math-4x4-8k-test-20261118.toml`). Arm A config `runs/olmicro-2.7b-math-rl-v1a-full-20261118.toml` validated and preflighted.
- `data-prep/filter_by_pass_rate.py` now also reads sweep `*.scored.json` (tested on synthetic scores).
- 14:50 — **4+4 / 8k test phase 1 passed** (exit 0, workflow `stopped` after updates 0–1 as designed):
  - Weight equality: all `[check_tensors]` reports list every tensor equal (652 entries), at startup and after each update.
  - Grad norm 0.100 / 0.080; train/rollout logprob abs diff 0.0061 / 0.0055 (KL ~1e-4), same as the 2-GPU smoke.
  - Checkpoint `rollout_0000001` committed with world_size 4. A response reached 5,399 tokens (past the old 4,096 cap); no truncation.
  - Only tracebacks are the usual `/freeze_gc` startup retries.
- 14:51 — Relaunched the same config to test resume at world size 4: https://beaker.org/ex/01M7QJ96XQN48TM6SHVTGBVD79 (job `01M7QJ977P0XMC54W5YYBW8F83`).
- 15:10 — **4+4 / 8k test resume passed** (exit 0, workflow `complete`, attempt 1):
  - Native restore from `rollout_0000001` on all 4 trainer ranks; update 3 completed.
  - Checkpoint `rollout_0000002`: completed_steps 3, world_size 4, cursor SHA OK.
  - Weight equality all equal after resume; grad norm 0.137; logprob diff 0.0062.
  - So the 4+4 topology at 8,192 context is qualified for training, publication, checkpointing and resume.
- 15:12 — **Arm A launched:** https://beaker.org/ex/01M7QKCMXE9TX17MBHN8H0WYBM (job `01M7QKCN9HJECTD42566AR0M3J`), 4 trainer + 4 inference, urgent.
- 15:29 — **Base-model pass-rate sweep done** (exit 0; 8 shards × ~51 min). Scored 212,016 samples on CPU in 186 s (MathVerifier, truncated = 0; `monitor/passrate-sweep-20261118/*.scored.json`).
  - Base model, R1 template, 7,680-token cap: mean reward 0.041; **~30% of prompts have ≥1 correct in 16**; truncation 2.1%; boxed 62%; mean 795 tokens.
  - Correct-of-16 histogram: 0: 9,222 · 1: 2,026 · 2: 906 · 3: 473 · 4: 233 · 5: 164 · 6: 101 · 7: 43 · 8: 46 · 9–14: 37 · 15–16: 0.
- 15:33 — **Arm B data** `data/dolci-math-base-r1-v1-passband-base16`: 4,029 prompts with 1–14 correct of 16. Dropped 9,222 never solved; none too easy. Mean base pass rate of the kept set ≈ 13%. 63% overlap with the v0-derived candidate. Eval/verifier files are byte-identical to Arm A's.
- 15:35 — **Arm B launched:** https://beaker.org/ex/01M7QMNVT21VP580HSS8HSTTTV (job `01M7QMNVZ5W5YKGKHWEYN2458Y`), 4 trainer + 4 inference, urgent. Config differs from Arm A only in name/root/W&B group and data directory.
- 15:37 — **Arm A early check (10 updates): healthy.**
  - Startup weight equality all equal; grad norms 0.057–0.188; logprob diff 0.0057–0.0070; tracebacks only startup retries.
  - **Step-0 eval (4 samples/prompt, 7,680 cap): MATH-500 31.3%, GSM8K Platinum 53.5%** (v0 at n=1, 4,096 cap: 29.2 / 51.0). MATH-500 eval truncation 5.7% at 7,680 (base rambling).
  - Startup: serving 193 s, step-0 eval 242 s.
  - Training step time is bimodal: 298 / 148 s for the first two updates, then ~40 s or 2.7–4.2 s. Tokens/update are the same as v0's start (120–180k), so this looks like FA4 JIT compiling new packed shapes. Fast steps reach 35–48k tok/s on 4 GPUs vs v0's 13k on 2. Recheck that it settles.
- 15:50 — Check at A update 50 / B update 1:
  - **A training step settled at ~3.4 s** (the earlier ~40 s steps were one-time compilation). Wall time is 20 s/update, now **generation-bound** (generation wait 14 s; 4 engines vs v0's 6). ~1.9 h remaining + evals; it will slow as responses grow. Batch reward 0.047 → 0.063; mean length ~700; truncation 0.
  - **B started cleanly**: startup weight equality all equal, grad norm 0.21, logprob diff 0.0057. Step-0 eval MATH-500 30.2%, GSM8K 54.4%, matching A's 31.3 / 53.5 within noise (same base model).
  - Signal (inference records): **A 22.4% mixed groups (reward 0.051/sample) vs B 45.9% (0.107)**, so the filter doubles the useful groups as intended.
  - Tuning note for later: with 4+4 the trainer is mostly idle; 3 trainer + 5 inference (or more engines) would likely be faster at this length, but would not resume v1 checkpoints.
- 16:37 — Both arms healthy (`tools/arm_status.py` summaries).

  | eval (4 samples/prompt) | 0 | 50 | 100 | 150 |
  |---|---|---|---|---|
  | A MATH-500 | 31.3 | 46.0 | 51.0 | 51.0 |
  | B MATH-500 | 30.2 | 48.9 | 51.9 | – |
  | A GSM8K Platinum | 53.5 | 72.1 | 77.6 | 77.6 |
  | B GSM8K Platinum | 54.4 | 76.8 | 79.1 | – |

  - A's MATH-500 at 100 and 150 is the identical count (1,013 / 1,988 samples) by coincidence: separate evals, different timestamps, and GSM8K differs (77.61 vs 77.65%). v0 also flattened between 100 and 150.
  - **A** (update 155): ~25 s/update excluding evals (train 4.5 s, generation wait 19.5 s); ~2–2.5 h left. Batch reward 0.051 → 0.073; mean length 639 → 1,121; truncation at 7,680 0.8% in the latest updates. Mixed groups 22.7% → 27.3%.
  - **B** (update 113): 22 s/update; ~1.8 h left. Batch reward 0.150 → 0.211; mean length 598 → 844; truncation 0.1%. **Mixed groups 55.8% → 62.4%** (vs A's ~27%).
  - Both: grad norms all nonzero (A 0.035–0.188, B 0.097–0.229); logprob diff median 0.0048–0.0049; startup weight checks equal; only startup-retry tracebacks; no NaN.
- 17:24 — Both healthy; both slower as responses lengthen (generation-bound).
  - **A** (update 228): MATH-500 at 200 = 52.0, GSM8K 78.6. ~43 s/update (generation wait 37 s, train 6 s); ~2 h left + evals. Batch reward 0.098 (200–227); mean length 1,240; truncation at 7,680 2.3%. Mixed groups 24% → 30%.
  - **B** (update 200): MATH-500 at 150 = **55.4** (A at 150: 51.0), GSM8K 79.3. ~38 s/update; ~2.1 h left. Batch reward 0.270 (150–199); mean length 1,055; truncation 0.8%. Mixed groups 58% → 68%; all-correct 1.1%.
  - Grad norms all nonzero; logprob diff median ~0.0045; no NaN; no new errors.

## Switch to the long-context (`main`) checkpoints (2026-11-18, 9:52 AM PST)

Jacob: reported results must use the `main` (final, long-context) checkpoint of each size. Cancelled both v1 arms (midtrain base) at 9:52 AM PST after a final snapshot; their checkpoints, logs and records stay on Weka as a reference/fallback in case the LC model misbehaves under RL.
- Last evals before cancel (4 samples/prompt):
  - A (unfiltered, update 263): MATH-500 31.3 / 46.0 / 51.0 / 51.0 / 52.0 / **54.2** at 0–250; GSM8K 53.5 → **79.3**.
  - B (base-filtered, update 233): MATH-500 30.2 / 48.9 / 51.9 / 55.4 / **60.1** at 0–200; GSM8K 54.4 → **84.1**.
  - So at update 200, B leads A by +8.1 MATH-500 and +5.5 GSM8K. Mixed groups: A 24 → 32%, B 59 → 70%.
- All midtrain reference evals (v0, v1 A, v1 B) are collected in `runs/midtrain-reference-evals-20261118.csv`.
- **`main` currently holds only README + .gitattributes at all five sizes** (the model cards say it will get the stage3/long-context final). Using the stage3 final branches: 275m/450m `stage3-step0047684`, 810m/1.4b/2.7b `stage3-step0023842`. To do: verify `main` matches via SHA256SUMS once populated.
- LC vs midtrain at every size: identical architecture; HF `max_position_embeddings` 65,536 (vs 8,192); Core configs differ only in attention backend and `lm_head.loss_implementation`, both overridden by MILES.

## Truncation, the barrier length cap, and the fix (2026-11-18, ~10:00 AM PST)

- **Filtering was barely affected by the 7,680 cap** (2.7B midtrain sweep):
  - 2.11% of 212k responses truncated, across 3,579 prompts.
  - Only 66 truncated texts already contained the correct answer: 28 of 9,222 "never solved" prompts (0.3%) would have been kept, and 36 kept prompts had their pass rate lowered.
  - Correct responses: median 514 tokens, p99 6,125; 2.8% > 4,096; 0.9% > 6,144.
  - Script: `scratchpad/truncation_analysis.py`.
- **RL's real length cap was the weight-update cycle.** With `publication_mode = "barrier"`, a prompt group still generating at publication is aborted and regenerated: barrier buffers require one policy version per group (`docs/miles/run-controls.md:86`); 513 `abort_request` calls in Arm A's log.
  - Arm A, per-update longest response vs median cycle: updates 10–149 ~2.2–2.5k tokens at 19–21 s; 150–199 4.3k at 32 s; 200–261 7.5k at 43 s.
  - Training-side truncation was 0% for updates 0–149, although the base model truncates 2.1% offline.
  - So v0/v1 trained with a bias against long responses early on, and the length "growth" partly tracked slowing cycles.
- Options considered:
  - engine_drain: documented for Core MoE only.
  - refresh: in-flight tokens continue across publication, TIS-corrected, staleness measured by the oldest token. Passes validate for our setup.
  - synchronous (`fully_async = false`): passes validate without records, but each update waits for its slowest response, which is impractical at long caps.
- Jacob (~10:10 AM PST): test and fix; use a **32k output cap**; start the long-context sweep meanwhile.
- 2.7B long-context final downloaded and verified: `models/olmicro-2.7b-base/stage3-step0023842` plus `olmo-core/stage3-step0023842/config.json` (SHA256SUMS OK).
  - All 568 tensors identical in name/shape/dtype to midtrain; `max_position_embeddings` 65,536.
  - `tokenizer.json` identical; `tokenizer_config.json` differs only in `model_max_length`.
- 10:07 AM PST — **LC 2.7B pass-rate sweep (32k cap)**: https://beaker.org/ex/01M7QXFFGVNRN6NC5M4BDZCYAV (job `01M7QXFG1KTE3Y8MEZ5SJ9586C`). Same settings as before except `--max-new 32768 --context-length 33280`; output `monitor/passrate-sweep-lc2.7b-32k-20261118/`.
- 10:09 AM PST — **Refresh qualification test** (`runs/olmicro-2.7b-lc-refresh-test-20261118.toml`): https://beaker.org/ex/01M7QXJ9YMZ24Q3DJ40Q8649QR (job `01M7QXJA8QREHB0AX1B9T7797E`).
  - Config: LC 2.7B, 4+4, `publication_mode = "refresh"`, `max_weight_staleness = 4`, 32k responses, packing 40,960, records on.
  - Stop after update 3, then relaunch to resume for 4–6.
  - Plan shows `use_tis: true`, `use_rollout_logprobs: false`, `max_policy_lag: 4`.
  - Pass criteria: startup weight equality; finite nonzero grads; logprob diff ≈ 0.005–0.007; at least one response spanning >1 policy version (records `policy_versions`); responses longer than one update cycle actually complete; no abort/requeue churn; clean resume.
- 10:10 AM PST — **LC sweep preempted** 20 s after scheduling: "preempted by job … at priority urgent … allocated workloads are scheduled ahead of unallocated ones" (an allocated job from another workspace).
  - Cause (corrected by Jacob): allocation is per job, not per workspace. A positive `context.minRuntime` selects allocated scheduling with a preemption-protected window (`docs/miles/launching.md:203`); after that window, or with no minRuntime as in my hand-written sweep spec, a job is unallocated and preemptible. Launcher runs set `launch.min_runtime` (2h so far).
  - Made the sweep preemption-tolerant: `format_probe.py --shard` takes several shards per engine, skips shards already written, and writes atomically (`.partial` → rename). The spec now uses 32 shards (4 per GPU), `autoResume: true`, `minRuntime: 2h`, `retry.allowedTaskRetries: 3`.
- 10:13 AM PST — Sweep resubmitted: https://beaker.org/ex/01M7QXRZK5NWVMGTYRXTTW4EBJ (job `01M7QXRZQ217QNJQ8FJ1C03RZ5`).
- 10:16 AM PST — Policy (Jacob): `min_runtime` = expected job length + buffer, max 8h. Resubmitted the not-yet-started sweep with `minRuntime: 4h` (expected 1.5–3 h): https://beaker.org/ex/01M7QXZ6NAE4N1DPQP67G6XGDT (job `01M7QXZ6RV0MK9A1SJ89S8C8ZZ`). RL runs will use the same rule, capped at 8h, with auto-resume beyond that.
- 10:27 AM PST — Jacob: after the LC 2.7B filtering, run two arms to compare: (1) barrier with the longer cap (previous settings), (2) refresh / option C. Choose small-model settings from that comparison.
  - **Cap decision delegated to me**: stay at 32k unless responses >16k are clearly broken and carry a large token share; then switch to 16k (rescore finished 32k shards with >16,384 counted as truncated; rerun the rest at 16k).
  - Both arms launch once filtering is done; refresh staleness set from the refresh test.
- 10:40 AM PST — **Refresh test phase 1** (updates 0–2) exited 0, workflow `stopped` as designed.
  - Startup weight equality all equal; grad norms 0.099 / 0.111 / 0.031; logprob diff 0.0074–0.0081 (barrier ~0.005–0.006); no NaN; only startup-retry tracebacks.
  - **0 `abort_request` calls** (barrier: 513).
  - **Long responses now reach training**: per-update longest 9,020 / 16,579 / 32,768 tokens (1, 15, 34 responses > 8k; 1 at the 32k cap). Barrier at this stage: ~2.5k.
  - Not yet exercised: all 1,280 recorded responses were generated under a single policy version (batch version span 0). Training dominated these first updates (382 / 206 / 360 s: compilation on new long shapes), so the producer had filled its buffer and nothing was in flight at publication. Cross-version continuation should occur once training is fast and generation is the bottleneck.
- 10:45 AM PST — Refresh test phase 2 (resume, updates 3–5): https://beaker.org/ex/01M7QZDQYTEVKP1JC6TST0CM51 (job `01M7QZDR30BV9DNB1KPGKQHB16`).
- 10:50 AM PST — **Decision (Jacob, after talking to Michael): use refresh; "barrier is unnecessarily conservative."** The barrier comparison arm is dropped (never launched). After filtering, launch only the refresh run, tuned from Michael's maintained refresh templates (`configs/miles/examples/node.toml` / `three-node.toml`: refresh, `max_policy_lag = 12`, 32k responses).
- 10:57 AM PST — **v2 refresh run approved by Jacob.** Generator `tools/make_v2_refresh_config.py RUNS --data <LC filtered set> --cap <32768|16384>`; both caps pass validate. Changes vs v1 B:
  - stage3 (long-context) checkpoint and Core config;
  - `publication_mode = "refresh"`, `max_weight_staleness = 12`;
  - response cap 32k (context 33,280; packing 40,960 as qualified) or 16k (16,896; 24,576);
  - `async_max_concurrent_samples` 512; `sglang_max_running_requests` / decode graph batch / server concurrency 128;
  - `sglang_mem_fraction_static` 0.6 (a KV pool of 128 × 33k ≈ 4.3M tokens ≈ 105 GB does not fit at 0.35);
  - `min_runtime = "8h"` (auto-resume and max_retries 3 beyond that).
  - Expected 7–10 h for 400 updates (measure in the first hour).
- Launch steps when the sweep finishes: score → length_tail / truncation analysis → filter 1/16–14/16 → generate config → preflight → run (urgent, ai2/olmo3p5-training) → push-notify.
- 11:14 AM PST — **Output cap decision: keep 32k** (first 8/32 sweep shards: 53,040 responses, 118.3M tokens; overall correct 5.79%).

  | length | % responses | % tokens | correct | loop-like tails |
  |---|---|---|---|---|
  | 0–4k | 87.5 | 35.3 | 5.3% | 1.8% |
  | 4–8k | 4.1 | 11.0 | **13.3%** | 12.8% |
  | 8–16k | 6.4 | 34.5 | **9.2%** | 4.7% |
  | 16–32k | 2.0 | 19.2 | 2.8% (11% hit 32k) | 12.1% |

  - A 16k cap would lose 30 correct answers (1.0% of all correct) for 4.4% fewer tokens; an 8k cap would lose 11.1% of correct answers.
  - The LC model writes ~2.8× more tokens per response than midtrain (mean ≈ 2.2k) and is more accurate (5.8% vs 4.1% per sample). Long answers are disproportionately correct.
  - Sweep pace: first wave of 8 shards in ~34 min → done ≈ 12:35 PM PST.
- 11:15 AM PST — **Refresh test phase 2** (resume, updates 3–5) completed all 3 updates (checkpoints 3–5; exit pending at time of writing). Resume worked.
  - Long responses again reach training (per-update max 8,299 / 15,529 / 32,768). Logprob diff ~0.0089.
  - Still **0 responses spanning >1 policy version** (2,560 recorded): training steps were 281–407 s, so generation finished before each publication.
- **Root cause of the slow steps: no persistent compiler cache.** Log: "Core compiler cache: no default WEKA mount; using ordinary local compiler caches". The default shared root is on `/weka/oe-training-default`, which I had dropped from the mounts when consolidating inputs onto oe-adapt-default. So every process recompiles and autotunes its Triton kernels (incl. the GDN kernels) from scratch, and 32k contexts mean many more shapes.
  - v1 settled to ~3.4 s/step after ~10 updates within one process. The 3-update test processes never got past warm-up (~50–95 s per 40k-token pack). Auto-resumes would also start cold.
  - Fix: `tools/make_v2_refresh_config.py` now mounts oe-training-default (as Michael's configs do).
- Prepared `runs/olmicro-2.7b-lc-refresh-warmup-20261118.toml`: v2's exact serving/publication settings, 20 updates in one process, compiler diagnostics on (validate + preflight OK).
  - Purpose: measure warm-up length and the settled step time, populate the shared compile cache for v2, and observe cross-version refresh once training is fast.
  - Launches when refresh-test phase 2 exits (2-node limit).
- 11:25 AM PST — Refresh test phase 2 exited 0; workflow complete (attempt 1, rollouts 3–5). Checkpoint 5: completed_steps 6, world_size 4, cursor SHA OK. **Refresh + resume qualified mechanically.** Note: ~15 min between the last update and exit (finalization/shutdown).
- 11:27 AM PST — **Warm-up test launched:** https://beaker.org/ex/01M7R204EGSM9TAZK0A9MK8G3K (job `01M7R204JFEZGMZGPM2WZMEZBK`).
- 12:05 PM PST — The warm-up test with the `oe-training-default` mount (job `01M7R204JFEZGMZGPM2WZMEZBK`) was scheduled at 11:39 and passed the health check at 11:41, but never started (no Ready/Started for 23 min; other jobs start ~1 s after the health check). The extra mount was its only new element. Cancelled (never created its output root).
  - Switched to `compiler_cache.shared_root = "/weka/oe-adapt-default/jacobm/olmicro/compiler-cache/tmp-30d"` (an expiry component is required on WEKA), no extra mount; the v2 generator was updated the same way.
  - Per `docs/miles/compiler-cache.md`, only Triton artifacts persist; FA4/CuTe DSL compiles still warm up per process.
- 12:07 PM PST — **Warm-up test b launched:** https://beaker.org/ex/01M7R47XJ9AN1TADYW312CWD34 (job `01M7R47XP8Q7EPHPB7VPTSRXVB`), config `runs/olmicro-2.7b-lc-refresh-warmup-b-20261118.toml`.
- 12:16 PM PST — Warm-up test b stalled exactly like the first attempt (health check passed 12:07, never started). **Both stalled jobs were on the same node, `holmes-cs-aus-511.reviz.ai2.in`** (node `01M6HAHM8KS3WD0N3C8S6GA09D`, Beaker status "ok"). Every job that started today ran elsewhere. So the stall was the node, not the oe-training-default mount.
  - It is probably often the only "free" 8-GPU node because nothing actually runs there.
  - Worth reporting to cluster ops (Jacob's call).
- 12:18 PM PST — Cancelled b. The launcher cannot exclude a node (it only has an unused hostname allowlist; Beaker makes cluster and hostname constraints mutually exclusive). Resubmitted the launcher-generated spec with `constraints.cluster` replaced by a hostname allowlist of the other 70 uncordoned Holmes nodes (`monitor/warmup-c-spec.json`; durations re-expressed as strings, budget ai2/oe-other, otherwise identical).
  - Warm-up test c: https://beaker.org/ex/01M7R4YEBRR9DV9Y1HA77DBRYY (job `01M7R4YEFJSDNNEXNPDN68PWQ7`). Queued: no other 8-GPU slot free right now.
  - v2 will need the same hostname-exclusion treatment.
- 12:15 PM PST — LC sweep at 24/32 shards (its watcher hit its own 2 h limit; re-armed). Expected done ≈ 12:50 PM PST.
- 12:25 PM PST — Wrote `tools/submit_excluding.py` (mirrors `launch.submit`; passes `launch.specification()` a hostname allowlist of every uncordoned node of the cluster except `--exclude`; writes the normal launch receipt; `--dry-run` renders and checks). Dry run on the warm-up config: 70 nodes, excluded node absent, context/budget unchanged.
- 12:26 PM PST — Cancelled the queued warm-up test c (never started): the cluster has no free 8-GPU node until the sweep's node frees (~12:50). v2's own first hour gives the same signals (step-time settling, cache population, cross-version spans), so v2 launches directly on that node via `submit_excluding.py --exclude holmes-cs-aus-511.reviz.ai2.in`, with close early monitoring.
- 12:43 PM PST — **LC 2.7B 32k sweep done** (32/32 shards, all GPUs exit 0, no tracebacks; ~2 h 20 min). Scored 212,016 samples in 501 s.
  - **LC base model: reward 0.055/sample, pass@16 0.370, truncation at 32k 0.25%, `\boxed` 76.6%, mean 2,241 tokens.**
  - Midtrain base at 7,680: 0.041 / 0.30 / 2.1% / 62% / 795.
- 12:52 PM PST — **LC filtered set** `data/dolci-math-base-r1-v1-passband-lc2.7b-32k`: 4,900 prompts with 1–14 correct of 16 (dropped 8,350 never solved, 1 always solved).
  - Histogram: 1: 2,271 · 2: 1,059 · 3: 631 · 4: 347 · 5: 221 · 6: 115 · 7: 98 · 8: 69 · 9–14: 89.
  - Mean base pass rate of kept ≈ 14.8%; 60% overlap with the midtrain-filtered set; eval/verifier files byte-identical.
- 12:53 PM PST — **v2 launched** (`runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-20261118.toml`, generated by `tools/make_v2_refresh_config.py`; validate + preflight OK) via `tools/submit_excluding.py --exclude holmes-cs-aus-511.reviz.ai2.in`: https://beaker.org/ex/01M7R6Z11XWGR46DA7TWM8HDH5 (job `01M7R6Z15JPX5VC6P9R4R3JWM0`).
  - Scheduled in 11 s; 70-node allowlist.
  - Settings: refresh, staleness 12, 32k cap, 4+4, 128 running/engine, mem 0.6, 512 in flight, Triton cache root `compiler-cache/tmp-30d`, min_runtime 8h, urgent, ai2/olmo3p5-training.
- 12:55 PM PST — v2 **started** (on node `01M67DHGC9JM9GDBPDTAGR1G2J`, the healthy node the sweep used), 2 min after creation.
- 1:30 PM PST — **v2 early check** (35 min after start):
  - Compiler cache now "Triton enabled", publishing under `compiler-cache/tmp-30d`. Startup weight equality all equal; grad norms 0.185 / 0.143; logprob diff 0.0071 / 0.0080; no NaN, no OOM at 128 running × 33k with mem 0.6; only startup-retry tracebacks.
  - **LC base step-0 eval (4 samples, 32k): MATH-500 37.4%, GSM8K Platinum 68.3%** (midtrain base at 8k: 31.3 / 53.5). Eval length: MATH-500 mean 2,896 tokens, 6.9% truncated at 32k; GSM8K 923 tokens, 2.0%.
  - Startup: trainer 41 s, serving 199 s, step-0 eval 675 s (4 × 1,706 prompts at 32k).
  - **Training is still slow**: 335 / 211 / 541 s for updates 0–2 (726k tokens in update 2 ≈ 1.3k tok/s vs v1's 35–48k), e.g. rank 1 step 1 = 232 s for 2 packs.
    - Consistent with first-process warm-up (v1: 298 / 148 / ~40 / then 3.4 s), especially as update 2 brought the first 32k sequences; FA4 compiles are not persisted.
    - Rollouts are 2–3 batches ahead, so no cross-version spans yet (generation waits on training).
  - Batches: update 0–4 mean length 952 → 2,879; up to 32 responses > 8k and 3 at 32k per update; reward 0.18 → 0.08–0.11 (longer, harder batches).
  - Next: confirm steps settle within ~10 updates. If not, investigate the training attention path for long packs (ask Jacob before changing the backend).
- 2:05 PM PST — **v2 throughput problem (needs Jacob's decision).** 10 updates in 70 min.
  - Training per update: 335 / 211 / 541 / 466 / 303 / 332 / 183 / 137 / 222 / 127 / 186 s; tok/s 0.7–1.9k (updates 0–5) → **2.5–4.3k (6–10), flat**. v1 B late: 58.7k tok/s.
  - Trainer-bound: generation_wait 0 s since update 1.
  - Token mix: 40–64% of tokens are in sequences > 8k (v1: 6% in > 4k).
  - Ruled out:
    - FA4 JIT: its compile messages stopped by 1:20 PM PST, and throughput stayed flat after.
    - Row specialization: already `dynamic`.
    - Standalone scoring pass: not run.
    - Not starved for generation.
  - Most likely: the trainer is ~15–20× slower per token on very long sequences (GDN recurrent chunk scan is sequential along each sequence, so packs with a few 10–32k sequences underuse the GPU; also 32k full-attention docs and recompute). Unverified: needs a profile or micro-benchmark.
  - At this rate, 400 updates ≈ 16–27 h, more as lengths grow.
  - Numerics healthy: logprob diff 0.0075–0.0092, KL ~1e-4, clipfrac 0, ESS 1.0, grad norms 0.08–0.19.
  - **First cross-version response at update 10** (max version span 1): refresh's continuation path is now exercised.
  - v2 left running unchanged pending Jacob.
- 2:10 PM PST — Jacob: "profile and then we try 8k and 16k"; this is to show the models work for smaller-scale RL, not an idealized setting. Q: could more generators help? A: not at 32k: the trainer is the bottleneck (generation_wait 0); likely yes at shorter caps (v1 at 8k was generation-bound). Decide topology per cap from the profile.
- 2:15 PM PST — **Benchmark** `tools/bench_long_seq.py`: Core fwd+bwd on 1 B300 mirroring MILES's trainer setup (fp32 params, bf16 autocast, full AC, flash_4, packed docs), same 32,768 tokens as 32×1k / 8×4k / 2×16k / 1×32k, with a kernel profile for 1k vs 32k, plus a no-AC timing run. https://beaker.org/ex/01M7RBNT8EEVAT9NS40C45CQHG (job `01M7RBNTC483DE8W22G6JN5GXF`, excludes holmes-cs-aus-511). Output `monitor/bench-long-seq-20261118/`.
- 2:14 PM PST — `filter_by_pass_rate.py --cap` (sweep mode: a response that reached the cap counts as truncated, which is exact). From the same LC 32k sweep:
  - **8k set (cap 7,680): 4,353 prompts**; **16k set (cap 16,384): 4,845**; 32k: 4,900.
  - 547 prompts are solved only with responses > 7,680 tokens.
  - Dirs: `data/dolci-math-base-r1-v1-passband-lc2.7b-8k`, `…-16k`.
- 2:18 PM PST — Jacob: up to **3 nodes** total. Plan: keep v2 (32k) running as the third data point and launch the 8k and 16k runs alongside it, with the GPU split per cap chosen from the benchmark.
- 2:30 PM PST — **Benchmark 1 result (kernels are fine):** single-B300 Core fwd+bwd on 32,768 tokens (MILES setup):
  - 32×1k docs: 29.5k tok/s; 8×4k: 29.2k; 2×16k: 28.2k; 1×32k: 26.1k (only 13% slower).
  - GDN chunk kernels 0.12 → 0.19 s and GDN other 0.03 → 0.09 s at 1k → 32k; matmuls 0.37 s both. (No-AC variant OOMed, as expected.)
  - So **no better GDN kernel is needed**; v2's trainer (~0.3–4k tok/GPU/s) is 10–30× below the model's long-document speed.
  - Also ruled out: memory (0 allocator retries/OOMs; peak reserved 151 GB); quadratic per-sample cost (step time vs Σlen² corr 0.29).
  - v2 step times are erratic: step 17 = 5 s (18.6k tok/GPU/s, near benchmark), neighbours 53–186 s.
  - Correction: FA4's compile message is a Python UserWarning, printed only once per process, so its disappearance did not show that compilation stopped.
  - New hypothesis: per-shape recompilation (FA4 CuTe varlen and/or Triton autotune) on every new packed-document layout.
- 2:33 PM PST — **Benchmark 2 (shape churn)** `tools/bench_shape_churn.py`: 8 packs built from real v2 response lengths (40,960-token budget), timed cold then warm, FA4 compile warnings counted per pack, Triton autotune printing on; flash_4 vs flash_2 in fresh processes; prints fla/flash_attn versions. https://beaker.org/ex/01M7RCKCNVNXZRM59M6FMTW5GJ (job `01M7RCKCSCSFKGBH3WKKCMY2A5`).
- 2:56 PM PST — **Root cause of the slow training: FLA Triton autotune keyed on pack length** (benchmark 2: 8 packs from real v2 lengths, cold vs warm, single B300).
  - Cold packs took 33–71 s (0.4–1.1k tok/s); the same packs warm took 1.1–1.9 s (20–29k tok/s). Same with flash_2 (cold 41–81 s), so not attention.
  - Every cold slowdown coincides with autotuning of `causal_conv1d_fwd/bwd_kernel` (32 configs each, ~33 s) and sometimes `l2norm_fwd/bwd_kernel` (25 each, ~30 s). GDN chunk kernels tune once.
  - fla-core 0.5.2 keys: conv `key=['D','W','NB']` with `NB = cdiv(B*T, 1024)`; l2norm `key=['D','NB']`. So each new pack length (rounded up to 1,024 tokens) triggers a fresh ~33–70 s autotune.
  - Results are cached per process (and `FLA_CACHE_RESULTS=1` passes `cache_results=True` to Triton), so this is warm-up that converges as the ~40 possible keys at 40,960-token packs are seen. v1 (16k packs, ≤16 keys) converged in ~10 updates.
  - Versions: fla 0.5.2, flash_attn 2.8.3.post1. No new kernel needed. A durable fix would pad packs to fixed lengths or pre-tune all NB values (code change; not done).
- **v2 has converged**: median training per update 262 s (0–9) → 98 → 80 → 19 → 31 s (40–48) while tokens/update grew 570k → 1.4M (≈55k tok/s, v1-level).
  - Last 10 updates: training 27 s + generation wait 26 s, **wall cycle 71 s**. Mean response length 5,685 tokens; max version span 3–6 per update (refresh continuing responses); 0–2 at 32k per update.
  - ~6.9 h left + evals (finish ≈ 10 PM PST).
- 2:57 PM PST — **8k and 16k runs launched** (3 nodes total, Jacob-approved), 4+4 like v2. Configs from `make_v2_refresh_config.py` differ from v2 only in cap-dependent fields and cap-matched data; validate + preflight OK; submitted via `submit_excluding.py`.
  - 8k (`runs/olmicro-2.7b-lc-math-rl-v2-refresh-8k-20261118.toml`, 4,353 prompts, 7,680 cap, packing 16,384): https://beaker.org/ex/01M7RDZW2BQJEVFWZFRYWDMAG5 (job `01M7RDZW5WV0FEDE9EZ5KJN5XA`).
  - 16k (`…-16k-20261118.toml`, 4,845 prompts, 16,384 cap, packing 24,576): https://beaker.org/ex/01M7RE04EEC9R1PP7ESR4X1Q0D (job `01M7RE04JHYBPQ36TE3EPEK6A8`).
- 3:07 PM PST — 8k and 16k both started (scheduled 3:01, started 3:07–3:08 PM PST; nodes `01M5FV8VTK2G0FYP5BF61TX5RE` (8k) and `01M5ZT6PVFNF5P264DENZK2J7E` (16k), neither the excluded 511). Three nodes in use now (32k, 8k, 16k).
- 3:40 PM PST — check of all three runs.
  - **32k (update 68/400):**
    - Update-50 eval: **MATH-500 62.6% (from 37.4), GSM8K Platinum 82.1% (from 68.3)**. After 50 updates it is already close to v0's final MATH-500 of 64.6.
    - Train reward 0.16 (0–49) → 0.35 (50–67).
    - Response length is growing fast: median 736 (versions 0–19) → 900 → 5,886 (40–54) → **9,194 (55–69)**.
    - Previews show R1-style "Okay, so I need to…" reasoning. Only 0.2% hit the 32,768 cap and nothing else is truncated, so this isn't runaway looping. Response text isn't stored (`responses_included: false`), so loop rate can't be measured directly.
    - Accuracy by length (versions 55–69): <2k 40%, 2–8k **58%**, 8–16k 26%, 16–32k 15%. New `tools/length_by_version.py` gives this from records without text.
    - Wall cycle 93 s/update (generation wait ~57 s, train ~30 s), longer because responses are longer; ETA ~8.6 h + evals (~12:15 AM PST). That's past the 8 h `min_runtime` (protected to ~8:55 PM PST); `auto_resume` with 3 retries and checkpoints every 50 updates cover a late preemption.
    - Grad norm 0.05–0.22, logprob diff median 0.0076, no NaN. The 5 tracebacks are the benign SGLang warm-up GC-freeze request at startup.
  - **8k and 16k (update 4/400):**
    - Step-0 evals: 8k MATH-500 37.3 / GSM8K 67.3; 16k 38.5 / 68.1 (same model and cap-matched eval lengths, so ≈ the 32k run's 37.4 / 68.3).
    - Training 277–634 s/update, the known FLA autotune warm-up.
    - The compiler cache does **not** help: `compiler-cache-progress.json` says restore `unavailable — External Core model-config files are not qualified by this wrapper`. Every run re-tunes from scratch (shared root holds only 108 KB).
    - Fewer pack lengths (8k: ≤16 NB keys; 16k: ≤24) should converge faster than 32k's ~40 updates.
    - 8k truncation 7.2% at 7,680 tokens (versions 0–19); 16k 1.6% at 16,384.
- 3:57 PM PST — **ETA correction for 32k:** update 79/400; wall 100 s/update; mean length has levelled at ~10k (2.5M response tokens/update).
  - Evals are blocking and slow at this length: 675 s at step 0, **993 s at update 50**.
  - Remaining ≈ 321 × 100 s ≈ 8.9 h + 7 evals × ~17–20 min ≈ 2.2 h → **finish ≈ 3 AM PST** (not the 12:15 AM I logged at 3:40, which ignored eval time).
  - The 8 h `min_runtime` window ends ≈ 8:55 PM PST, so the last ~6 h are preemptible (auto_resume from 50-update checkpoints).
- Resume mechanics checked for a possible generator increase:
  - The same `output.root` with a changed spec is refused ("Run configuration changed; choose a new output.root").
  - A new run can instead set `miles.load` to this run's `checkpoints/`: workflow reads `resume_manifest`, sets `start_rollout_id`, and restores the prompt cursor from `rollout/global_dataset_state_dict_N.pt` — a native Core RL resume (`constraints.py`).
  - Not yet tested by us.
- 4:01 PM PST — **16k run canceled** (Jacob: free its node for the 32k run, maybe rerun 16k later). It had 4 updates (still in FLA warm-up); step-0 eval MATH-500 38.5 / GSM8K 68.1. Its config/data remain for a rerun.
- 4:05 PM PST — **32k moves to 2 nodes (4 trainer + 12 inference) at the update-100 boundary** (Jacob-approved; stays at 3 nodes: 32k ×2 + 8k).
  - New config `runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-4x12-20261118.toml`: same recipe; `inference.gpus = 12`; new `output.root`/name; `[miles] load = <4+4 run>/checkpoints` (native Core RL resume).
  - `async_max_concurrent_samples` stays 512, so in-flight work and off-policy lag per update are unchanged; the launcher warns 512 < 1,536 serving slots, which is intended.
  - Resume only requires the same trainer world size/EP (`validate_topology`); the inference count is free. A resumed run skips the initial eval (`start_rollout_id != 0`), so the old run finishes its update-100 eval first.
  - validate / plan / preflight OK; dry run shows 2 × 8-GPU replicas, bad node 511 excluded.
  - `tools/switch_32k_to_4x12.sh` (background, log `monitor/v2-switch-4x12.log`): waits for checkpoint 99 + eval 99, checks cursor 99, stops the old experiment, submits the new run via `submit_excluding.py`, writes `monitor/v2-4x12-job-id`, and starts a `wait_job.sh` watcher (`monitor/v2-4x12-status.log`) and a 25-min start alarm. On anything unexpected it logs and exits without submitting.
  - Expected handoff ≈ 4:50 PM PST.
- 4:10 PM PST — **Warm-up fix found (background investigation; notes `runs/warmup-fix-notes-20261118.md`)**.
  - FLA 0.5.2's own config cache: `FLA_CACHE_MODE=default` + `FLA_CONFIG_DIR=/weka/oe-adapt-default/jacobm/olmicro/fla-configs/b300-fla0.5.2-v1` makes `causal_conv1d_*` and `l2norm_*` use a fixed winning config instead of autotuning per NB. Config/env only, no image rebuild.
  - Measured warm-up in the live runs: ~89 min (32k), ~33 min (8k).
  - `layer_norm_gated` (~2 min) still tunes.
  - Also an optional compiler-cache patch (`tools/patches/`, needs an image rebuild; not applied).
- 4:12 PM PST — **validation submitted while only 2 nodes are in use**: https://beaker.org/ex/01M7RJ41AFJECKMSAEVNVY8Q58 (job `01M7RJ41SE49YQECJP9WHBPNT5`).
  - Spec `tools/bench_fla_config_cache_parallel_beaker.json`: 2 GPUs, baseline and fix arms in parallel, same seeded packs; output in `monitor/bench-fla-config-cache/`.
  - Pass = no conv/l2norm autotune in the fix arm, cold packs ≤5 s, warm tok/s within 3%, loss parity.
  - If it passes before the 32k handoff submits (~4:48 PM PST), add the two env lines to the 4+12 config so the resumed run skips most of its ~89 min re-tune.
- 4:16 PM PST — Jacob: do the FLA fix; the 3-node limit isn't strict ("you can use a few more as needed").
  - The 4+12 config now has `FLA_CACHE_MODE=default`, `FLA_CONFIG_DIR=…/fla-configs/b300-fla0.5.2-v1` and `TRITON_PRINT_AUTOTUNING=1` under `[launch.env]`; it validates and the env shows in the plan.
  - The handoff script was replaced by `tools/switch_32k_to_4x12_v2.sh` (log `monitor/v2-switch-4x12.log`). It submits as soon as old checkpoint 99 + cursor are committed (≈4:29 PM PST) and stops the old run after its update-100 eval (≈4:47 PM PST), so the two overlap for ~15 min.
  - If `monitor/fla-fix-failed` exists at submit time, it strips the FLA env block first.
  - The validation bench was still unscheduled at 4:16 PM PST.
- 4:33 PM PST — **The 4+4 32k run's update-100 eval failed**: `RayTaskError(TimeoutError)` in `RolloutExecutor.eval()`; workflow status failed.
  - Cause: before a blocking eval, refresh mode drains in-flight generation with `core.engine_drain_timeout` (default **180 s**). The update-50 drain took 147 s; at ~10k-token responses (some to 32k) the update-100 drain exceeded 180 s.
  - Checkpoint 99 + cursor were committed first (4:29 PM PST), so no training is lost. **The update-100 eval point for 32k is lost** (a resumed run skips the initial eval); it could be recovered later by an offline eval of `iter_0000099` while it is retained.
  - The handoff script took the job ending as its stop condition and stopped the experiment (which also prevented a pointless auto-retry).
  - First 4+12 submission (`01M7RKBAHR9WXTAFPWJ9RD5P3W`, job id recorded as None because Beaker hadn't created jobs yet) was stopped before scheduling (no run root created).
- 4:36 PM PST — **4+12 resubmitted** with `core.engine_drain_timeout = 1800.0` (= `refresh_request_timeout`) plus the FLA config cache env: https://beaker.org/ex/01M7RKNZWBXJDDX8VQ8CQ2P84R (jobs `01M7RKP065VG96S0GTVZZY9RRC`, `01M7RKP0E160NV1X17NSY9DE9H`).
  - validate OK; receipt `monitor/v2-4x12-launch-receipt.json`; watcher `monitor/v2-4x12-status.log`; 25-min start alarm.
  - Stale watchers (old 32k, 16k, "None") were cleaned up.
- **8k (update 93/400)**:
  - Update-50 eval **MATH-500 52.3 / GSM8K 77.4** (step 0: 37.3 / 67.3; the 32k run had 62.6 / 82.1 at update 50).
  - Wall 43 s/update (train 21 s, generation wait 18 s): the warm-up ended without the fix. ETA ~3.7 h + evals ≈ 8:45 PM PST.
  - Mean length 1,620 (0–49) → 4,436 (50–92), and **36.5% of responses hit the 7,680 cap** (reward 0) in 50–92: the cap is binding.
  - Its eval drain is bounded by the 7,680-token cap (update-50 eval took 279 s in total), but update 100 is the riskier one.
  - The new traceback is a transient router 500 on one group, which was retried.
- 4:49 PM PST — **8k update-100 eval passed** (drain 54 s, eval 425 s): **MATH-500 66.6, GSM8K 87.9** (update 50: 52.3 / 77.4; step 0: 37.3 / 67.3). 103/400 updates.
- 4:52 PM PST — **4+12 run not scheduled yet** (16 min), and the 2-GPU FLA validation not either (45 min).
  - `beaker cluster usage ai2/holmes`: 413 urgent + 136 high slots allocated out of 576, 19 unassigned scattered (max 4 per node), 8 preemptible. No whole node is free.
  - The nodes we released (16k at 4:01, old 32k at 4:33 PM PST) were taken by others.
  - Keep the 2-node job queued: it beats a 1-node fallback unless the wait exceeds ~6 h (2-node remaining ≈ 4.5 h incl. evals vs 1-node ≈ 10.5 h).
  - 8k watcher re-armed.
- 5:20 PM PST — **FLA config-cache fix validated** (job `01M7RJ41SE49YQECJP9WHBPNT5`, 2 GPUs in parallel, same seeded 8 real-length 40,960-token packs; fla 0.5.2, triton 3.7.1, torch 2.13.0+cu130):

  | | baseline | fix |
  |---|---|---|
  | cold packs (s) | 77, 41, 76, 42, 76, 1.5, 1.4, 41 (**355 s total**) | 16, 3.9, 16, 3.6, 17, 1.5, 1.4, 3.6 (**63 s**) |
  | warm tok/s | 21.5–28.5k | 25.2–28.3k (same) |
  | loss | | **identical to every printed digit**, cold and warm |
  | autotunes | 53 | 17 |

  - The autotunes removed are all 28 `causal_conv1d_*` and 8 `l2norm_*`.
  - Left: one-time GDN chunk kernels (9) plus `layer_norm_gated_fwd/bwd` (4 + 4, ~12 s per new NB; plain `triton.autotune`, which the FLA cache doesn't cover). That accounts for the 16 s packs.
  - Expected remaining per-run warm-up: a few minutes instead of 30–90.
  - `tools/make_v2_refresh_config.py` now emits the FLA env block and `engine_drain_timeout = 1800.0` for all future configs; dry run for 16k diffs only in those lines.
- 4+12 32k run still unscheduled at 5:20 PM PST (submitted 4:36).
- 5:27 PM PST — 4+12 scheduled after ~51 min in queue: cluster capacity, not a job problem. Beaker gives no stated reason; the job just had no `scheduled` time while 549/576 GPUs were allocated to urgent/high jobs inside their protected windows and no two whole nodes were free.
  - Nodes: `holmes-cs-aus-495` (replica 0) + `01M65HW1CM5Z6JB0HX3KC4CA83` (replica 1, which hosts the driver: **its log is the one to read**; replica 0's log is 18 lines).
  - Started 5:31 PM PST. Startup: placement 40 s, trainer 102 s, serving 186 s, initial publication 20 s.
- 5:54 PM PST — **4+12 healthy and resumed correctly at rollout 100**: reward 0.38 (old run 0.36 at updates 50–78), mean length ~10.5k, grad norm 0.05–0.09, logprob diff 0.0075.
  - **FLA fix active in the Ray trainer workers: zero `causal_conv1d`/`l2norm` autotunes.** Remaining: `layer_norm_gated` fwd/bwd (58 across 4 ranks) + one-time chunk kernels.
  - Training per update over rollouts 100–106: 342, 205, 114, 55, 50, 61, **36 s**: warm in ~7 updates (~15 min) instead of ~40 updates / ~89 min in the first run.
  - **Generation wait ≈ 0 s from rollout 101**: now trainer-bound (2–3.2M response tokens/update ≈ 40–55 s at ~55k tok/s). 12 engines are more than enough; 4+8 would likely match for future 32k runs.
  - 14 tracebacks, all the benign SGLang warm-up GC-freeze connection errors (more engines, more of them).
  - ETA ≈ 293 × ~45 s ≈ 3.7 h + 6 evals ≈ **10:20 PM PST**.
- 6:02 PM PST — **Recovering the 32k update-100 eval** (Jacob asked).
  - It was not lost because we ended the run early: the eval itself crashed on the 180 s drain deadline, and a resumed run skips the initial eval.
  - Exact path, same MILES eval code, data, sampling and verifiers:
    1. Export job `runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-u100-export-20261118.toml`: `miles.load` = 4+4 checkpoints (99), `num_rollouts = 100`, so the training loop is empty and the run only does `final_hf_export` of rollout 99 to `<root>/export-hf`. https://beaker.org/ex/01M7RRPQ6XQ0ZHAN7PAK202GFZ.
    2. Eval-only run `…-32k-u100-eval-20261118.toml`: model.source = that export; its step-0 eval = the update-100 eval. `num_rollouts = 1`, no checkpoints/export; stopped right after the initial eval.
  - Chained by `tools/run_u100_eval_chain.sh` (log `monitor/u100-eval-chain.log`).
- 6:05 PM PST — check.
  - **8k 200/400**: evals MATH-500 37.3 → 52.3 → 66.6 → **69.5** (update 150); GSM8K 67.3 → 77.4 → 87.9 → **87.9**. The update-200 eval is running.
    - Cap hits at 7,680 by 50-update block: 8% → 38% → 42% → **29%** (150–199; mean length 4.9k): the model is learning to finish inside the cap.
    - Eval drains 54 / 56 / 47 s (deadline 180), 37 s/update, ETA ≈ 8:30 PM PST.
    - The new traceback is another transient router 500 (group retried).
  - **32k 4+12 at rollout 118**: training is warm (32–46 s); **generation wait is back to 10–35 s**, so wall ≈ 62 s/update (4+4 was ~100).
    - Generation is now limited by the 512 in-flight cap (Little's law), not the engine count. Raising `async_max_concurrent_samples` would speed it up but increases off-policy lag; I'm not changing it mid-run.
    - Remaining 281 × ~62 s ≈ 4.9 h + 6 evals: **ETA ≈ 11:30 PM–midnight PST**.
    - No conv/l2norm autotunes; tracebacks are only the benign startup ones.
  - u100 export job `01M7RRPQC7RS308D0BW0PPNEQT` scheduled 6:03 PM PST.
- 6:12 PM PST — **Checkpoint retention** (Jacob: we'll want broader evals later).
  - All runs have `core.checkpoint_keep_last = 3`, so older 50-update checkpoints are pruned.
  - **8k u50 was already gone**; 8k u100 would have been pruned at u250.
  - Started `tools/preserve_checkpoints.py` (nohup, 14 h, log `monitor/preserve-checkpoints.log`). It hardlinks every committed checkpoint into `preserved-checkpoints/<run>/u<update>/`, each a self-contained `miles.load` root that can be exported to HF with an empty-loop run like the u100 export.
  - Hardlinks verified on Weka (link count 2). They cost no space until the run prunes its copy; then ~31 GB each, ≈ 465 GB total for both runs at the end.
  - Preserved now: 32k u50/u100, 8k u100/u150/u200. The 4+12 run is picked up from u150.
  - **Caveat:** the preserver lives in this session's container.
  - Future configs: `make_v2_refresh_config.py` adds `core.checkpoint_keep_every = 50` (keep every eval-interval checkpoint natively); validated.
- 6:33 PM PST — **32k update-100 eval recovered: MATH-500 70.9, GSM8K Platinum 88.3** (8k at update 100: 66.6 / 87.9).
  - Export job `01M7RRPQC7RS308D0BW0PPNEQT`: exit 0, workflow complete (6:06–6:12 PM PST), HF at `runs/…-32k-u100-export-20261118/export-hf`.
  - Eval-only run https://beaker.org/ex/01M7RS71B90TDDSCHBMB74BTYH: initial eval passed 6:33 PM PST, then stopped. Logs `monitor/u100-eval-logs.txt`.
  - 32k series so far: 0: 37.4 / 68.3; 50: 62.6 / 82.1; **100: 70.9 / 88.3**.
- 6:52 PM PST — check.
  - **32k update-150 eval passed with the new drain limit**: drain 135 s (deadline now 1,800 s), eval 539 s with 12 engines (vs 993 s on 4 at update 50). **MATH-500 72.8, GSM8K 88.2.**
    - 32k series: 37.4 / 68.3 → 62.6 / 82.1 → 70.9 / 88.3 → **72.8 / 88.2**.
    - Checkpoint 149 saved (20 s) and preserved (`preserved-checkpoints/…-32k-4x12-20261118/u0150`).
    - Update 153; ~62 s/update steady; mean length 10.8k; reward 0.41.
    - No conv/l2norm autotunes; the one new traceback is a transient router 500 (group retried).
    - ETA ≈ 247 × 62 s + 5 evals ≈ **11:50 PM PST**.
  - **8k update 274**: u250 eval **MATH-500 70.8, GSM8K 88.9** (u200 68.5 / 88.5). Drains 47–56 s.
    - **Cap hits at 7,680 keep falling**: 42% (100–149) → 29% → 17% (200–249) → **6.7% (250–273)**; mean length 5.5k → 4.1k. The 8k run is learning to be concise.
    - ETA ≈ **8:20 PM PST** incl. 3 evals + HF export.
  - Preserver alive (8k u250 and 32k u150 added).
  - Both `wait_job.sh` watchers had hit their 2 h limit and were re-armed at 6:53 PM PST.
- 7:54 PM PST — check.
  - **32k u200 eval: MATH-500 73.1, GSM8K 89.3** (drain 148 s, eval 540 s). Series: 37.4 / 68.3 → 62.6 / 82.1 → 70.9 / 88.3 → 72.8 / 88.2 → **73.1 / 89.3**.
    - Checkpoint 199 preserved. Update 204; ~62 s/update; reward 0.48 (200–203); mean length 10.6k.
    - ETA ≈ 196 × 62 s + 4 evals ≈ **11:55 PM PST**.
  - **8k update 380/400**: u300 **71.0 / 89.7**, u350 **72.7 / 90.2**. Cap hits 4.6% → 3.4% → 2.5%; mean length ~3.9k; drains 38–56 s.
    - Finishing ≈ 8:15 PM PST (final eval + HF export).
  - Preserver alive (8k u300/u350, 32k u200).
- 8:12 PM PST — **8k run finished cleanly**: job exit 0, `workflow.json` status complete (attempt 0); final checkpoint 399, final eval (368 s), `final_hf_export` → `runs/olmicro-2.7b-lc-math-rl-v2-refresh-8k-20261118/export-hf` (config.json, 5.4 GB bf16 model.safetensors, tokenizer, `.complete`); u400 preserved.
  - **Final 8k evals (u0 … u400, every 50):**
    - MATH-500: 37.3, 52.3, 66.6, 69.5, 68.5, 70.8, 71.0, 72.7, **73.6**
    - GSM8K Platinum: 67.3, 77.4, 87.9, 87.9, 88.5, 88.9, 89.7, 90.2, **90.9**
  - Cap hits at 7,680 by block: 8% → 38% → 42% → 29% → 17% → 4.6% → 3.4% → 2.5%; final mean length ~3.9k; train reward 0.14 → 0.43.
  - Note: `beaker job logs` for the finished job streams only part of the log (exit 1 at ~59k–170k lines). The complete console log is `wandb/wandb/run-*/files/output_*.log` in the run root (source of the final evals).
- 9:02 PM PST — **32k u250 eval: MATH-500 73.9, GSM8K 89.0** (eval 606 s, passed).
  - Series: 37.4 / 68.3 → 62.6 / 82.1 → 70.9 / 88.3 → 72.8 / 88.2 → 73.1 / 89.3 → **73.9 / 89.0**.
  - Checkpoint 249 preserved. Update 251; training 32–41 s/update.
  - ETA ≈ 149 × 62 s + 3 evals ≈ **12:05 AM PST**.
  - Preserver alive. The 4+12 watcher expired and was re-armed.
  - The chart artifact and `runs/plots/math500-2.7b-runs-20261118.csv` gained the u250 point.
- 8:58 PM PST — small-model filtering/RL plan drafted: `runs/small-models-filtering-plan-20261118.md` (not launched).
  - All sizes have head_dim 128 (attention + GDN key) and GDN value 256, same as 2.7B.
  - **u250 drain took 198.9 s.** It would have failed under the old 180 s `engine_drain_timeout`, so the 1,800 s setting is what kept this eval. Drains so far: 135 → 148 → 199 s.
- 9:32 PM PST — **Extra evals: full Minerva MATH + AIME 2026** on the 8k final and the 32k u250 checkpoint (Jacob: "normal minerva_math", no OCW).
  - Data `data/olmicro-extra-evals-r1-v1` (`data-prep/prepare_extra_evals.py`, R1 template, revisions pinned):
    - minerva_math = full Hendrycks MATH test, 5,000 problems from `EleutherAI/hendrycks_math` (olmo-eval's minerva_math suite); label = last `\boxed{}` of the reference solution; prompt limit 2,048 so none are dropped (longest 1,326 tokens).
    - aime2026 = `MathArena/aime_2026` (as olmo-eval) × 32 repeats.
    - The OCW file exists but is unused.
  - Protocol: zero-shot R1 template (olmo-eval's minerva variants are few-shot base prompts with 1,024 tokens). Temperature 1.0, top-p 0.95 (= in-loop evals and olmo-eval `aime_2026:pass_at_32_rlzero`), up to 32,768 new tokens, stop `\nUser:`. MATH 1 sample/problem, AIME 32.
  - Generation: `tools/format_probe.py` (now with `--top-p` and `--eval-files`; defaults unchanged), 8 engines per job.
  - Scoring (`tools/score_extra_evals.py`) with two graders:
    - MathVerifier as in training (truncated = 0);
    - olmo-eval `MathExtractor` + `is_equiv` (`MinervaMathScorer` semantics, golds as olmo-eval builds them; olmo-eval checkout `1eddca07`).
    - Gold self-check: 100% for both graders on MATH and AIME.
  - Reported: MATH micro + macro over 7 subjects (olmo-eval suite aggregation); AIME pass@1/4/8/16/32 plus pass@1 with responses > 16,384 tokens counted wrong (olmo-eval rlzero length).
  - 32k u250 HF export: job `01M7S3N7Z4EQVTS86Y6BVKAR6S`, exit 0 (9:17–9:23 PM PST), `runs/…-32k-u250-export-20261118/export-hf`.
  - Jobs:
    - 8k final https://beaker.org/ex/01M7S4FN0MQJPJAB0ERBP3QMX7
    - 32k u250 https://beaker.org/ex/01M7S4FRAT4Z92M3YBK0YQ1WRV
    - Outputs `monitor/extra-evals-<tag>-20261118/`; scoring chain `tools/run_extra_eval_scoring.sh` (log `monitor/extra-evals-scoring.log`).
- 9:36 PM PST — **1.4B pass-rate sweep launched** (Jacob: test the 2.7B-prefiltered sweep plan on the second-largest model first): https://beaker.org/ex/01M7S4NBZ091D1QSR3DF1XKZTP.
  - Pool `data/dolci-math-base-r1-v1-sweep-pool-lc2.7b`, 5,701 prompts:
    - the 4,901 the 2.7B long-context final solved ≥ 1/16 at 32k;
    - plus a seeded control of 800 of the 8,350 it never solved, which measures what the prefilter misses.
  - In the 2.7B sweep the never-solved 63% took 67% of generated tokens.
  - Same sampling as the 2.7B sweep (16 samples, temperature 1.0, top-p 1.0, 32,768 tokens, 32 resumable shards), hostname allowlist without 511. Output `monitor/passrate-sweep-lc1.4b-32k-20261118`.
  - Post-processing chain `tools/run_lc1.4b_sweep_postprocess.sh` (log `monitor/passrate-sweep-lc1.4b-postprocess.log`):
    - score;
    - bands for solved pool vs control;
    - pass-band sets at 32k and 16k (`data/dolci-math-base-r1-v1-passband-lc1.4b-{32k,16k}`);
    - length tail.
- **Setup agent stopped** (by Jacob's interrupt ~9:25 PM PST; its work counts as cancelled). State it left, checked:
  - All downloads present and verified (`sha256sum -c --ignore-missing`): 275m/450m `stage3-step0047684`, 810m/1.4b `stage3-step0023842`, HF files + Core `config.json`.
  - Smoke TOMLs `open-instruct/runs/olmicro-{275m,450m,810m,1.4b}-lc-smoke-20261118.toml` (2 trainer + 2 inference, 6 updates; trainer must stay 2 for `native_smoke.py`).
  - **Four smokes running** since 9:20–9:24 PM PST: 1.4b `01M7S3WPFZ7EGD0AJPY402C626`, 810m `01M7S3WZDYYM4SFEXR1ZFB7GCS`, 450m `01M7S3X5BWSWJEC848Q2ZAX7NF`, 275m `01M7S3XGE79PGVWP4D0S2JN0HY` (4 GPUs each). I'm watching them.
  - `tools/make_v2_refresh_config.py` now has `--size`; for 2.7B it emits the same config as before apart from corrected comments.
  - It did not write a log section.
- 9:49 PM PST — **First extra-eval result, 8k final (u400)** (10 min on 1 node; T=1.0, top-p 0.95, R1, 32k tokens):
  - **Minerva MATH (full 5,000 × 1): 73.0 macro over subjects / 75.5 micro (olmo-eval grader); MathVerifier 72.0 / 74.5.**
    - Algebra 90.6, prealgebra 86.2, number theory 79.3, C&P 68.6, geometry 65.6, intermediate algebra 61.4, precalculus 59.7.
    - Mean 2.4k tokens, 0.14% truncated.
  - **AIME 2026 (30 × 32): pass@1 9.8**; pass@4/8/16/32 = 17.5 / 22.3 / 27.6 / 33.3 (both graders identical). pass@1 at the 16k rlzero cap is also 9.8. Mean 5.6k tokens, 0.2% truncated.
- 9:55 PM PST — Jacob: also evaluate the base models and the 8k run at u250 (to match the 32k u250).
  - **Base models**, one job for all five long-context bases (2.7B, 1.4B, 810M, 450M, 275M, sequential per GPU): https://beaker.org/ex/01M7S66WRNEQAT0JTT4E248147.
  - **8k u250 HF export** (from the 8k run's own config so its prompt cursor matches; `miles.load` = `preserved-checkpoints/…-8k-20261118/u0250`): https://beaker.org/ex/01M7S6698A58GYKDR280E47913. Its eval is submitted automatically after the export.
  - Chain `tools/run_extra_eval_chain2.sh` (log `monitor/extra-evals-chain2.log`) scores everything with both graders.
- 10:02 PM PST — **32k u250 extra evals** (same protocol):
  - **Minerva MATH: 73.1 macro / 75.2 micro (olmo-eval); MathVerifier 72.0 / 74.2.** Algebra 89.4, prealgebra 85.8, number theory 79.6, C&P 69.0, geometry 67.2, precalculus 60.3, intermediate algebra 60.2. Mean 6.4k tokens (8k final: 2.4k), 0.9% truncated at 32k.
  - **AIME 2026: pass@1 11.2**; pass@4/8/16/32 = 20.9 / 26.6 / 33.3 / 40.0 (12 of 30 solved at least once). At the 16k rlzero cap pass@1 = 9.6. Mean 17.1k tokens, 2.9% truncated at 32k.
  - vs 8k final (u400): Minerva identical (73.0 macro); AIME pass@1 9.8 vs 11.2, pass@32 33.3 vs 40.0. The 32k model uses ~2.6× (MATH) and ~3× (AIME) the tokens.
- 10:15 PM PST — **32k u300 eval: MATH-500 74.9, GSM8K 89.4** (drain 207 s; deadline 1,800).
  - Series: 37.4 / 68.3 → 62.6 / 82.1 → 70.9 / 88.3 → 72.8 / 88.2 → 73.1 / 89.3 → 73.9 / 89.0 → **74.9 / 89.4**.
  - Checkpoint 299 preserved (u0300). Training ~40 s/update, cycle ~75–78 s.
  - ETA ≈ 12:30 AM PST.
  - Chart artifact and CSV updated.
- 10:20 PM PST — **All four small-model qualification smokes passed**. Each: exit 0, workflow complete, `native-smoke-complete.json` status passed (6 updates, resume of update 5 in a separate process), `graph-eager-parity.json` passed, no conv/l2norm autotunes (FLA cache env set).

  | size | finished (PST) | trainer–rollout logprob abs diff | first update (warm-up) → later updates |
  |---|---|---|---|
  | 1.4B | 9:55 PM | 0.0053–0.0059 | 179 s → 4–22 s |
  | 810M | 9:55 PM | 0.0085–0.0135 | 181 s → 1–21 s |
  | 450M | 10:04 PM | 0.0082–0.0114 | 171 s → 3–7 s |
  | 275M | 10:20 PM | 0.0144–0.0179 | 174 s → 1–11 s |

  - 2.7B runs: 0.0073–0.0077. The smaller models' larger mismatch is expected with higher-entropy distributions in bf16 and is TIS-corrected; watch it in the RL runs.
- 10:40 PM PST — **1.4B sweep done and filtered** (sweep 9:54–10:31 PM PST, all 32 shards exit 0; scored 10:40).
  - **2.7B-solved pool (4,901): 2,213 in band (45.2%)**, 2,688 never solved. **Control (800 the 2.7B never solved): 68 in band (8.5%)**, almost all at 1/16.
  - Mean 1.5k tokens; 0.3–0.5% at the 32k cap.
  - **1.4B set at 32k: 2,281 prompts** (`data/dolci-math-base-r1-v1-passband-lc1.4b-32k`), ≈ 5.6 epochs in 400 updates. At 16k: 2,278.
  - Kept the prefilter: sweeping the other 7,550 unsolved prompts would add ~640, nearly all at 1/16.
- 10:44 PM PST — Jacob (going offline): launch a 1.4B RL run like the 2.7B 32k run, then the full AIME/Minerva sweep over the 8k and 32k runs.
  - **1.4B RL launched**: https://beaker.org/ex/01M7S8K2VDV78BHR9N0B0BNYF8 (job `01M7S8K3600WTADBHFGKQNBZW0`).
    - Config `runs/olmicro-1.4b-lc-math-rl-refresh-32k-20261118.toml` (`make_v2_refresh_config.py --size 1.4b`). Its diff vs the 2.7B 32k config is only model, Core config, data, FLA cache env, `checkpoint_keep_every = 50` and `engine_drain_timeout = 1800`.
    - 4 trainer + 4 inference, refresh, staleness 12, 512 in flight, 32k cap, 400 updates, evals every 50, min_runtime 8h, urgent, 511 excluded. validate + preflight OK.
    - Watcher `monitor/1.4b-rl-status.log`; 25-min start alarm.
  - **Eval sweep orchestrator started** (`tools/eval_sweep_orchestrator.py --max-jobs 2 --hours 8`, nohup, log `monitor/eval-sweep-orchestrator.log`):
    - HF-export and evaluate 8k u100/u150/u200/u300/u350 and 32k u50/u150/u200/u300/u350 from preserved checkpoints; 32k u100 from its existing export; 32k u400 once that run completes.
    - 8k u250 and the base models come from chain2; 8k final and 32k u250 are done. 8k u50 was pruned before preservation.
    - Dry run validated all 10 export configs. Final table: `runs/plots/extra-evals-2.7b-20261118.csv`.
- 10:43 PM PST — 1.4B RL **started** (about 2 min after submission). The orchestrator's first two exports (8k u100, u150) were submitted at 10:41 PM PST.
- 1.4B sweep length tail (91,216 responses, 141.7M tokens, 5.1% correct):
  - 0–4k: 93% of responses / 50% of tokens. 4k–8k: 3% / 12% (37% repetition-loop tails). 8k–16k: 3% / 22%. 16k–32k: 1% / 16%, 1.5% correct, 30% truncated.
  - A 16k cap would lose 0.3% of correct answers (5.6% of tokens saved); a 7,680 cap would lose 4.5%.
- 11:15 PM PST — check.
  - **32k** at update 342 (u350 eval ≈ 11:35 PM PST).
  - **1.4B RL healthy**: update 37 at **~18 s/update** (training 8 s, generation wait 9 s); reward 0.11; mean length 1.3k; logprob diff median 0.0066; grad norm 0.06–0.21.
    - **Step-0 evals: MATH-500 23.5, GSM8K 55.8.** Tracebacks are only the benign startup connection retries.
    - ETA ~1.8 h + evals (lengths will grow).
  - **8k-u250 extra eval only got 2/8 shards**: 6 SGLang engines exited 137 at startup with "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.6 (minimum viable 0.601–0.653)".
    - That means ~35–40% of those GPUs was held by something else at 10:14 PM PST on node `01M6F0DP7X4ATCTRCD5YQSP1ES`, most likely a previous job still tearing down.
    - The 1.4B run started fine on the same node at 10:42 PM.
    - Resubmitted the same spec (format_probe skips finished shards): `01M7SAD9RECBYP90CC94WGRB88`; it is scored when done.
  - Orchestrator alive: 8k u100 and u150 exported OK; their evals are running (`01M7S93JWQ5DE75283SZWP12WK`, `01M7S9GSWBM084D718FH0Y852B`). Its incomplete-eval handling only logs, so check `eval incomplete` lines and resubmit by hand.
  - The base-model eval job (5 models) has been running since 10:01 PM PST.
  - 32k 4+12 watcher re-armed.
- 11:43 PM PST — check.
  - **32k u350 eval: MATH-500 76.1, GSM8K 89.4** (drain 202 s). Checkpoint 349 preserved. Update 356. Chart + CSV updated.
  - **1.4B** at update 100, ~12 s/update. Evals:

    | update | MATH-500 | GSM8K |
    |---|---|---|
    | 0 | 23.5 | 55.8 |
    | 50 | 28.8 | 58.3 |
    | 100 | **34.6** | **63.9** |

  - **Bug found and fixed: the eval specs I generated after the first two dropped `CUDA_VISIBLE_DEVICES=$gpu`.**
    - Affected: the base ×5 job, 8k-u250 (both attempts) and the orchestrator's 8k u100/u150 template. All 8 SGLang engines started on GPU 0.
    - The first took 60% of the GPU; most others failed with "Loaded weights leave no GPU memory for the KV cache" (exit 137); the survivors shared one GPU, hence the slowness.
    - This is not leftover memory from other jobs (my earlier guess was wrong). The 8k-final and 32k-u250 specs had the variable and are fine.
    - Fix:
      - patched every `tools/extra_evals_*_beaker.json`, including the orchestrator's template, which it reads per eval;
      - stopped the four affected jobs, keeping their 15 finished shards (valid: same generation, just slower);
      - resubmitted as two jobs, base ×5 `01M7SCABB46FF20NMDBW1PWDQ4` and 8k u100/u150/u250 combined `01M7SCAEJXKGKDKJNMGEZ5127D`. format_probe skips finished shards.
    - Scoring chain `tools/run_cvd_resubmit_scoring.sh` (log `monitor/extra-evals-cvd-scoring.log`).
    - The orchestrator marked 8k u100/u150 "incomplete" (handled by the combined job) and continues with the remaining exports/evals using the fixed template.
- 12:34 AM PST — check.
  - **32k at update 389** (~96 s/update incl. eval amortization). u400 eval + final export ≈ 1:10 AM PST.
  - **1.4B at update 200** (~38 s/update; lengths growing). Evals MATH-500 / GSM8K: 23.5 / 55.8 → 28.8 / 58.3 → 34.6 / 63.9 → **37.6 / 64.7 (u150)**; the u200 eval is running.
  - **Extra-eval sweep results so far** (olmo-eval grader; Minerva = macro over 7 subjects; AIME 2026 pass@1 / pass@32 from 32 samples; T=1.0, top-p 0.95, 32k):

    | checkpoint | Minerva MATH | AIME pass@1 | AIME pass@32 | mean tokens MATH / AIME |
    |---|---|---|---|---|
    | 2.7B base | 36.8 | 1.5 | 16.7 | 2.8k / 5.6k (6–12% truncated) |
    | 8k u100 | 68.6 | 8.2 | 30.0 | 4.1k / 11.9k |
    | 8k u150 | 69.2 | 9.7 | 40.0 | 3.4k / 9.8k |
    | 8k u200 | 69.1 | 9.3 | 40.0 | |
    | 8k u250 | 70.4 | 9.5 | 33.3 | 2.4k / 6.4k |
    | 8k u300 | 71.2 | 9.4 | 30.0 | |
    | 8k u350 | 73.0 | 10.7 | 36.7 | |
    | 8k u400 (final) | 73.0 | 9.8 | 33.3 | 2.4k / 5.6k |
    | 32k u250 | 73.1 | 11.2 | 40.0 | 6.4k / 17.1k |

  - Base models: 1.4B 21.8 / 0.1 / 3.3; 810M 10.3 / 0.1 / 3.3; 450M 5.0 / 0 / 0; 275M being scored.
  - Orchestrator alive: 32k u50 and u100 evals running; 32k u150/u200/u300/u350 exports to come; u400 after the run ends.
  - Preserver and watchers alive.
- 12:40 AM PST — 275M base: Minerva 3.0 macro, AIME 0 / 0. All five base models now evaluated.
  - Minerva (macro): 2.7B 36.8, 1.4B 21.8, 810M 10.3, 450M 5.0, 275M 3.0.
  - Bases truncate 6–10% of Minerva responses at 32k (rambling).
- 1:02 AM PST — **32k run finished**: workflow complete, final eval passed (drain 190 s, eval 670 s), `final_hf_export` → `runs/olmicro-2.7b-lc-math-rl-v2-refresh-32k-4x12-20261118/export-hf` (`.complete`), u400 preserved.
  - **Final u400: MATH-500 77.3, GSM8K 90.3.**
  - Full 32k series: 37.4 / 68.3 → 62.6 / 82.1 → 70.9 / 88.3 → 72.8 / 88.2 → 73.1 / 89.3 → 73.9 / 89.0 → 74.9 / 89.4 → 76.1 / 89.4 → **77.3 / 90.3**.
  - vs 8k final 73.6 / 90.9.
  - Chart (lc32 marked final) and CSV updated. The orchestrator picks up the final export for its Minerva/AIME eval.
- 2:05 AM PST — check.
  - **1.4B** at update 291 (~48 s/update; training 13–16 s).
    - Evals MATH-500 / GSM8K: 23.5 / 55.8 → 28.8 / 58.3 → 34.6 / 63.9 → 37.6 / 64.7 → **45.6 / 68.4 (u200)** → **47.7 / 69.6 (u250)**.
    - Mean length jumped 1.3k → 4.0k (150–199) → 6.9k → 7.7k, like the 2.7B did; ~0.05% at the 32k cap.
    - Drains 71–126 s. Watcher re-armed. ETA ≈ 4:15 AM PST.
  - **Extra-eval sweep: all checkpoints done except 32k u400** (eval submitted 1:52 AM PST). 32k (Minerva macro / AIME pass@1 / pass@32):

    | update | Minerva | AIME pass@1 | AIME pass@32 |
    |---|---|---|---|
    | u50 | 61.3 | 8.8 | 33.3 |
    | u100 | 70.9 | 9.9 | 36.7 |
    | u150 | 72.0 | 11.2 | 33.3 |
    | u200 | 72.3 | 10.7 | 36.7 |
    | u250 | 73.1 | 11.2 | 40.0 |
    | u300 | 74.7 | 12.8 | 40.0 |
    | u350 | **76.4** | **15.2** | **50.0** |

  - Unlike 8k, the 32k run keeps improving on AIME late in training.
- 2:20 AM PST — **Minerva/AIME sweep complete** (`runs/plots/extra-evals-2.7b-20261118.csv`). Final columns (olmo-eval grader):

  | checkpoint | Minerva macro | AIME p@1 | AIME p@32 | AIME p@1 at 16k cap | mean tokens MATH / AIME |
  |---|---|---|---|---|---|
  | base 2.7B | 36.8 | 1.5 | 16.7 | 1.5 | 2.8k / 5.6k |
  | 8k final | 73.0 | 9.8 | 33.3 | 9.8 | 2.4k / 5.6k |
  | **32k final** | **77.1** | **14.5** | 36.7 | 11.5 | 7.9k / 20.5k |
  | 32k u350 (best AIME) | 76.4 | 15.2 | 50.0 | 12.3 | 7.2k / 19.5k |

  - 32k beats 8k by ~4 Minerva points and ~4–5 AIME pass@1 points at the end.
  - It uses ~3.3× (MATH) / ~3.7× (AIME) the tokens. About a third of its AIME gain needs responses longer than 16k.
  - **8k AIME is flat (~9–11) from u100**; 32k keeps rising late (9.9 at u100 → 14.5–15.2 at u350–400).
  - Published as a new section of https://claude.ai/artifact/1z2YjPmJBHeaqggTUDzhEj: two charts (Minerva, AIME pass@1) plus a table. Builder: `scratchpad/build_extra_section.py`.
- 3:29 AM PST — **1.4B** at update 359 (training 14–19 s/update; the wall average includes the 14-min u350 eval).
  - Evals MATH-500 / GSM8K: … 47.7 / 69.6 (u250) → **49.7 / 71.3 (u300)** → **50.7 / 71.5 (u350)**.
  - Mean length ~8.5k (0.08% at the 32k cap); drains 85–116 s.
  - `checkpoint_keep_every = 50` works natively (rollout_0000049 … 349 all kept).
  - ETA ≈ 4:25 AM PST.
  - Chain `tools/run_1.4b_final_chain.sh` (log `monitor/1.4b-final-chain.log`): wait for completion + export → Minerva/AIME eval on the final export (spec `tools/extra_evals_1.4b-final_beaker.json`, has CUDA_VISIBLE_DEVICES) → score.
- 4:21 AM PST — **1.4B RL finished cleanly**: job exit 0, workflow complete, `export-hf/.complete`, all 8 checkpoints kept natively (rollout_0000049 … 399). Wall time 5 h 39 min on one node.
  - **Final in-loop: MATH-500 49.9, GSM8K 73.2** (u350 50.7 / 71.5; base 23.5 / 55.8).
  - 4:37 AM PST — **Minerva/AIME on the 1.4B final**: https://beaker.org/ex/01M7SW346DETNN8B7BYNRMKWBM, 8/8 shards.
    - Minerva 49.6 macro / 53.2 micro (base 21.8); AIME pass@1 0.9 (base 0.1), pass@32 13.3 (base 3.3).
    - Mean tokens 6.8k / 13.3k; truncated 3.8% / 7.6% at 32k (more than the 2.7B).
  - Added a 1.4B section to https://claude.ai/artifact/1z2YjPmJBHeaqggTUDzhEj: in-loop MATH-500 chart + base-vs-final table; builder `scratchpad/build_size_section.py`.
- 4:40 AM PST — **All overnight work finished.**
  - The checkpoint preserver was stopped (all runs done). `preserved-checkpoints/` holds 452 GB: 2.7B 8k u100–u400, 32k u50–u400 (both runs). The 1.4B keeps its own checkpoints natively.
  - The fallback wake-up was deleted. Nothing is running.

## 2026-11-19 — small-model difficulty filtering

- 8:49 AM PST — Jacob: launch difficulty filtering on the smaller models. **Sweeps submitted in parallel** (one node each, nothing else running): 810M `01M7T7YHQPZ4X8QGPTXEMWX56A`, 450M `01M7T7YMTQ62TQB8CZ03K0VY3T`, 275M `01M7T7YQZGAPFASG3EDHC2MBV4`.
  - Same as the 1.4B: pool `data/dolci-math-base-r1-v1-sweep-pool-lc2.7b` (4,901 2.7B-solved + 800 control), 16 samples, T=1.0, top-p 1.0, 32,768 tokens, 32 resumable shards, CUDA_VISIBLE_DEVICES per engine, 511 excluded.
  - Specs `tools/passrate_sweep_lc{810m,450m,275m}_32k_beaker.json`.
  - Post-processing per size: `tools/run_lc_sweep_postprocess.sh SIZE EXP` (nohup; logs `monitor/passrate-sweep-lc<size>-postprocess.log`). It scores, splits pool vs control bands, writes pass-band sets `data/dolci-math-base-r1-v1-passband-lc<size>-{32k,16k}`, and runs the length tail.
- 8:51 AM PST — all three sweeps started (810M and 450M at 8:51:21, 275M at 8:51:53).
- 8:56 AM PST — Jacob: Minerva and AIME plots with three curves (2.7B 8k finished run, 2.7B 32k, 1.4B 32k).
  - The 1.4B had only base + final, so I'm evaluating its intermediate checkpoints. Hardlinked them to `preserved-checkpoints/olmicro-1.4b-lc-math-rl-refresh-32k-20261118/u0050…u0400`.
  - Export template `open-instruct/runs/olmicro-1.4b-lc-math-rl-refresh-32k-u250-export-20261118.toml`.
  - Orchestrator gained `--set 1.4b` (writes `runs/plots/extra-evals-1.4b-20261118.csv`); running with `--max-jobs 3`, log `monitor/eval-sweep-orchestrator-1.4b.log`.
  - Colors: the validator fails green (and violet) beside blue/orange for colour-blind readers under all-pairs. The palette's first three slots (blue, orange, aqua) are the only all-pairs-safe set, so the 1.4B is aqua in the new charts and in the 1.4B section. Aqua is also the dashed midtrain v0 line in the MATH-500 chart, a separate figure with its own legend.
- 10:02 AM PST — **Small-model sweeps done** (all 32/32 shards, exit 0; sweeps 8:51–9:20 AM PST, ~25–28 min each; scored + filtered by ~9:40):

  | size | 2.7B-solved pool in band | control in band | kept @32k (16k) | in-band at 1/16 | ≥4/16 | est. extra from the other 7,550 | epochs at 400 updates |
  |---|---|---|---|---|---|---|---|
  | 1.4B (ref) | 2,213 (45%) | 8.5% | 2,281 (2,278) | 50% | 296 | ~640 | 5.6 |
  | 810M | 1,528 (31%) | 6.1% | **1,577** (1,576) | 63% | 82 | ~460 | 8.1 |
  | 450M | 1,026 (21%) | 4.8% | **1,064** (1,063) | 70% | 50 | ~360 | 12.0 |
  | 275M | 919 (19%) | 2.8% | **941** (939) | 70% | 24 | ~210 | 13.6 |

  - Sets: `data/dolci-math-base-r1-v1-passband-lc{810m,450m,275m}-{32k,16k}`.
  - Length tails: a 16k cap loses ≤ 0.5% of correct answers at every size. The >16k responses are mostly loops: 810M 0.5% of responses / 8% of tokens; 450M 1.2% / 28% (50% truncated at 32k); 275M 2.1% / 32% (43% truncated).
  - A 7,680 cap would lose 3–5% of correct answers.
- Recommendation (for Jacob):
  - Sweep the remaining 7,550 prompts the 2.7B never solved for 450M and 275M (~35 min each, one node each). Projected sets ~1,420 and ~1,150, still at or below the floor, nearly all at 1/16.
  - 810M is above the floor (1,577); sweeping its remainder (~2,040) is optional.
  - Then either keep 400 updates for consistency (450M/275M ≈ 9–11 epochs) or cut them to ~250.
  - Cap: keep 32k for consistency with 2.7B/1.4B. 16k would cut ~30% of the small models' sweep tokens (loops) at no measurable loss, so it's a reasonable compute saver for 450M/275M if Jacob prefers.
- 10:36 AM PST — **1.4B checkpoint evals done** (`runs/plots/extra-evals-1.4b-20261118.csv`).
  - Minerva (macro): 21.8 → 27.5 → 32.4 → 36.2 → 43.8 → 45.7 → 47.6 → 47.4 → 49.6 (u0 … u400).
  - AIME pass@1: 0.1 → 0.4 → 0.6 → 0.4 → 0.8 → 1.1 → 0.9 → 1.0 → 0.9; pass@32 3.3 → 6.7–16.7.
  - Mean AIME tokens 3.7k (u100) → 13.3k (final).
  - Published the three-curve Minerva and AIME charts (2.7B 32k blue, 2.7B 8k orange, 1.4B 32k aqua) on https://claude.ai/artifact/1z2YjPmJBHeaqggTUDzhEj.
- 10:39 AM PST — Jacob: sweep the rest of the pool ("can't hurt"); keep 400 updates for every run for comparability. Cap stays 32k.
  - **Rest-of-pool sweeps submitted for all three sizes**: 810M `01M7THPHB3BZ9Z21DF55B95DVM`, 450M `01M7THPMQG74ME7YX5D9RMXKDS`, 275M `01M7THPQZPQ14CT08TW4AMEQA7`.
    - Data `data/dolci-math-base-r1-v1-sweep-pool-lc2.7b-rest`: the 7,550 prompts the 2.7B never solved minus the 800 control; pool + rest = all 13,251.
    - Specs `tools/passrate_sweep_lc<size>_32k_rest_beaker.json`.
  - Post-processing `tools/run_lc_rest_postprocess.sh SIZE EXP` (nohup; logs `monitor/passrate-sweep-lc<size>-rest-postprocess.log`):
    - score;
    - symlink both sweeps' scored files into `monitor/passrate-sweep-lc<size>-32k-combined-20261118`;
    - filter over all 13,251 prompts → `data/dolci-math-base-r1-v1-passband-lc<size>-{32k,16k}-full`.
- 10:45 AM PST — Jacob: launch the **810M RL run first**; **hold 450M and 275M** (others may need workspace GPUs). The 810M run launches once its full set exists; the 450M/275M sweeps finish but their RL runs wait.
- 11:26 AM PST — **810M full set: 2,043 prompts** (`data/dolci-math-base-r1-v1-passband-lc810m-32k-full`).
  - The rest sweep (10:39–11:15 AM PST, 32/32 exit 0) added 466 in band (6.2% of 7,550; 399 at 1/16), matching the control's projection (~460). All 13,251 prompts observed. The 16k set would be 2,042.
- 11:27 AM PST — **810M RL launched** (Jacob: 810M first; hold 450M/275M): https://beaker.org/ex/01M7TME1MYB7DQSCJJQ8WKQZ2T (job `01M7TME1RMK3DW0MSWCMQNF68C`).
  - Config `runs/olmicro-810m-lc-math-rl-refresh-32k-20261119.toml` (`make_v2_refresh_config.py --size 810m`). Its diff vs the 1.4B config is only model, Core config, data, and 2 trainer + 6 inference.
  - 400 updates, 32k, refresh, 512 in flight, evals every 50, `checkpoint_keep_every = 50`, FLA cache, drain 1800, min_runtime 8h, urgent, 511 excluded. validate + preflight OK.
  - Watcher `monitor/810m-rl-status.log`.
- 11:29 AM PST — 810M RL **started** (~2 min after submission). The 450M/275M rest sweeps finished at 11:17 AM PST (32/32, exit 0); scoring and filtering are in progress.
- 11:44 AM PST — **Rest-of-pool filtering done for all three sizes** (full sets over all 13,251 prompts, 32k cap):

  | size | first pool | + rest (rate, at 1/16) | **full set** | epochs at 400 updates |
  |---|---|---|---|---|
  | 810M | 1,577 | +466 (6.2%, 399) | **2,043** | 6.3 |
  | 450M | 1,064 | +344 (4.6%, 297) | **1,408** | 9.1 |
  | 275M | 941 | +287 (3.8%, 251) | **1,228** | 10.4 |

  - Sets `data/dolci-math-base-r1-v1-passband-lc{810m,450m,275m}-32k-full`; the 16k versions differ by ≤ 2 prompts.
  - 450M/275M remain just below the ~1,500 floor. Per Jacob, keep 400 updates for comparability; their RL runs are on hold.
- 12:46 PM PST — **810M RL at update 300/400**, very fast: ~3.3 s/update (training 2.4 s, generation wait 0.1 s); evals dominate the wall time. ETA ≈ 1:05 PM PST.
  - Evals MATH-500 / GSM8K: 10.2 / 32.1 → 12.5 / 34.5 → 15.8 / 39.0 → 18.8 / 43.9 → 21.5 / 45.8 → **23.4 / 48.3 (u250)**.
  - Reward 0.05 → 0.16. **Mean length shrinks** 1,292 → 1,155 → 924 → 616 → 493 → 510 (unlike 2.7B/1.4B, which grew to ~10k/~8k).
  - Logprob diff median 0.0081; grad norm 0.02–0.28; drains 31–82 s; no conv/l2norm autotunes.
  - The 14 tracebacks are startup connection retries plus 5 transient router 500s whose groups were retried. The one "nan/inf" hit is the word "inf" in a response preview.
- 1:04 PM PST — **810M RL finished cleanly**: job exit 0, workflow complete, `export-hf/.complete`, all 8 checkpoints kept. ~1 h 35 min wall.
  - MATH-500: 10.2, 12.5, 15.8, 18.8, 21.5, **23.4 (u250 peak)**, 21.3, 19.0, **21.4 (final)**.
  - GSM8K: 32.1, 34.5, 39.0, 43.9, 45.8, 48.3, 47.2, 46.7, **48.6**.
  - MATH-500 plateaus or slips after u250; at ~6 epochs over 2,043 mostly-1/16 prompts with very short responses (mean ~460–580 tokens late), this may be over-training. GSM8K holds.
  - No GPU jobs of ours are running now; 450M/275M RL are on hold with data ready.
- 9:40 PM PST — Michael (via Jacob): use a larger dataset for the small models, since they may learn to solve prompts they initially failed. Agreed: the 810M's MATH-500 peaked at u250 and slipped after ~6 epochs on 2,043 mostly-1/16 prompts.
  - **Combined datasets built** (own full set ∪ 2.7B 32k pass band), `data/dolci-math-base-r1-v1-combined-lc{810m,450m,275m}-32k`: 810M **5,416**, 450M **5,283**, 275M **5,209** (~2.4 epochs at 400 updates). The 1.4B set is ~97% inside the 2.7B set, so the 2.7B set is the right superset.
  - **Unlock check submitted**: the 810M RL final on the 3,373 2.7B-band prompts its base solved 0/16 (16 samples, T=1.0, top-p 1.0, 32k, as the base sweep): https://beaker.org/ex/01M7VQM52763BJ87GZ1D25BYP0.
    - Data `data/check-810m-final-on-base-unsolved-lc2.7b`; output `monitor/check-810m-final-on-base-unsolved-20261119`.
  - 2.7B "add prompts back" (Jacob's question): not all. The midtrain v1 A/B found unfiltered 8 MATH-500 points worse at u200, and the 2.7B already has ~2.6 epochs.
    - A targeted version (base-unsolved prompts the trained 32k model now solves) would need ~1–2 node-hours of sampling the 2.7B final on the 8,350 base-unsolved prompts.
    - Decide after the 810M check.
- 9:55 PM PST — **Unlock check result** (810M RL final vs the 3,373 2.7B-band prompts its base solved 0/16; same 16-sample, T=1.0, 32k protocol; job 9:44–9:53 PM PST):
  - **1,124 (33.3%) are now solved at least once in 16**; 221 at ≥ 4/16; 2 at ≥ 15/16. Per-sample accuracy 5.1% (base: 0). Mean 470 tokens.
  - Distribution: 0: 2,249; 1: 545; 2: 227; 3: 131; 4: 78; 5: 37; …
  - That is about as many "new" learnable prompts as the 810M's whole base-filtered set had in the 2.7B pool (1,528).
  - This supports Michael's suggestion: training on the combined set would keep feeding the model prompts it grows into, instead of ~6 epochs over a fixed small band.
- 10:05 PM PST — **Unlock result → launched all four combined-set RL runs**, one node each (Jacob). In-loop evals only for now; Minerva/AIME decided tomorrow. Jacob may also want a 2.7B filter with its final checkpoint tomorrow.

  | size | data | prompts | GPUs | experiment | job |
  |---|---|---|---|---|---|
  | 810M | `combined-lc810m-32k` | 5,416 | 2 + 6 | https://beaker.org/ex/01M7VRXVR1HGERV149NCFCZJRG | `01M7VRXVW90882JATW499R6PRY` |
  | 450M | `combined-lc450m-32k` | 5,283 | 2 + 6 | https://beaker.org/ex/01M7VRY1TVPNCGD4P5AGP4D5PQ | `01M7VRY22B17HDCEGF7Y1B07C6` |
  | 275M | `combined-lc275m-32k` | 5,209 | 2 + 6 | https://beaker.org/ex/01M7VRY7X435MBNVJGHCKRP175 | `01M7VRY81HCXB69JNRRWGMVB8C` |
  | 1.4B | `combined-lc1.4b-32k` | 4,969 | 4 + 4 | https://beaker.org/ex/01M7VRYE0P8B3RWXA168DPBBMK | `01M7VRYEB30JD106EYCTB5QM5J` |

  - Configs `runs/olmicro-<size>-lc-math-rl-refresh-32k-combined-20261119.toml` (`make_v2_refresh_config.py`). Each diffs from its earlier counterpart only in name/root/data.
  - The 1.4B combined set is its own pool-only band ∪ the 2.7B band (no rest-of-pool sweep for 1.4B; ~640 mostly-1/16 prompts likely missing).
  - validate + preflight OK for all; watchers `monitor/<size>-combined-rl-status.log`.
  - Estimates: small three ~1.5–2.5 h each, 1.4B ~5–6 h.
- All four combined-set runs started: 810M 10:08 PM, 275M 10:28 PM, 1.4B 10:33 PM, 450M 10:38 PM PST.
- 11:40 PM PST — combined-set runs check (no conv/l2norm autotunes in any; tracebacks only startup/transient):

  | run | update | s/update | in-loop MATH-500 so far (u0, u50, …) | single-set run at the same updates | mixed-group fraction (signal) |
  |---|---|---|---|---|---|
  | 810M | 300 | 3.8 | 10.5, 12.1, 13.5, 15.8, 19.5, **21.7** | 10.2, 12.5, 15.8, 18.8, 21.5, 23.4 | 20% → 34% |
  | 1.4B | 100 | 17.5 | 22.5, **26.4** | 23.5, 28.8 | 32% → 38% |
  | 450M | 140 | 14 | 5.3, 5.3, **6.0** | — | **~14% flat** |
  | 275M | 173 | 14 | 2.6, 3.5, 2.7, **3.3** | — | **~12% flat** |

  - 810M combined learns a bit slower than the single-set run (zero-signal prompts dilute the per-token-normalized loss) but is still rising; the question is whether it keeps climbing past u250 where the single-set run peaked.
  - **450M/275M: only ~12–14% of groups carry any reward signal; reward is flat at ~0.02 and evals barely move** (GSM8K 19–21 and 6–7). With ~85% zero-advantage groups the effective step is ~7× smaller.
    - Options for Jacob tomorrow: their own-band sets (1,408 / 1,228, ~9–10 epochs), or the combined sets with `filter_zero_std_groups = true` (drops no-signal groups online; generation is cheap at these sizes).
    - Not relaunching tonight: recipe change = Jacob's call; these runs finish in ~1–1.5 h anyway.
  - logprob diff: 810M 0.0088, 1.4B 0.0068, 450M 0.0135, 275M 0.0149 (as in the smokes).
- 12:41 AM PST — **810M combined finished cleanly** (exit 0 at 11:52 PM PST; workflow complete; `export-hf/.complete`; ~1 h 45 min). MATH-500 u0 … u400:
  - combined: 10.5, 12.1, 13.5, 15.8, 19.5, 21.7, **21.7, 24.0, 22.2**
  - single-set: 10.2, 12.5, 15.8, 18.8, 21.5, 23.4, **21.3, 19.0, 21.4**
  - **Combined keeps improving through u350 (peak 24.0) and ends at 22.2 vs 21.4; it avoids the single-set run's late decline**, after learning a bit slower in u100–u250. GSM8K final 47.7 vs 48.6 (same).
  - **450M (u305) and 275M (u396) are picking up**:
    - 450M MATH-500 5.3, 5.3, 6.0, 5.1, 7.1, 8.6, **10.9 (u300)**; GSM8K 19.3 → 27.0.
    - 275M MATH-500 2.6 … **4.3 (u350)**; GSM8K 6.3 → 9.4.
    - [CORRECTED 11:50 AM PST 11/20] The numbers originally here came from `rollout_flow.jsonl` `mixed_responses`, which counts responses spanning more than one policy version, not reward-mixed groups. The true signal fraction (from records) rose more modestly: 450M 14% → 25%, 275M 11% → 18%; see the 11/20 11:50 AM PST entry.
  - **1.4B combined** u200: 22.5, 26.4, 33.5, 36.0 vs single-set 23.5, 28.8, 34.6, 37.6 (slightly behind, tracking).
  - Watchers re-armed for 450M/275M/1.4B.
- 1:29 AM PST — **450M and 275M combined finished cleanly** (exit 0; workflow complete; `export-hf/.complete`). Both still rising at the end.
  - **450M** (done 1:10 AM PST):
    - MATH-500 5.3, 5.3, 6.0, 5.1, 7.1, 8.6, 10.9, 11.2, **12.1**; GSM8K 19.3 → **29.9**.
    - [CORRECTED] the multi-version response fraction, not signal; true signal 14% → 25%.
  - **275M** (done 12:47 AM PST):
    - MATH-500 2.6, 3.5, 2.7, 3.3, 3.8, 3.6, 3.4, 4.3, **4.8**; GSM8K 6.3 → **9.6**.
    - [CORRECTED] multi-version fraction, not signal; true signal 11% → 18%.
  - **1.4B combined** u242 (~52 s/update): MATH-500 22.5, 26.4, 33.5, 36.0, **43.6 (u200)** vs single-set 45.6 at u200; GSM8K 68.7 vs 68.4. (The 0.23 → 0.72 figure is the multi-version fraction, not signal.) ETA ≈ 4:50 AM PST.
- 5:05 AM PST — **1.4B combined finished cleanly** (exit 0; workflow complete; `export-hf/.complete`; ~6.4 h wall).
  - MATH-500 22.5, 26.4, 33.5, 36.0, 43.6, 45.5, 47.4, 48.5, **48.1**; GSM8K 56.0 … **71.5**.
  - Single-set run: 49.9 / 73.2 final (peak 50.7). So combined is ~2 points lower for the 1.4B: there the own set (5.6 epochs) wasn't badly overfit, and the extra zero-signal prompts slowed learning.
  - Mean length grew to ~10k (as in the single-set run).
- **Morning summary (combined-set runs vs single-set):**

  | size | single-set final MATH-500 (peak) | combined final (peak) | GSM8K single → combined |
  |---|---|---|---|
  | 1.4B | 49.9 (50.7 @ u350) | 48.1 (48.5 @ u350) | 73.2 → 71.5 |
  | 810M | 21.4 (23.4 @ u250, then decline) | **22.2 (24.0 @ u350)** | 48.6 → 47.7 |
  | 450M | not run | 12.1 (still rising; base 5.3) | — → 29.9 (base 19.3) |
  | 275M | not run | 4.8 (still rising; base 2.6) | — → 9.6 (base 6.3) |

  - Takeaway: the combined set helps where the own set is small enough to overfit (810M: no late decline, higher peak and final) and costs a little where it isn't (1.4B).
  - The 450M/275M learn slowly but steadily; their signal fraction roughly doubled over training, and they are still rising at u400.
  - Open decisions for Jacob:
    1. Minerva/AIME for these four runs (finals only, or full checkpoint sweeps).
    2. 450M/275M: rerun with `filter_zero_std_groups` and/or more updates.
    3. Which data recipe to report per size.
    4. The 2.7B final-checkpoint filter (targeted add-back).
  - No GPU jobs of ours are running.
- 10:49 AM PST — Jacob: relaunch the two smallest with the zero-std filter. Signal fractions confirm his intuition (bigger models leave far fewer zero-std groups as training proceeds):
  - [CORRECTED] The figures originally here (0.16–0.23 rising to 0.35–0.85) were the multi-policy-version response fraction (`mixed_responses`), not zero-std. True signal fractions are in the 11/20 11:50 AM PST entry; the conclusion (fewer zero-std groups at larger sizes) holds, more strongly.
  - **Submitted**: 450M https://beaker.org/ex/01M7X4NY110MHC1CFEZKR4KH00 (job `01M7X4NYGFADHE8W2A73NY2G90`); 275M https://beaker.org/ex/01M7X4PB0J12ZXG8WTWT4JYNEX (job `01M7X4PB99QNRG3T2V8JH93XK8`).
  - Configs `runs/olmicro-{450m,275m}-lc-math-rl-refresh-32k-combined-zstd-20261120.toml` = the combined configs with `filter_zero_std_groups = true` (only change besides name/root/wandb group). validate + preflight OK.
  - Expect ~3–5 h each (~5–6× generation per update early, while only ~16–19% of groups carry signal).
- 10:55 AM PST — **Scaling plot published** (new artifact for sharing): https://claude.ai/artifact/8VehshHhrNLYgqEv6H5gnm.
  - MATH-500 and GSM8K Platinum vs RL update, one line per size: best run by final MATH-500 (2.7B 32k own set, 1.4B own set, 810M/450M/275M combined).
  - Ordinal blue ramp validated with `--ordinal` in both modes; direct end labels; table view.
  - Source `scratchpad/olmicro-scaling.html`. Update it if the zero-std reruns beat the current 450M/275M runs.
- 10:52–10:53 AM PST — both zero-std reruns started (450M, 275M).
- 11:50 AM PST — **Correction: `mixed_responses` in `rollout_flow.jsonl` counts responses whose tokens span more than one policy version (refresh continuations; it grows with response length), not reward-mixed groups.**
  - True signal fraction = share of groups with mixed rewards, from inference records, by policy version 0–49 / 50–199 / 200–349 / 350–399:

    | run | 0–49 | 50–199 | 200–349 | 350–399 |
    |---|---|---|---|---|
    | 2.7B 32k | 60% | 75–76% | 74% | 71% (14% all-right) |
    | 1.4B own | 51% | 64% | 74% | 77% |
    | 1.4B combined | 32% | 42% | 56% | 56% |
    | 810M own | 31% | 45% | 53% | 52% |
    | 810M combined | 19% | 25% | 35% | 35% |
    | 450M combined | 14% | 15% | 22% | 25% |
    | 275M combined | 11% | 12% | 15% | 18% |

  - Zero-std groups shrink sharply with size (2.7B ~25–40%; 275M 82–89%).
  - **Zero-std reruns** (11:45 AM PST): collection is working, not stalling: ~228 all-wrong groups dropped per 32 kept, keep rate ~12%, matching the signal fraction above.
    - It is slow, though: 450M u26 at ~92 s/update (generation wait ~90 s, training 3–11 s); 275M u19 at ~127 s/update.
    - Evals at step 0: 5.1 / 18.7 and 3.1 / 6.8.
    - ETA at the current rate ~9 h (450M) and ~13 h (275M), shrinking as the keep rate rises; probably ~6–10 h.
- 1:49 PM PST — zero-std reruns check (watchers re-armed):
  - **450M zstd** u117, ~52 s/update (down from ~92; keep rate 13.7% → 19.7% by 25-update blocks).
    - MATH-500 5.1, 5.8, **8.0 (u100)** vs non-filtered 5.3, 5.3, 6.0; GSM8K 18.7, 21.1, **23.6** vs 19.3, 19.0, 20.7. **Ahead by ~2–3 points at u100.**
    - ETA ≈ 4 h + evals (~6 PM PST).
  - **275M zstd** u87, ~96 s/update (keep 11–13%, flat so far).
    - MATH-500 3.1, 3.6 (u50) vs 2.6, 3.5; GSM8K 6.8, 7.2 vs 6.3, 6.6.
    - ETA ≈ 8 h + evals (~10–11 PM PST).
  - Grad norms 0.06–0.17 with no zero-gradient updates (vs zeros=2–3 before, since every kept group now has signal). Logprob diff 0.013–0.014 as before. Errors are only transient router 500s (retried).
- 3:57 PM PST — zero-std reruns:
  - **450M** u375, ~14 s/update; keep rate 13.9% → 30.8%.
    - MATH-500 u0–u350: 5.1, 5.8, 8.0, 9.2, 13.5, 14.1, 14.4, 13.9 vs non-filtered 5.3, 5.3, 6.0, 5.1, 7.1, 8.6, 10.9, 11.2: **+3–6 points**.
    - GSM8K 33.9 vs 27.5 at u350.
  - **275M** u212, ~33 s/update; keep 11.7% → 21.3%.
    - MATH-500 3.1, 3.6, 3.7, 3.7, 4.3 vs 2.6, 3.5, 2.7, 3.3, 3.8; GSM8K 9.5 vs 6.8 at u200.
    - ETA ≈ 6:15 PM PST.
  - Watchers re-armed.
- 4:07 PM PST — **450M zero-std rerun finished cleanly** (exit 0; workflow complete; `export-hf/.complete`; ~5.3 h).
  - MATH-500 5.1, 5.8, 8.0, 9.2, 13.5, 14.1, 14.4, 13.9, **14.8** vs non-filtered final 12.1; GSM8K **33.6** vs 29.9.
  - **Better: swapped into the scaling plot** (https://claude.ai/artifact/8VehshHhrNLYgqEv6H5gnm, notes updated).
- 5:23 PM PST — **275M zero-std rerun finished cleanly** (exit 0; workflow complete; `export-hf/.complete`; ~6.5 h).
  - MATH-500 3.1, 3.6, 3.7, 3.7, 4.3, 6.1, 6.5, 6.3, **6.3** vs non-filtered final 4.8; GSM8K **11.5** vs 9.6.
  - Swapped into the scaling plot.
  - **Final best-run table** (in-loop, final checkpoint):

    | size | run | MATH-500 | GSM8K |
    |---|---|---|---|
    | 2.7B | own set | 77.3 | 90.3 |
    | 1.4B | own set | 49.9 | 73.2 |
    | 810M | combined | 22.2 | 47.7 |
    | 450M | combined + zero-std filter | 14.8 | 33.6 |
    | 275M | combined + zero-std filter | 6.3 | 11.5 |

  - No GPU jobs of ours are running.
