# Olmicro RL-zero on math: results

RL-zero (RL from the base model, no SFT) on math for the Olmicro long-context base models from 275M to 2.7B parameters, trained with `python -m open_instruct.miles` on the MILES/OLMo-core stack. Work ran 2026-11-17 to 2026-11-20 on `ai2/holmes` (B300). This folder is a reference snapshot: results, run configs, the tools used, and the full run log. Checkpoints, datasets and raw logs stay on Weka; paths are below.

Interactive charts (private claude.ai pages; ask Jacob for access):
- Scaling plot (MATH-500 and GSM8K, best run per size): https://claude.ai/artifact/8VehshHhrNLYgqEv6H5gnm
- Detailed 2.7B page (all 2.7B runs on MATH-500; Minerva MATH and AIME 2026 over training; 1.4B section): https://claude.ai/artifact/1z2YjPmJBHeaqggTUDzhEj

## Headline results

Best run per size, judged by the final checkpoint on MATH-500. All runs use 400 RL updates and in-loop evals at temperature 1.0, top-p 0.95, 4 samples per prompt and up to 32,768 tokens.

| size | run | MATH-500 base → final | GSM8K Platinum base → final | wall time (GPUs) |
|---|---|---|---|---|
| 2.7B | own pass band, 32k cap | 37.4 → **77.3** | 68.3 → **90.3** | 3.6 h (8) + 7.5 h (16) |
| 1.4B | own pass band | 23.5 → **49.9** | 55.8 → **73.2** | 5.6 h (8) |
| 810M | own band + 2.7B band | 10.5 → **22.2** | 31.9 → **47.7** | 1.7 h (8) |
| 450M | own band + 2.7B band, zero-std filter | 5.1 → **14.8** | 18.7 → **33.6** | 5.2 h (8) |
| 275M | own band + 2.7B band, zero-std filter | 3.1 → **6.3** | 6.8 → **11.5** | 6.5 h (8) |

Offline evals with olmo-eval's Minerva grader. Minerva MATH is the full 5,000-problem Hendrycks MATH test set, olmo-eval's `minerva_math` suite averaged over its 7 subjects, 1 sample per problem. AIME 2026 is `MathArena/aime_2026` with 30 problems × 32 samples.

| checkpoint | Minerva MATH | AIME 2026 pass@1 | AIME pass@32 | AIME pass@1 at a 16k cap |
|---|---|---|---|---|
| 2.7B base | 36.8 | 1.5 | 16.7 | 1.5 |
| **2.7B 32k final** | **77.1** | **14.5** | 36.7 | 11.5 |
| 2.7B 8k final (ablation) | 73.0 | 9.8 | 33.3 | 9.8 |
| 1.4B base | 21.8 | 0.1 | 3.3 | 0.1 |
| 1.4B own-band final | 49.6 | 0.9 | 13.3 | 0.8 |
| 810M / 450M / 275M base | 10.3 / 5.0 / 3.0 | 0.1 / 0 / 0 | 3.3 / 0 / 0 | |

Every checkpoint of the 2.7B 8k and 32k runs and of the 1.4B own-band run is in `tables/minerva_aime_evals.csv`. The 810M, 450M and 275M trained models have no Minerva/AIME evals yet.

## Main findings

1. **The R1-style prompt is what makes RL-zero work on these base models.** The template is "A conversation between User and Assistant … The final answer is enclosed within \boxed{} … User: {q}\nAssistant:", with stop string `\nUser:`.
   - A plain "Question/Answer" prompt left most responses truncated.
   - See `notes/run-log.md` and the format probe in `tools/format_probe.py`.
2. **Use the long-context (stage3) final, not the midtrain checkpoint.**
   - At the same 7,680-token eval cap and 4 samples, the long-context base starts at 37.3 on MATH-500 vs 30–31 for midtrain.
   - At update 200 it reaches 68.5 (8k run) vs 60.1 for the best midtrain run (v1b), though the publication mode also differs.
   - The midtrain runs (v0/v1) are kept only as a reference.
3. **Use refresh publication, not barrier.** Barrier aborts every group that spans a weight update, so responses can't grow past what one update cycle allows (v0/v1 were length-limited). Following Michael's advice ("barrier is unnecessarily conservative"), all long-context runs use `core.publication_mode = "refresh"` with `max_weight_staleness = 12`. We did not run a barrier-vs-refresh A/B on the long-context checkpoint.
4. **Difficulty filtering helps.** We keep prompts the base model solves 1–14 times out of 16.
   - On the midtrain 2.7B, the filtered run led unfiltered by 8 MATH-500 points at update 200 (60.1 vs 52.0).
   - The filtering method is described under "Difficulty filtering" below.
5. **A 32k response cap is worth it at 2.7B; an 8k cap is a decent cheaper ablation.**
   - **32k vs 8k, in-loop MATH-500:** 32k reaches 73% by update 150–200 and ends at 77.3. 8k gets to 73.6 by update 400.
   - **Offline evals:** 32k is +4 on Minerva MATH and +4.7 on AIME pass@1. Most of the AIME gain (about 3 of the 4.7 points) needs responses longer than 16k tokens: at a 16k cap the 32k model scores 11.5.
   - **Lengths:** the 32k model's mean response grows to ~10k tokens (~20k on AIME). The 8k model learns to stay short (~4k).
6. **Larger models keep many more prompt groups informative.** These are groups whose 8 rewards are not all the same; all-correct and all-wrong groups give zero gradient.

   | run | updates 0–49 | 50–199 | 200–349 | 350–399 |
   |---|---|---|---|---|
   | 2.7B 32k | 60% | 75% | 74% | 71% |
   | 1.4B own band | 51% | 64% | 74% | 77% |
   | 810M own band | 31% | 45% | 53% | 52% |
   | 450M combined | 14% | 15% | 22% | 25% |
   | 275M combined | 11% | 12% | 15% | 18% |

   Measured from inference records (`tables/signal_fraction.csv`).
7. **Small models run out of prompts; a bigger, partly too-hard pool helps.** This was Michael's suggestion.
   - **The problem:** the 810M on its own band (2,043 prompts, ~6 epochs) peaked at update 250 (23.4) and then declined. Its responses also shrank to ~500 tokens.
   - **The check:** after RL, the 810M solved 33% of the 2.7B-band prompts its base had solved 0/16 times (1,124 of 3,373).
   - **The fix:** training on its own band ∪ the 2.7B band (5,416 prompts) removed the late decline: peak 24.0, final 22.2 vs 21.4.
   - **The limit:** at 1.4B the combined set cost about 2 points (48.1 vs 49.9), since its own band wasn't overfitting. At 450M/275M the combined set is mostly prompts they can't solve yet (11–25% signal), and learning was slow.
8. **For the smallest models, drop zero-signal groups.** With `filter_zero_std_groups = true` on the combined sets, the 450M improved 12.1 → 14.8 MATH-500 and 29.9 → 33.6 GSM8K, and the 275M 4.8 → 6.3 and 9.6 → 11.5.
   - The cost is generation: early on, only ~12% of groups survive, so each update samples ~8× more groups.
   - At these sizes that is affordable: the runs took 5–6.5 h instead of 2–2.5 h.
   - We have not tried this on 810M/1.4B.

## Setup

- **Models.** `allenai/olmicro-{275m,450m,810m,1.4b,2.7b}-base` at the stage3 (long-context, 64k) finals: `stage3-step0047684` for 275M and 450M, `stage3-step0023842` for 810M, 1.4B and 2.7B. HF `main` held only a README at the time.
  - All files are verified against `SHA256SUMS`.
  - OLMo-core model configs come from the HF bucket (`stage3/checkpoints/step…/config.json`).
  - All sizes share head_dim 128 (attention and GDN key), GDN value dim 256, a NoPE full-attention layer every 5th layer, and the same tokenizer.
- **Stack.** MILES image `01M6HMYBFW8D9ZK4SKGJ1XD6HW`; this repo at `codex/olmicro-mainline-sync` (`15413672d`); native SGLang Olmicro serving (TP1); OLMo-core trainer (FSDP, fp32 params, bf16 autocast, full activation checkpointing, FA4).
- **Training data.** `allenai/Dolci-RL-Zero-Math-7B`, 13,251 prompts after a 512-token prompt cap, rendered with the R1 template (`tools/data-prep/prepare_olmicro_math.py --template r1`).
- **In-loop evals.**
  - Sets: MATH-500 (497) and GSM8K Platinum (1,209), every 50 updates.
  - Sampling: 4 samples per prompt, T=1.0, top-p 0.95, eval length = training cap.
  - Scoring: MathVerifier / GSM8KVerifier, with truncated responses scored 0.
- **Offline evals** (`tools/data-prep/prepare_extra_evals.py`, `tools/format_probe.py --eval-files`, `tools/score_extra_evals.py`): same R1 template and sampling, scored with both MathVerifier and olmo-eval's `MathExtractor` + `is_equiv` (`MinervaMathScorer` semantics, olmo-eval `1eddca07`).
  - Both graders score 100% on the gold answers.
  - olmo-eval's few-shot `minerva_math` base variants use a different prompt and 1,024 tokens, so they are not comparable.
- **Final recipe** (`tools/make_v2_refresh_config.py --size <size> --data <set>`):
  - **Batch and schedule:** 32 prompts × 8 samples per update, lr 1e-6 constant, no KL, clip 0.2 / 0.272, TIS clip 2.0, 400 updates.
  - **Publication:** refresh, max staleness 12, 512 samples in flight.
  - **Lengths:** 32,768-token responses (context = cap + 512); sequence packing to a multiple of 8,192 covering the context (40,960 at 32k).
  - **Fixes and settings from this work:**
    - FLA config cache env: `FLA_CACHE_MODE=default` plus `FLA_CONFIG_DIR` (see the infrastructure section).
    - `core.engine_drain_timeout = 1800`.
    - `core.checkpoint_keep_every = 50`.
    - `min_runtime = "8h"`, urgent priority.
  - **Topology:** 4 trainer + 4 inference GPUs (2.7B, 1.4B) or 2 + 6 (810M and below), one 8-GPU node. The 2.7B 32k run moved to 4 + 12 on two nodes at update 100 (`miles.load` resume into a new root).

## Difficulty filtering

- **Sweep.** Sample each base model 16 times per prompt at T=1.0, top-p 1.0 with a 32k cap (`tools/format_probe.py`, 32 resumable shards on one node). Score with MathVerifier (`tools/score_format_probe.py`).
- **Band.** Keep prompts solved 1–14 of 16 (`tools/data-prep/filter_by_pass_rate.py`). Nothing was ever too easy: at most 1 prompt per size was above 14/16.
- **Lower caps.** `filter_by_pass_rate.py --cap N` rescores the 32k sweep as if generated with cap N; this is exact, because a response that reached N would have been truncated. That is how the 8k/16k sets were made.
- **Prefiltering with the 2.7B.** Smaller models were swept on the 4,901 prompts the 2.7B solved at least once, plus an 800-prompt random control from the 8,350 it never solved.
  - The never-solved 63% of prompts cost 67% of the 2.7B sweep's tokens.
  - The control showed the prefilter misses some prompts (6% for 810M, 4–5% for 450M, 3% for 275M, 8.5% for 1.4B), almost all at 1/16. So the rest of the pool was then swept too for 810M, 450M and 275M.

| size | band (2.7B pool) | control in band | + rest of pool | band over all 13,251 | + 2.7B band (combined) |
|---|---|---|---|---|---|
| 2.7B | 4,900 (8k: 4,353; 16k: 4,845) | | | 4,900 | — |
| 1.4B | 2,281 | 68 / 800 | not swept | — | 4,969 |
| 810M | 1,577 | 49 / 800 | +466 | 2,043 | 5,416 |
| 450M | 1,064 | 38 / 800 | +344 | 1,408 | 5,283 |
| 275M | 941 | 22 / 800 | +287 | 1,228 | 5,209 |

Most in-band prompts at small sizes are 1/16: 63–70% for 810M and below, against 50% for 1.4B. At 1.4B and below, a 16k cap changes the set by ≤ 3 prompts; the >16k responses are mostly repetition loops. At 2.7B a 16k cap drops 55 prompts and an 8k cap 547.

## All runs

See `tables/runs.csv` for experiment links, topology, wall time and first/final/peak scores, and `tables/in_loop_evals.csv` for every eval point. In short:

| run | MATH-500 final (peak) | note |
|---|---|---|
| 2.7B midtrain v0 (barrier, 4k, unfiltered) | 64.6 | reference |
| 2.7B midtrain v1a / v1b (barrier, 7,680) | 54.2 @ u250 / 60.1 @ u200 | stopped when we switched to the long-context checkpoint |
| **2.7B long-context 32k** | **77.3** | u0–u99 on 4+4, u100–u399 on 4+12; the u100 point was re-run offline |
| 2.7B long-context 8k | 73.6 | ablation |
| 2.7B long-context 16k | — | canceled at u4 to free a node |
| **1.4B own band** | **49.9** (50.7) | |
| 1.4B combined | 48.1 (48.5) | |
| 810M own band | 21.4 (23.4 @ u250) | declines after u250 |
| **810M combined** | **22.2** (24.0 @ u350) | |
| 450M combined | 12.1 | |
| **450M combined + zero-std filter** | **14.8** | |
| 275M combined | 4.8 | |
| **275M combined + zero-std filter** | **6.3** (6.5) | |

## Infrastructure findings and fixes

Each of these cost real time; the fixes are in the configs and tools here.

- **Trainer warm-up of 30–90 min per run (and per resume) came from FLA Triton autotuning, not slow kernels.**
  - `causal_conv1d` and `l2norm` autotune per `NB = cdiv(B*T, 1024)`, so every new pack length re-tunes (~30–70 s each).
  - Fix: `FLA_CACHE_MODE=default` + `FLA_CONFIG_DIR` pointing at fixed configs (`tools/fla-configs/b300-fla0.5.2-v1`, built by `tools/make_fla_config_cache.py`).
  - Validated with identical loss and cold-pack time 355 s → 63 s (`notes/warmup-fix-notes-20261118.md`). `layer_norm_gated` still tunes; it uses plain `triton.autotune`.
  - The GDN kernels themselves run at 26–29.5k tokens/s/GPU at any document length.
- **The MILES compiler cache refuses to restore for runs with an external Core `model_config`** ("External Core model-config files are not qualified by this wrapper"), so nothing carries over between runs. A candidate fix is in `tools/patches/` (not applied; needs an image rebuild).
- **Evals failed at long response lengths.** Before a blocking eval, refresh drains in-flight generation with `core.engine_drain_timeout`, default 180 s. With ~10k-token responses the 2.7B 32k drains ran 135–207 s; the update-100 drain exceeded 180 s, the eval timed out, and the run crashed. Set 1,800 s.
- **Bad node.** `holmes-cs-aus-511` passed health checks but jobs never started there. `tools/submit_excluding.py` submits with a hostname allowlist that excludes it, since Beaker cluster and hostname constraints are mutually exclusive.
- **Allocation is per job.** A positive `minRuntime` protects the job for that window (≤ 8 h); without it, jobs were preempted within seconds.
- **`core.checkpoint_keep_last = 3` prunes older checkpoints.** Use `checkpoint_keep_every = 50` to keep every eval-interval checkpoint. `tools/preserve_checkpoints.py` hardlinks committed checkpoints for runs that didn't set it. The 2.7B 8k update-50 checkpoint was lost to pruning.
- **Changing inference topology or other settings mid-run.** The same `output.root` with a changed spec is refused. Instead, start a new root with `[miles] load = <old run>/checkpoints`; only the trainer world size must match.
  - An empty loop (`num_rollouts = N` with checkpoint N−1) exports any saved checkpoint to HF.
  - A resumed run skips the initial eval.
- **Evaluating an arbitrary checkpoint.** Export it (as above), then either run an eval-only MILES run from the export or `tools/format_probe.py` + scoring; the orchestration is `tools/eval_sweep_orchestrator.py`.
  - Multi-engine eval specs must set `CUDA_VISIBLE_DEVICES=$gpu` per engine. Otherwise all engines start on GPU 0 and fail with "Loaded weights leave no GPU memory for the KV cache".
- **Monitoring gotchas.**
  - `beaker job logs` for finished jobs returns only part of the log; the complete console log is the run's `wandb/…/files/output_*.log`.
  - `rollout_flow.jsonl` `mixed_responses` counts responses spanning more than one policy version (refresh continuations), not reward-mixed groups. Measure signal from inference records or `queue_metrics` `rollout/dynamic_filter/drop_zero_std_*`.
- **Repo workflow note.** This repo's `CLAUDE.md` asks Olmicro launches to update `/home/mnoukhov/olmicro/Plan.md`. That file isn't reachable from the machine these runs were launched from; the plan and receipts live in `notes/run-log.md` instead.

## Open questions and suggested next steps

See `notes/paper-followups-todo.md` for a prioritized, implementor-ready list of follow-ups tied to Section 4.2 of the Olmicro technical report (highest priority: `810M combined + filter_zero_std_groups = true` and CPU-only response-length extraction from Weka records).

- **Minerva/AIME for the small-model runs** (810M combined, 450M/275M zero-std, 1.4B combined): HF exports exist, and each eval takes ~10–20 min of one node.
- **Zero-std filtering at 810M and 1.4B.** It helped a lot below 810M. At 810M the combined set has 19–35% signal, so the filter would likely help there too.
- **More updates for 450M/275M.** Both were still rising at update 400.
- **A targeted add-back for the 2.7B.** Sample the 2.7B 32k final on the 8,350 prompts its base never solved, and add the ones it now solves. Adding all unsolved prompts lost 8 points in the midtrain v1 A/B; this targeted version is untested. Cost: ~1–2 node-hours.
- **A 1.4B rest-of-pool sweep** (~640 more prompts at 1/16) if the 1.4B moves to the combined recipe.
- **A 16k cap for 450M/275M**: no measurable loss in the sweeps, and ~30% fewer tokens, because >16k responses are mostly loops.
- **Confirm HF `main` = stage3 finals** via `SHA256SUMS` once `main` is populated.

## Where things are

On Weka (`/weka/oe-adapt-default/jacobm/olmicro/`):
- Run roots: `runs/<run name>/` (checkpoints `checkpoints/core/rollout_*`, HF exports `export-hf/`, wandb logs).
- Kept checkpoints of the 2.7B and 1.4B runs: `preserved-checkpoints/<run>/u<update>/` (each a self-contained `miles.load` root).
- Datasets: `data/` (manifests copied to `notes/data-manifests/`).
- Pass-rate sweeps: `monitor/passrate-sweep-lc<size>-32k-*`.
- Offline evals: `monitor/extra-evals-<checkpoint>-20261118/` (raw generations + `summary.json`).
- Inference records: `inference-records/`.
- Models: `models/olmicro-<size>-base/`.

In this folder:
- `tables/`:
  - `runs.csv`, `in_loop_evals.csv`, `minerva_aime_evals.csv` (also per subject and pass@k);
  - `signal_fraction.csv`, `difficulty_filtering.csv`;
  - `math500-2.7b-runs-20261118.csv` (the 2.7B chart data).
- `configs/`: every run TOML used (training runs, export-only and eval-only runs, smokes). Paths inside point at Weka.
- `tools/`: data prep, sweep/eval generation and scoring, config generators, submit/preflight helpers, monitoring and orchestration scripts, Beaker specs (`tools/beaker-specs/`), FLA configs, the compiler-cache patch, and `gather_results.py`, which rebuilds `tables/`. These are snapshots of working scripts, not maintained library code.
- `notes/`: the full chronological run log (`run-log.md`; times in PST), the small-model plan, the warm-up investigation, and dataset manifests.
