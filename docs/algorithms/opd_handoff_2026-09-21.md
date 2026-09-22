# OPD campaign handoff (2026-09-21)

Start-of-session brief for the agent taking over the Qwen3.5 / EOPD on-policy-distillation
campaign from Kevin Farhat. Everything below is committed state or a pointer to it; the
running log lives in [`opd_validation_program.md`](opd_validation_program.md) (append-only,
dated Log entries).

## 0. Read this first

1. This file, then the "Where we are" and "Decisions so far" sections of
   [`opd_validation_program.md`](opd_validation_program.md).
2. [`qwen35_math_opd_sync_rerun.md`](qwen35_math_opd_sync_rerun.md) for the Open Instruct
   sync-vs-async mechanics and run ids.
3. In the Miles worktree: `docs/miles/opd.md` (the OPD wrapper), `docs/miles/async-pipeline.md`
   and `docs/miles/configuration.md` (the async mode), `docs/miles/throughput-profiles.md`.
4. The results page (tables only): <https://claude.ai/artifact/GotcfjHj91rF6ECQa3Xqpn>.

## 1. How the work is organised

### Repositories and branches

| Repo | Path on Kevin's Mac | Branch | HEAD (2026-09-21) | Pushed? |
|---|---|---|---|---|
| Open Instruct | `/Users/kevinfarhat/repos/open-instruct` | `codex/qwen35-math-opd` | `13866c4d5` | Yes, `origin` = `allenai/open-instruct`, 0 ahead / 0 behind. 733 commits ahead of `main`, no PR, not merged. |
| Miles (git worktree of the same repo) | `/Users/kevinfarhat/repos/open-instruct-miles` | `robertb/miles-qwen35-opd` | `14080e06c` | Yes, same remote, 0 ahead / 0 behind. 495 commits ahead of `main`. |

- Both branches live in `allenai/open-instruct`; there is no separate Miles GitHub repo on our
  side. The Miles *framework* itself is pinned inside the Beaker image (built by
  `scripts/miles/build_and_launch.sh` from `runtime/miles/Dockerfile`, pins in
  `runtime/miles/runtime.lock.json`). Our code is the wrapper `open_instruct/miles/` plus run
  files under `configs/miles/`.
- `robertb/miles-qwen35-opd` is Robert's branch; Kevin got it cleared for direct commits
  (decision D5 in the tracker). Miles work is committed there directly, no review gate.
- Commit convention: end every commit message with
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- `docs/experiments/qwen35_math_opd_results.md` is Kevin's **untracked** results write-up in
  the Open Instruct repo. Edit it if asked, never `git add` it.
- Nothing is uncommitted or unpushed in either repo as of this handoff.

### What lives where

| Topic | Open Instruct (`codex/qwen35-math-opd`) | Miles (`robertb/miles-qwen35-opd`) |
|---|---|---|
| Tracker / handoffs | `docs/algorithms/opd_validation_program.md`, `qwen35_math_opd_sync_rerun.md`, `qwen35_math_opd_rerun_guide.md`, this file | `docs/miles/opd.md`, `docs/miles/index.md` |
| Sync vs async | `--synchronous_rollouts` in `open_instruct/data_loader.py`, `grpo_fast.py`, `actor_manager.py`; launch script `scripts/general_agent/terminal/rl/qwen35_math_opd_sync_rerun.sh` (`ARM=2b|4b`); fixed-batches variant `qwen35_math_opd_fixed_batches_rerun.sh` | `miles.fully_async` + `core.max_policy_lag` (see section 2); example `configs/miles/examples/grpo-async-disaggregated.toml`; wrapper `open_instruct/miles/async_rollout.py`, `async_buffer.py`, `async_capacity.py` |
| OPD training code | `grpo_fast.py` / `grpo_utils.py` (`--opd_pure`, teacher log-probs computed in the trainer) | `open_instruct/miles/opd_*.py`, SGLang teacher hook; EOPD in `eopd_math.py`, `eopd_loss.py` |
| Run files used | `scripts/general_agent/terminal/rl/qwen35_{2b,4b}_opd_from_*_4node.sh` (canonical async), `qwen35_math_opd_sync_rerun.sh` (sync) | `configs/miles/opd/qwen35-{2b,4b}-opd-from-verifier-*-math.toml` (phase 2), `eopd-opd-qwen3-4b-base-dapo14k.toml`, `eopd-opd-qwen3-1.7b-base-math.toml`, `eopd-eopd-qwen3-4b-base-dapo14k.toml` (paper replication) |
| Post-hoc eval | `scripts/general_agent/terminal/rl/qwen35_math_posthoc_eval.sh` (greedy, our verifier); `scripts/eopd/qwen25_math_harness_eval.py` + `open_instruct/qwen25_math_harness.py` (paper's grader) | in-run Avg@8 eval in the wrapper (`eval_*` keys of the TOML) |
| Diagnostics | `scripts/eopd/teacher_entropy_diagnostic.py` | `opd_audit.py`, runtime contract |
| Data prep | DAPO split dataset `01M1TKR7BQE4D5CYKYM1TZX0AM`, MATH500 `01M23KKN8GJJY4TYE6B2131799` | `scripts/miles/prepare_qwen35_math_prompts.py`, `prepare_eopd_math_prompts.py`; prompts on Weka under `.../deletable_checkpoint/kevinfarhat/miles-opd/data/{qwen35-math-v1,eopd-math-v1,eopd-math-v2-train}` |

### Infrastructure pointers

- W&B project `allenai-team1/opd` (run URL pattern `https://wandb.ai/allenai-team1/opd/runs/<id>`).
  The API works from Kevin's laptop.
- Beaker: Miles TOMLs launch in `ai2/open-instruct-dev` (budget `ai2/oe-other`); Open Instruct
  `priority: urgent` launches must use `--workspace ai2/olmo-instruct`, which has no allocation
  on `ai2/ceres` for min-runtime jobs. `ai2/holmes` has B300 nodes (`compute_103a`) that the
  campaign image's vLLM/flashinfer cannot JIT for, so keep holmes out of eval yamls. Training
  jobs request `min_runtime 8h` (Beaker's cap); Miles runs auto-resume across windows.
- Images: Open Instruct canonical `01M01EPKXMXR4502S1HNJVYN0M` + code-patch dataset
  `01M2HYMSMXT9WPEWRBKB2BBPTZ` overlaid at `/patch`; Miles current image
  `01M2V7V946ZN9N9STPK0H04YWM` with a code overlay of the branch HEAD at launch. Base image tag
  for Miles builds: `olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks` (re-tag `fe34fb1fef49` if it
  goes stale; build with `DOCKER_DEFAULT_PLATFORM=linux/amd64`).
- Secrets: reference `kevinfarhat_WANDB_API_KEY` by name only, never read its value.
- Checkpoints and run roots: `/weka/oe-adapt-default/allennlp/deletable_checkpoint/kevinfarhat/`
  (Open Instruct `<run>__42__<ts>_checkpoints/step_N`; Miles `miles-opd/runs/<name>-vN/`
  with `checkpoints/iter_*` and `hf-*` exports). Weka was 100 % full on 2026-09-16; do not
  delete anything without Kevin's explicit OK.
- Session scratch copied to `/Users/kevinfarhat/repos/opd-campaign-scratch/` (outside both
  repos): harness eval yamls (`qwen25math_eval_*.yaml`), `watch_all.py` / `watch_eval.py`
  watchers, `tables/opd_results_tables.html` (source of the results page), harness summaries,
  paper text `paper_2603.07079.txt`, `state.md`. Full harness outputs are Beaker datasets
  (OPD hf-219 from job `01M3119ZR9TQPFG8S8MMQJVDFN`; EOPD `01M32MXFBDWAV4WGPY6QZM5H01`; arm 1
  `01M32NNHA91ZRZJJRDCNFAF1AP`).
- Artifacts: results tables page <https://claude.ai/artifact/GotcfjHj91rF6ECQa3Xqpn>; team deck
  <https://claude.ai/artifact/3xXYu2ygGfJzEnnBc6qTxm>; manager deck
  <https://claude.ai/artifact/M9FicVCAyiRAow3wFvmUcf> (stale, superseded by the tables page).

## 2. Correction: Miles does have an async mode

Earlier messages and the results page said Miles "was never run async because its loop is
strictly on-policy by construction". The first half is true, the second is not:

- Miles has a **bounded fully-asynchronous pipeline**: `miles.fully_async = true` with an
  explicit positive `core.max_policy_lag` (optimizer-step age at consumption), resident
  disaggregated rollout engines (`colocate = false`), FIFO completed-group buffer sized by
  `async_data_buffer_capacity_factor`, stale-group discards, and truncated importance sampling
  (TIS) on by default. Documented in `docs/miles/async-pipeline.md`, `configuration.md` (the
  `async` section, `max_weight_staleness` ↔ `core.max_policy_lag`), `grpo.md`; example run file
  `configs/miles/examples/grpo-async-disaggregated.toml` (lag 1, TIS). Metrics land under
  `rollout/fully_async/*` (consumer wait, rejected policy groups).
- **Every Miles OPD run in this campaign used the default synchronous barrier mode** (the OPD
  TOMLs never set `fully_async`; the page and tracker say "strictly on-policy" in that sense).
- The wrapper **refuses async for Core-backend OPD** (`open_instruct/miles/config.py`: "Core OPD
  requires synchronous barrier publication and trainer-scored log probabilities"). Our OPD arms
  use `backend = "megatron"`, where async OPD is not refused by the schema but has never been
  exercised, and the SGLang teacher-scoring hook plus `opd_audit` were only validated
  synchronously. Treat Miles async as **available for GRPO, unproven for OPD**.
- Consequence: the sync-vs-async comparison exists only in Open Instruct so far. A Miles async
  arm is part of the throughput plan below (GRPO first).

## 3. Results so far

Five findings. Each says where it ran (stack), how rollouts were sampled (sync = every batch from
the current weights; async = Open Instruct `async_steps 4` with in-flight updates), then the
numbers. Full tables and links: <https://claude.ai/artifact/GotcfjHj91rF6ECQa3Xqpn>. Run ids are
collected at the end of this section.

### 3.1 Async sampling broke the 2B OPD run; sync fixed it

Open Instruct only. Qwen3.5, DAPO math, DPPO loss, 4 nodes, greedy post-hoc eval.

| Arm | Async (canonical) | Sync (rerun) | Verdict |
|---|---|---|---|
| 2B ← verifier-2B, LR 1e-6 | collapses: 0.065 at step 20, 0.007 at step 100 | stable 0.44–0.50 from step 10 | async is the bug |
| 4B ← verifier-9B, LR 5e-7 | ≈ sync within noise | ≈ async within noise | no effect at 4B |
| Wall clock per step | 2B 1.01 min, 4B 0.65 min | 2B 2.15 min, 4B ~2.6 min | sync costs 2–4× |

Why: async batches are length-sorted by completion order and 16–18k of 25.6k episodes are dropped
as stale. Pure-OPD advantages are per-token and not group-centered, so batch composition steers
the update. Still open: a 2 h staleness-only isolation run.

### 3.2 Open Instruct and Miles agree

Same Qwen3.5 recipe, both sync. Open Instruct = the 4-node sync reruns above (DPPO loss, vLLM
log-probs). Miles = Megatron trainer + separate SGLang teacher, PPO-clip loss, one 8-GPU node.

| Checkpoints compared | Match | Verdict |
|---|---|---|
| 2B steps 10–100, 4B steps 10–20 (greedy post-hoc eval) | 20 / 20 within noise | the OPD math is right in both; Open Instruct's flaws are pipeline-level |

Cost is not matched topology: Miles 136 GPU-min per 4B step (1 node) vs Open Instruct sync 83
(4 nodes) vs async 21. Section 7 makes this comparison fair.

### 3.3 The paper's OPD baselines replicate

Miles only, sync, PPO-clip, 4 PPO mini-batch steps per rollout, trainer-side student log-probs
(the paper's verl setup). Not run in Open Instruct. Grader = the paper's Qwen2.5-Math harness,
Avg@8 / Pass@8 over six benchmarks.

| Arm | MATH500 Avg@8 (ours / paper) | Six-benchmark Avg@8 (ours / paper) | Six-benchmark Pass@8 (ours / paper) |
|---|---|---|---|
| 2: Qwen3-4B-Base ← Qwen3-8B, DAPO-14k | 79.45 / 78.81 | 41.88 / 41.45 | 60.29 / 56.71 |
| 1: Qwen3-1.7B-Base ← Qwen3-8B, MATH | 68.12 / 67.76 | 29.98 / 30.22 | 49.89 / 48.35 |

Only Minerva is below the paper (29.78 vs 40.08) and that is a grader artifact: the harness
rejects LaTeX scientific notation; the same samples re-grade to 41.91.

### 3.4 EOPD does not replicate (one seed each)

Miles only, sync, same setup as arm 2 plus the gated forward-KL term (α 1.0, τ 0.8, k 16).

| Metric (six-benchmark mean) | OPD | EOPD | Δ ours | Δ paper |
|---|---|---|---|---|
| Avg@8 | 41.88 | 41.84 | −0.04 | +1.80 |
| Pass@8 | 60.29 | 60.59 | +0.30 | +5.05 |

In-run MATH500 Avg@8 was within ±0.5 of OPD at all five evals. Every specified setting matches
the paper; known deviations: top-16 proxy entropy gate, SGLang bf16 teacher log-probs vs verl's
in-trainer reference, k 16 vs the README's 32, 8×H100 vs 4×A100, single seeds. Honest statement:
no evidence for the claimed gain, not a disproof. A second seed of both arms would settle it.

### 3.5 Teacher entropy: EOPD is not degenerate here

Offline Open Instruct script over vLLM rollouts, no training.

| Pair | Tokens with H > 0.8 | Teacher top-16 mass |
|---|---|---|
| Qwen3-8B on Qwen3-4B-Base (paper pair) | 31 % | 0.91 |
| Verifier-9B on Qwen3.5-4B (ours) | 11 % | 0.9994 |

During EOPD training the gate fired on 27–33 % of tokens.

### Run ids

- Open Instruct async canonical: W&B `v171addf` (2B), `rdcupvki` (4B); Beaker
  `01M1WQ2DRFJF2C1019HMZ02318`, `01M22AHC3VMFDQVBFJCQHPPZ6G`.
- Open Instruct sync reruns: W&B `rg1a2gel` (2B, Beaker `01M2JXTZPDXTZDR217DKGAW48W`),
  `zz3q6skv` + `8nlm6azd` (4B, Beaker `01M2JXV53B6Z12BFNH3V74FVB6` + resume). Fixed-batches
  async rerun `01M2SPXTBP04X0HD7FPDTKN3WZ`.
- Miles Qwen3.5 replication: Beaker `01M2NJB3HF380VQ1554FZPSKN7` (2B), `01M2NJB63SB0E636G6JPJ63KZ2`
  (4B), audit `01M2S39293NTSRJQGYWB2Y183R`.
- Miles paper arms: arm 2 OPD `01M30JAV4K6F2F7XQYV1HY0R6N` (W&B `c4four8o`, continuation of
  `2zs3yb46`); arm 1 OPD `01M30N24RT0JCCGAXPPG0X4F08` (W&B `7csinhqe`); arm 2 EOPD
  `01M30P1BP5N6SJVSRJMNMQHN7A` (W&B `tqou539j`).
- Paper-grader harness evals: OPD hf-219 `01M3119ZR9TQPFG8S8MMQJVDFN`, EOPD hf-219
  `01M32MXFB7R8M00VSR9EKBDE0A`, arm 1 hf-175 `01M32NNHA27J7W1ZT282CQWCV1`.
- Teacher entropy: `01M2RZXXTT0GBCQSJCFKYAA76C` (paper pair), `01M2S08RST44V0RFTGRFRSS72Y` (ours).

## 4. Standing constraints (carry these over verbatim)

- New compute needs Kevin's OK. Read-only CPU Beaker listing jobs are fine.
- Do not read or copy Beaker secret values. Do not delete Kevin's Weka data without his
  explicit OK (pending candidates: `-v3/checkpoints/iter_0000179` and `iter_0000199`, ~120 GB).
- Ruff errors in `open_instruct/test_data_loader_gpu.py` are pre-existing; do not fix them.
- Tracker entries: append-only dated Log entries with the real UTC time from `date -u` captured
  in the same command as the write (earlier entries were stamped a minute ahead and had to be
  corrected); update the "Where we are" line and the step table when work lands; commit and
  push to `origin` each time.
- Local background tasks die when the app restarts or the laptop sleeps; Beaker jobs do not.
- Do not watch finished experiment `01M2NJB3HF380VQ1554FZPSKN7` (Miles 2B phase 2, reference only).
- Miles gotchas: resume double-stepped the LR scheduler until `17e236a11`
  (`--use-checkpoint-opt-param-scheduler`); auto-resume reuses the launch image, so code fixes
  after launch are not picked up; a resumed job starts a new W&B run id (join by rollout);
  Miles refuses to reuse an `output.root` whose spec changed (bump `-vN`); `ai2/open-instruct-dev`
  was far over its allocation target and lost preemption tiebreaks.
- Open Instruct gotchas: mason launches need `--no_auto_dataset_cache`; `keep_last_n_checkpoints`
  prunes DeepSpeed states only; per-dataset eval scores are printed in the job log after
  "Evaluation responses received".

## 5. Open decisions for Kevin (nothing running)

1. Second seed of the OPD and EOPD arms (2 × ~16 h on one node) to put error bars on the
   −0.04 / +0.30 EOPD result. Proposed, not approved.
2. Async isolation rerun in Open Instruct (staleness only, fixed composition; 4 nodes ~2 h).
3. Weka deletion above.
4. Later options: port EOPD to the Qwen3.5 verifier setting; TIS ablation; Thinking Machines
   8B ← 32B recipe.
5. **New focus (this handoff): throughput and performance, Open Instruct vs Miles, for RL and
   OPD.** Plan in section 7.

## 6. Where the numbers we already have come from

| Quantity | Open Instruct | Miles |
|---|---|---|
| Per-step wall clock | W&B `time/total`, `time/training`, `time/getting_response`, `time/weight_sync`, `time/saving`; `time/generation_idle_waiting_for_trainer` from the data loader | W&B `perf/*` per rollout; Beaker job wall clock; `rollout/*` |
| Learner throughput | `learner_tokens_per_second_step`, `learner_tokens_per_second_overall` | `model_tokens_per_gpu_second`, `active_response_tokens_per_gpu_second` (`open_instruct/miles/performance.py`, logged from `actor.py`) |
| Generation throughput | `val/actor_tokens_per_second`, `eval/actor_tokens_per_second` | `gen_throughput` via `pipeline_observer.py` (needs `core.pipeline_observation_interval > 0`); engine metrics |
| Staleness / drops | stale-drop counts in the grpo_fast log (`parse4b.py` style parsing; train-time lengths were not in W&B) | `rollout/fully_async/rejected_policy_groups`, `completed_queue/consumer_wait_seconds` |
| Sizing before launch | none | `python -m open_instruct.miles plan <toml>` prints async capacity and throughput warnings |

## 7. Plan: throughput and performance, Open Instruct vs Miles, RL and OPD

### Question

For the same recipe on the same hardware, how do the two stacks compare on (a) wall clock and
GPU-hours per optimizer step, (b) sampled and trained tokens per GPU-second, (c) idle fraction
(trainer waiting on generation and vice versa), and (d) GPU-hours to a fixed learning target,
in both synchronous and asynchronous modes, for OPD and for verifier RL?

### Design rules

- Same node count and GPU type per pair (H100, `ai2/jupiter` or `ai2/saturn`), same model pair,
  same prompts, same batch (128 prompts × n samples), same max response length, same LR, same
  number of steps, same eval (greedy post-hoc via `qwen35_math_posthoc_eval.sh`, our verifier).
- Charge every GPU the job holds, including Miles' dedicated teacher GPU and Open Instruct's
  in-trainer teacher forward; report GPU-minutes per optimizer step and per 1k sampled tokens.
- Short runs for throughput (20 steps after warm-up, first 2 steps excluded), long runs only
  where a learning-parity claim is needed.
- One node (8 GPUs) per arm keeps the matrix affordable and matches how Miles has been run.
  Open Instruct must then be given a one-node layout (learners + vLLM engines on one node; the
  existing `*_smoke_1node.sh` scripts are the template).

### Matrix

| # | Algorithm | Stack | Mode | Exists? | Cost (8×H100) |
|---|---|---|---|---|---|
| T1 | OPD 4B ← verifier-9B, 128×2, 16k | Miles | sync (barrier) | Yes: phase-2 4B run, 17 min/rollout. Re-measure 20 steps with `pipeline_observation_interval` on | ~6 h |
| T2 | same | Open Instruct | sync (`--synchronous_rollouts`) | Only on 4 nodes. New one-node run | ~6 h |
| T3 | same | Open Instruct | async (`async_steps 4`, in-flight) | Only on 4 nodes. New one-node run | ~3 h |
| T4 | same | Miles | async (`fully_async`, `max_policy_lag` 3, TIS) | **Never run; not refused on Megatron but unproven.** Attempt only after T6 works | ~3 h if it runs |
| T5 | Verifier RL (DPPO/GRPO) Qwen3.5-4B on DAPO math, 128×8, 16k | Open Instruct | async (production default) and sync | Scripts exist (`qwen35_4b_dppo_opd_smoke_1node.sh`, `qwen35_9b_verifier_dppo_math_4node.sh` as template) | ~3 h + ~6 h |
| T6 | same | Miles | sync and async | **No Qwen3.5 GRPO run file yet.** Write `configs/miles/<name>.toml` from `grpo-disaggregated.toml` / `grpo-async-disaggregated.toml` with the Megatron backend, `qwen35-math-v1` prompts and `verifiers.json` already on Weka | ~6 h + ~3 h |

Total for the short-run matrix: roughly 36 GPU-node-hours; each arm is one `min_runtime 8h`
window. Needs Kevin's approval before any launch.

### Phases

**P0, no GPU (do first).**
1. Metric map: one table pairing each Open Instruct timing key with the Miles equivalent
   (section 6 is the start). Decide the normalisations: GPU-min per optimizer step, tokens per
   GPU-second sampled and trained, idle fraction.
2. Write the Miles Qwen3.5-4B GRPO run file (T6) and validate it with
   `python -m open_instruct.miles plan`; write the one-node Open Instruct OPD and RL launch
   scripts (T2, T3, T5) from the smoke templates; check both use the same DAPO prompts and the
   same verifier.
3. Re-derive the existing numbers into the normalised units from W&B (`rdcupvki`, `zz3q6skv`,
   `8nlm6azd`, `rg1a2gel`, `v171addf`, Miles 4B/2B) so there is a baseline row before new runs.
4. Add a "Throughput" section skeleton to the tables page and a step-7 row to the tracker.

**P1, throughput short runs (after approval).** Launch T1, T2, T3, T5-async, T6-sync; 20
optimizer steps each; pull W&B and job logs; fill the table. Parity check: greedy post-hoc eval
of the step-20 checkpoints must agree within noise between stacks for the same algorithm and
mode, otherwise the throughput comparison is comparing different training.

**P2, Miles async.** T6-async (GRPO) first: confirm `rollout/fully_async/*` metrics, drop rate,
and that learning matches Open Instruct async at step 20. Then T4 (OPD async on Megatron) as an
explicit experiment: if the wrapper or the audit rejects it, record why and stop.

**P3, learning-to-target (optional, expensive).** Only if P1 shows a gap larger than ~20 % in
GPU-hours per step: run the winning configuration of each stack to the same eval target (4B
OPD: MATH500 greedy 0.89 / DAPO holdout 0.75 at step 20) and report GPU-hours to target.

### Deliverables

- Tables page: "Throughput and cost" section with the matrix above filled in, plus the metric
  map. Tracker: step-7 row and dated Log entries per launch and result.
- A one-paragraph recommendation on which stack to use for production OPD and for RL at 4B
  scale, with the async caveats from section 2 and the Open Instruct async flaw (length-sorted
  batches, stale drops) stated explicitly.

### Risks and gotchas for this plan

- Open Instruct on one node has never been run for a full 4B OPD step here; vLLM engine count
  and learner count must be re-balanced (`--num_learners_per_node`, `--vllm_num_engines`).
- Miles OPD needs a teacher GPU; the RL arms do not. Report per-GPU numbers, not per-node.
- Miles under-provisioned inference in phase 2 (3 student engines for 4 trainers); tune
  `[inference].gpus` vs `[trainer].gpus` once with `plan` before measuring, and record the
  split.
- Miles auto-resume starts a new W&B run id; keep each throughput arm inside one 8 h window.
- `ai2/holmes` B300 nodes break the Open Instruct image; pin clusters to jupiter/saturn for
  the Open Instruct arms, and use the same clusters for Miles so GPU types match.
