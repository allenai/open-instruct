# Qwen3.5 math OPD: on-policy reruns and Miles replication handoff

Status notes for the Qwen3.5 math OPD / DPPO campaign (Kevin Farhat). Written
September 15, 2026 so a fresh session can continue from the committed state.

## Why

The campaign brief (canvas, September 9) found pure OPD peaks near step 60-80 and
regresses by step 100 (verifier-2B -> 2B collapses at LR 1e-6; verifier-9B -> 4B
peaks at step 20). The canonical runs used `async_steps=4` with in-flight weight
updates, so each batch was sampled by weights up to four updates old and single
responses could span a weight swap. The first diagnostic is to rerun the two most
informative arms strictly on-policy, changing nothing else, before replicating in
Miles.

## What exists on `codex/qwen35-math-opd`

- `--synchronous_rollouts` in `open_instruct/data_loader.py` (gate in the
  data-preparation actor), `open_instruct/grpo_fast.py` (post-update flag on the
  weight sync trigger) and `open_instruct/actor_manager.py` (on-policy step).
  Requires `async_steps=1`, `inflight_updates=false`, no zero-std filtering or
  active sampling, and ZeRO-3. Tests: `TestSynchronousRolloutsConfig` in
  `open_instruct/test_data_loader_gpu.py`.
- `scripts/general_agent/terminal/rl/qwen35_math_opd_sync_rerun.sh`: `ARM=2b`
  (verifier-2B teacher -> 2B, LR 1e-6) or `ARM=4b` (verifier-9B teacher -> 4B,
  LR 5e-7), pinned to the canonical image `01M01EPKXMXR4502S1HNJVYN0M`, DAPO split
  `01M1TKR7BQE4D5CYKYM1TZX0AM` and Weka teacher paths; saves every 10 steps.
- `scripts/general_agent/terminal/rl/make_math_patch_dataset.sh`: builds the
  code-patch Beaker dataset (five `open_instruct/*.py` files) that the launch
  scripts overlay onto the pinned image at `/patch`.

## Launch sequence (from a beaker-session with Beaker and Weka access)

```bash
git checkout codex/qwen35-math-opd
scripts/general_agent/terminal/rl/make_math_patch_dataset.sh            # note the dataset ID
PATCH_DATASET=<id> ARM=2b RUN_MODE=smoke scripts/general_agent/terminal/rl/qwen35_math_opd_sync_rerun.sh
```

The smoke is one node and 192 episodes (three steps; the gate only engages from
the second data-preparation step). In its log, every data-preparation step
after the first should print `weights synced after ... queueing 64 on-policy
prompts`. That confirms the gate engages and that the patch overlay is compatible
with the image (image commit `154e1f701` is not in this repository's history, so
compatibility could not be checked statically). Then:

```bash
PATCH_DATASET=<id> ARM=2b scripts/general_agent/terminal/rl/qwen35_math_opd_sync_rerun.sh
PATCH_DATASET=<id> ARM=4b scripts/general_agent/terminal/rl/qwen35_math_opd_sync_rerun.sh
```

Evaluate every saved checkpoint with the matched greedy protocol
(`qwen35_math_posthoc_eval.sh`, `EVAL_MODE=greedy`), never the inline numbers.
Compare against canonical W&B `v171addf` (2B) and `rdcupvki` (4B).

## Miles replication plan (next, on `robertb/miles-qwen35-opd`)

Robert offered direct commits to that branch. Its Qwen OPD route
(`open_instruct/miles/opd_config.py`) is a closed two-update prototype. Order:

1. Relax the schema: local teacher path (copy the check from `core_opd.py`),
   learner/teacher revisions, `num_rollouts`, save cadence, response/context
   lengths, eval temperature and sample count, online W&B.
2. Data and evals: allow `rl_manifest` in the OPD config, write the
   `qwen_instruct_user_boxed_math` template out as a template file, generalize
   `opd_hooks.evaluate` from one GSM8K holdout to named eval sets with the math
   verifier. Evaluate exports post hoc in Open Instruct regardless.
3. Run verifier-9B -> 4B in Miles (needs only 1 and 2). Miles uses PPO clipping,
   not DPPO with a TV mask, so label it framework-native OPD.
4. Add a Qwen3.5-2B Megatron profile, then verifier-2B -> 2B.
5. Verifier-RLVR 2B in Miles via upstream Megatron GRPO as the positive control.

Confirm whether Miles scores the student side of the reverse KL with rollout or
trainer log-probabilities before comparing curves; Open Instruct uses rollout
log-probabilities.

## Results (2026-09-16)

Both strictly-on-policy reruns finished. Beaker and W&B IDs:

| Run | Beaker | W&B | Notes |
|-----|--------|-----|-------|
| 2B sync (verifier-2B -> 2B, LR 1e-6) | `01M2JXTZPDXTZDR217DKGAW48W` | `rg1a2gel` | exit 0, 2.15 min/step |
| 4B sync (verifier-9B -> 4B, LR 5e-7), steps 1-85 | `01M2JXV53B6Z12BFNH3V74FVB6` | `zz3q6skv` | preempted at the 4h `min_runtime` |
| 4B sync resume, steps 81-100 | `01M2M9EXJHA9FEW1Y5K0N2QJ46` | `8nlm6azd` | `--checkpoint_state_dir` from step 80, exit 0 |

Checkpoints: `.../deletable_checkpoint/kevinfarhat/qwen35_2b_opd_from_verifier_2b_sync_lr1e6_100step_4node__42__1789491103_checkpoints/step_N`,
`.../qwen35_4b_opd_from_verifier_9b_sync_lr5e7_100step_4node__42__1789505099_checkpoints/step_N` (N <= 80) and
`..._4node__42__1789542980_checkpoints/step_N` (N = 90, 100).

### In-loop eval (temperature 1.0, one sample per prompt, `eval/scores`)

| step | 2B canonical `v171addf` | 2B sync | 4B canonical `rdcupvki` | 4B sync |
|-----:|------:|------:|------:|------:|
| 0 | 0.365 | 0.395 | 0.743 | 0.731 |
| 20 | 0.065 | 0.510 | 0.795 | 0.792 |
| 40 | 0.469 | 0.493 | 0.760 | 0.757 |
| 60 | 0.469 | 0.514 | 0.753 | 0.757 |
| 80 | 0.526 | 0.524 | 0.764 | 0.757 |
| 100 | 0.007 | 0.498 | 0.745 | 0.760 |

### Greedy post-hoc eval (matched harness, accuracy %)

| Model | Checkpoint | DAPO | AIME | BRUMO | MATH-500 | W&B |
|-------|-----------|-----:|-----:|------:|---------:|-----|
| Base 2B | - | 30.7 | 13.3 | 23.3 | - | `1ywzoe3p` |
| Verifier-DPPO 2B (teacher) | step 100 | 49.4 | 26.7 | 43.3 | - | `l7ioqxec` |
| 2B canonical async | step 80 | 49.8 | 26.7 | 33.3 | - | `aa4lbbsg` |
| 2B canonical async | step 100 | 1.0 | 0.0 | 0.0 | - | `pszko9zn` |
| 2B sync | step 30 | 50.2 | 26.7 | 26.7 | 80.0 | `3kefkwtq` |
| 2B sync | step 80 | 47.7 | 30.0 | 26.7 | 81.2 | `ho39pobq` |
| 2B sync | step 100 | 47.7 | 26.7 | 36.7 | 80.2 | `4vadl3oy` |
| Base 4B | - | 64.5 | 33.3 | 63.3 | 87.0 | `xr2sg6zd` |
| Verifier-DPPO 9B (teacher) | step 100 | 82.4 | 66.7 | 63.3 | - | `7njterw1` |
| 4B canonical async | step 20 | 78.5 | 56.7 | 60.0 | 91.0 | `ofh0xefj` |
| 4B canonical async | step 100 | 74.8 | 50.0 | 60.0 | 89.6 | `vs4dibqv` |
| 4B sync | step 20 | 77.5 | 50.0 | 53.3 | 90.2 | `3k3vpjsp` |
| 4B sync | step 80 | 77.9 | 50.0 | 60.0 | 90.4 | `miaylnye` |
| 4B sync | step 100 | 76.2 | 53.3 | 60.0 | 89.8 | `ymr3pohq` |

The full 2B and 4B sync sweeps (every 10 steps) are in W&B under `qwen35_math_posthoc_{2b,4b}_sync_stepN_greedy`.
2B sync DAPO sits at 44-50 from step 10 on; 4B sync DAPO sits at 74-78 from step 10 on. Neither collapses.
AIME and BRUMO are 30 questions each, so one question is 3.3 points.

### Miles replication (framework-native OPD, greedy post-hoc on hf exports)

Miles runs on `robertb/miles-qwen35-opd` use the same prompts, teachers, students, batch
shape, lengths, LR, seed and rollout-engine student log-probs, with strictly on-policy
rollouts, but PPO clipping instead of the DPPO TV mask. `hf-N` is the export after
rollout N (N+1 updates), so `hf-9` pairs with `step_10`.

| Model | Export | DAPO | AIME | BRUMO | MATH-500 | Beaker |
|-------|--------|-----:|-----:|------:|---------:|--------|
| 2B sync (Open Instruct) | step 10 | 44.1 | 30.0 | 30.0 | 81.2 | `8cwnr6ku` (W&B) |
| 2B Miles v5 | hf-9 | 53.1 | 26.7 | 40.0 | 81.0 | `01M2P1W1DRB0BCG1SW5AKA5ZXE` |
| 2B sync (Open Instruct) | step 20 | 45.1 | 26.7 | 33.3 | 77.0 | `810bopdx` (W&B) |
| 2B Miles v5 | hf-19 | 49.0 | 30.0 | 40.0 | 80.0 | `01M2P5B859FGCX0GM6WZ5PRHPK` |
| 2B sync (Open Instruct) | step 30 | 50.2 | 26.7 | 26.7 | 80.0 | `3kefkwtq` (W&B) |
| 2B Miles v5 | hf-29 | 49.0 | 30.0 | 33.3 | 81.0 | `01M2PCYN9JK4KKEYA2CQG62QXM` |
| 2B sync (Open Instruct) | step 40 | 49.0 | 33.3 | 33.3 | 79.0 | `dks8tr3b` (W&B) |
| 2B Miles v5 | hf-39 | 50.0 | 33.3 | 33.3 | 80.0 | `01M2PKZC7SA36MGK00MWDZ70P3` |
| 2B sync (Open Instruct) | step 50 | 49.0 | 23.3 | 23.3 | 80.0 | `01M2KCV6A1SRAKRX1VMPY4VE8A` |
| 2B Miles v5 | hf-49 | 51.0 | 33.3 | 26.7 | 82.0 | `01M2PXJ9JSNW5HF4R79DY3TNA9` |
| 2B sync (Open Instruct) | step 60 | 49.0 | 36.7 | 33.3 | 80.0 | `01M2KJV6B5FA4T1TXGMV0H90AZ` |
| 2B Miles v5 | hf-59 | 49.0 | 26.7 | 26.7 | 80.0 | `01M2Q591VE48CZYYWV1BTZ33MC` |
| 2B sync (Open Instruct) | step 70 | 50.0 | 36.7 | 33.3 | 81.0 | `01M2KMAPBHDGWSFGGGN4WD2N5P` |
| 2B Miles v5 | hf-69 | 46.0 | 30.0 | 43.3 | 79.0 | `01M2QCZG9901Q5QY6ZZJJV4SXM` |
| 2B sync (Open Instruct) | step 80 | 48.0 | 30.0 | 26.7 | 81.0 | `01M2KNYX9KR6Y7MK0PKVZSGYER` |
| 2B Miles v5 | hf-79 | 46.0 | 23.3 | 33.3 | 81.0 | `01M2QY17CYPTGMZ9TVWVADKTZ4` |
| 2B sync (Open Instruct) | step 90 | 49.0 | 33.3 | 30.0 | 81.0 | `01M2KQK4FBYPS341HHPPXV0PAE` |
| 2B Miles v5 | hf-89 | 51.0 | 33.3 | 43.3 | 80.0 | `01M2R72P5H7GSD2TZFBT28PNR4` |
| 2B sync (Open Instruct) | step 100 | 48.0 | 26.7 | 36.7 | 80.0 | `01M2M8Y5BSKVDDN2FW0DJK9RHF` |
| 2B Miles v5 | hf-99 | 49.0 | 26.7 | 23.3 | 80.0 | `01M2RD932J83K39RG0NN6FJNVN` |
| 4B sync (Open Instruct) | step 10 | 73.6 | 33.3 | 53.3 | 90.0 | `fk3gvxmc` (W&B) |
| 4B Miles v4 | hf-9 | 75.0 | 43.3 | 66.7 | 89.0 | `01M2P1W3G8ED6042K0F8KARJEX` |
| 4B Miles v5 | hf-9 | 75.0 | 53.3 | 60.0 | 89.0 | `01M2P5BD1R4TKSV4MZ4PQ8NZ01` |
| 4B sync (Open Instruct) | step 20 | 77.5 | 50.0 | 53.3 | 90.2 | `3k3vpjsp` (W&B) |
| 4B Miles v5 | hf-19 | 79.0 | 46.7 | 66.7 | 91.0 | `01M2PCYRS3T1C093AZT3MKR4SY` |
| 4B sync (Open Instruct) | step 30 | 74.0 | 36.7 | 76.7 | 90.0 | `01M2MGAC03SMWQAXSZG1VW7Q2D` |
| 4B Miles v5 | hf-29 | 77.0 | 50.0 | 60.0 | 90.0 | `01M2Q5956F0W2TMGVB22ATJNKS` |
| 4B sync (Open Instruct) | step 40 | 78.0 | 46.7 | 60.0 | 89.0 | `01M2MHBPW00N58JADA8GPB8384` |
| 4B Miles v5 | hf-39 | 76.0 | 50.0 | 66.7 | 91.0 | `01M2QFQA6KRJ59ZH9MZP9M2KQW` |
| 4B sync (Open Instruct) | step 50 | 76.0 | 46.7 | 63.3 | 90.0 | `01M2NHSTPQ33W41BP7FYMX86FS` |
| 4B Miles v5 | hf-49 | 76.0 | 50.0 | 66.7 | 91.0 | `01M2QNSHDE9HA9HQRW63W7VFCG` |
| 4B sync (Open Instruct) | step 60 | 75.0 | 40.0 | 56.7 | 90.0 | `01M2NJD18ZM2QKBYA32BSD1NVF` |
| 4B Miles v5 | hf-59 | 76.0 | 36.7 | 63.3 | 90.0 | `01M2R1GDDQBB8K8Y0CQBQQ6YKJ` |
| 4B sync (Open Instruct) | step 70 | 78.0 | 40.0 | 63.3 | 90.0 | `01M2NK0A62CKCM84AYA5ER6A8N` |
| 4B Miles v5 | hf-69 | 77.0 | 50.0 | 56.7 | 91.0 | `01M2RBHY5BRYCQX148D6HZQFKC` |

Both frameworks land in the same band: 2B DAPO 44-53 and 4B DAPO 74-78 from step 10
on, with AIME/BRUMO differences within the noise of 30-question sets. The 2B Miles hf-9
DAPO of 53.1 is not sustained at hf-19 (49.0), so it reads as noise rather than a PPO-clip
advantage. Miles reverse KL tracks the Open Instruct sync runs at matched steps (2B 0.033 -> ~0.007
by rollout 10, 4B flat ~0.07). The 2B Miles run finished all 100 rollouts (three 8h Beaker
windows, two auto-resumes) and every one of its ten exports sits in the Open Instruct sync band,
ending at hf-99 DAPO 49.0 vs sync step 100 48.0. Its post-training `opd_audit` step then crashed
(the dumps carry `rollout_log_probs`, not `log_probs`, under `use_rollout_logprobs = true`; fixed on
`robertb/miles-qwen35-opd`) so the job exited 1 after training; the exports were unaffected. Remaining
4B exports (hf-79 onward) are evaluated as the run produces them; the Miles output roots are
`.../deletable_checkpoint/kevinfarhat/miles-opd/runs/qwen35-{2b-opd-from-verifier-2b,4b-opd-from-verifier-9b}-math-v5/hf-N`.

### What the reruns show

1. **The canonical 2B collapses were an artifact of asynchronous sampling, not of pure OPD.**
   With `async_steps=4` and in-flight updates, the trainer consumes the first 256 of ~1024
   in-flight responses, so each batch is length-sorted by completion order. The canonical 2B
   train batches sawtooth between ~4.9k tokens / stop rate 1.0 and ~15.9k tokens / stop rate
   0.09 with a ~6-step period (corr(batch length, batch score) = -0.87); the all-truncated
   batches coincide with the eval collapses at steps 20 and 100, and reverse KL never falls
   below 0.02. Pure OPD advantages are per-token and not group-centered, so batch composition
   steers the update directly. The strictly-on-policy 2B has batch lengths of 11-14k tokens
   with stop rate 0.40-0.70 at every step, reverse KL decays monotonically to 0.0005, and the
   greedy DAPO score matches the verifier-2B teacher (47.7-50.2 vs 49.4).
2. **The async pipeline never trained on the long tail.** The canonical runs dropped ~16.4k (2B)
   and ~17.7k (4B) results as stale (`stale_results_dropped`) against 25.6k episodes trained;
   the sync runs dropped zero. Canonical 4B train batches averaged 1.2-2.4k tokens with stop
   rate 1.00 at 99 of 100 steps and train reward 0.94; sync 4B batches averaged 4.2-6.6k tokens
   with train reward 0.79, and truncations fell from 38 per step to 0 by step 20. The higher
   canonical train reward is selection, not learning.
3. **For 4B the schedule did not change the outcome, only the cost.** Canonical and sync 4B
   in-loop and greedy curves agree within eval noise at every step, and both keep the step-20
   peak / step-100 softening pattern. Reverse KL for the 9B -> 4B pair is flat at 0.06-0.08
   in both schedules (and in the Miles replication), so it is a property of the pair at LR
   5e-7. The on-policy schedule costs 2.1x (2B: 2.15 vs 1.01 min/step) to 4x (4B: ~2.6 vs
   0.65 min/step) because every batch waits for its longest response.

Per-step train stats were parsed from the grpo_fast job logs (`val/sequence_lengths`,
`val/stop_rate`, `stale_results_dropped`), not from W&B, which did not log train-time lengths
for the canonical runs.

### Operational notes

- Beaker caps `min_runtime` at 8h; the 4B rerun was preempted at its 4h `min_runtime` by
  workspace-group rebalancing (exit 134 after SIGTERM). Resume with
  `--checkpoint_state_dir /weka/.../deletable_checkpoint_states/kevinfarhat/<run>` and a new
  run name; grpo_fast logs `Resuming training from step N`. The 4-node full-run launch
  scripts now request the 8h maximum.
- `/weka/oe-adapt-default` reached 100% during the campaign (451T of 455T) and killed the
  first Miles attempts with ENOSPC.
