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
