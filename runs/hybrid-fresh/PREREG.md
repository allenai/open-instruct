# Fresh hybrid: MOPD 60 → mixed RL 240 with full optimizer continuity, seed 42 (preregistered 2026-11-11 PT)

Written before any stage of this run was launched. It supersedes the design in `runs/hybrid/PLAN.md` (kept as the
record of the earlier proposal; see the note at its top).

## Decided by Kevin

- 2026-11-10: "i want to test mopd as a starting training like 20% of steps then continue with RL if that helps."
- 2026-11-11, after learning that the original runs' update-59 optimizer state was pruned: "yes let's just do a fresh
  one then and continue from there". Relayed scope:
  - one fresh chain, 300 student updates in total: MOPD for updates 0–59, then mixed RL for updates 60–299;
  - full optimizer continuity at the switch: Adam moments, optimizer step, RNG and the intended (constant) LR schedule
    carry over, not a fresh optimizer;
  - the prompt/data cursor continues at the boundary (the earlier D2-a);
  - the first planned seed only (seed 42 = seed 1), the existing model, base, teachers, recipes and matched eval;
    no other seeds or arms;
  - the full update-59 checkpoint is kept permanently as a boundary anchor, plus the final state;
  - new output roots on `/weka/olmo-3p5-post-checkpoints/kevinfarhat`.
- D3 (prediction) and D4 (practical margin) are **not** decided. Nothing here claims they are.

## Defaulted by the agent (review these)

- The prediction and falsifier below are the agent's, recorded before launch so the result cannot shape them. They
  are not Kevin-approved (D3 open).
- No margin. The decision rule is the study's existing CI-sign rule, descriptive only (D4 open).
- Stage B's `model.source` is the base model, as in mixed RL 300. Its weights come from the loaded Megatron checkpoint
  and are published to the SGLang engines before the first rollout, as in the MOPD 300 resume.
- Stage A keeps 2 checkpoints (Nov 11 OPD default). Stage B keeps every save: Megatron GRPO cannot prune.

## Design

| | Stage A (MOPD) | Stage B (mixed RL) |
|---|---|---|
| Config | `hybrid-fresh-mopd60-qwen35-9b-all3.toml` | `hybrid-fresh-rl240-qwen35-9b-all3.toml` |
| Parent (diff) | `mopd-qwen35-9b-all3-100u.toml` (`stage-a.diff`) | `mixrl-qwen35-9b-all3-300u.toml` (`stage-b.diff`) |
| Updates | 0–59 (`num_rollouts = 60`) | 60–299 (`num_rollouts = 300`, `initial_rollout = 60`) |
| Objective | MOPD reverse KL, DPPO-TV 0.03, routed teachers (v4 math/IF/code hf-99) | verifier GRPO, DPPO-TV 0.1, verifiers v4 |
| Shared | base Qwen3.5-9B, data v4 `mix_all3` (same prepared file, sha256 `6859d61a…`), seed 42, 64 prompts × 8, LR 1e-6 constant, Adam β (0.9, 0.999), ε 1e-8, clip 1.0, weight decay 0, domain-balanced token mean, TP2 × DP2 trainer | |
| GPUs | 4 trainer + 3 student inference + 1 teacher | 4 trainer + 4 inference |
| Root | `/weka/olmo-3p5-post-checkpoints/kevinfarhat/mopd-phase1/runs/hybrid-fresh-mopd60-qwen35-9b-all3` | `.../hybrid-fresh-rl240-qwen35-9b-all3` |

**Handoff.** When stage A completes, `oi-mopd-phase1-runs/hybrid-fresh/seed_b.py` (a CPU job):
- hardlinks stage A's `iter_0000059` into stage B's `checkpoints/`;
- writes marker 59 and copies `global_dataset_state_dict_59.pt`.

Stage B then takes the ordinary GRPO resume path, which:
- loads weights, Adam moments and step, and RNG with `--load`;
- loads the LR scheduler with `--use-checkpoint-opt-param-scheduler`;
- has MILES restore the data cursor;
- starts at rollout 60.

`training.initial_rollout = 60` does two things. It refuses to start if the seeded state is missing, so there is no silent fresh start. It also limits stage B's evidence and audit to rollouts 60–299.

**Compatibility evidence (CPU, before launch).** Jobs 01M76JEEZT96WGPASMYJ9X4T7W and 01M76K728S0FRNM1BW43RHFB0Y compared an OPD-written checkpoint (MOPD 100 s1 `iter_0000099`) with GRPO-written ones (mixed RL 300 s1):
- Tensors: identical key sets (391), with 0 shape, dtype or chunking differences. That includes the 12 distributed-optimizer tensors.
- Objects: identical key sets (106), including per-rank RNG.
- Data cursors: identical schema (`sample_offset`, `epoch_id`, `sample_group_index`, `sample_index`, `olmo_async_pending` schema 1).
- Prepared prompt file: identical.

**Exposure.** H sees prompt batches 0–299 in the same seeded order as mixed RL 300 s1 and MOPD 300 s1; the order check is 01M73Q1BNVRT9RE2TASW3QRNF9. The fresh MOPD stage is not a bitwise replay of MOPD 100 s1, because async rollouts are not deterministic.

**Cost accounting.** 60 MOPD + 240 RL student updates, plus 300 sunk teacher updates: **600 total updates**. H is matched to RL 300 on student updates and prompts, **not on compute**.

## Question, prediction, falsifier

- **Question:** at 300 student updates on the same prompts, is H (MOPD 0–59 → RL 60–299) ahead of, behind, or unclear relative to mixed RL 300 s1 on internal math, IF and code? Secondary: H vs MOPD 300 s1.
- **Prediction (agent's, not approved; D3 open):** H is ahead of mixed RL 300 s1 on IF.
- **Falsifier:** an IF upper bound below 0 (behind) falsifies "an MOPD warm start helps IF at a fixed student budget" for this seed. "Unclear" means not supported. Math and code are guards with no directional prediction, reported under the same rule.

## Analysis

- **Data:** the matched internal evaluation of `hf-299`, 256 prompts × 8 answers per task at temperature 1, scored as in `seedstats3.py`.
- **Paired comparisons, within seed 1:**
  - H − RL300 s1;
  - H − MOPD300 s1.
- **Interval:** 95% prompt-level bootstrap, 4,000 resamples.
- **Rule:** ahead if the lower bound > 0, behind if the upper bound < 0, unclear otherwise. No equivalence claim.
- **Single seed:** training-seed variance cannot be estimated, so any result is a one-seed observation, not a general claim.
- **Seed context:** mixed RL 300's seed-to-seed differences (3 seeds) are reported beside the result, descriptively. They play no decision role.
- **Secondary benchmarks:** standard benchmarks and boxed AIME are not part of this authorization.

## Gates (mechanical only)

- **A0 (before launch):**
  - configs validate;
  - diffs only in the listed fields;
  - roots absent;
  - new volume mounted and writable, 135 TB free (job 01M76JEEZT96WGPASMYJ9X4T7W);
  - CPU tests pass.
- **A1 (stage A startup):**
  - all 3 teachers up;
  - step 0 metrics finite.
- **A2 (stage A end):**
  - workflow complete, marker 59;
  - `iter_0000059`, cursor 59 and `hf-59/.complete` present;
  - audit passes.
- **S (seeding):**
  - stage B's `iter_0000059` shares inodes with stage A's;
  - cursor sha256 equal;
  - stage A's anchor directories made read-only;
  - `ANCHOR.json` written.
- **B1 (stage B startup, from `training.log`):**
  - "successfully loaded checkpoint … at iteration 59";
  - cursor `global_dataset_state_dict_59.pt` loaded;
  - first trained step 60, LR 1e-6;
  - finite metrics;
  - no optimizer- or RNG-skipped load message.
- **B1b (continuity receipt):** at stage B's first save (`iter_0000074`), the optimizer step and scheduler counters continue from stage A's (stage A at 59 → 75 updates). A fresh optimizer would show 15.
- **B2 (stage B end):**
  - `hf-299` exported;
  - audit over rollouts 60–299 passes.
- **H3 (evaluation):** 256 × 8 answers per task, 0 failed.

No exclusions and no score-driven retries; relaunches are mechanical only.

## Retention

- **Kept permanently:**
  - stage A `iter_0000059`, `hf-59` and `checkpoints/rollout` (the boundary anchor, read-only after seeding);
  - stage B `iter_0000299` and `hf-299` (the final state).
- **Rolling retention:**
  - stage A keeps the newest 2, and its last save is 59, so the anchor is never a pruning candidate;
  - stage B keeps all saves (74, 99, …, 299 at `save_interval` 25), about 1.4 TB.

## Budget (estimate, GPU-h)

| Item | Basis | GPU-h |
|---|---|---|
| Stage A, 60 MOPD updates | MOPD 100 s1: 3.75 min/update, about 0.4 h startup, 8 GPUs | ≈35 |
| Stage B, 240 RL updates | mixed RL 300 s1: 3.47 min/update, about 0.5 h startup, 8 GPUs | ≈115 |
| Internal eval of `hf-299` | MOPD 300 eval8: about 0.5 h × 8 GPUs | ≈4 |
| Seeding / readouts | CPU | 0 |
| Preemption replay allowance | about 10% | ≈15 |
| **Total** | | **≈170** |

Peak 8 GPUs (Holmes; budget ai2/oe-other; workspace ai2/olmo-instruct).
