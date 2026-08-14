# Backend Comparison: DeepSpeed vs OLMo-core

Matched-config A/B benchmarks (150 steps; steady-state = mean over steps 20–150;
one Docker image built from branch `backend-parity` commit `eedd78a9f`, with
launch-script fixes through `93a5b3c51`; cluster ai2/jupiter H100s). Scripts:
`scripts/train/debug/backend_ab/`. Spec:
`docs/superpowers/specs/2026-08-05-backend-consolidation-design.md`.

## Caveats (read before quoting numbers)

- Each backend runs its production memory strategy (DeepSpeed: ZeRO-3/stage-2 +
  full gradient checkpointing; OLMo-core: FSDP/HSDP + `torch.compile` + budget
  activation checkpointing). This compares *backends as production-configured*,
  not isolated kernels.
- SFT data semantics differ structurally: DeepSpeed feeds padded per-example
  batches (measured avg ~490 real tokens/seq on this mixture); OLMo-core packs
  fixed 4096-token FSL blocks (measured 0.3 pad tokens/instance). tokens/s counts
  non-padding tokens on both sides.
- **GRPO metric warning:** `grpo_fast.py`'s reported `val/num_step_tokens`,
  `learner_tokens_per_second_*`, `learner_mfu`, and `actor_mfu` are inflated by
  `world_size / sequence_parallel_size` (= 4× in these runs) because the data-prep
  actor hands every rank the same unsharded global metrics
  (`data_loader.py:1550`) and `grpo_fast.py:1600-1603` concatenates them across
  ranks. Historical DeepSpeed GRPO MFU/TPS claims should be treated as suspect.
  All GRPO numbers below are **corrected**. Only `time/total` is directly
  comparable end-to-end; `time/training` brackets different work on each side
  (DS: data-wait + compute; OC: compute only, excluding weight sync).
- OLMo-core SFT has a wandb sync defect (runs sync ~0 history rows; tracked
  separately) — its SFT numbers below are derived from exact token counts over
  log-timestamped step windows, not wandb.

## SFT — OLMo-2-1124-7B, tulu-3 mixture (60k), seq 4096, bs1 × ga2, 2 nodes × 8

| Backend | tok/s/GPU (non-pad) | MFU | loss | step-0 loss | Beaker | wandb |
|---|---|---|---|---|---|---|
| DeepSpeed (`finetune.py`, ZeRO-3) | 818 | n/a (not instrumented) | 1.007 steady | 1.528 | [01KZA7J8CR](https://beaker.org/ex/01KZA7J8CR03TSJ9V62KPAXPCM) | y45du2vs |
| OLMo-core (`olmo_core_finetune.py`) | ~5,700–7,400 (log-derived) | ~33–37% (derived) | CE 1.06 @ step 150 | n/a (sync bug) | [01KZF959PN](https://beaker.org/ex/01KZF959PN9YQ8JS29VVJQS676) | nflyli07 (0 rows) |

OLMo-core numbers derived from the exact packed token count (150 × 32 × 4096 =
19.66M tokens) over the log-timestamped training window (167–215 s for 150 steps,
≤1.4 s/step) across 16 GPUs. Even the most conservative bound is **~7× DeepSpeed's
818 tok/s/GPU**; the gap is packing density (0.3 pad tokens per 4096-token
instance vs padded ~490-real-token examples). Loss sanity: OC CE 1.06 at step 150
vs DS steady 1.007 — within ~5%, expected given structurally different batch
composition.

## DPO — Olmo-3-7B instruct-SFT ckpt, olmo-3 pref mix (30k), seq 16384, bs1 × ga4, 4 nodes × 8

Unpacked (olmo3 production config) — reference rows:

| Backend | tok/s/GPU | MFU | loss (steady) | step-0 (=log 2) | Beaker | wandb |
|---|---|---|---|---|---|---|
| DeepSpeed (`dpo_tune_cache.py`, ZeRO-3) | ~456 | 1.94% | 0.5411 | 0.6931 exact | [01KZA7JAR1](https://beaker.org/ex/01KZA7JAR1B7A0NR5K6HWMGMTW) | 4bvtcmwu |
| DeepSpeed (replicate) | ~455 | 1.94% | 0.5407 | 0.6931 exact | [01KZEJR73P](https://beaker.org/ex/01KZEJR73PD110PVENFQPK1J1Z) | y01kxrkf |
| OLMo-core | **could not run** (4 failures, see below) | — | — | — | — | — |

Packed (`--packing` both sides; the matched pair — user decision after the
unpacked OC failures):

| Backend | throughput | MFU | loss (steady) | step-0 (=log 2) | Beaker | wandb |
|---|---|---|---|---|---|---|
| DeepSpeed packed | ~651 tok/s/GPU (20.83k total) | 2.77% | 0.5408 | 0.6931 exact | [01KZG13JC1](https://beaker.org/ex/01KZG13JC1WJA5CHRPKJBQ8N83) | 89ibnowv |
| OLMo-core packed (budget 0.1 + compile) | **3.436 s/step** (steps 20→149 in 443.2 s, log-derived) — ~8× DeepSpeed's implied ~27 s/step at the same 128-pairs/step workload | pending wandb pull | pending wandb pull | pending wandb pull | [01KZTS8JQ9](https://beaker.org/ex/01KZTS8JQ929YA1Y8JA5DKTYC6) | sivsja72 |

The pending cells require a wandb API pull of run `sivsja72` (API access was
unavailable at write time); the step-time comparison is from console-log
timestamps and stands on its own. Loss equivalence should be confirmed from
`sivsja72` before treating the DPO verdict as final: expected step-0 = 0.6931
(policy == reference invariant, hit exactly by all three DS runs) and steady loss
≈ 0.54.

**OLMo-core DPO cannot run the unpacked 16k production config on the current
pin** (four documented failures):

1. budget AC 0.1 — OOM (76 GiB): compile graph breaks on variable-length 32k
   concatenated (chosen+rejected) forwards defeat the budget partitioner
   ([01KZA7JD](https://beaker.org/ex/01KZA7JDBNGS22RZB0Y6VPDP7R)).
2. budget AC 0.05 — identical OOM, confirming the partitioner never engaged
   ([01KZEJR9](https://beaker.org/ex/01KZEJR9TSES795EFS2D9415XT)).
3. selected-modules AC (`blocks.*`) + compile — tensor-stride assertion during
   recompute ([01KZF95D](https://beaker.org/ex/01KZF95D6NM27PCTG3W8NNC0RK)).
4. packed + budget 0.5 — dry-run OOM; retaining 50% of 16k-token activations for
   a 7B cannot fit ([01KZG13R](https://beaker.org/ex/01KZG13RB4M680XGMD3SK0TN0T)).

Packed + budget 0.1 (the configuration PR #1466 validated) trains cleanly.
DeepSpeed's ZeRO-3 + full recompute handles the unpacked config with defaults.

## GRPO — Qwen3-4B-Base, DAPO-Math-17k, 128 samples/step, 1 node (4 learner + 4 vLLM GPUs)

Corrected numbers (see metric warning above):

| Backend | time/total (s/step) | global tok/step | tok/s per learner GPU | learner MFU | reward (steady) | Beaker | wandb |
|---|---|---|---|---|---|---|---|
| DeepSpeed (`grpo_fast.py`, ZeRO-2) | 24.83 | ~286k (reported 1.143M ÷ 4) | ~2,877 | reported 44.3 is 4×-inflated; ~11.1 over its data-wait-inclusive window | 4.26 | [01KZB01ZSX](https://beaker.org/ex/01KZB01ZSXCGT3MVMGNGGC8Y7N) | 923rvrm2 |
| OLMo-core (`grpo.py`, FSDP) | 23.56 | ~286k (248.5k response + ~37k prompt) | ~2,637 (response-only) / ~3,029 (incl. prompt) | 42.9 (structurally correct, over its 5.24 s compute window) | 4.12 | [01KZB022JN](https://beaker.org/ex/01KZB022JNSHRNVXQHJNKPJ5KN) | dzk3z8gl |

End-to-end step latency and per-GPU token throughput are a **tie within ~10%**
(OLMo-core ~5% faster on `time/total`). GRPO steps are generation-bound: the
DeepSpeed learner spends most of the step waiting for vLLM
(`time/training` ≈ `time/total`), while OLMo-core's isolated compute window shows
~43% MFU. Loss and reward trajectories are in the same range (loss 0.082 vs
0.070; reward 4.26 vs 4.12 under temperature-1.0 sampling noise).

## Verdicts

| Stage | Verdict | Basis |
|---|---|---|
| SFT | **PASS (OLMo-core)** | ~7× tokens/s/GPU at matched config (packing-driven); loss sanity within ~5% |
| DPO | **PASS (OLMo-core), provisional** | ~8× faster step time at the packed config both backends can run; pending loss-equivalence confirmation from wandb run `sivsja72`. Finding: OC requires packing at 16k — the unpacked production config only runs on DeepSpeed |
| GRPO | **PASS (tie)** | corrected throughput within ~10%, OC slightly faster end-to-end; rewards/losses track |

Per the spec's failure-branch rules, a PASS greenlights the Part 2 rename for
that stage; the DPO PASS is provisional until the loss check lands.

## Reliability observations (part of the comparison)

- OLMo-core DPO required four rounds of memory-strategy intervention to fit a
  production 16k workload that ZeRO-3 + full recompute handles with defaults.
  Root cause: budget-mode AC depends on `torch.compile`'s partitioner, which
  variable-length batches defeat via graph breaks.
- The benchmark surfaced five pre-existing repo defects, each tracked separately:
  the `olmo3_hybrid_7B` config lost in the #1723 olmo-core pin bump; the stale
  `--gradient_checkpointing` flag in committed DPO scripts; an unstable dataset
  config hash in the OC SFT cache flow; the OC SFT wandb sync failure (plus
  `PerfCallback` silently no-oping on the SFT path); and the 4× GRPO metrics
  inflation in `grpo_fast.py`.
- DeepSpeed DPO reproducibility was excellent: three runs (two unpacked, one
  env-var variant) agreed on steady loss to the third decimal (0.5407 / 0.5408 /
  0.5411) and hit the step-0 log(2) invariant exactly every time.
