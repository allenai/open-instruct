# Backend Comparison: DeepSpeed vs OLMo-core

Matched-config A/B benchmarks (150 steps; steady-state = mean over steps 20–150;
one Docker image lineage from branch `backend-parity` (base commit `eedd78a9f`;
instrumentation + launch-script fixes through `93a5b3c51` — the only Python
change post-launch was a metrics callback, results-neutral); cluster ai2/jupiter
H100s). Scripts:
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
| OLMo-core packed (budget 0.1 + compile) | **62,940 tok/s total (~1,967/GPU)** — **3.0× DeepSpeed** (3.454 s/step, matching the log-derived 3.436) | **8.56%** (3.1× DS — consistent with the TPS ratio) | 0.437 (see note) | 0.6924 (within 0.1% of log 2) | [01KZTS8JQ9](https://beaker.org/ex/01KZTS8JQ929YA1Y8JA5DKTYC6) | sivsja72 |

Note on the packed loss columns: tokens-per-step differ between the two packed
implementations (their collators fill the 16k budget differently), so the raw
step-time ratio (~8×) overstates the backend difference — tokens/s and MFU
(both ~3×) are the fair comparison. For the same reason, step-indexed loss
values are not directly comparable (each backend has seen a different number of
pairs by step N): OC's steady 0.437 vs DS's 0.5408 reflects differing batch
schedules, not a training defect. The step-0 invariant (policy == reference ⇒
loss = log 2) holds on both backends (DS: 0.6931 exact ×3 runs; OC: 0.6924,
within 0.1%, consistent with a first-log-after-first-update offset), which is
the conversion-correctness check. A downstream-eval check during the Part 2 DPO
PR is recommended as the definitive loss-equivalence confirmation.

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

## Downstream eval equivalence (the DPO follow-up, expanded to all three stages)

Step-indexed losses are not comparable across the two packed implementations, so
the definitive parity check is downstream: train the same model / data / steps /
lr / seed once per backend, export both to HF format, and evaluate on a fixed
suite. Equivalent backends should score within task noise.

Setup: six deterministic (temperature-0) tasks via oe-eval, all `::tulu` chat
configs — `gsm8k`, `bbh:cot-v1`, `minerva_math`, `codex_humanevalplus`,
`ifeval`, `popqa` — against the checkpoints from the A/B runs above. Full run
list in PR #1827.

### DPO — matched data, matched steps: scores match

Both DPO runs consumed the same dataset fraction (wandb `epoch` at step 150:
0.640 DeepSpeed vs 0.638 OLMo-core), so this is a genuine matched-data
comparison, not just matched-config.

| task | DeepSpeed | OLMo-core | Δ |
|---|---|---|---|
| bbh:cot-v1 | 0.0000 | 0.0000 | 0.000 |
| codex_humanevalplus | 0.5169 | 0.5136 | −0.003 |
| gsm8k | 0.0227 | 0.0250 | +0.002 |
| ifeval | 0.6562 | 0.6229 | −0.033 |
| minerva_math | 0.0000 | 0.0000 | 0.000 |
| popqa | 0.1529 | 0.1558 | +0.003 |
| **avg** | **0.2248** | **0.2196** | **−0.005** |

The near-zero math/bbh rows are an eval-format artifact of this model family,
not model damage: the *base* (sft-1115) emits immediate-EOS/empty continuations
under the `::tulu` 8-shot CoT chat format (median 1 output token on gsm8k), and
both trained models behave the same terse way. The pair agrees beyond scores, on
output-shape distributions: gsm8k median output 4 vs 4 tokens, mean 471 vs 479,
fraction ≤10 tokens 0.81 vs 0.79. The backends are statistically
indistinguishable even in the degenerate regime.

### GRPO — matched rollout budget: scores match

| task | DeepSpeed | OLMo-core | Δ |
|---|---|---|---|
| bbh:cot-v1 | 0.2560 | 0.2560 | 0.000 |
| codex_humanevalplus | 0.5756 | 0.5488 | −0.027 |
| gsm8k | 0.1895 | 0.1615 | −0.028 |
| ifeval | 0.1978 | 0.2015 | +0.004 |
| minerva_math | 0.1136 | 0.1081 | −0.006 |
| popqa | 0.1693 | 0.1684 | −0.001 |
| **avg** | **0.2503** | **0.2407** | **−0.010** |

bbh identical to four decimals; every task within 2.8 points under
temperature-1.0 rollout noise.

### GRPO — seed-variance noise floor: the backend gap is seed noise

Whether 2–3 point gaps mean anything depends on the same-backend noise floor,
so the DeepSpeed GRPO run was repeated at seeds 2 and 3 (identical config
otherwise) and evaluated on the same suite:

| task | DS seed 1 | DS seed 2 | DS seed 3 | seed min–max | OLMo-core |
|---|---|---|---|---|---|
| bbh:cot-v1 | 0.2560 | 0.2720 | 0.2480 | 0.2480–0.2720 | 0.2560 ✓ |
| codex_humanevalplus | 0.5756 | 0.5711 | 0.5561 | 0.5561–0.5756 | 0.5488 (−0.007 below) |
| gsm8k | 0.1895 | 0.1243 | 0.1744 | 0.1243–0.1895 | 0.1615 ✓ |
| ifeval | 0.1978 | 0.1774 | 0.2015 | 0.1774–0.2015 | 0.2015 ✓ |
| minerva_math | 0.1136 | 0.1136 | 0.1007 | 0.1007–0.1136 | 0.1081 ✓ |
| popqa | 0.1693 | 0.1695 | 0.1678 | 0.1678–0.1695 | 0.1684 ✓ |
| **avg** | **0.2503** | **0.2380** | **0.2414** | 0.2380–0.2503 | **0.2407 ✓** |

The OLMo-core average lands inside the DeepSpeed seed spread, as do five of six
individual tasks (codex is 0.7 points below the seed minimum — small next to
gsm8k's 6.5-point seed swing). Changing the *backend* moves scores less than
changing the *seed*: the cross-backend gap is indistinguishable from seed noise.

### SFT — functional health check only (not matched data)

The SFT pair is matched-config but not matched-data: OLMo-core packs 4096-token
blocks while DeepSpeed pads ~490-token examples, so over 150 steps OLMo-core saw
roughly 8× the tokens (see data-semantics caveat above). Reported as a health
check of training + the new HF export path; the OLMo-core edge is consistent
with the extra data:

| task | DeepSpeed | OLMo-core |
|---|---|---|
| bbh:cot-v1 | 0.0000 | 0.0000 |
| codex_humanevalplus | 0.1095 | 0.1366 |
| gsm8k | 0.0197 | 0.0227 |
| ifeval | 0.2569 | 0.3438 |
| minerva_math | 0.0000 | 0.0000 |
| popqa | 0.2220 | 0.1871 |
| **avg** | **0.1014** | **0.1151** |

### SFT — epoch-matched: scores match

To remove the packing asymmetry, the pair was re-run with `--num_epochs 1` and
no step cap: both backends traverse the identical 60k examples exactly once
(DeepSpeed ~1875 padded steps, OLMo-core ~224 packed steps — different step
counts, same data). This is the genuine matched-data SFT comparison:

| task | DeepSpeed ep1 | OLMo-core ep1 | Δ |
|---|---|---|---|
| bbh:cot-v1 | 0.0000 | 0.0000 | 0.000 |
| codex_humanevalplus | 0.1426 | 0.1328 | −0.010 |
| gsm8k | 0.0167 | 0.0227 | +0.006 |
| ifeval* | 0.3771 | 0.3789 | +0.002 |
| minerva_math | 0.0000 | 0.0000 | 0.000 |
| popqa | 0.2000 | 0.2124 | +0.012 |
| **avg** | **0.1227** | **0.1245** | **+0.002** |

Average gap 0.2 points, max task gap 1.2 points — tighter than the DPO and GRPO
pairs and well inside the measured seed noise floor above.

*ifeval for this pair ran at eval `--max-length 4096` (see caveat below), so its
row is comparable within the pair but not to the 150-step SFT table above.

### Eval caveats and operational notes

- **Base rows are confounded; only within-pair columns are read strictly.**
  Training stamps its chat template into the export while base models eval with
  their stock template, so trained-vs-base deltas mix training effects with
  template effects. References: sft-1115 avg 0.4303, Qwen3-4B-Base avg 0.4998.
- **SFT base reference dropped.** Raw OLMo-2-1124-7B has no chat template, so
  the `::tulu` chat configs fail before generating. The pair comparison is
  unaffected — and the failure itself confirms both backends installed the chat
  interface during SFT.
- **Tokenizer patches for eval-image compatibility.** The training image's newer
  transformers writes `tokenizer_class: TokenizersBackend` and list-typed
  `extra_special_tokens`; both crash oe-eval's older transformers at model load.
  Exports were patched in place (class → `PreTrainedTokenizerFast`; list-typed
  key removed) before evaluation. Worth tracking for the next oe-eval image bump.
- **Eval `--max-length` must respect the model's position limit.** OLMo-2-1124-7B
  has `max_position_embeddings: 4096`; evaluating it with `--max-length 8192`
  intermittently kills vLLM with a device-side index assert
  (`0 <= tmp16 < 4096`) once a generation runs past position 4096 — reliably on
  ifeval (longest generations), occasionally on codex. The epoch-matched pair's
  ifeval was re-run at 4096 for both backends. The 150-step SFT table above ran
  at 8192 and survived; treat its ifeval row with that in mind.
- **Datalake outage.** `oe-eval-datalake.allen.ai` was unreachable from Beaker
  compute nodes throughout; metrics were harvested from each experiment's Beaker
  result dataset (`metrics-all.jsonl`), and relaunches used `--no-datalake`.
- The OLMo-core SFT eval used the final HF export added in the stacked PR
  (`olmo_core_finetune.py` previously ended training with no HF-format save).

## Verdicts

| Stage | Verdict | Basis |
|---|---|---|
| SFT | **PASS (OLMo-core)** | ~7× tokens/s/GPU at matched config (packing-driven); loss sanity within ~5%. Downstream eval equivalence **confirmed** on the epoch-matched pair (same 60k examples, one epoch each): avg Δ +0.002, max task gap 1.2 points, through the new HF export path |
| DPO | **PASS (OLMo-core)** | 3.0× tokens/s and 3.1× MFU at the packed config both backends can run; step-0 invariant holds on both. Downstream eval equivalence **confirmed** on matched data (epoch 0.640 vs 0.638): six tasks within 3.3 points, avg Δ −0.005. Finding: OC requires packing at 16k — the unpacked production config only runs on DeepSpeed |
| GRPO | **PASS (tie)** | corrected throughput within ~10%, OC slightly faster end-to-end; rewards/losses track; downstream evals within 2.8 points on all six tasks, bbh identical — and the OLMo-core average sits **inside the measured DeepSpeed seed-variance spread** (seeds 1–3: 0.2380–0.2503; OC: 0.2407) |

Per the spec's failure-branch rules, a PASS greenlights the Part 2 rename for
that stage. The DPO PASS originally carried one follow-up — an eval-based
equivalence check, since step-indexed loss is not comparable across the two
packed implementations — which is now complete (see "Downstream eval
equivalence" above): all three stages' checkpoint pairs score within task noise
of each other, the seed-variance runs put a measured floor under "task noise"
(same-backend seed changes move the average by 1.2 points; the backend change
moves it 1.0), and the epoch-matched SFT pair closes the packing-asymmetry gap
in the original SFT comparison.

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
