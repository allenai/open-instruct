# Backend Comparison: DeepSpeed vs OLMo-core

Can OLMo-core replace DeepSpeed as the training backend? We ran matched A/B
pairs (same model, data, steps, lr, seed) for SFT, DPO, and GRPO, compared
speed and training behavior, then evaluated the resulting checkpoints on a
fixed suite. **Answer: yes.** OLMo-core is as fast or much faster, and
checkpoints from either backend score the same on downstream evals — the
cross-backend difference is smaller than the same-backend seed-to-seed
difference we measured.

## Verdicts

| Stage | Verdict | Why |
|---|---|---|
| SFT | **PASS (OLMo-core)** | ~7× tokens/s/GPU (packing-driven). Epoch-matched evals: avg Δ +0.002, max task gap 1.2 pts |
| DPO | **PASS (OLMo-core)** | 3.0× tokens/s, 3.1× MFU at the packed config. Matched-data evals: avg Δ −0.005, max gap 3.3 pts. Caveat: OC cannot run the *unpacked* 16k config |
| GRPO | **PASS (tie)** | Throughput within ~10% (OC slightly faster). Evals within 2.8 pts on all tasks; OC's average sits inside the measured seed-noise spread |

A PASS greenlights renaming that stage's DeepSpeed path to `_deepspeed`
(demote, not delete).

## Setup

150-step runs (steady-state = mean over steps 20–150), one Docker image
lineage from branch `backend-parity`, ai2/jupiter H100s. Scripts in
`scripts/train/debug/backend_ab/`; full run links in PR #1827. Each backend
uses its production memory strategy (DeepSpeed: ZeRO-3 + full gradient
checkpointing; OLMo-core: FSDP + `torch.compile` + budget activation
checkpointing), so this compares backends as configured in practice, not
isolated kernels.

## Speed

**SFT** (OLMo-2-1124-7B, tulu-3 60k, seq 4096, 2×8 GPUs):
DeepSpeed 818 tok/s/GPU; OLMo-core ~5,700–7,400 tok/s/GPU (~33–37% MFU) —
**~7×**, driven by sequence packing (OC packs 4096-token blocks with ~0 pad;
DS pads ~490-real-token examples). Loss agrees within ~5% (OC CE 1.06 vs DS
1.007 steady; exact match isn't expected since batch composition differs).
Runs: DS [01KZA7J8CR](https://beaker.org/ex/01KZA7J8CR03TSJ9V62KPAXPCM),
OC [01KZF959PN](https://beaker.org/ex/01KZF959PN9YQ8JS29VVJQS676).
(OC's numbers are log-derived: its wandb runs sync ~0 history rows — a known
defect, tracked separately.)

**DPO** (Olmo-3-7B instruct-SFT, olmo-3 pref mix 30k, seq 16384, 4×8 GPUs),
both sides packed: DeepSpeed ~651 tok/s/GPU (2.77% MFU); OLMo-core ~1,967
tok/s/GPU (8.56% MFU) — **3.0×**. The step-0 invariant (policy == reference ⇒
loss = log 2) holds on both backends, which is the conversion-correctness
check. Step-indexed losses are *not* comparable across the two packed
implementations (their collators fill the 16k budget differently), which is
why the eval comparison below exists.
Runs: DS [01KZG13JC1](https://beaker.org/ex/01KZG13JC1WJA5CHRPKJBQ8N83),
OC [01KZTS8JQ9](https://beaker.org/ex/01KZTS8JQ929YA1Y8JA5DKTYC6).

**Known limitation:** OLMo-core cannot run the *unpacked* 16k DPO production
config on the current pin — four documented failures (OOMs from budget-AC
compile graph breaks on variable-length forwards; a tensor-stride assertion
under selected-modules AC; runs
[01KZA7JD](https://beaker.org/ex/01KZA7JDBNGS22RZB0Y6VPDP7R),
[01KZEJR9](https://beaker.org/ex/01KZEJR9TSES795EFS2D9415XT),
[01KZF95D](https://beaker.org/ex/01KZF95D6NM27PCTG3W8NNC0RK),
[01KZG13R](https://beaker.org/ex/01KZG13RB4M680XGMD3SK0TN0T)). Packed +
budget-0.1 AC trains cleanly. Until fixed, the unpacked config runs only on
DeepSpeed. DeepSpeed unpacked reference: ~456 tok/s/GPU, steady loss
reproducible to the third decimal across three runs
([01KZA7JAR1](https://beaker.org/ex/01KZA7JAR1B7A0NR5K6HWMGMTW),
[01KZEJR73P](https://beaker.org/ex/01KZEJR73PD110PVENFQPK1J1Z)).

**GRPO** (Qwen3-4B-Base, DAPO-Math-17k, 128 samples/step, 4 learner + 4 vLLM
GPUs): end-to-end step time 24.83 s (DS) vs 23.56 s (OC) — a tie within ~10%,
OC ~5% faster. GRPO steps are generation-bound, so learner speed barely
matters. Rewards and losses track (reward 4.26 vs 4.12 under temperature-1.0
sampling). Runs: DS
[01KZB01ZSX](https://beaker.org/ex/01KZB01ZSXCGT3MVMGNGGC8Y7N),
OC [01KZB022JN](https://beaker.org/ex/01KZB022JNSHRNVXQHJNKPJ5KN).

> **Metric warning:** `grpo_fast.py`'s reported `learner_tokens_per_second_*`,
> `learner_mfu`, `actor_mfu`, and `val/num_step_tokens` are inflated by
> `world_size / sequence_parallel_size` (4× in these runs): every rank
> receives the same unsharded global stats (`data_loader.py:1550`) and
> `grpo_fast.py:1600-1603` concatenates them across ranks. All GRPO numbers
> here are corrected; historical DeepSpeed GRPO MFU/TPS claims are suspect.

## Downstream eval equivalence

The definitive parity check: export both checkpoints of each pair to HF format
and evaluate on six deterministic (temperature-0) oe-eval `::tulu` tasks —
`gsm8k`, `bbh:cot-v1`, `minerva_math`, `codex_humanevalplus`, `ifeval`,
`popqa`. Equivalent backends should score within task noise, and we measured
what "task noise" is (see the seed table below).

### DPO — matched data, matched steps

Both runs consumed the same dataset fraction (epoch 0.640 vs 0.638), so this
is a genuine matched-data comparison:

| task | DeepSpeed | OLMo-core | Δ |
|---|---|---|---|
| bbh:cot-v1 | 0.0000 | 0.0000 | 0.000 |
| codex_humanevalplus | 0.5169 | 0.5136 | −0.003 |
| gsm8k | 0.0227 | 0.0250 | +0.002 |
| ifeval | 0.6562 | 0.6229 | −0.033 |
| minerva_math | 0.0000 | 0.0000 | 0.000 |
| popqa | 0.1529 | 0.1558 | +0.003 |
| **avg** | **0.2248** | **0.2196** | **−0.005** |

The zero math/bbh rows are an eval-format artifact of this model family, not
model damage: the *base* model already emits immediate-EOS continuations under
the `::tulu` 8-shot CoT chat format (median 1 output token on gsm8k), and both
trained models behave the same terse way. The pair also agrees on output-shape
distributions (gsm8k median output 4 vs 4 tokens; fraction ≤10 tokens 0.81 vs
0.79) — indistinguishable even in the degenerate regime.

### GRPO — matched rollout budget, plus the measured noise floor

To answer "is a 2–3 point gap real?", the DeepSpeed run was repeated at seeds
2 and 3 (identical config otherwise):

| task | DS seed 1 | DS seed 2 | DS seed 3 | seed min–max | OLMo-core |
|---|---|---|---|---|---|
| bbh:cot-v1 | 0.2560 | 0.2720 | 0.2480 | 0.2480–0.2720 | 0.2560 ✓ |
| codex_humanevalplus | 0.5756 | 0.5711 | 0.5561 | 0.5561–0.5756 | 0.5488 (−0.007) |
| gsm8k | 0.1895 | 0.1243 | 0.1744 | 0.1243–0.1895 | 0.1615 ✓ |
| ifeval | 0.1978 | 0.1774 | 0.2015 | 0.1774–0.2015 | 0.2015 ✓ |
| minerva_math | 0.1136 | 0.1136 | 0.1007 | 0.1007–0.1136 | 0.1081 ✓ |
| popqa | 0.1693 | 0.1695 | 0.1678 | 0.1678–0.1695 | 0.1684 ✓ |
| **avg** | **0.2503** | **0.2380** | **0.2414** | 0.2380–0.2503 | **0.2407 ✓** |

The OLMo-core average lands inside the DeepSpeed seed spread, as do five of
six tasks (codex is 0.7 points below the seed minimum — small next to gsm8k's
6.5-point seed swing). **Changing the backend moves scores less than changing
the seed.**

### SFT — epoch-matched

The 150-step SFT pair is not matched-data (packing means OC saw ~8× the
tokens; it served only as a health check of training + the new HF export
path). For the real test, both backends traversed the identical 60k examples
exactly once (`--num_epochs 1`, no step cap; ~1875 padded DS steps vs ~224
packed OC steps — different step counts, same data):

| task | DeepSpeed ep1 | OLMo-core ep1 | Δ |
|---|---|---|---|
| bbh:cot-v1 | 0.0000 | 0.0000 | 0.000 |
| codex_humanevalplus | 0.1426 | 0.1328 | −0.010 |
| gsm8k | 0.0167 | 0.0227 | +0.006 |
| ifeval* | 0.3771 | 0.3789 | +0.002 |
| minerva_math | 0.0000 | 0.0000 | 0.000 |
| popqa | 0.2000 | 0.2124 | +0.012 |
| **avg** | **0.1227** | **0.1245** | **+0.002** |

Average gap 0.2 points, max task gap 1.2 — the tightest pair of the campaign,
well inside the seed noise floor. (*ifeval ran at eval max-length 4096 on both
sides; see the notes below.)

### Eval notes and gotchas

- **Only within-pair columns are read strictly.** Training stamps its chat
  template into the export while base models eval with their stock template,
  so trained-vs-base deltas mix training and template effects.
- **Eval `--max-length` must respect the model's position limit.**
  OLMo-2-1124-7B has `max_position_embeddings: 4096`; evaluating at 8192
  intermittently kills vLLM with a device-side index assert once a generation
  passes position 4096 (reliably on ifeval, occasionally on codex). It looks
  like a transient CUDA failure but isn't.
- **transformers version skew breaks eval on fresh exports.** The training
  image's newer transformers writes `tokenizer_class: TokenizersBackend` and
  list-typed `extra_special_tokens`; both crash oe-eval's older transformers
  at model load. Exports were patched in place (class →
  `PreTrainedTokenizerFast`; list key removed).
- The oe-eval datalake was unreachable from Beaker compute nodes throughout;
  metrics were harvested from each experiment's Beaker result dataset
  (`metrics-all.jsonl`).

## Defects found along the way (each tracked separately)

- 4× GRPO metrics inflation in `grpo_fast.py` (see warning above).
- OLMo-core SFT wandb runs sync ~0 history rows; `PerfCallback` silently
  no-ops on the SFT path.
- `olmo3_hybrid_7B` TransformerConfig was lost in the #1723 olmo-core pin bump.
- Stale `--gradient_checkpointing` flag crashes committed DPO launch scripts.
- The OC SFT dataset-cache hash is environment-dependent (tokenizer revision).
