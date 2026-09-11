# Researcher workflow: 8 × 8, async TIS, September 11

The public configuration-based launch completed successfully, and an independent
CPU audit verified every retained training/evaluation sample and the trainer
counters. Both Beaker jobs exited 0.

- Training: [01M29AQXT4NDWG5W5BPGZSTPG6](https://beaker.org/ex/01M29AQXT4NDWG5W5BPGZSTPG6)
- Read-only Saturn audit: [01M29CBC996BG4THEF32YHF3NB](https://beaker.org/ex/01M29CBC996BG4THEF32YHF3NB)
- [Exact run file](../../configs/miles/qualification/workflow-async-gsm8k.toml)
- [Machine-readable evidence](miles-researcher-workflow-20260911.json)

Training source was `7aa5a3d51`; auditor source was `39014ae4f`.
The immutable runtime image was `01M29AQG8QTBJW02V7MR4N5R79`.
The run used the existing full-SFT HF checkpoint from the frozen September 10
GSM8K campaign. No Megatron conversion or trainer was involved.

## Configuration and execution

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  python -m open_instruct.miles run configs/miles/qualification/workflow-async-gsm8k.toml
python -m open_instruct.miles status configs/miles/qualification/workflow-async-gsm8k.toml
```

The command froze the resolved configuration, built the committed source image
through `build_image_and_launch.sh --miles`, and launched on urgent Holmes in
`ai2/open-instruct-dev`, with a one-hour minimum runtime. Two B300s trained with
Core EP2; one dedicated B300 served through SGLang TP1. Client/engine admission
was 64, with decode graphs through 64, recurrent cache 128 and KV token pool
524288. Checkpoint saves and final HF export were disabled for this exercise.

Fresh preparation selected 32 GSM8K training questions and 16 disjoint held-out
questions from the pinned source. The workflow retained source/question IDs,
template/token hashes, resolved arguments, reports and offline W&B data under
`/weka/oe-training-default/robertb/open-instruct/workflow-exercise/workflow-async-gsm8k`.
These are held-out entries for this run, not the previous 128-question benchmark.

## Verified behavior

- Four optimizer updates on both ranks, 64 samples/update, eight groups of eight.
- 256 consumed responses and 534,413 response tokens; 32 distinct consumed group
  IDs covering 30 unique prepared questions. Async completion order and dataset
  wrap allow a question to recur in a new group.
- Finite, nonzero dense/expert/router gradients and sampled parameter updates;
  no skipped optimizer steps; learning rate `1e-6` throughout.
- Maximum policy age one optimizer update. Consumed versions were `[0]`, `[0]`,
  `[1,2]`, `[2,3]` at trainer versions 0, 1, 2, 3 respectively.
- TIS enabled with trainer-scored old log probabilities. Buffer factor two,
  retry policy, and group-level submission were used.
- Nine publication operations: initial version 0, versions 1–4, and four same-
  version diagnostic re-publications. Each transferred 29,669 tensors in 35 buckets.
- Every retained prompt/token identity, reward and train/eval membership passed
  the independent audit, including direct GSM8K rescoring.
- Initial and final held-out results both **12/16 (75%)**. Four updates on 16
  held-out questions do not establish a learning effect.

## Timing and clipping observations

Initial/final blocking evaluation took **31.8 / 51.6 seconds**. This does not
qualify the separate 128-question evaluation target. The first Core scoring pass
took 147.9 seconds and the first full training call 423.4 seconds. Later scoring
passes took 5.3–6.1 seconds; forward/backward/optimizer work took 22.8–25.2 seconds.
Per-update publication stages took 42.8–48.1 seconds because they included full
snapshot/reset/equality checks; ordinary tensor transfer took roughly four
seconds. Startup, cold compilation and these diagnostics prevent interpreting
the four-update cycle mean as steady training throughput.

PPO clipping remained zero. TIS clipping was zero on updates 1–2, then
`2.111e-5` and `2.053e-5` on updates 3–4. These are response-averaged clip
fractions, not direct counts of clipped tokens. The native TIS bounds were
`[0,2]`, applied to `exp(Core_score - serving_logprob)`, so upper clipping
requires a positive log-probability difference above `ln(2) ≈ 0.693`.

Two effects can contribute: trainer/serving numerical differences and the
allowed one-update policy lag. Before any optimizer update, mean absolute
scoring discrepancy was 0.0187 and maximum absolute discrepancy 0.857; the
absolute maximum does not establish the sign or upper clipping. The clipped
updates mixed current and preceding policy versions. Current reports do not
identify which exact tokens clipped or whether they came from stale groups;
causal attribution remains unmeasured.

## Scope

This qualifies the structured HF-input → named GSM8K preparation → config-based
Beaker launch → async TIS/8 × 8 training → shared-engine evaluation → clean exit
path on the stated topology. It does not qualify native conversion, final export,
checkpoint/restart, fault recovery, replay-plus-async, multiple nodes or learning
parity. Those capabilities keep their existing separate evidence and limits.

Local validation: 155 focused CPU tests, 35 runtime lifecycle tests, Ruff,
compilation and type checks passed. The new workflow uses the existing MILES
and Core dependency pins; this task did not modify either dependency branch.
