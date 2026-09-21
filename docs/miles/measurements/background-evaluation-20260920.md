# Background olmo-eval qualification — September 21, 2026

This is a tiny Olmo MoE mechanics qualification, not model-quality evidence or a
qualification of other architectures, tasks, tensor-parallel sizes or clusters.
Existing experiments and maintained example configurations were not changed.

## Consistent chart schema follow-up

The status-only reopening behavior was accepted by the user. Publication is
now enabled, and every MILES writer in background mode declares the same ten
metric definitions as the evaluator. This preserves native training and rollout
axes while assigning all `eval/*` scores to `eval/checkpoint_update`.

- Application source: `e9fde68d2`; trainer image `01M32D2NFAGYNWQK1MSFYNN1BW`.
- Evaluator image/revision remain `01M31EGDWC2D57T1ZC2S19241V` and
  `55461d9bf09c6ff7027d7a977f4ae45b8a07bbcf`.
- [Live W&B writer contract](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/z7h715td):
  one evaluator published while the primary was active, then another published
  late. Training rows, both evaluation points, public configuration and all ten
  metric definitions were retained. Different task groups use the common
  `eval/*` wildcard, so their axes do not depend on which writer finishes last.
- The built trainer image passed an import/contract check: background writers
  declare ten consistent definitions; default shared-engine runs retain their
  original nine definitions.
- [Final MoE run](https://beaker.org/ex/01M32D37XG76QMZRPTFJXV67XT),
  [W&B](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/7596x9ru):
  exit 0, four optimizer updates completed. Ignored config `small-v5.toml` uses
  the same fixture below, with a fresh output root ending `-v5`. Training exited
  at 16:42:30 UTC, before the evaluator started at 16:42:53.
- [Final update-2 evaluator](https://beaker.org/ex/01M32DHERDHSJ6Z4DW6KRVM2N8):
  exit 0, two predictions retained, GSM8K exact-match accuracy 0.0. It published
  after training finished. W&B history grew from eight to nine rows; all four
  training steps, all ten metric definitions, the name/group and all 978 public
  configuration fields were preserved. The score uses the checkpoint-update
  axis. The dashboard status changed to `running`, as accepted above.
- [Final deliberate failure](https://beaker.org/ex/01M32DHNHV2B5A3QJSB176S1NQ):
  exit 1 for the invalid task; diagnostics and the update-4 snapshot retained.
- The final GSM8K submission received `BeakerExperimentConflict`. Its receipt
  records `submission_failed` and the diagnostic. It was not automatically
  retried, and the dashboard has no update-4 point for this run. This is an
  expected best-effort gap; the earlier run below exercised both milestones.

The final implementation passes the bounded mechanics qualification with the
accepted W&B status semantics. The evaluator does not finish the training run,
wait on training, restore lifecycle state, retry missed points or delete snapshots.

## Initial GPU qualification artifacts

- Implementation branch: `robertb/miles-background-evaluation`, isolated from
  unrelated development changes; live application source `16fcaf63d`.
- Trainer image: `01M31EJVZXFJDJXKZJESJT159D`.
- Evaluator image: `01M31EGDWC2D57T1ZC2S19241V`
  (`robertb/miles-evaluator-55461d9-v2-20260921`).
- Publisher W&B SDK: `0.30.0`.
- Evaluator Docker ID:
  `sha256:06039420309baa2e4bc0e9769b58c3ac3312f8ea861e8b1775836b2815225992`.
- olmo-eval revision: `55461d9bf09c6ff7027d7a977f4ae45b8a07bbcf`.
- Serving foundation: MILES source `ff9570b31497`, including the Olmo MoE
  SGLang extension. olmo-eval's `vllm_server` client connects to a separate
  SGLang process; the evaluator does not launch vLLM.
- Ignored run configuration: `runs/background-evaluation-20260921/small-v4.toml`,
  copied from `configs/miles/examples/small.toml`.
- Initial HF fixture:
  `/weka/oe-training-default/robertb/open-instruct/runs/standard-examples-20260918/tiny/hf`.
- Output root:
  `/weka/oe-training-default/robertb/open-instruct/runs/background-eval-qualification-20260921-v4`.

Training used four optimizer updates, one trainer GPU and one rollout GPU on
`ai2/holmes`, barrier publication, online W&B and rollout capture. Native
checkpoint saving and final HF export were disabled. Background export still
captured immutable snapshots at updates 2 and 4, under `eval-snapshots/`.

Each successful evaluation requests two GSM8K examples, zero few-shot examples,
32 generated tokens, temperature zero and a 512-token serving context matching
this tiny model. An independent invalid task at update 4 uses different generation
settings to form a separate failure group. Coincident periodic/final GSM8K at
update 4 forms one submission.

## Runs and observed behavior

1. [Training](https://beaker.org/ex/01M31ETMQWRSTZAZGZ4AW2KWTX): exit 0, all four
   optimizer updates completed. The update-2 snapshot completed at 07:53:49 UTC;
   update 3 completed at 07:53:52 and the update-4 snapshot at 07:53:56.
   Training exited at 07:54:09, before any evaluator started (07:54:38–39).
2. [Update-2 GSM8K](https://beaker.org/ex/01M31F9YPHK965HQCQ7H0ZFNPA): exit 0; two examples saved, exact-match accuracy 0.0.
3. [Update-4 GSM8K](https://beaker.org/ex/01M31FA5SFMK7PN528D8XDE08Y): exit 0; two examples saved, exact-match accuracy 0.0.
4. [Deliberately invalid task](https://beaker.org/ex/01M31FA6WJPAS0GE6RGZP047CZ): exit 1; unknown task diagnostic retained, no evaluation score published.

All three receipts recorded distinct Beaker experiment IDs. There was exactly
one GSM8K submission at each milestone and one shared snapshot at update 4.
The main [W&B run](https://wandb.ai/ai2-llm/olmo-rl-comparison/runs/c83p1rp3)
was already `finished` with eight training history rows before evaluation.
Both points were published late, in arrival order update 4 then update 2. The
history grew from eight to ten rows; all four training-update rows remained.
The exact score metric uses `eval/checkpoint_update` as its chart axis. Public
training configuration (978 fields), name and group were unchanged.

The first publisher correctly uploaded scores, but late shared-mode attachment
reopened the finished run as `running`, despite `x_primary=False` and
`x_update_finish_state=False`. The user subsequently accepted this dashboard
status change: it does not restart or interfere with training. Evaluators still
do not mark the main run finished on exit or rewrite its public configuration.

The first publisher also replaced the nine native metric definitions with its
two evaluation definitions. This was a separate chart issue. The implementation
now uses one common metric schema in every background-mode MILES writer and
evaluator, retaining training/rollout axes and using `eval/checkpoint_update`
for the `eval/*` wildcard. The initial publication block has been removed.
The consistent-schema follow-up is recorded above.

Credentials were inherited from the training allocation in
`ai2/open-instruct-dev`: `robertb_BEAKER_TOKEN` for submission and
`robertb_WANDB_API_KEY` for tracking. `robertb_HF_TOKEN` also exists there but was
not needed for this local model/public task. Only secret names were inspected;
no credentials were created, copied into configuration, or placed in receipts.

## Validation and integration fixes

- 116 focused CPU tests passed across evaluation, configuration, launch,
  workflow/resume, checkpoint retention and documentation.
- Ruff check/format and focused type checks passed. MkDocs built successfully
  with existing unrelated link/anchor warnings.
- A real two-example olmo-eval mock-provider run passed inside the evaluator
  image; it wrote predictions, requests and metrics that the results publisher
  parsed successfully. This exercised more than CLI argument parsing.
- Run `plan` and `validate` passed, and the actual launches used the clean,
  committed MILES image wrapper. No local source overlay was injected into jobs.
- GPU-dependent lifecycle tests cannot import on the local CPU-only Docker host
  because `libcuda.so.1` is unavailable. The CPU driver test uses actor stand-ins
  to prove shared evaluation is bypassed and training returns while submission
  remains blocked; the live experiments supply the GPU lifecycle evidence.

Earlier fresh qualification attempts exposed and corrected integration issues:

- [First trainer](https://beaker.org/ex/01M31CW7EM1JNFZYA9YWGFC15R) completed
  despite evaluator failures. Beaker SDK 2.5.7 returns the experiment ID through
  `workload.experiment.id`; the original caller used the wrong attribute after
  successful API acceptance. The fix has a regression test. Evaluators also
  required `trust_remote_code` for the exported model configuration.
- [Second trainer](https://beaker.org/ex/01M31DX3GTXQVXK57W52XXN7XZ) recorded
  submission IDs correctly and completed despite evaluator startup failures.
- [Third trainer](https://beaker.org/ex/01M31EKCP62JXWFHXRYQVXNKR2) used the
  corrected images but retained the inappropriate 4,096-token context setting.
  The final fixture uses its actual 512-token context and zero-shot GSM8K.
- Actual provider import/execution checks caught an optional Beaker dependency
  required by olmo-eval's provider and its flat sampling-override API. Both were
  corrected before the final attempt. A CLI dry run alone did not detect them.

Failed attempts and snapshots remain available for manual diagnosis and cleanup.
There are no automatic retries or automatic snapshot deletion. Inspect every
referencing job before removing snapshots; never remove them while a job is
pending or running.
