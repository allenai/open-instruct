# Background evaluation implementation checks — September 20, 2026

This is a **prequalification record**, not a completed GPU/W&B qualification.
Existing training experiments were not changed.

## Artifacts

- Implementation branch: `robertb/miles-background-evaluation`, isolated from
  unrelated in-progress router changes in the development checkout.
- Evaluator Beaker image: `01M31BJJXPFWNZM3J1BE0P8CPB`
  (`robertb/miles-evaluator-55461d9-20260920`).
- Evaluator Docker ID:
  `sha256:1272092fd57730d634961c2593a1ed095bc809374274eb75406095d2a08bd999`.
- olmo-eval revision: `55461d9bf09c6ff7027d7a977f4ae45b8a07bbcf`.
- Evaluator serving foundation: MILES image source `ff9570b31497`, with its
  qualified Olmo SGLang extension. This foundation does **not** by itself qualify
  the new evaluator integration.
- Prepared ignored configuration: `runs/background-evaluation-20260920/small.toml`,
  copied from `configs/miles/examples/small.toml` with a fresh output path.
- Tiny HF fixture:
  `/weka/oe-training-default/robertb/open-instruct/runs/standard-examples-20260918/tiny/hf`.
- Planned training: four optimizer updates, one trainer GPU and one rollout GPU,
  barrier publication, rollout capture retained, no native checkpoints or final
  HF export. Online W&B is explicitly selected to qualify publishing.
- Planned evaluation: one-GPU GSM8K jobs at updates 2 and 4, two examples and 32
  generated tokens; an independent invalid-task job at update 4 is the deliberate
  evaluator failure case. Periodic/final evaluation at update 4 is deduplicated.
- Planned dashboard: main run in `ai2-llm/olmo-rl-comparison`, group
  `background-eval-qualification-20260920`.

## Completed checks

- 119 focused CPU tests passed across evaluation, configuration, launch,
  preparation/resume, checkpoint retention and documentation.
- Ruff checks and formatting passed for all MILES modules and modified tests.
- Focused type checking passed for evaluation/submission/publishing, run specs,
  workflow and launch modules.
- Generated documentation check and MkDocs build passed. The documentation build
  reports existing unrelated missing-link warnings.
- The evaluator image built successfully and its actual olmo-eval CLI accepted
  the provider, frozen model path, generation settings and task overrides in a
  dry run.
- The training image built from the committed isolated checkout, including the
  pinned Beaker submission SDK.
- Qualification TOML `plan` and `validate` passed.

## Remaining evidence

No live experiment has been submitted, no MoE export has been served through
this new evaluator, and no new evaluation curve has been observed in W&B.
The GPU-dependent lifecycle suite cannot import on the local CPU-only Docker
host (`libcuda.so.1` is unavailable); the new CPU driver test uses actor stand-ins
and verifies evaluator bypass and return while the submitter remains blocked.

Live qualification is awaiting an existing accessible Beaker-token secret or
explicit authorization to create one for the new training allocation. W&B uses
the main run's existing `ROBERTB_WANDB_API_KEY` reference in `ai2/robertb`.
An automatic approval review rejected creating a persistent submitter credential
without explicit authorization. No credential was created or copied.

After authorization, use the committed-image wrapper, record the training and
external evaluator experiment IDs, inspect the latest job attempts and scheduler
events when queued, and verify the successful results in the main run's history
with `eval/checkpoint_update` axes. Confirm that training completes despite the
invalid-task evaluator's nonzero exit. Until then, treat this image as an
unqualified candidate, not a recommended runtime default.
