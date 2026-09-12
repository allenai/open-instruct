# Full-SFT Core rollout router replay qualification

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Full-model router replay is qualified for synchronous Core EP2 with
TP=PP=CP=1. Eight updates consumed 512 samples in 128 prompt groups, including
38 mixed-reward groups. The independent retained-data audit passed.

- Training: [01M2931SC4WNX57GB3AXWKV03W](https://beaker.org/ex/01M2931SC4WNX57GB3AXWKV03W),
  source `44c6aff33`, image `01M2931KARFP3Y2W2FPADGRFEP`.
- Passing independent CPU audit on Saturn:
  [01M295HY6X4NMBGB8R0JW24EBQ](https://beaker.org/ex/01M295HY6X4NMBGB8R0JW24EBQ),
  exit 0; auditor image `01M295B3XHQCP02ZHTKNQ4YSR3` (source `0de110f51`),
  launcher source `8e0a709f3`.
- [Machine-readable measurements](core-replay-full-sft-20260911.json).

There were zero expert-ID mismatches across all 19 routed layers: 9,728 scoring
router calls and 19,456 training/recomputation router calls. Each training
microbatch returned routes twice per layer, including backward recomputation.
Captured routes covered 1,069,081 prediction-input tokens. Router gradient norms
were finite and nonzero at every update (0.01068–0.01400), with nonzero sampled
router parameter changes. Mean active-token train/serving log-probability gaps
were 0.00645–0.00758.

The GPU process completed training but exited 1 in its original post-run auditor,
which expected nine publications and overlooked the eight extra same-version
verification round trips. The corrected independent auditor checked all 17 in
order, plus retained tokens, rewards, sample accounting, versions, and routes.
This is completed training with a separately passing audit, not an exit-zero GPU
job. No held-out evaluation or checkpoint was requested; this establishes replay
mechanics, not learning benefit or unrestricted parallelism.

This trial uses the frozen full-SFT GSM8K campaign, two B300 Core EP ranks and one
TP1 SGLang engine, eight synchronous updates, 16 prompts × 4 generations per
update, and the qualified 64-way admission/cache/graph settings. Activation
checkpointing stays enabled. Auxiliary coefficients stay at 0.01 and 1e-5.

Enable replay with `miles.use_rollout_routing_replay=true` and
`miles.use_miles_router=true`. The opt-in `core.replay_diagnostics=true` adds hooks
that compare supplied IDs with the actual router outputs, and counts router calls
in scoring, training and backward recomputation. It has synchronization overhead
and is not enabled in normal starter profiles.

Acceptance requires all eight updates, nine distinct policy versions, and eight
additional same-version reset/republish verification round trips; independent token,
reward and policy-version audits; complete per-rank/per-sample/per-routed-layer
coverage; zero mismatched expert IDs; and observed returned routes during backward
recomputation. Gradient diagnostics are retained each update. Passing routing
checks does not establish a learning benefit or match Megatron's auxiliary loss.

Routes cover `tokens-1` prediction inputs. The final unscored token receives the
existing deterministic expert assignment; its auxiliary contribution is not
claimed to match serving. Retained rollout dumps preserve captured route arrays.

Launch from a clean committed checkout:

```bash
MILES_BASE_IMAGE=olmo-miles:gate-01m24e7msdgn2qfw1t8z31bcks \
  ./scripts/train/build_image_and_launch.sh --miles \
  scripts/train/debug/miles_control_exercise.sh --replay-only
```

Placement: urgent, `ai2/holmes`, `ai2/open-instruct-dev`, minimum runtime one hour,
three GPUs, two-hour timeout. Results are under
`/weka/oe-training-default/robertb/open-instruct/control-exercise/$BEAKER_EXPERIMENT_ID/replay-admission64`.

The first attempt failed before its first optimizer update in the new diagnostic hook.
Captured IDs remained on CPU while returned router IDs were on GPU. The diagnostic
now compares both on one device without changing the replay inputs. A local CUDA
regression exercises CPU route payloads and GPU forward/backward, including
intentional route corruption. The final targeted suite passed 33 tests with no
skips; style, lint and type checks passed. The first CPU audit launch was canceled
because its shell wrapper omitted the replay-only argument; the passing retry
used the corrected wrapper without repeating GPU training.

- Experiment: [01M291KSAXN8P2M0QCE26NE1TZ](https://beaker.org/ex/01M291KSAXN8P2M0QCE26NE1TZ)
- Source: `9004a8cda`; immutable image: `01M291KK92RHGYEDN33N5F8BNR`.
- Core development revision pinned by the image: `48bb6d7e1554`.
- Submitted 2026-09-11 20:12:45 UTC; scheduled 20:13:05 UTC.
