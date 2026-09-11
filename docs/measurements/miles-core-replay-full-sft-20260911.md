# Full-SFT Core rollout router replay qualification

The earlier tiny-model live replay test passed three optimizer updates across a
restart, but all rewards and policy advantages were zero. The full-model GSM8K
learning comparisons ran with replay disabled. Neither qualifies full-model replay.

This trial uses the frozen full-SFT GSM8K campaign, two B300 Core EP ranks and one
TP1 SGLang engine, eight synchronous updates, 16 prompts × 4 generations per
update, and the qualified 64-way admission/cache/graph settings. Activation
checkpointing stays enabled. Auxiliary coefficients stay at 0.01 and 1e-5.

Enable replay with `miles.use_rollout_routing_replay=true` and
`miles.use_miles_router=true`. The opt-in `core.replay_diagnostics=true` adds hooks
that compare supplied IDs with the actual router outputs, and counts router calls
in scoring, training and backward recomputation. It has synchronization overhead
and is not enabled in normal starter profiles.

Acceptance requires all eight updates and nine publications; independent token,
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

Status: prepared; full-model results pending.
