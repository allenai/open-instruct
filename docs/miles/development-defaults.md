# Development defaults

This page is light guidance from developing the MILES GRPO path. Each entry gives
a starting point and what to weigh when changing it. Unless an entry says
otherwise, a default is **not established as optimal, or even good**, for any
particular model, scale or task; it is simply where to begin. Where the
maintained example TOMLs encode a choice, the example is authoritative.

## Objective and data

- **Policy lag** (`async.max_weight_staleness = 6`): a higher limit discards fewer slow groups but trains on older samples; reconsider it when learning rate, group size or response length changes.
- **Off-policy correction** (TIS on, `use_rollout_logprobs = false`): TIS reweights samples toward the current policy; turning it off assumes inference and training probabilities agree closely.
- **Group filtering** (`training.filter_zero_std_groups = true`): groups where every response gets the same reward carry no policy gradient, so dropping them saves training work at the cost of extra generation; disable it for tiny checks that cannot produce mixed rewards.
- **Router auxiliary losses** (coefficients zero in the examples): the balancing and z-losses keep expert load even during pretraining; whether RL needs them is open, so watch the `train/moe/` load metrics if you leave them off.
- **Truncated responses** (scored normally): excluding them avoids rewarding unfinished reasoning but removes long responses from the batch; worth considering when many responses hit the length cap.

## Publication and capacity

- **Publication mode** (`refresh` for full-model async runs, `barrier` for small synchronous checks): refresh keeps generating across weight updates, so long responses span several policy versions; barrier is simpler to reason about but stalls generation at each update.
- **Engine admission** (generic default 64; matching B300 learning profiles propose 512): size it from GPU memory, with `sglang_max_total_tokens = admission × context`; higher values add throughput only while memory and the router keep up. See the [example evidence and limits](../../configs/miles/examples/README.md#capacity-and-qualification-settings) before transferring 512 to another model or GPU.
- **HTTP admission** (generic async default twice running admission; matched B300 profile 512/512): permits cover transit and response processing as well as generation. Additional permits can refill GPU slots or just add queueing; measure occupancy and residence time before increasing them.
- **Producer budget** (automatic when `async_max_concurrent_samples` is unset): the automatic value scales with the engine fleet; a larger budget fills engines more reliably but leaves more work to age out. The learning examples omit this setting so it follows the resolved profile and fleet size; matched 512-slot node/large profiles give 6,144/19,456 samples. These larger budgets require policy-age and useful-throughput validation with the unchanged batch and lag limits.
- **Completed buffer** (one collection): separate from unfinished-request headroom; enlarging it retains more completed work without fixing response transport.
- **Submission granularity** (`"sample"`): engines refill as individual responses finish rather than waiting for whole groups.
- **Activation recomputation** (on for long contexts): trades extra compute for memory; turning it off is reasonable for short contexts once memory fit is checked.
- **Radix caching** (automatic for matched profiles: on for B300, off for H100; off for generic/colocated setups): the realistic-prompt comparison favored cached 512 on B300 and uncached 512 on H100. Explicit `inference.radix_cache` wins. Cached hybrid serving needs the pinned FP32 tracked-state implementation; rebuild images after changing runtime pins. Recheck this choice for different prompt reuse and lengths.
- **Compiler cache** (on): shortens repeat startup and does not change training.
- **Weight equality audit** (off in learning templates, on in mechanics templates): startup snapshot/reset qualification is separate from the required publication protocol.
- **Trainer scoring** (`scoring_pass_required = false`): reuse the training forward only when the recipe permits it. Learning templates set `scoring_check_interval = 0` to retain startup/resume validation without periodic extra scoring forwards; TIS and required scoring remain active.
- **Native checkpoints** (every 25 updates in learning templates): synchronous saves interrupt training. Adjust using measured save time and acceptable lost work after preemption; keep recovery enabled.

## Opt-in features

- **Expert-aware packing** (off): may reduce MoE load imbalance, but planning cost can cancel the savings; measure end-to-end update time before adopting it.
- **FP32 output head** (off): reduces probability mismatch between training and serving at some memory cost; its effect on learning is unknown.
- **Engine drain** (off): an experimental alternative to refresh that finishes requests before swapping weights; no example uses it.

## Measuring

- **Timing:** early updates include compilation and warmup, so exclude them and run long enough for update times to settle before comparing throughput.
- **Wasted work:** report dropped response tokens as well as dropped samples, since a few discarded long responses can cost more than many short ones.
- **Learning comparisons:** compare at matched wall time or generated-token budget, and evaluate on the full held-out set; small subsets are noisy enough to reverse a comparison.
- **Run length:** Beaker protects at most eight hours of minimum runtime, so longer runs should save checkpoints and enable auto-resume.
