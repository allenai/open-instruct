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
- **Engine admission** (64 requests per engine): size it from GPU memory, with `sglang_max_total_tokens = admission × context`; higher values add throughput only while memory and the router keep up.
- **Producer budget** (`async_max_concurrent_samples` unset): the automatic value scales with the engine fleet; a larger budget fills engines more reliably but leaves more work to age out.
- **Submission granularity** (`"sample"`): engines refill as individual responses finish rather than waiting for whole groups.
- **Activation recomputation** (on for long contexts): trades extra compute for memory; turning it off is reasonable for short contexts once memory fit is checked.
- **Radix caching** (as in the examples): pays off when long prompt prefixes repeat; with short prompts and long responses it matters little.
- **Compiler cache** (on): shortens repeat startup and does not change training.

## Opt-in features

- **Expert-aware packing** (off): may reduce MoE load imbalance, but planning cost can cancel the savings; measure end-to-end update time before adopting it.
- **FP32 output head** (off): reduces probability mismatch between training and serving at some memory cost; its effect on learning is unknown.
- **Engine drain** (off): an experimental alternative to refresh that finishes requests before swapping weights; no example uses it.

## Measuring

- **Timing:** early updates include compilation and warmup, so exclude them and run long enough for update times to settle before comparing throughput.
- **Wasted work:** report dropped response tokens as well as dropped samples, since a few discarded long responses can cost more than many short ones.
- **Learning comparisons:** compare at matched wall time or generated-token budget, and evaluate on the full held-out set; small subsets are noisy enough to reverse a comparison.
- **Run length:** Beaker protects at most eight hours of minimum runtime, so longer runs should save checkpoints and enable auto-resume.
