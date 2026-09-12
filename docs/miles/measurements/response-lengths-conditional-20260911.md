# Response length conditioned on correctness and cap status

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

[The paired question data](response-lengths-conditional-20260911.json) extend the [original 100 audit](response-lengths-100-20260911.md). No additional rollout generation or GPU computation was performed.

Megatron's held-out shortening persists among questions answered correctly without reaching the cap at both endpoints:

| Interval | Core stable-correct uncapped questions | Core mean tokens before → after | Megatron stable-correct uncapped questions | Megatron mean tokens before → after |
|---|---:|---:|---:|---:|
|60→80|86|1168→1064|82|1210→738|
|80→100|81|1013→955|89|737→682|
|60→100|84|1115→989|85|1240→688|

Each comparison pairs the same question within its backend. The selected question subsets differ across backends and intervals. This demonstrates shortening within a stable-correct subset; it does not establish that shortening caused improvement, or compare identical subsets across backends.

The 60→100 accuracy changes also have different question transitions: Core has 16 correct→wrong and 7 wrong→correct; Megatron has 8 correct→wrong and 16 wrong→correct. Newly wrong answers tend to become much longer (Core 2269→3765 mean tokens; Megatron 1881→2559), while newly correct answers shorten (Core 3518→972; Megatron 3111→1331). Cap and correctness transitions overlap and are not independent explanations.

Removing capped training answers leaves the late 80–99 mean lengths nearly equal: Core 1205 versus Megatron 1197. Restricting further to correct uncapped training answers yields 1176 versus 1181. Therefore the late held-out gap is not accompanied by a comparable mean-length gap in these conditioned training subsets.

The JSON also contains next-rollout associations after all-uniform reward groups versus any mixed group. In both backends, the next rollout after an all-uniform batch is longer and less rewarded on average. These are different questions and fresh stochastic samples, and selecting a uniform batch changes the reward distribution. They must not be interpreted as a causal optimizer effect or evidence that auxiliary updates hurt learning. The last rollout is excluded because no next training rollout exists, leaving 21 such transitions in Core and 28 in Megatron.

Reproduce from the validated response-length JSON with `python -m scripts.miles.response_length_transitions report.json > conditional.json`. Nine focused tests cover the collector corruption checks and paired-question/cap filtering.
