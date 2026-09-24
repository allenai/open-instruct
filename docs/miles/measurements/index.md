# MILES measurements

These records preserve their original experiment or proposal scope. They are not
current operating defaults; start with the [MILES guide](../index.md).

| Record | Topic |
|---|---|
| [Hero long-run startup investigation](hero-long-startup-20260924.md) | Publication timeout, partial Python-module cache reproduction and fix, evaluator execution fixes; long-run qualification pending |
| [Hero SFT end-to-end MILES smoke](hero-rl-smoke-20260924.md) | Paired H100 four-update barrier checks, automatic fused rounding, nonzero gradients and exact publication |
| [Multi-node automatic resume](multinode-resume-20260922.md) | Two replicas interrupted and restarted into the same run directory; rollout cursor continues, no update repeated or skipped |
| [Lessons from retired integration branches](retired-branch-lessons-20260921.md) | Checkpoint lifecycle, architecture preservation, packing/replay contracts, evaluation discipline, and local archive provenance |
| [Router investigation: findings worth keeping](router-findings-20260921.md) | Completed aux-off sweep, routing dose response, causal padding correction, rejected hypotheses, and archived research provenance |
| [Exploratory gradient noise scale](critical-batch-20260921.md) | Noise-dominated retained-group gradients; corrected uncertainty limits and implications for batch-size experiments |
| [Online group filtering: small qualification](online-filtering-20260919.md) | Four-update barrier and refresh checks; all-zero/all-one rejection, batch replenishment and checkpoint-cursor audit |
| [Router controls and standard-example qualification](router-controls-20260918.md) | September 18 image identity; completed dev/small checks and remaining qualification scope |
| [Dense GSM8K full test: Core versus original after 200 updates](gsm8k-dense-test-20260916.md) | Same-checkpoint framework comparison on 1,319 questions; original +48, Core −16, paired p=6e-7 |
| [Reward by response length on the basket](length-reward-20260916.md) | Where reward is earned by length per domain and model; cost of the 32K cap |
| [Overfit diagnostic: Core and Megatron on 16 fixed GSM8K prompts](overfit-20260915.md) | Fully async update-correctness check on both arms, plus Core null, sign-flip, learning-rate and re-score controls |
| [Faster trainer and 200-update task baseline](full-sft-basket-20260914.md) | EP2 qualification, fixed multi-domain data and EP8 baseline |
| [Throughput and queue qualification](throughput-20260913.md) | September 13 steady-state configuration comparisons, occupancy figures and limitations |
| [Throughput campaign log](throughput-campaign-20260913.md) | Chronological controls, runtime repairs and allocation decisions |
| [Sharing consolidation and documentation audit](sharing-20260913/README.md) | Branch decisions, complete document inventory and candidate validation |
| [Length guidance exercise](length-guidance-20260913/README.md) | Measured 16K/32K/64K serving capacity and audited natural long-response Core RL probes |
| [Colleague readiness qualification](colleague-20260913/README.md) | Dense resume/export/reload, mixed services, long contexts, replay, and remaining gates |
| [Fixed FLA tuner choices eliminate the measured prefill divergence](autotune-pinned-controls-20260911.md) | autotune-pinned-controls-20260911 |
| [Compiler-cache publication: local proxy, 2026-09-12](cache-publication-20260912/README.md) | README |
| [Native checkpoint performance qualification](checkpoint-perf-20260911.md) | checkpoint-perf-20260911 |
| [Core scoring, scheduling and configuration exercise](control-exercise-20260911.md) | control-exercise-20260911 |
| [Actual Core cold/restored compiler-cache qualification](core-cache-trial.md) | core-cache-trial |
| [Core500 update-100 save: planning latency observation](core-checkpoint-planning-20260911.md) | core-checkpoint-planning-20260911 |
| [Core native scorer versus serving, full SFT checkpoint at update zero](core-native-routes-20260911.md) | core-native-routes-20260911 |
| [Full-SFT Core rollout router replay qualification](core-replay-full-sft-20260911.md) | core-replay-full-sft-20260911 |
| [Opt-in runtime-row SwiGLU integration](core-row-specialization-20260911.md) | core-row-specialization-20260911 |
| [Frozen Core100 scorer profile](core-score-profile-20260911.md) | core-score-profile-20260911 |
| [Successive-batch Core scorer qualification](core-score-variants-20260911.md) | core-score-variants-20260911 |
| [MILES/Core feature parity audit — September 11, 2026](feature-parity-audit-20260911.md) | feature-parity-audit-20260911 |
| [Frozen100 GSM8K configuration and timing audit](gsm8k-configuration-differences-20260911.md) | gsm8k-configuration-differences-20260911 |
| [Paired GSM8K generations and routing investigation](gsm8k-generation-behavior-20260911.md) | gsm8k-generation-behavior-20260911 |
| [GSM8K: completed 100-update comparison](gsm8k-results-20260911.md) | gsm8k-results-20260911 |
| [Router, response length, and serving reproducibility investigation](gsm8k-router-investigation-20260911.md) | gsm8k-router-investigation-20260911 |
| [olmo-miles field inventory](knob-inventory.md) | knob-inventory |
| [Learning comparisons and performance analysis](learning-comparisons-20260911.md) | learning-comparisons-20260911 |
| [Light SFT1000 GSM8K comparison](light-sft1000-gsm8k.md) | light-sft1000-gsm8k |
| [Mixed-source qualification](mixture-qualification.md) | mixture-qualification |
| [Tiny multi-node mixed-task and named-judge exercise](multinode-judges-20260912.md) | multinode-judges-20260912 |
| [MILES/Core project working branches](project-state.md) | project-state |
| [Publication transport profile (phase 0)](publication-profile-20260912.md) | publication-profile-20260912 |
| [Researcher workflow: 8 × 8, async TIS, September 11](researcher-workflow-20260911.md) | researcher-workflow-20260911 |
| [Original GSM8K 100 response-length audit](response-lengths-100-20260911.md) | response-lengths-100-20260911 |
| [Response length conditioned on correctness and cap status](response-lengths-conditional-20260911.md) | response-lengths-conditional-20260911 |
| [Native auxiliary and policy router gradients: tiny EP1 qualification](router-gradient-decomposition-20260911.md) | router-gradient-decomposition-20260911 |
| [Router precision in the completed and current GSM8K runs](router-precision-20260911.md) | router-precision-20260911 |
| [Scoring-pass merge review](scoring-pass-merge-20260912.md) | scoring-pass-merge-20260912 |
| [Sequence packing qualification, 2026-09-12](sequence-packing-20260912/README.md) | README |
| [Two-node async GSM8K: first EP8 run at 64 × 8](two-node-async-gsm8k-20260912.md) | two-node-async-gsm8k-20260912 |

* [Two-engine concurrency-32 follow-up](throughput-c32-20260913.md): same useful throughput with fewer serving GPUs.
