# Active MILES / Core work

> Historical proposal. For current operating instructions, start at the [MILES guide](../index.md).

This is the continuation ledger for the September10–11 session. Multiple Beaker
jobs are authorized. Run GPU trials on Holmes in `ai2/open-instruct-dev` at
urgent priority with a positive minimum runtime. CPU-only WEKA jobs belong on
Saturn. Preserve every attempt and keep the old SFT architecture separate from
hero. Do not publish this branch to the public open-instruct repository without
separate authorization.

| Track | State | Next evidence needed |
| --- | --- | --- |
| Old SFT, Core,100 GSM8K updates | [Completed, exit0](https://beaker.org/ex/01M26P6XX6SN886DCVZ68WMQK2); heldout97/101/94/102/97/93 out of128 at0/20/40/60/80/100 | [Independent audit passed](../measurements/gsm8k-core-final-20260910.json):1,600training and768held-out responses;100steps/101publications. User notified. |
| Same SFT, olmo-miles/Megatron,100 updates | [r3 running](https://beaker.org/ex/01M26YNP5E64YGNXR85TA2RP4Q); same shared data and recipe; held-out96/93/98/99 out of128 at0/20/40/60;76 updates observed03:20UTC | Full curve; audit `megatron-r3`, then compare both curves, truncation and measured cadence/allocation time. |
| Hero native/HF conversion | [step75500 passed](https://beaker.org/ex/01M26YP78T0JJ545H914AEYDDZ); both directions exact after export cast, 23,441 tensors | [Recorded conversion evidence](../measurements/hero-conversion-20260910.json); 579 seconds, 72 GiB peak RSS. |
| Hero serving |85 local tests and tiny Core/HF/SGLang parity passed; live gain/scale updates passed | [Full-checkpoint TP1 scoring failed its 0.1 logprob gate](https://beaker.org/ex/01M26ZBDKPDBRJ03TYT6V2GYRJ). All eight greedy tokens match; Core max full-vocabulary error 0.5473, SGLang max top-20 error 0.5619. [Layerwise Core/HF diagnosis completed](https://beaker.org/ex/01M270S3NYZE11QW0H03NHQGP7); diagnose before training qualification. |
| Native reduction/clipping stress | [EP1/EP2 passed](https://beaker.org/ex/01M270A2J2E3WC60978EM8G5SV); independent global norm, active clipping and Adam moments verified | [Evidence](../measurements/core-ep-stress-20260910.json); max gradient relative L2 difference 5.002e-6; does not establish exact update parity near zero gradients. |
| Three-source mixture | [Two-update 8192-response trial passed](https://beaker.org/ex/01M270TQ57VM13CMYAP8AYAZA8); both updates and independent audit passed; 16 focused tests passed | [Evidence](../measurements/mixture-20260910.json): all three sources supplied mixed-reward groups; all 16 math training responses hit the cap. No learning claim. |
| Standard dense Olmo3 | Isolated standard trainer plus local full/sliding HF math and real update/resume passed | Full-checkpoint serving and a bounded training trial when prioritized; no claim of a full recipe run yet. |

The first Megatron attempt failed at initialization; r2 was stopped at the user's
request before training; the user then explicitly restored the paired comparison.
The active r3 recipe changes only run/output/W&B identities from r2. The repaired
router precision image is retained. Earlier failed/stopped attempts are not
learning curves or steady-state throughput measurements.

The integration lives in `.worktrees/miles-integration` on
`robertb/miles-olmo-core`. Core changes are isolated in `miles-core-adapter`;
serving changes are isolated in `miles-serving`. The original Core100 image and
source remain unchanged. `runtime/miles/runtime.lock.json` records the hero pins.

## Other open work retained

The detailed acceptance sequence remains in
[miles-qualification-plan.md](qualification-plan.md), and feature/default
coverage in [miles-feature-parity.md](../feature-parity.md):

- Mixed open-instruct datasource GPU qualification passed above.
  The prior full-SFT math slice was almost entirely capped at4096 tokens; the
  new8192-token limit is a measured extension, not an assumption that math
  responses will now be uncapped.
- Fixed-token Core/Megatron gradient and optimizer-state comparison, including
  auxiliary-loss semantics. A similar GSM8K curve cannot substitute for this.
- Full-model EP2 durable continuation and same-next-update comparison.
- Bounded async/sync comparison, followed by endurance, failure and restart checks.
- Resident colocation fit with matched engine counts and useful-work throughput.
- Fingerprinted compiler-cache lifecycle and cold-versus-warm measurements.
- Broader hero EP/global-balancing, replay and multi-node qualification. Tiny
  operator and single-GPU tests are not distributed training evidence.

The short100-update experiment is a descriptive screen, not a proof of learning
rate superiority or backend equivalence. Keep numerical correctness, task quality
and performance conclusions separate.

Held-out GSM8K evaluation uses temperature 0 and one response per question.
Training uses temperature 1. The observed held-out score changes are greedy
outputs on the same fixed set; they are not repeated stochastic eval samples.

The [full hero graph/chunk matrix](https://beaker.org/ex/01M27118R2ER9QG0PYP3ARCHBE)
completed with all eight greedy tokens matching in all four modes. Graphs on/off
produce exactly the same checked probabilities at each fixed chunk size. Changing
chunk size produces up to 0.5712 logprob difference; the original 0.1 probability
gate still fails. [Evidence](../measurements/hero-matrix-20260910.json).

The [hero layerwise report](../measurements/hero-layerwise-20260910.json)
found exact dense-block and same-input KDA attention outputs. The first mismatch
is in the first latent MoE block. A [Core-compatible HF MoE control](https://beaker.org/ex/01M271KNVXQ7N0EKK99MFHMHHX)
completed but did not eliminate the local MoE difference.
[Evidence](../measurements/hero-moe-control-20260910.json). Operator-level
diagnosis now separates native inference-only activation math from the actual
gradient-enabled training path; default execution and gates remain unchanged.

The [fixed-batch Core/Megatron comparison](../measurements/backend-fixed-policy-20260910.json)
starts from exactly matching model weights and FP32 masters. Both production
optimizers pass independent clipping/Adam checks. Across the tiny EP1 fixture,
preclip gradient relative L2 delta is 1.132% (cosine 0.999937); FP32 master-update
delta is 7.022% (cosine 0.997534). These are descriptive, with no cross-backend
acceptance threshold. Auxiliary objectives are disabled in this fixture and
remain a separate comparison.

The [full hero operator trace](../measurements/hero-moe-operators-20260910.json)
confirms that grad-enabled Core and the controlled HF path match exactly throughout
the first latent MoE block at both saved prefixes. The no-grad path first differs
at SwiGLU. Ordinary HF retains a separate packed-layout/reduction difference.
Core commit `290d2ca4521373bef0bf7fe4244673cc79dcc004` corrects the scoring
call to round the SiLU intermediate as eager training does, preserving the fused
kernel default for its explicit forward/backward users. Fourteen local GPU kernel
tests pass; the [full-checkpoint block rerun](../measurements/hero-scoring-rounding-20260910.json)
now matches grad-enabled Core and controlled HF exactly at every captured
branch/GEMM boundary for both prefixes. This fix does not change either active
100-update image. The old Core image contains the same shortcut; its full-model
impact on that run is not yet quantified.

[Full-model layerwise follow-up](https://beaker.org/ex/01M273R04XVRRQZA0HMRD2M5ZG)
is running after the scoring correction; ordinary HF/SGLang qualification remains
separate from the controlled-HF diagnostic.

[Full-SFT async four-update trial](https://beaker.org/ex/01M274MRNFSFF34QRYRGAASH52)
and [matching synchronous scheduling control](https://beaker.org/ex/01M274P7X8F52HTW6SNEPTRD4T)
are running from source `778a91cd9` with corrected Core scoring. Both use rollout
probabilities as the policy anchor, eager serving, 4096 response tokens, and no
evaluation/checkpoint saves. Async permits one step of lag and completion-order
prompt selection; audits require unique prepared prompts and complete groups.
These are scheduling checks, not a replacement 100-update learning comparison.
Twelve adversarial/config tests and both real pinned-parser checks passed.

The corrected hero layerwise control matches all KDA-only blocks on the same HF
input. Remaining first drift begins at full-attention block7. An explicit
[HF SDPA control](https://beaker.org/ex/01M274P8Z2D9A6FQ0BSHGBX6NK) is running to
separate attention implementation arithmetic; the original serving gate is not
being widened.

## September11 learning extension and qualification updates

The user authorized a historical lightly-SFT-trained Core comparison and a fresh
500-update pair on Abhishek's SFT checkpoint, plus full effective-configuration
and phase-by-phase performance analysis. The detailed plan is
[miles-learning-comparisons-20260911.md](../measurements/learning-comparisons-20260911.md).
Shared500 preparation passed on Saturn: experiment`01M277CCDPHQDBXBETS04X77YP`,
source`58cfd1a7f`, exit0. Core500 remains gated on full-sized native continuation;
Megatron500 configuration and aligned sampler/token-pool preflight are ready.

The controlled full-hero HF/Core comparison now matches final logits and
logprobs exactly at both16/81-token prefixes when using the same grouped-MoE
and SDPA paths. This isolates the remaining default-HF discrepancy to execution
arithmetic; it does not qualify ordinary SGLang probabilities or hero training.
[Evidence](../measurements/hero-controlled-forward-20260911.json).

Actual Core compiler-cache qualification failed: both tiny updates ran, but
Triton/Inductor artifacts containing private paths were rejected, so Triton
recompiled in both processes; scores/gradients/updates also differed. Cache reuse
remains opt-in and unqualified. [Evidence](../measurements/core-cache-trial-20260911.json).

The async four-update job completed training but its original audit incorrectly
required a single version across the entire batch. The runtime contract instead
requires homogeneous prompt groups and bounded lag per group. A corrected strict
per-group/rank-consumption audit is running on retained data on Saturn:
`01M277SGJZ0YMSZHK6DY2NWE4H`. The synchronous eager control hit its45-minute timeout
at`03:14:57UTC` (exit143); do not report it as a completed scheduling comparison.
Full native EP2 durable-continuation trial`01M276Y2J577FPD6G6JYMS9NV6` is running.
