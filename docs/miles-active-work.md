# Active MILES / Core work

This is the continuation ledger for the September10–11 session. Multiple Beaker
jobs are authorized. Run GPU trials on Holmes in `ai2/open-instruct-dev` at
urgent priority with a positive minimum runtime. CPU-only WEKA jobs belong on
Saturn. Preserve every attempt and keep the old SFT architecture separate from
hero. Do not publish this branch to the public open-instruct repository without
separate authorization.

| Track | State | Next evidence needed |
| --- | --- | --- |
| Old SFT, Core,100 GSM8K updates | [Running](https://beaker.org/ex/01M26P6XX6SN886DCVZ68WMQK2); heldout97/128→101/128→94/128 at0/20/40 | Finish100; audit every rollout, reward, policy version, publication and optimizer step. Notify the user when it finishes. |
| Same SFT, olmo-miles/Megatron,100 updates | [r3 running](https://beaker.org/ex/01M26YNP5E64YGNXR85TA2RP4Q); same shared data and recipe | Initial evaluation and full curve; audit `megatron-r3`, then compare both curves, truncation and measured cadence/allocation time. |
| Hero native/HF conversion | [step75500 passed](https://beaker.org/ex/01M26YP78T0JJ545H914AEYDDZ); both directions exact after export cast, 23,441 tensors | [Recorded conversion evidence](measurements/miles-hero-conversion-20260910.json); 579 seconds, 72 GiB peak RSS. |
| Hero serving |85 local tests and tiny Core/HF/SGLang parity passed; live gain/scale updates passed | [Full-checkpoint TP1 scoring submitted](https://beaker.org/ex/01M26ZBDKPDBRJ03TYT6V2GYRJ), including Core/HF/SGLang, mixed lengths, chunking and graphs. Then real training/EP/publication qualification. |
| Standard dense Olmo3 | Isolated standard trainer plus local full/sliding HF math and real update/resume passed | Full-checkpoint serving and a bounded training trial when prioritized; no claim of a full recipe run yet. |

The first Megatron attempt failed at initialization; r2 was stopped at the user's
request before training; the user then explicitly restored the paired comparison.
The active r3 recipe changes only run/output/W&B identities from r2. The repaired
router precision image is retained. Earlier failed/stopped attempts are not
learning curves or steady-state throughput measurements.

Hero implementation lives in `.worktrees/miles-hero-integration` on
`robertb/miles-hero-support`. Core changes are isolated in `miles-core-hero`;
serving changes are isolated in `olmo-sglang-hero`. The original Core100 image and
source remain unchanged. `runtime/miles/runtime.lock.json` records the hero pins.

## Other open work retained

The detailed acceptance sequence remains in
[miles-qualification-plan.md](miles-qualification-plan.md), and feature/default
coverage in [miles-feature-parity.md](miles-feature-parity.md):

- Mixed open-instruct datasource manifest and per-source policy-signal audit.
  The prior full-SFT math slice was almost entirely capped at4096 tokens, so
  longer-response qualification is needed before comparing math quality.
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
